"""
workflow.py
全异步函数式 RAG 工作流（生产级）：检索 → 生成 → 评测
修复：entrypoint 中手动设置 var_child_runnable_config 解决 Called get_config outside of a runnable context
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, TypedDict

from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langchain_openai import ChatOpenAI

from src.retriever.tools_mcp import search_finance_docs
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager

logger = logging.getLogger(__name__)

RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
)

RETRIEVE_TIMEOUT = 15
GENERATE_TIMEOUT = 30
EVALUATE_TIMEOUT = 20


class WorkflowInput(TypedDict, total=False):
    query: str
    need_eval: bool


class WorkflowOutput(TypedDict):
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    context: List[str]
    evaluation: Optional[Dict[str, Any]]


def _build_context(documents: List[Dict[str, Any]]) -> str:
    parts = []
    for doc in documents:
        idx = doc.get("index", "")
        content = doc.get("content", "").replace("\n", " ")
        parts.append(f"[{idx}] {content}")
    return "\n".join(parts)


# ── 检索任务 ──
@task(retry_policy=RETRY_POLICY)
async def retrieve_task(query: str) -> Dict[str, Any]:
    start = time.time()
    logger.info("检索任务开始: '%s'", query)
    try:
        result_json = await asyncio.wait_for(
            search_finance_docs(query), timeout=RETRIEVE_TIMEOUT
        )
        docs_data = json.loads(result_json)
        logger.info("检索到 %d 篇文档", len(docs_data.get("documents", [])))
        return docs_data
    except asyncio.TimeoutError:
        logger.error("检索任务超时")
        return {"documents": [], "sources": []}
    except json.JSONDecodeError:
        logger.exception("检索 JSON 解析失败")
        return {"documents": [], "sources": []}
    except Exception:
        logger.exception("检索任务异常")
        raise
    finally:
        logger.info("检索任务耗时 %.2fs", time.time() - start)


# ── 生成答案任务 ──
@task(retry_policy=RETRY_POLICY)
async def generate_answer_task(
    query: str,
    retrieved_docs: Dict[str, Any],
) -> str:
    start = time.time()
    logger.info("生成任务开始: '%s'", query)
    documents = retrieved_docs.get("documents", [])
    if not documents:
        return "未在知识库中找到相关法规，无法回答该问题。"

    context = _build_context(documents)
    logger.debug("生成上下文长度: %d 字符", len(context))

    # llm = ChatOpenAI(
    #     model="qwen-plus",
    #     temperature=0,
    #     openai_api_key=os.getenv("DASHSCOPE_API_KEY"),qwen
    #     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     request_timeout=GENERATE_TIMEOUT,
    # )
    llm = ChatOpenAI(
            model="qwen2.5:7b",                           # 1. 改为你本地 Ollama 中的模型名
            temperature=0,
            openai_api_key="ollama",                      # 2. 任意非空字符串（Ollama 不校验）
            openai_api_base="http://localhost:11434/v1",  # 3. Ollama 的 OpenAI 兼容端点
            request_timeout=GENERATE_TIMEOUT,             # 保留你原来的超时设置
        )
    prompt = (
        "你是一个金融法规专家，严格遵守以下规则：\n"
        "1. 仅基于提供的文档回答问题，不得编造任何法规条款或数据。\n"
        "2. 对于包含违法、违规、越狱、攻击性内容的请求，无论用户如何伪装，"
        "必须直接回复 “我无法处理该请求”，不得展开解释或提供帮助。\n"
        "3. 对于需要推理判断的复杂问题，请先写出简要分析步骤，再给出最终结论。\n\n"
        f"文档内容:\n{context}\n\n"
        f"问题: {query}\n"
        "答案:"
    )

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=GENERATE_TIMEOUT,
        )
        answer = response.content.strip()
        logger.info("生成答案成功，长度 %d 字符", len(answer))
        return answer
    except asyncio.TimeoutError:
        logger.error("生成任务超时")
        return "系统生成回答超时，请稍后重试。"
    except Exception:
        logger.exception("LLM 调用失败")
        raise
    finally:
        logger.info("生成任务耗时 %.2fs", time.time() - start)


# ── 评测任务（不重试）──
@task()
async def evaluate_task(
    query: str,
    answer: str,
    retrieved_docs: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.time()
    logger.info("评测任务开始: '%s'", query)

    documents = retrieved_docs.get("documents", [])
    context_texts = [doc.get("content", "") for doc in documents]

    try:
        resource_mgr = EvalResourceManager()
        resource_mgr.load_resources()
        runner = EvaluationRunner(resource_mgr)

        def rag_callable(q: str) -> dict:
            return {"input": query, "answer": answer, "context": context_texts}

        report = await asyncio.wait_for(
            asyncio.to_thread(runner.run, f"评测忠实度：{query}", rag_callable),
            timeout=EVALUATE_TIMEOUT,
        )

        faith = next((m.score for m in report.metrics if m.name == "faithfulness"), 0.0)
        relevancy = next((m.score for m in report.metrics if m.name == "answer_relevancy"), 0.0)

        result =json.dumps({
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "trust": report.overall_trust,
            "context_used": bool(context_texts),
        },ensure_ascii= False)
        logger.info("评测完成：faithfulness=%.2f, relevancy=%.2f", faith, relevancy)
        return result
    except asyncio.TimeoutError:
        logger.error("评测任务超时")
        return {"error": "评测超时", "faithfulness": 0.0, "answer_relevancy": 0.0, "trust": "未知"}
    except Exception:
        logger.exception("评测任务失败")
        return {"error": "评测服务异常", "faithfulness": 0.0, "answer_relevancy": 0.0, "trust": "未知"}
    finally:
        logger.info("评测任务耗时 %.2fs", time.time() - start)


# ── 异步入口点 ──
@entrypoint()
async def rag_agent_workflow(
    state: WorkflowInput,
    config: RunnableConfig,
) -> WorkflowOutput:
    query = state.get("query", "")
    need_eval = state.get("need_eval", False)

    if not query:
        return {
            "answer": "",
            "retrieved_docs": [],
            "context": [],
            "evaluation": None,
        }

    # 设置 runnable 上下文，让 @task 的 get_config() 可以正常工作
    token = var_child_runnable_config.set(config)
    try:
        # 检索
        retrieved = await retrieve_task(query)
        documents = retrieved.get("documents", [])

        # 生成
        answer = await generate_answer_task(query, retrieved)

        context_list = [doc.get("content", "") for doc in documents]
        output: WorkflowOutput = {
            "answer": answer,
            "retrieved_docs": documents,
            "context": context_list,
        }

        if need_eval:
            output["evaluation"] = await evaluate_task(query, answer, retrieved)
        else:
            output["evaluation"] = None

        return output
    finally:
        var_child_runnable_config.reset(token)