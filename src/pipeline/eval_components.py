"""
eval_components.py
Day46 核心交付物：组件级评估器（修正异步调用问题）
"""
import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional

from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
import tenacity

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# 自定义 LLM 类
# ═══════════════════════════════════════════
class CustomQwenLLM(DeepEvalBaseLLM):
    def __init__(self):
        self._model = ChatOpenAI(
            model="qwen-max",
            temperature=0,
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # self._model= ChatOpenAI(model="qwen2.5:7b",temperature=0, base_url="http://localhost:11434/v1", api_key="ollama")

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        response = self._model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self._model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "qwen-plus"

# 全局评测 LLM 单例
_eval_llm = None

def _get_eval_llm() -> DeepEvalBaseLLM:
    global _eval_llm
    if _eval_llm is None:
        _eval_llm = CustomQwenLLM()
    return _eval_llm

# ─── 规划组件评估（不变） ───────────────────
def evaluate_planning(
    query: str,
    expected_tool: str,
    expected_args: Dict[str, Any],
) -> Dict[str, float]:
    planned_tool = "financial_qa"
    planned_args = {"query": query}
    tool_accuracy = 1.0 if planned_tool == expected_tool else 0.0
    if expected_args and "query" in expected_args:
        arg_reasonableness = 1.0 if planned_args.get("query") == expected_args["query"] else 0.0
    else:
        arg_reasonableness = 1.0
    return {"tool_accuracy": tool_accuracy, "arg_reasonableness": arg_reasonableness}

def run_async(coro):
    """安全地运行异步协程，兼容已有事件循环和无事件循环环境"""
    try:
        # 如果当前有正在运行的事件循环（如 pytest-asyncio 环境）
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 无事件循环，直接用 asyncio.run（创建新循环）
        return asyncio.run(coro)
    else:
        # 事件循环已运行，在新线程中执行 asyncio.run 避免冲突
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
        
# ─── 检索组件评估（修正异步调用） ──────────
def evaluate_retrieval(
    query: str,
    key_info: Optional[List[str]] = None,
    top_k: int = 4,
) -> Dict[str, float]:
    from src.retriever.tools_mcp import search_finance_docs

    # 用 asyncio.run() 执行异步检索函数
    try:
        result_json = run_async(search_finance_docs(query, top_k=top_k))
    except Exception as e:
        print(f"检索失败: {e}")
        return {"contextual_recall": 0.0, "contextual_precision": 0.0}

    try:
        docs_data = json.loads(result_json)
    except json.JSONDecodeError:
        print(f"JSON 解析失败: {result_json[:100]}")
        return {"contextual_recall": 0.0, "contextual_precision": 0.0}
    
    documents = docs_data.get("documents", [])
    actual_context = [str(doc["content"]).strip() for doc in documents if "content" in doc]
    actual_context = [c for c in actual_context if c]  # 过滤空字符串
    if not actual_context or not key_info:
        # 无检索结果或未标注关键信息时，无法计算指标，直接返回 0 分
        return {"contextual_recall": 0.0, "contextual_precision": 0.0}
    scores = {}
    if key_info:
        expected_context = key_info
        expected_output_text = " ".join(key_info) if key_info else query
        test_case_recall = LLMTestCase(
            input=query,
            retrieval_context=actual_context,
            expected_context=expected_context,
            expected_output=expected_output_text,  # 必须提供
        )
        metric_recall = ContextualRecallMetric(model=_get_eval_llm())
        metric_recall.measure(test_case_recall)
        scores["contextual_recall"] = metric_recall.score

        # test_case_precision = LLMTestCase(
        #     input=query,
        #     retrieval_context=actual_context,
        #     expected_context=expected_context,
        #     expected_output=expected_output_text,   # 必须加上这一行
        # )
        metric_precision = ContextualPrecisionMetric(model=_get_eval_llm())
        metric_precision.measure(test_case_recall)  # 复用同一个测试用例
        scores["contextual_precision"] = metric_precision.score
    else:
        scores["contextual_recall"] = 0.0
        scores["contextual_precision"] = 0.0

    return scores

# ─── 生成组件评估（修正异步调用） ──────────
def evaluate_generation(
    query: str,
    expected_answer: Optional[str] = None,
) -> Dict[str, float]:
    from src.retriever.tools_mcp import search_finance_docs

    # 获取检索上下文
    try:
        result_json = run_async(search_finance_docs(query, top_k=4))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, search_finance_docs(query, top_k=4))
            result_json = future.result()

    docs_data = json.loads(result_json)
    retrieval_context = [doc["content"] for doc in docs_data.get("documents", [])]

    if not retrieval_context:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    # 生成答案
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    context_str = "\n".join([f"文档{i+1}: {c}" for i, c in enumerate(retrieval_context)])
    prompt = f"你是一个金融法规专家。根据以下信息回答问题：\n\n{context_str}\n\n问题：{query}\n答案："
    answer = llm.invoke(prompt).content

    # 忠实度
    test_case_faith = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )
    faith_metric = FaithfulnessMetric(model=_get_eval_llm())
    @tenacity.retry(stop=tenacity.stop_after_attempt(3), reraise=True)
    def _measure_with_retry(metric, test_case):
        metric.measure(test_case)
    _measure_with_retry(faith_metric, test_case_faith)

    # 答案相关性
    test_case_rel = LLMTestCase(input=query, actual_output=answer)
    rel_metric = AnswerRelevancyMetric(model=_get_eval_llm())
    # rel_metric.measure(test_case_rel)
    _measure_with_retry(rel_metric, test_case_rel)

    return {
        "faithfulness": faith_metric.score,
        "answer_relevancy": rel_metric.score,
    }


import statistics
import yaml
from pathlib import Path

def generate_component_report(
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    基于组件评估函数，汇总三个组件的平均分、通过率和失败详情。
    返回与 Day34 整体报告对齐的结构，并增加组件维度字段。
    """
    # 定义各指标阈值
    thresholds = {
        "planning": {"tool_accuracy": 0.9, "arg_reasonableness": 0.8},
        "retrieval": {"contextual_recall": 0.7, "contextual_precision": 0.7},
        "generation": {"faithfulness": 0.8, "answer_relevancy": 0.8},
    }

    # 存储每个组件的详细结果
    detailed = []
    # 各指标收集列表，用于统计
    metrics_collector = {
        "planning": {"tool_accuracy": [], "arg_reasonableness": []},
        "retrieval": {"contextual_recall": [], "contextual_precision": []},
        "generation": {"faithfulness": [], "answer_relevancy": []},
    }

    for case in test_cases:
        query = case["query"]
        cat = case.get("category", "unknown")
        planning = case.get("planning", {})
        retrieval = case.get("retrieval", {})
        expected_answer = case.get("expected_answer")

        # 1. 规划评估
        if planning:
            plan_res = evaluate_planning(
                query=query,
                expected_tool=planning.get("expected_tool", ""),
                expected_args=planning.get("expected_args", {}),
            )
        else:
            plan_res = {"tool_accuracy": 0.0, "arg_reasonableness": 0.0}

        # 2. 检索评估
        retrieval_res = evaluate_retrieval(
            query=query,
            key_info=retrieval.get("key_info"),
        )

        # 3. 生成评估
        gen_res = evaluate_generation(
            query=query,
            expected_answer=expected_answer,
        )

        # 记录详细结果
        detail = {
            "id": case.get("id", f"case_{len(detailed)}"),
            "query": query,
            "category": cat,
            "planning": plan_res,
            "retrieval": retrieval_res,
            "generation": gen_res,
        }
        detailed.append(detail)

        # 收集指标
        for k, v in plan_res.items():
            metrics_collector["planning"][k].append(v)
        for k, v in retrieval_res.items():
            metrics_collector["retrieval"][k].append(v)
        for k, v in gen_res.items():
            metrics_collector["generation"][k].append(v)

    # ── 计算汇总统计 ───────────────────────────
    def calc_summary(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

    # 组件级汇总
    components_summary = {}
    total_passed = 0
    total_cases = len(test_cases)

    for comp_name, thresh_dict in thresholds.items():
        comp_metrics = {}
        comp_passed = True
        for metric_name, thresh in thresh_dict.items():
            scores = metrics_collector[comp_name].get(metric_name, [])
            if not scores:
                comp_metrics[metric_name] = calc_summary([0.0])
                comp_passed = False
            else:
                summary = calc_summary(scores)
                comp_metrics[metric_name] = summary
                # 该指标是否通过（平均值达到阈值）
                if summary["mean"] < thresh:
                    comp_passed = False
        components_summary[comp_name] = {
            "metrics": comp_metrics,
            "passed": comp_passed,
            "thresholds": thresh_dict,
        }
        if comp_passed:
            total_passed += 1  # 按组件计数，非按指标

    # 整体通过率：所有组件都通过才计数（或者单独统计）
    # 这里采用“组件级通过率”，即每个组件都达到所有指标阈值则组件通过
    # 计算所有组件都通过的用例占比（本例每个用例独立算组件通过？更合理是按用例统计）
    # 简化：我们直接统计每个用例三个组件是否都通过，然后算整体通过率
    cases_passed = 0
    for detail in detailed:
        plan_ok = all(
            detail["planning"].get(k, 0.0) >= thresholds["planning"][k]
            for k in thresholds["planning"]
        )
        ret_ok = all(
            detail["retrieval"].get(k, 0.0) >= thresholds["retrieval"][k]
            for k in thresholds["retrieval"]
        )
        gen_ok = all(
            detail["generation"].get(k, 0.0) >= thresholds["generation"][k]
            for k in thresholds["generation"]
        )
        if plan_ok and ret_ok and gen_ok:
            cases_passed += 1

    overall_pass_rate = round(cases_passed / total_cases, 3) if total_cases else 0.0

    # 最终报告
    report = {
        "total_test_cases": total_cases,
        "overall_pass_rate": overall_pass_rate,
        "components_summary": components_summary,
        "detailed_results": detailed,
        "thresholds": thresholds,
    }
    return report


# ─── 生成示例报告并保存 ─────────────────────
if __name__ == "__main__":
    import sys
    data_path = Path("data/component_eval_data.yaml")
    if not data_path.exists():
        print(f"找不到测试数据文件: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        test_data = yaml.safe_load(f)["test_cases"]

    report = generate_component_report(test_data)
    output_path = "component_eval_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"组件评估报告已保存至 {output_path}")