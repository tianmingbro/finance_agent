import asyncio
import time
import logging
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.a2a.a2a_types import Task, Message, TextPart, AgentCard, AgentSkill
import uvicorn
from dataclasses import asdict

app = FastAPI()
logger = logging.getLogger(__name__)

RAG_AGENT_URL = "http://localhost:8101"
EVAL_AGENT_URL = "http://localhost:8102"

# 🆕 简单的内存会话存储，带 TTL（300 秒过期）
session_store: dict = {}
SESSION_TTL = 300  # 秒

def get_session(session_id: str) -> dict:
    """获取会话上下文，若过期则清除并返回空字典"""
    if session_id in session_store:
        entry = session_store[session_id]
        if time.time() - entry["timestamp"] < SESSION_TTL:
            logger.debug("命中会话 %s, context keys: %s", session_id, list(entry["data"].keys()))
            return entry["data"]
        else:
            del session_store[session_id]
            logger.info("会话 %s 已过期并被清理", session_id)
    return {}

def set_session(session_id: str, data: dict):
    """保存会话上下文"""
    summary = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v) for k, v in data.items()}
    session_store[session_id] = {"data": data, "timestamp": time.time()}
    logger.debug("更新会话 %s, context: %s", session_id, summary)

# 后台定期清理过期会话
async def cleanup_expired_sessions():
    while True:
        await asyncio.sleep(60)  # 每分钟执行一次
        now = time.time()
        expired = [sid for sid, entry in session_store.items()
                   if now - entry["timestamp"] >= SESSION_TTL]
        for sid in expired:
            del session_store[sid]
            logger.debug("定期清理过期会话 %s", sid)
        if expired:
            logger.info("清理了 %d 个过期会话", len(expired))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：创建后台清理任务
    task = asyncio.create_task(cleanup_expired_sessions())
    yield
    # 关闭时：取消任务并清理
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

class TaskRequest(BaseModel):
    task: Task

@app.post("/a2a/task")
async def handle_task(req: TaskRequest):
    task = req.task
    user_text = task.messages[-1].parts[0].text if task.messages else ""

    # 意图解析
    if "评测" in user_text:
        # 1. 优先使用请求中自带的 context
        if task.context and task.context.get("query") and task.context.get("answer"):
            query = task.context["query"]
            answer = task.context["answer"]
        else:
            # 提取可能的手动输入（格式：评测 问题|答案）
            content = user_text.replace("评测", "", 1).strip()
            parts = content.split("|")
            if len(parts) == 2:
                query, answer = parts[0].strip(), parts[1].strip()
            else:
                # 没有手动提供，从会话存储中获取上一次问答
                session_data = get_session(task.session_id)
                query = session_data.get("query", "")
                answer = session_data.get("answer", "")
                if not query or not answer:
                    task.status = "failed"
                    task.artifacts = [{"error": "没有找到上一次对话，请先提问或手动输入 评测 问题|答案"}]
                    return task

        # 构造 context 并调用 Eval Agent
        # context = {"query": query, "answer": answer}
        payload = Task(
            id=task.id,
            session_id=task.session_id,
            messages=[Message(role="user", parts=[TextPart(text=f"{query}|{answer}")])],
            context=task.context,
        )
        target_url = EVAL_AGENT_URL
    else:
        # 问答请求
        payload = Task(
            id=task.id,
            session_id=task.session_id,
            messages=[Message(role="user", parts=[TextPart(text=user_text)])],
            context=task.context  # 透传上游 context（如果有）
        )
        target_url = RAG_AGENT_URL

    logger.info("发送任务到 %s, session=%s, context keys=%s", target_url, task.session_id, list(task.context.keys()))
    # 调用下游 Agent
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{target_url}/a2a/task", json={"task": asdict(payload)})
            resp.raise_for_status()
            task = Task(**resp.json())
        logger.info("调用 %s 成功，耗时 %.3fs", target_url, time.perf_counter()-start)
    except Exception as e:
        logger.error("调用 %s 失败: %s", target_url, e)
        task.status = "failed"
        task.artifacts = [{"error": f"服务不可用: {str(e)}"}]
        return task

    # 如果调用的是 RAG Agent 且成功，将结果存入 session
    if target_url == RAG_AGENT_URL and task.status == "completed":
        ctx = task.context if task.context else {}
        set_session(task.session_id, {
            "query": ctx.get("query", user_text),
            "answer": ctx.get("answer", ""),
            "sources": ctx.get("sources", [])
        })

    return task

@app.get("/.well-known/agent-card")
def agent_card():
    return AgentCard(
        name="orchestrator",
        description="主控代理，根据意图分发任务，支持上下文透传",
        url="http://localhost:8100",
        skills=[AgentSkill(id="dispatch", name="任务分发")],
        capabilities={"streaming": False}
    ).to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)