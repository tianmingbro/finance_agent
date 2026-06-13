import time
import logging
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import asdict
from src.agent.a2a.a2a_types import Task, Message, TextPart, AgentCard, AgentSkill
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI()

# 其他 Agent 的地址
RAG_AGENT_URL = "http://localhost:8101"
EVAL_AGENT_URL = "http://localhost:8102"

class TaskRequest(BaseModel):
    task: Task

@app.post("/a2a/task")
async def handle_task(req: TaskRequest):
    task = req.task
    user_text = task.messages[-1].parts[0].text if task.messages else ""

    # 意图解析与目标 Agent 选择
    if "评测" in user_text:
        target_url = EVAL_AGENT_URL
        # 提取评测内容
        content = user_text.replace("评测", "", 1).strip()
        parts = content.split("|")
        if len(parts) != 2:
            task.status = "failed"
            task.artifacts = [{"error": f"评测请求格式：评测 问题|答案，当前输入：{user_text}"}]
            return task
        query = parts[0].strip()
        answer = parts[1].strip()
        # 构造发往下游的 Task
        downstream_payload = Task(
            id=task.id,
            session_id=task.session_id,
            messages=[Message(role="user", parts=[TextPart(text=f"{query}|{answer}")])]
        )
    else:
        target_url = RAG_AGENT_URL
        downstream_payload = Task(
            id=task.id,
            session_id=task.session_id,
            messages=[Message(role="user", parts=[TextPart(text=user_text)])]
        )

    # 调用下游 Agent（带耗时记录与异常处理）
    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{target_url}/a2a/task",
                json={"task": asdict(downstream_payload)}
            )
            resp.raise_for_status()  # 4xx/5xx 均会抛出异常
            task_data = resp.json()
            task = Task(**task_data)
            elapsed = time.perf_counter() - start_time
            logger.info("成功调用 %s，耗时 %.3f 秒", target_url, elapsed)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        elapsed = time.perf_counter() - start_time
        logger.error("连接 %s 失败（耗时 %.3f 秒）: %s", target_url, elapsed, e)
        task.status = "failed"
        task.artifacts = [{"error": f"Agent 服务不可用: {target_url}"}]
    except httpx.HTTPStatusError as e:
        elapsed = time.perf_counter() - start_time
        logger.error("调用 %s 返回错误状态 %d（耗时 %.3f 秒）: %s",
                     target_url, e.response.status_code, elapsed, e)
        task.status = "failed"
        task.artifacts = [{"error": f"下游服务错误（{e.response.status_code}）"}]
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.exception("调用 %s 时发生未知错误（耗时 %.3f 秒）", target_url, elapsed)
        task.status = "failed"
        task.artifacts = [{"error": f"内部错误: {str(e)}"}]

    return task

@app.get("/.well-known/agent-card")
def agent_card():
    return AgentCard(
        name="orchestrator",
        description="主控代理，根据意图分发任务给 RAG 或评测 Agent",
        url="http://localhost:8100",
        skills=[AgentSkill(id="dispatch", name="任务分发")],
        capabilities={"streaming": False}
    ).__dict__

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)