import logging
import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.a2a.a2a_types import Task, Message, TextPart, AgentCard, AgentSkill
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager
import json
import uvicorn
from src.agent.a2a.decorators import agent_task
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)

app = FastAPI()
Instrumentator().instrument(app).expose(app)
_eval_runner = None

def get_runner():
    global _eval_runner
    if _eval_runner is None:
        mgr = EvalResourceManager()
        mgr.load_resources()
        _eval_runner = EvaluationRunner(mgr)
    return _eval_runner

class TaskRequest(BaseModel):
    task: Task

@app.post("/a2a/task")
@agent_task("eval_agent")
async def handle_task(req: TaskRequest):
    task = req.task
    # 🆕 优先从 context 获取
    ctx = task.context if task.context else {}
    query = ctx.get("query", "")
    answer = ctx.get("answer", "")
    retrieval_context = ctx.get("retrieval_context", [])   # 🆕 获取检索上下文
    # 若 context 缺失，回退到消息文本解析
    if not query or not answer:
        text = task.messages[-1].parts[0].text if task.messages else ""
        parts = text.split("|", 1)
        if len(parts) == 2:
            query = parts[0].strip()
            answer = parts[1].strip()
        else:
            raise ValueError("需要提供 query 和 answer（通过 context 或 '问题|答案' 格式）")
    logger.info("Eval Agent 收到任务 session=%s, context=%s", task.session_id, task.context if task.context else "无")
    runner = get_runner()
    def rag_callable(q):
        return {"input": query, "answer": answer, "context": retrieval_context}
    # report = runner.run(f"评测：{query}", rag_callable)
    report = await runner.async_run(f"评测：{query}", rag_callable)
    faith = next((m.score for m in report.metrics if m.name=="faithfulness"), 0.0)
    rel = next((m.score for m in report.metrics if m.name=="answer_relevancy"), 0.0)

    task.artifacts = [{"faithfulness": faith, "answer_relevancy": rel,
                       "query": query, "answer": answer}]
    return task

@app.get("/.well-known/agent-card")
def agent_card():
    return AgentCard(
        name="eval_agent",
        description="评测代理，评估答案的忠实度和相关性",
        url="http://localhost:8102",
        skills=[AgentSkill(id="evaluate", name="评测回答")],
        capabilities={"streaming": False}
    ).to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8102)