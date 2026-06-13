import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.a2a.a2a_types import Task, Message, TextPart, AgentCard, AgentSkill
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager
import json
import uvicorn
from src.agent.a2a.decorators import agent_task

app = FastAPI()

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
    text = task.messages[-1].parts[0].text if task.messages else ""
    # 期望格式 "query|answer"
    parts = text.split("|", 1)
    if len(parts) != 2:
        task.status = "failed"
        task.artifacts = [{"error": "需要 query|answer 格式"}]
        return task

    query, answer = parts[0], parts[1]
    runner = get_runner()

    def rag_callable(q):
        return {"input": query, "answer": answer, "context": []}

    report = runner.run(f"评测：{query}", rag_callable)
    faith = next((m.score for m in report.metrics if m.name=="faithfulness"), 0.0)
    rel = next((m.score for m in report.metrics if m.name=="answer_relevancy"), 0.0)

    task.artifacts = [{"faithfulness": faith, "answer_relevancy": rel}]
    task.status = "completed"
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