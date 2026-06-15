import logging
import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.a2a.a2a_types import Task, Message, TextPart, AgentCard, AgentSkill
from src.retriever.tools_mcp import search_finance_docs, warmup
from langchain_openai import ChatOpenAI
import json
import uvicorn
from src.agent.a2a.decorators import agent_task
logger = logging.getLogger(__name__)

app = FastAPI()
warmup()

class TaskRequest(BaseModel):
    task: Task

@app.post("/a2a/task")
@agent_task("rag_agent")
async def handle_task(req: TaskRequest):
    task = req.task
    if not task.messages:
        return task

    query = task.messages[-1].parts[0].text
    # 检索
    result_json = await search_finance_docs(query, top_k=4)
    docs = json.loads(result_json)["documents"]
    context = "\n".join([f"[{i+1}] {d['content']}" for i,d in enumerate(docs)])
    logger.info("RAG Agent 收到任务 session=%s, context=%s", task.session_id, task.context if task.context else "无")
    # 生成
    # llm = ChatOpenAI(
    #     model="qwen-plus",
    #     temperature=0,
    #     openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    llm = ChatOpenAI(
        model="qwen2.5:7b",                     # Ollama 中的模型名
        temperature=0,
        openai_api_key="ollama",                # 任意非空字符串
        openai_api_base="http://localhost:11434/v1",
        )
    prompt = f"根据以下文档回答问题：\n{context}\n\n问题：{query}\n答案："
    answer = llm.invoke(prompt).content

    task.artifacts = [{"answer": answer,
                        "sources": [d["source"] for d in docs]}]
    task.context = {
    "query": query,
    "answer": answer,
    "sources": [d["source"] for d in docs]
}
    task.status = "completed"
    return task

@app.get("/.well-known/agent-card")
def agent_card():
    return AgentCard(
        name="rag_agent",
        description="金融法规 RAG 代理，提供检索增强生成回答",
        url="http://localhost:8101",
        skills=[AgentSkill(id="rag_query", name="RAG问答")],
        capabilities={"streaming": False}
    ).to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)