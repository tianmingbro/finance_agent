"""
check_context_direct.py
直接调用 work flow 内部逻辑，避免 LangGraph 上下文问题。
"""
import asyncio
import json
from src.retriever.tools_mcp import search_finance_docs
from langchain_openai import ChatOpenAI
import os

async def main():
    query = "核心一级资本充足率是多少？"
    
    # 1. 检索（直接调用 MCP 工具）
    retrieved_json = await search_finance_docs(query)
    retrieved = json.loads(retrieved_json)
    documents = retrieved.get("documents", [])
    print(f"✅ 检索到 {len(documents)} 篇文档")
    if documents:
        print(f"   第一条内容: {documents[0]['content'][:80]}...")
        context_list = [doc["content"] for doc in documents]
    else:
        context_list = []
        print("   ⚠️ 文档列表为空")
    
    # 2. 生成答案（直接调用 LLM）
    if documents:
        context = "\n".join([f"[{d['index']}] {d['content']}" for d in documents])
        llm = ChatOpenAI(
            model="qwen-plus",
            temperature=0,
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        prompt = f"你是一个金融法规专家。严格依据以下信息回答问题：\n{context}\n\n问题：{query}\n答案："
        answer = llm.invoke(prompt).content
    else:
        answer = "未找到相关法规。"
    
    print(f"🤖 生成答案: {answer[:100]}...")
    print(f"📚 context 长度: {len(context_list)}")
    if context_list:
        print(f"   第一条上下文: {context_list[0][:80]}...")

asyncio.run(main())