#!/usr/bin/env python3
"""
mcp_finance_server.py
金融文档检索 MCP 服务器。
将 search_finance_docs 工具通过 stdio 传输暴露给 MCP 客户端。
"""
import asyncio
from mcp.server.fastmcp import FastMCP
from src.retriever.tools_mcp import search_finance_docs, warmup,list_available_topics

# 创建 MCP 服务器实例
mcp = FastMCP("FinanceSearchServer")

@mcp.tool()
async def search_finance_docs_tool(query: str, top_k: int = 4) -> str:
    """
    检索金融法规相关文档。
    输入：
      - query: 查询文本
      - top_k: 返回的文档数量，默认 4
    输出：格式化字符串，包含检索到的文档摘要及其来源。
    """
    return await search_finance_docs(query, top_k)

@mcp.tool()
async def list_available_topics_tool() -> str:
    """列出知识库覆盖的所有主题"""
    return await list_available_topics()

if __name__ == "__main__":
    # 预热检索器，避免首次调用超时
    warmup()
    mcp.run(transport="stdio")