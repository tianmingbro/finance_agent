"""
test_mcp_tools.py
MCP 工具测试：验证 search_finance_docs 的功能、参数校验与工具注册。
需要先执行 warmup 预热向量库，避免首次调用超时。
"""
"""
test_mcp_tools.py
通过启动子进程的方式测试 MCP 工具功能与注册。
需要先执行 warmup 预热向量库。
"""
import json
import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from src.retriever.tools_mcp import search_finance_docs, warmup,list_available_topics
import sys

from mcp import StdioServerParameters
from mcp.client.session import ClientSession

# 模块级预热
warmup()

@pytest.mark.asyncio
async def test_search_finance_docs_returns_results():
    result = await search_finance_docs("资本充足率", top_k=2)
    data = json.loads(result)
    assert "documents" in data
    assert "sources" in data
    assert isinstance(data["documents"], list)
    assert len(data["documents"]) > 0
    # 确认 sources 列表包含已知来源
    assert "capital_management_measures_2024.txt" in data["sources"]

@pytest.mark.asyncio
async def test_top_k_zero_returns_warning():
    result = await search_finance_docs("测试", top_k=0)
    assert "必须大于 0" in result


@pytest.mark.asyncio
async def test_tool_list_via_mcp():
    """通过启动 MCP 服务器子进程，使用 list_tools 验证工具已注册"""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_finance_server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 获取结果对象，再取 tools 列表
            result = await session.list_tools()
            tools = result.tools
            tool_names = [t.name for t in tools]
            assert "search_finance_docs_tool" in tool_names

@pytest.mark.asyncio
async def test_search_returns_valid_json():
    result = await search_finance_docs("资本充足率", top_k=2)
    data = json.loads(result)
    assert "documents" in data
    assert "sources" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["sources"], list)
    if data["documents"]:
        doc = data["documents"][0]
        assert "content" in doc
        assert "source" in doc
        assert "index" in doc

@pytest.mark.asyncio
async def test_search_top_k_zero_returns_error():
    result = await search_finance_docs("测试", top_k=0)
    data = json.loads(result)
    assert "error" in data
    assert "必须大于 0" in data["error"]

@pytest.mark.asyncio
async def test_list_available_topics():
    result = await list_available_topics()
    data = json.loads(result)
    assert "topics" in data
    assert isinstance(data["topics"], list)
    assert len(data["topics"]) > 0
    # 常见主题应至少包含一个
    assert any(t in data["topics"] for t in ["资本充足率", "存款保险", "外汇管理", "LPR"])

@pytest.mark.asyncio
async def test_search_result_contains_sources():
    result = await search_finance_docs("个人购汇额度")
    data = json.loads(result)
    if data["documents"]:
        # sources 列表应与 documents 中的 source 一致
        sources_from_docs = {doc["source"] for doc in data["documents"]}
        sources_list = set(data["sources"])
        assert sources_from_docs.issubset(sources_list)