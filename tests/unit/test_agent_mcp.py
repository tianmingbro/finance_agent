"""
test_agent_mcp.py
验证 Agent 通过 MCP 工具检索金融法规并生成回答。
"""
import os
import pytest
from langchain_core.messages import HumanMessage
from src.agent.agent import build_agent
from langgraph.checkpoint.memory import MemorySaver

requires_api = pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="请设置 DASHSCOPE_API_KEY"
)

@requires_api
@pytest.mark.asyncio                      # 标记为异步测试
async def test_agent_uses_mcp_tool():
    """提问资本充足率，Agent 应通过 MCP 调用检索工具并返回答案"""
    agent = await build_agent(checkpointer=MemorySaver())  # 等待异步函数
    config = {"configurable": {"thread_id": "test-mcp-1"}}

    query = "商业银行的核心一级资本充足率要求是多少？"
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config
    )
    messages = response["messages"]
    # 至少应有一条 AIMessage 包含最终答案
    from langchain_core.messages import AIMessage, ToolMessage
    assert any(isinstance(m, ToolMessage) for m in messages), "应有工具调用消息"
    final_answer = messages[-1].content
    assert isinstance(messages[-1], AIMessage)
    assert "不低于5%" in final_answer or "核心一级资本充足率" in final_answer