import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from src.tools import rag_workflow_query, evaluate_answer      # 使用本地工具
from langchain_openai import ChatOpenAI

requires_api = pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="请设置 DASHSCOPE_API_KEY"
)

@pytest.mark.asyncio
@requires_api
async def test_agent_uses_workflow_tool():
    """Agent 使用工作流工具回答金融问题（本地工具，绕过 MCP 兼容性问题）"""
    # model = ChatOpenAI(
    #     model="qwen-plus",
    #     temperature=0,
    #     openai_api_key=os.environ["DASHSCOPE_API_KEY"],
    #     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    model = ChatOpenAI(
    model="qwen2.5:7b",                     # Ollama 中的模型名
    temperature=0,
    openai_api_key="ollama",                # 任意非空字符串
    openai_api_base="http://localhost:11434/v1",
)
    agent = create_agent(
        model=model,
        tools=[rag_workflow_query, evaluate_answer],
        system_prompt=(
            "你是专业的金融法规助手。"
            "当用户询问金融法规问题时，使用 rag_workflow_query 工具获取准确答案。"
            "当用户要求评测某段回答的质量时，使用 evaluate_answer 工具进行评估。"
            "对于无关闲聊，请直接友好回复，不要调用工具。"
        ),
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "e2e-test-local"}}
    query = "商业银行的核心一级资本充足率要求是多少？"
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config
    )
    messages = response["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) > 0, "应有工具调用消息"
    final_msg = messages[-1]
    assert isinstance(final_msg, AIMessage)
    assert "不低于5%" in final_msg.content or "核心一级资本充足率" in final_msg.content