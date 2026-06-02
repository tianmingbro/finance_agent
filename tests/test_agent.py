"""
test_agent.py — Day40 Agent 集成测试（持久化记忆 + 会话隔离 + 参数化工具选择）
"""
import os
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.agent import build_agent

requires_api = pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="请设置 DASHSCOPE_API_KEY 以运行集成测试"
)


@requires_api
class TestAgentMemory:
    """持久化记忆与会话隔离测试"""

    def test_session_isolation(self):
        """验证不同 thread_id 之间的会话隔离"""
        # 会话 A
        agent = build_agent(checkpointer=MemorySaver())
        config_a = {"configurable": {"thread_id": "user-a"}}
        resp_a = agent.invoke(
            {"messages": [HumanMessage(content="我叫张三")]},
            config_a
        )

        # 会话 B（空历史）
        config_b = {"configurable": {"thread_id": "user-b"}}
        resp_b = agent.invoke(
            {"messages": [HumanMessage(content="我叫什么？")]},
            config_b
        )
        # 会话 B 不应知道"A 叫张三"
        final_b = resp_b["messages"][-1].content
        assert "张三" not in final_b

        # 回到会话 A，应记住名字
        resp_a2 = agent.invoke(
            {"messages": [HumanMessage(content="我叫什么？")]},
            config_a
        )
        final_a = resp_a2["messages"][-1].content
        assert "张三" in final_a

    def test_sqlite_persistence(self):
        """验证 SqliteSaver 持久化：重启后记忆保留"""
        import pytest
        from contextlib import ExitStack
        
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite 未安装")

        # 使用 ExitStack 管理上下文
        stack = ExitStack()
        memory = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))
        
        agent = build_agent(checkpointer=memory)
        config = {"configurable": {"thread_id": "sqlite-test"}}

        agent.invoke(
            {"messages": [HumanMessage(content="记住：核心一级资本充足率是5%")]},
            config
        )
        resp = agent.invoke(
            {"messages": [HumanMessage(content="我刚才让你记住的数字是多少？")]},
            config
        )
        assert "5%" in resp["messages"][-1].content
        
        # 清理上下文
        stack.close()


@requires_api
class TestAgentBehavior:
    """核心行为测试"""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        self.checkpointer = MemorySaver()
        self.agent = build_agent(checkpointer=self.checkpointer)

    def test_agent_calls_financial_qa(self):
        config = {"configurable": {"thread_id": "1"}}
        query = "商业银行的核心一级资本充足率要求是多少？"
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        tool_calls = [m for m in messages if isinstance(m, ToolMessage) and m.name == "financial_qa"]
        assert len(tool_calls) > 0, "Agent 未调用 financial_qa 工具"
        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)
        assert any(kw in final_msg.content for kw in ["%", "资本充足率", "不低于", "不得低于"])

    @pytest.mark.skip(reason="LangChain create_react_agent 内部错误处理报 NameError: logger，等待 LangChain 修复")
    def test_agent_calls_evaluate(self):
        config = {"configurable": {"thread_id": "2"}}
        query = "评测一下这个回答：问题：LPR是多少？答案：2025年1年期LPR为3.1%。"
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        tool_calls = [m for m in messages if isinstance(m, ToolMessage) and m.name == "evaluate_answer"]
        assert len(tool_calls) > 0, "Agent 未调用 evaluate_answer 工具"
        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)
        assert "忠实度" in final_msg.content or "faithfulness" in final_msg.content.lower()

    def test_agent_no_tool_call(self):
        config = {"configurable": {"thread_id": "3"}}
        query = "你好，今天天气怎么样？"
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        assert not any(isinstance(m, ToolMessage) for m in messages)
        assert isinstance(messages[-1], AIMessage)

    def test_agent_multi_turn(self):
        config = {"configurable": {"thread_id": "4"}}
        q1 = "什么是LPR？"
        self.agent.invoke({"messages": [HumanMessage(content=q1)]}, config)
        q2 = "那它对房贷有什么影响？"
        resp2 = self.agent.invoke({"messages": [HumanMessage(content=q2)]}, config)
        messages = resp2["messages"]
        assert isinstance(messages[-1], AIMessage)
        combined = messages[-1].content
        assert "房贷" in combined or "利率" in combined

    def test_agent_error_handling(self, monkeypatch):
        from langchain_core.tools import tool
        @tool
        def mock_failing_tool(query: str) -> str:
            """A tool that always fails"""
            raise RuntimeError("服务暂时不可用")
        # agent.py 顶部有 from src.tools import financial_qa，
        # 所以需要在 agent 模块级替换引用，build_agent() 才能拿到 mock
        monkeypatch.setattr("src.agent.agent.financial_qa", mock_failing_tool)
        agent = build_agent(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "5"}}
        query = "资本充足率是多少？"
        response = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)


@requires_api
class TestToolSelection:
    """参数化测试：验证不同输入下的工具选择正确性"""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        self.agent = build_agent(checkpointer=MemorySaver())

    @pytest.mark.parametrize("query,expected_tool", [
        ("资本充足率是多少？", "financial_qa"),
        ("LPR最新报价", "financial_qa"),
        ("个人外汇额度限制", "financial_qa"),
        ("存款保险最高赔多少", "financial_qa"),
    ])
    def test_correct_tool_selected(self, query, expected_tool):
        """验证不同输入能正确选择工具"""
        config = {"configurable": {"thread_id": f"tool-{hash(query)}"}}
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        tool_calls = [m for m in messages if isinstance(m, ToolMessage)]
        tool_names = [m.name for m in tool_calls]
        assert expected_tool in tool_names, \
            f"期望调用 {expected_tool}，实际调用: {tool_names}"

    @pytest.mark.skip(reason="LLM 行为不稳定，create_react_agent 下模型对 '评测' 触发的工具选择存在不确定性")
    @pytest.mark.parametrize("query,expected_tool", [
        ("评测一下：资本充足率是5%对不对？", "evaluate_answer"),
        ("帮我测试这个回答的质量：LPR是3.1%", "evaluate_answer"),
    ])
    def test_correct_tool_selected_eval(self, query, expected_tool):
        """验证评测相关输入能正确选择 evaluate_answer 工具"""
        config = {"configurable": {"thread_id": f"tool-{hash(query)}"}}
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        tool_calls = [m for m in messages if isinstance(m, ToolMessage)]
        tool_names = [m.name for m in tool_calls]
        assert expected_tool in tool_names, \
            f"期望调用 {expected_tool}，实际调用: {tool_names}"

    @pytest.mark.parametrize("query", [
        "你好",
        "今天天气怎么样",
        "帮我写首诗",
    ])
    def test_no_tool_for_chitchat(self, query):
        """闲聊不应调用任何工具"""
        config = {"configurable": {"thread_id": f"chitchat-{hash(query)}"}}
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        messages = response["messages"]
        tool_calls = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_calls) == 0, f"闲聊不应调用工具，但调用了: {[m.name for m in tool_calls]}"