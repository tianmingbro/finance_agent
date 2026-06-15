"""
test_a2a_context.py
Day51 TDAD 第一步：验证跨 Agent 上下文透传
依赖：pytest, httpx, respx, fastapi
"""
import pytest
import json
from httpx import Response
from fastapi.testclient import TestClient

# 导入应用实例
from src.agent.a2a.orchestrator_agent import app as orch_app
from src.agent.a2a.a2a_types import Task, Message, TextPart

import time
import src.agent.a2a.orchestrator_agent as orch
from src.agent.a2a.orchestrator_agent import app as orch_app

# ---------- 工具函数 ----------
def make_orchestrator_task(text: str, session_id: str = "test", context: dict = None) -> dict:
    """构造发给 Orchestrator 的任务请求"""
    task = Task(
        id="test-1",
        session_id=session_id,
        messages=[Message(role="user", parts=[TextPart(text=text)])],
        context=context or {}
    )
    return {"task": task.to_dict()}

# ---------- 测试类 ----------
class TestContextPassing:
    @pytest.fixture(autouse=True)
    def setup(self, respx_mock):
        self.client = TestClient(orch_app)
        # 模拟 RAG Agent 返回的答案
        def mock_rag(request):
            task_data = json.loads(request.content)["task"]
            query = task_data["messages"][-1]["parts"][0]["text"]
            return Response(200, json={
                "id": task_data["id"],
                "session_id": task_data["session_id"],
                "status": "completed",
                "artifacts": [{"answer": f"关于'{query}'的法规答案"}],
                "context": {                         # 🆕 RAG 返回上下文
                    "query": query,
                    "answer": f"关于'{query}'的法规答案",
                    "sources": ["capital.txt"]
                }
            })
        respx_mock.post("http://localhost:8101/a2a/task").mock(side_effect=mock_rag)

        # 模拟 Eval Agent 读取 context 并评测
        def mock_eval(request):
            task_data = json.loads(request.content)["task"]
            ctx = task_data.get("context", {})
            # 优先从 context 中获取 query 和 answer
            query = ctx.get("query") or task_data["messages"][-1]["parts"][0]["text"]
            answer = ctx.get("answer", "")
            return Response(200, json={
                "id": task_data["id"],
                "session_id": task_data["session_id"],
                "status": "completed",
                "artifacts": [{"faithfulness": 0.95, "answer_relevancy": 0.92,
                               "query": query, "answer": answer}]
            })
        respx_mock.post("http://localhost:8102/a2a/task").mock(side_effect=mock_eval)

    def test_context_passing_rag_to_eval(self):
        resp1 = self.client.post("/a2a/task", json=make_orchestrator_task("资本充足率是多少？", session_id="ctx1"))
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "completed"

        resp2 = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="ctx1"))
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["status"] == "completed"
        # 验证评测结果中包含了原始问题（可能以 "query|answer" 形式存在）
        assert "资本充足率是多少？" in data["artifacts"][0]["query"]
        # 同时应包含评测分数
        assert "faithfulness" in data["artifacts"][0]
        
    def test_orchestrator_session_store(self):
        """同一 session_id 的两次请求可获取之前存储的上下文"""
        session = "store-test"
        # 先问答
        self.client.post("/a2a/task", json=make_orchestrator_task("LPR是什么？", session_id=session))
        # 然后评测（不带具体答案）
        resp = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id=session))
        assert resp.status_code == 200
        # 应能正确评测上次的问答内容
        art = resp.json()["artifacts"][0]
        assert "LPR" in art["query"]

    def test_eval_agent_reads_context(self):
        """Eval Agent 能正确解析 context 中的 query 和 answer 执行评测"""
        # 直接调用 Orchestrator 的评测路由，但我们验证 mock 的 Eval Agent 读取了 context
        # 这里通过 Orchestrator 发送一个带有 context 的评测请求
        ctx = {"query": "存款保险限额", "answer": "最高50万元"}
        payload = make_orchestrator_task("评测一下", session_id="eval-ctx", context=ctx)
        resp = self.client.post("/a2a/task", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifacts"][0]["faithfulness"] == 0.95
        assert data["artifacts"][0]["query"] == "存款保险限额"
        assert data["artifacts"][0]["answer"] == "最高50万元"

    def test_context_isolation(self):
        """不同 session_id 的上下文互不干扰"""
        # Session A
        self.client.post("/a2a/task", json=make_orchestrator_task("资本充足率", session_id="iso-A"))
        # Session B
        self.client.post("/a2a/task", json=make_orchestrator_task("反洗钱规定", session_id="iso-B"))

        # Session A 评测（应得到资本充足率的上下文）
        resp_a = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="iso-A"))
        assert "资本充足率" in resp_a.json()["artifacts"][0]["query"]

        # Session B 评测（应得到反洗钱的上下文）
        resp_b = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="iso-B"))
        assert "反洗钱" in resp_b.json()["artifacts"][0]["query"]



class TestContextExpiry:
    @pytest.fixture(autouse=True)
    def setup(self, respx_mock, monkeypatch):
        # 将 TTL 设为 1 秒以加速过期测试
        monkeypatch.setattr(orch, "SESSION_TTL", 1)
        self.client = TestClient(orch_app)

        # 模拟 RAG Agent 正常响应
        def mock_rag(request):
            task_data = json.loads(request.content)["task"]
            query = task_data["messages"][-1]["parts"][0]["text"]
            return Response(200, json={
                "id": task_data["id"],
                "session_id": task_data["session_id"],
                "status": "completed",
                "artifacts": [{"answer": f"答案：{query}"}],
                "context": {"query": query, "answer": f"答案：{query}", "sources": ["test.txt"]}
            })
        respx_mock.post("http://localhost:8101/a2a/task").mock(side_effect=mock_rag)

        # 模拟 Eval Agent 正常响应（仅用于不关心具体内容的评测场景）
        def mock_eval(request):
            task_data = json.loads(request.content)["task"]
            ctx = task_data.get("context", {})
            return Response(200, json={
                "id": task_data["id"],
                "session_id": task_data["session_id"],
                "status": "completed",
                "artifacts": [{"faithfulness": 0.9, "query": ctx.get("query", ""), "answer": ctx.get("answer", "")}]
            })
        respx_mock.post("http://localhost:8102/a2a/task").mock(side_effect=mock_eval)

    def test_context_expired_then_refetch(self):
        """上下文过期后评测请求应返回失败，不会使用过期数据"""
        # 第一次问答，存入 session
        self.client.post("/a2a/task", json=make_orchestrator_task("问题1", session_id="exp1"))
        # 等待超过 TTL（1 秒）
        time.sleep(1.5)
        # 评测时不应能获取到过期的上下文
        resp = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="exp1"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "上一次对话" in data["artifacts"][0]["error"]

    @pytest.mark.parametrize("session_id", ["s1", "s2"])
    def test_multiple_sessions_ttl(self, session_id, monkeypatch):
        """多个会话的 TTL 独立，不会互相干扰"""
        # 会话 A
        self.client.post("/a2a/task", json=make_orchestrator_task("问题A", session_id="A"))
        time.sleep(0.3)  # 尚未过期
        # 会话 B
        self.client.post("/a2a/task", json=make_orchestrator_task("问题B", session_id="B"))
        
        # 评测 A 应成功（在有效期内）
        resp_a = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="A"))
        assert resp_a.json()["status"] == "completed"
        
        # 评测 B 应成功
        resp_b = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="B"))
        assert resp_b.json()["status"] == "completed"
        
        # 等待 A 过期（A 已经过了 1 秒）
        time.sleep(0.8)  # 总时间 > 1 秒
        resp_a_expired = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="A"))
        assert resp_a_expired.json()["status"] == "failed"
        
        # B 应该仍然有效（因为 B 刚刚创建）
        resp_b_still = self.client.post("/a2a/task", json=make_orchestrator_task("评测一下", session_id="B"))
        assert resp_b_still.json()["status"] == "completed"