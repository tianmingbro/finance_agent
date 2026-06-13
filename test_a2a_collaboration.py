"""
test_a2a_collaboration.py
测试多 Agent 协作（基于自定义 FastAPI A2A 服务）
依赖：pip install pytest httpx respx
"""
import httpx
import pytest
import json
from fastapi.testclient import TestClient

# 导入各 Agent 的 FastAPI 应用实例
from src.agent.a2a.rag_agent import app as rag_app
from src.agent.a2a.eval_agent import app as eval_app
from src.agent.a2a.orchestrator_agent import app as orch_app
from src.agent.a2a.a2a_types import Task, Message, TextPart

# ---------- 工具函数：将 Task 递归转换为可序列化的字典 ----------
def task_to_dict(task: Task) -> dict:
    return {
        "id": task.id,
        "session_id": task.session_id,
        "status": task.status,
        "artifacts": task.artifacts,
        "messages": [
            {
                "role": msg.role,
                "parts": [{"text": p.text} for p in msg.parts]
            }
            for msg in task.messages
        ]
    }


def make_task_request(query: str, session_id="test") -> dict:
    """将自然语言查询包装成符合 A2A 端点要求的 JSON"""
    task = Task(
        id="test-1",
        session_id=session_id,
        messages=[Message(role="user", parts=[TextPart(text=query)])]
    )
    return {"task": task_to_dict(task)}


# ---------- 独立 Agent 测试 ----------
class TestStandaloneAgents:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.rag_client = TestClient(rag_app)
        self.eval_client = TestClient(eval_app)

    def test_rag_agent_standalone(self):
        """直接调用 RAG Agent 的 A2A 接口，获得检索增强答案"""
        payload = make_task_request("资本充足率要求是多少？")
        resp = self.rag_client.post("/a2a/task", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        # 检查 artifacts 中是否包含答案
        # test_rag_agent_standalone
        assert any(
        ("资本" in art.get("answer", "")) or ("5%" in art.get("answer", ""))
        for art in data["artifacts"]
        )
    def test_eval_agent_standalone(self):
        """直接调用 Eval Agent 的 A2A 接口，获得忠实度评分"""
        # Eval 要求格式 "query|answer"
        payload = make_task_request("资本充足率|核心一级资本充足率不低于5%")
        resp = self.eval_client.post("/a2a/task", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        artifacts = data["artifacts"]
        assert any("faithfulness" in art for art in artifacts)


# ---------- Orchestrator 协作测试（Mock 下游 Agent） ----------
class TestOrchestrator:
    @pytest.fixture(autouse=True)
    def setup(self, respx_mock):
        # 模拟 RAG Agent 的正常响应
        def mock_rag_response(request):
            body = json.loads(request.content)
            task = body["task"]
            query = task["messages"][-1]["parts"][0]["text"]
            return httpx.Response(
                200,
                json={
                    "id": task["id"],
                    "session_id": task.get("session_id", ""),
                    "status": "completed",
                    "artifacts": [{"answer": f"关于'{query}'的答案：核心一级资本充足率不低于5%"}],
                    "messages": task["messages"]
                }
            )

        # 模拟 Eval Agent 的正常响应
        def mock_eval_response(request):
            body = json.loads(request.content)
            task = body["task"]
            return httpx.Response(
                200,
                json={
                    "id": task["id"],
                    "session_id": task.get("session_id", ""),
                    "status": "completed",
                    "artifacts": [{"faithfulness": 0.92, "answer_relevancy": 0.95}],
                    "messages": task["messages"]
                }
            )

        # 匹配下游请求（根据 URL 区分）
        respx_mock.post("http://localhost:8101/a2a/task").mock(side_effect=mock_rag_response)
        respx_mock.post("http://localhost:8102/a2a/task").mock(side_effect=mock_eval_response)

        self.orch_client = TestClient(orch_app)

    def test_orchestrator_routes_to_rag(self):
        """发送金融问题，Orchestrator 调用 RAG Agent，返回答案"""
        payload = make_task_request("资本充足率是多少？")
        resp = self.orch_client.post("/a2a/task", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert any("不低于5%" in art.get("answer", "") for art in data["artifacts"])

    def test_orchestrator_routes_to_eval(self):
        """发送评测请求，Orchestrator 调用 Eval Agent，返回评测分数"""
        # 包含“评测”关键字，以及“|”分隔的问题和答案
        payload = make_task_request("评测 资本充足率|核心一级资本充足率不低于5%")
        resp = self.orch_client.post("/a2a/task", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        artifacts = data["artifacts"]
        assert any("faithfulness" in art for art in artifacts)

    def test_orchestrator_multi_turn(self):
        """多轮对话（先问答再评测）保持上下文"""
        # 第一轮：问答
        payload1 = make_task_request("LPR是什么？", session_id="multi-1")
        resp1 = self.orch_client.post("/a2a/task", json=payload1)
        assert resp1.status_code == 200
        # 第二轮：评测（使用相同 session_id）
        payload2 = make_task_request("评测 LPR|贷款市场报价利率", session_id="multi-1")
        resp2 = self.orch_client.post("/a2a/task", json=payload2)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert any("faithfulness" in art for art in data2["artifacts"])



def make_task(text):
    task = {"id": "1", "session_id": "s1",
            "messages": [{"role":"user","parts":[{"text":text}]}]}
    return {"task": task}


class TestAgentFailover:
    @pytest.fixture(autouse=True)
    def setup(self, respx_mock):
        self.client = TestClient(orch_app)

    @pytest.mark.parametrize("downstream_status,downstream_error", [
        (500, "下游服务内部错误"),
        (None, "服务不可用"),
    ])
    def test_rag_agent_unavailable(self, downstream_status, downstream_error, respx_mock):
        if downstream_status:
            respx_mock.post("http://localhost:8101/a2a/task").mock(
                return_value=httpx.Response(downstream_status)
            )
        else:
            # 模拟连接异常（如服务未启动）
            respx_mock.post("http://localhost:8101/a2a/task").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )

        payload = make_task("资本充足率是多少？")
        resp = self.client.post("/a2a/task", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert any(
            "不可用" in art.get("error", "") or str(downstream_status) in art.get("error", "")
            for art in data.get("artifacts", [])
        )