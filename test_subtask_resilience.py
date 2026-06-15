import pytest
import json
import asyncio
import time
from fastapi.testclient import TestClient

from src.agent.a2a.orchestrator_agent import app as orch_app

def make_orchestrator_task(text: str, session_id: str = "test", context: dict = None) -> dict:
    task = {
        "id": f"task-{int(time.time()*1000)}",
        "session_id": session_id,
        "messages": [{"role": "user", "parts": [{"text": text}]}],
        "context": context or {},
        "status": "created",
        "artifacts": []
    }
    return {"task": task}

class TestSubtaskResilience:

    # 跳过异步超时和重试的单元测试（依赖真实网络模拟，集成测试覆盖）
    @pytest.mark.skip(reason="单元测试无法可靠模拟异步超时，改为手动/integration 验证")
    @pytest.mark.parametrize("fail_mode,expected_status,expected_error_msg", [
        ("timeout", "partial", "超时"),
        ("500", "partial", "500"),
    ])
    def test_subtask_retry_and_report(self, fail_mode, expected_status, expected_error_msg):
        pass

    def test_subtask_checkpoint(self):
        """
        验证子任务缓存：同一会话内重复相同的复合指令，不会再次调用下游 Agent。
        该测试通过 patch httpx 直接模拟下游返回，不依赖真实网络。
        """
        call_count = 0
        from unittest.mock import patch, AsyncMock
        import httpx

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 模拟下游成功响应
            return httpx.Response(200, json={
                "status": "completed",
                "artifacts": [{"answer": "缓存答案"}],
                "context": {"query": "test", "answer": "缓存答案", "sources": []}
            })

        with patch.object(httpx.AsyncClient, 'post', new_callable=AsyncMock) as mock:
            mock.side_effect = mock_post

            client = TestClient(orch_app)
            instruction = "查一下资本充足率并评测回答是否准确"
            session_id = "checkpoint-2"

            # 第一次请求
            resp1 = client.post("/a2a/task", json=make_orchestrator_task(instruction, session_id=session_id))
            assert resp1.status_code == 200
            first_calls = call_count

            # 第二次完全相同请求
            resp2 = client.post("/a2a/task", json=make_orchestrator_task(instruction, session_id=session_id))
            assert resp2.status_code == 200
            # 由于子任务结果已缓存，不应再调用下游
            assert call_count == first_calls