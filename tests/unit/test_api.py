"""
test_api.py
Day45 TDAD 第一步：FastAPI 端点测试（红灯状态）
使用 httpx.AsyncClient 和 pytest-asyncio
"""
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app  # 待实现

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_rag_query_returns_answer(client):
    payload = {"query": "商业银行资本充足率是多少？"}
    response = await client.post("/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "retrieved_docs" in data

@pytest.mark.asyncio
async def test_rag_query_without_question(client):
    response = await client.post("/rag/query", json={})
    assert response.status_code == 422  # FastAPI 的 Pydantic 校验失败

@pytest.mark.asyncio
async def test_eval_score_returns_json(client):
    payload = {"query": "资本充足率", "answer": "核心一级资本充足率不低于5%"}
    response = await client.post("/eval/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness" in data
    assert "answer_relevancy" in data
    assert 0 <= data["faithfulness"] <= 1

@pytest.mark.asyncio
async def test_eval_missing_fields(client):
    response = await client.post("/eval/score", json={"query": "test"})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_concurrent_requests(client):
    import asyncio
    async def make_request():
        payload = {"query": "存款保险最高限额"}
        return await client.post("/rag/query", json=payload)
    tasks = [make_request() for _ in range(10)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for resp in responses:
        if isinstance(resp, Exception):
            pytest.fail(f"并发请求失败: {resp}")
        assert resp.status_code == 200

@pytest.mark.asyncio
async def test_streaming_response(client):
    payload = {"query": "LPR是什么？"}
    async with client.stream("POST", "/rag/query/stream", json=payload) as response:
        assert response.status_code == 200
        chunks = []
        async for chunk in response.aiter_text():
            chunks.append(chunk)
        full_text = "".join(chunks)
        assert len(full_text) > 0

@pytest.mark.asyncio
async def test_value_error_handler(client):
    """触发 ValueError 应返回 422 和错误详情"""
    response = await client.get("/debug/raise-error?error_type=value_error")
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "测试" in data["detail"]

@pytest.mark.asyncio
async def test_connection_error_handler(client):
    """触发 ConnectionError 应返回 502 和固定消息"""
    response = await client.get("/debug/raise-error?error_type=connection_error")
    assert response.status_code == 502
    data = response.json()
    assert data["detail"] == "上游服务连接失败，请稍后重试。"

@pytest.mark.asyncio
async def test_http_exception_still_works(client):
    """测试普通的 HTTPException 仍能正常返回（状态码 500）"""
    response = await client.get("/debug/raise-error?error_type=unknown")
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data

import time

@pytest.mark.asyncio
async def test_concurrent_requests_with_latency(client):
    import asyncio
    payload = {"query": "存款保险最高限额"}
    latencies = []

    async def make_request():
        start = time.perf_counter()
        resp = await client.post("/rag/query", json=payload)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        return resp

    tasks = [make_request() for _ in range(10)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # 确保没有异常
    for resp in responses:
        if isinstance(resp, Exception):
            pytest.fail(f"并发请求失败: {resp}")
        assert resp.status_code == 200

    # 计算 P50, P99
    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.5)]
    p99 = latencies[int(n * 0.99)] if n > 1 else latencies[0]

    print(f"\n并发请求延迟 (秒) - P50: {p50:.3f}, P99: {p99:.3f}")
    # 可添加断言：P99 < 10 秒等
    assert p99 < 30  # 30 秒为上限，避免测试挂死