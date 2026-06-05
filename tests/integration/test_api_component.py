"""
test_api_component.py
验证 POST /eval/component 端点返回正确的组件评价格式
"""
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_component_eval_endpoint(client):
    payload = {
        "query": "商业银行核心一级资本充足率要求",
        "expected_tool": "financial_qa",
        "expected_args": {"query": "商业银行核心一级资本充足率要求"},
        "key_info": ["不低于5%"],
        "expected_answer": "核心一级资本充足率不得低于5%。"
    }
    response = await client.post("/eval/component", json=payload)
    assert response.status_code == 200
    data = response.json()
    # 检查顶层字段
    assert "planning" in data
    assert "retrieval" in data
    assert "generation" in data
    # 检查规划字段
    assert "tool_accuracy" in data["planning"]
    assert "arg_reasonableness" in data["planning"]
    # 检查检索字段
    assert "contextual_recall" in data["retrieval"]
    assert "contextual_precision" in data["retrieval"]
    # 检查生成字段
    assert "faithfulness" in data["generation"]
    assert "answer_relevancy" in data["generation"]

@pytest.mark.asyncio
async def test_component_eval_missing_query(client):
    """缺少必填参数 query 应返回 422"""
    response = await client.post("/eval/component", json={})
    assert response.status_code == 422