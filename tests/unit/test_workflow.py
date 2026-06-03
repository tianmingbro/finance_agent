import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, Mock
from src.agent.workflow import rag_agent_workflow

MOCK_RETRIEVED = {
    "documents": [
        {"index": 1, "content": "核心一级资本充足率不得低于5%。", "source": "capital.txt"},
        {"index": 2, "content": "资本充足率不得低于8%。", "source": "capital.txt"}
    ],
    "sources": ["capital.txt"]
}

MOCK_ANSWER = "根据《商业银行资本管理办法》，核心一级资本充足率要求为不低于5%。"


def _make_future(value):
    """创建预完成的 asyncio.Future，用于 mock @task 的 await 调用"""
    f = asyncio.Future()
    f.set_result(value)
    return f


class TestRAGAgentWorkflow:

    @patch("workflow.retrieve_task")
    @patch("workflow.generate_answer_task")
    @patch("src.skill.ai_test_skill.EvaluationRunner.run")
    @patch("src.skill.ai_test_skill.EvalResourceManager.load_resources")
    async def test_workflow_returns_answer(self, mock_load, mock_run, mock_gen, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_gen.return_value = _make_future(MOCK_ANSWER)

        fake_faith = Mock()
        fake_faith.name = "faithfulness"
        fake_faith.score = 0.92
        fake_relevancy = Mock()
        fake_relevancy.name = "answer_relevancy"
        fake_relevancy.score = 0.95

        fake_report = MagicMock()
        fake_report.metrics = [fake_faith, fake_relevancy]
        fake_report.overall_trust = "高"
        mock_run.return_value = fake_report

        result = await rag_agent_workflow.ainvoke({"query": "资本充足率是多少？", "need_eval": True})

        assert result["answer"] == MOCK_ANSWER
        assert result["retrieved_docs"][0]["content"] == "核心一级资本充足率不得低于5%。"
        assert result["evaluation"]["faithfulness"] == 0.92
        assert result["evaluation"]["answer_relevancy"] == 0.95

    @patch("workflow.retrieve_task")
    @patch("workflow.generate_answer_task")
    async def test_retrieve_task_called(self, mock_gen, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_gen.return_value = _make_future(MOCK_ANSWER)

        result = await rag_agent_workflow.ainvoke({"query": "测试", "need_eval": False})
        mock_ret.assert_called_once()
        assert len(result["retrieved_docs"]) > 0

    @patch("workflow.retrieve_task")
    @patch("workflow.generate_answer_task")
    async def test_generate_task_uses_retrieved_docs(self, mock_gen, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_gen.return_value = _make_future(MOCK_ANSWER)

        await rag_agent_workflow.ainvoke({"query": "资本充足率", "need_eval": False})
        call_args = mock_gen.call_args[0]
        assert call_args[0] == "资本充足率"
        assert call_args[1] == MOCK_RETRIEVED

    @patch("workflow.retrieve_task")
    @patch("workflow.generate_answer_task")
    @patch("src.skill.ai_test_skill.EvaluationRunner.run")
    @patch("src.skill.ai_test_skill.EvalResourceManager.load_resources")
    async def test_evaluate_task_runs(self, mock_load, mock_run, mock_gen, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_gen.return_value = _make_future(MOCK_ANSWER)

        fake_faith = Mock()
        fake_faith.name = "faithfulness"
        fake_faith.score = 0.8
        fake_relevancy = Mock()
        fake_relevancy.name = "answer_relevancy"
        fake_relevancy.score = 0.9

        fake_report = MagicMock()
        fake_report.metrics = [fake_faith, fake_relevancy]
        fake_report.overall_trust = "中"
        mock_run.return_value = fake_report

        result = await rag_agent_workflow.ainvoke({"query": "资本充足率", "need_eval": True})
        mock_run.assert_called_once()
        assert "evaluation" in result
        assert result["evaluation"]["faithfulness"] == 0.8
        assert result["evaluation"]["answer_relevancy"] == 0.9

    @patch("workflow.retrieve_task")
    @patch("workflow.generate_answer_task")
    @patch("workflow.evaluate_task")
    async def test_workflow_without_evaluation(self, mock_eval, mock_gen, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_gen.return_value = _make_future(MOCK_ANSWER)

        result = await rag_agent_workflow.ainvoke({"query": "测试", "need_eval": False})
        mock_eval.assert_not_called()
        assert result.get("evaluation") is None


class TestRetryMechanism:

    @patch("workflow.retrieve_task")
    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    async def test_generate_task_retry_on_llm_failure(self, mock_invoke, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_invoke.side_effect = [
            Exception("模拟 LLM 暂时不可用"),
            MagicMock(content="重试后生成的答案"),
        ]

        result = await rag_agent_workflow.ainvoke({"query": "资本充足率", "need_eval": False})
        assert result["answer"] == "重试后生成的答案"
        assert mock_invoke.call_count == 2

    @patch("workflow.retrieve_task")
    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    async def test_generate_task_retry_exhausted(self, mock_invoke, mock_ret):
        mock_ret.return_value = _make_future(MOCK_RETRIEVED)
        mock_invoke.side_effect = Exception("LLM 服务中断")

        with pytest.raises(Exception) as exc_info:
            await rag_agent_workflow.ainvoke({"query": "资本充足率", "need_eval": False})

        assert "LLM 服务中断" in str(exc_info.value)
        assert mock_invoke.call_count == 3