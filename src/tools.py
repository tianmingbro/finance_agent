# tools.py
import time
import logging
from langchain_core.tools import tool
from src.skill.financial_rag_skill import FinancialRAGSkill
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager
logger = logging.getLogger(__name__)

_financial_skill = None
_eval_runner = None

def _get_financial_skill():
    global _financial_skill
    if _financial_skill is None:
        _financial_skill = FinancialRAGSkill()
    return _financial_skill

def _get_eval_runner():
    global _eval_runner
    if _eval_runner is None:
        resource_mgr = EvalResourceManager()
        resource_mgr.load_resources(model_name="qwen-plus")   # 修正：参数名 model_name，不是 model
        _eval_runner = EvaluationRunner(resource_mgr)
    return _eval_runner

import json  # 新增导入

@tool
def financial_qa(query: str) -> str:
    """回答金融法规相关问题，返回结构化 JSON，包含答案和检索上下文"""
    start = time.time()
    result = _get_financial_skill().run_with_context(query)
    elapsed = time.time() - start
    # 返回 JSON 字符串，方便评测脚本提取 context
    response = {
        "answer": result["answer"],
        "context": result.get("context", [])
    }
    logger.info("financial_qa 工具调用完成，耗时 %.2fs，查询: %s", elapsed, query)
    return json.dumps(response, ensure_ascii=False)

@tool
def evaluate_answer(query: str, answer: str) -> str:
    """对指定的问答对进行质量评测，返回忠实度、答案相关性等分数。
    当用户要求评测某个回答的质量时调用此工具。
    参数 query：原始问题（可以从对话历史中获取）。
    参数 answer：待评测的回答文本（可以从对话历史中获取）。"""
    start = time.time()
    runner = _get_eval_runner()
    def rag_callable(q: str) -> dict:
        return {"input": query, "answer": answer, "context": []}
    report = runner.run(f"全面评测：{query}", rag_callable)
    # faith = next((m.score for m in report.metrics if m.name == "faithfulness"), None)
    # relevancy = next((m.score for m in report.metrics if m.name == "answer_relevancy"), None)
    # trust = report.overall_trust
    # 安全提取指标，缺失时返回 "N/A"
    def safe_score(metric_name: str) -> str:
        m = next((m for m in report.metrics if m.name == metric_name), None)
        return f"{m.score:.2f}" if m else "N/A"

    faith = safe_score("faithfulness")
    relevancy = safe_score("answer_relevancy")
    trust = report.overall_trust or "未知"

    elapsed = time.time() - start
    logger.info("evaluate_answer 工具调用完成，耗时 %.2fs", elapsed)
    return (
    f"忠实度: {faith}, 答案相关性: {relevancy}, 综合信任等级: {trust}"
)