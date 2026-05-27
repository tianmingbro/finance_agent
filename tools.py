# tools.py
import time
import logging
from langchain_core.tools import tool
from financial_rag_skill import FinancialRAGSkill
from ai_test_skill import EvaluationRunner, EvalResourceManager
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

@tool
def financial_qa(query: str) -> str:
    """回答金融法规相关问题"""
    start = time.time()
    result = _get_financial_skill().run_with_context(query)
    elapsed = time.time() - start
    context_str = "【检索上下文】" + "；".join(result.get("context", []))
    logger.info("financial_qa 工具调用完成，耗时 %.2fs，查询: %s", elapsed, query)
    print("DEBUG financial_qa 返回长度:", len(f"答案：{result['answer']}\n{context_str}"))  # 临时日志
    return f"答案：{result['answer']}\n{context_str}"

@tool
def evaluate_answer(query: str, answer: str) -> str:
    """评测回答质量"""
    start = time.time()
    runner = _get_eval_runner()
    def rag_callable(q: str) -> dict:
        return {"input": query, "answer": answer, "context": []}
    report = runner.run(f"全面评测：{query}", rag_callable)
    faith = next((m.score for m in report.metrics if m.name == "faithfulness"), None)
    relevancy = next((m.score for m in report.metrics if m.name == "answer_relevancy"), None)
    trust = report.overall_trust
    
    elapsed = time.time() - start
    logger.info("evaluate_answer 工具调用完成，耗时 %.2fs", elapsed)
    return f"忠实度: {faith:.2f}, 答案相关性: {relevancy:.2f}, 综合信任等级: {trust}"