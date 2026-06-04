"""
api/dependencies.py
全局单例管理：确保 FinancialRAGSkill 和 EvaluationRunner 只初始化一次，
使用 asyncio.Lock 防止并发初始化冲突。
"""
import asyncio
import logging

from src.skill.financial_rag_skill import FinancialRAGSkill
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager

logger = logging.getLogger(__name__)

_financial_skill = None
_eval_runner = None
_skill_lock = asyncio.Lock()
_eval_lock = asyncio.Lock()


async def get_financial_skill() -> FinancialRAGSkill:
    """获取 FinancialRAGSkill 单例，若未初始化则创建并加载资源"""
    global _financial_skill
    if _financial_skill is not None:
        return _financial_skill

    async with _skill_lock:
        if _financial_skill is not None:          # 双重检查
            return _financial_skill

        logger.info("首次初始化 FinancialRAGSkill ...")
        skill = FinancialRAGSkill()
        try:
            # 主动加载资源（向量库、LLM、缓存等）
            skill.resource_mgr.load_resources()
        except Exception as e:
            logger.warning("预加载资源失败，将在首次调用时加载: %s", e)

        _financial_skill = skill
        logger.info("FinancialRAGSkill 单例初始化完成")
        return _financial_skill


async def get_eval_runner() -> EvaluationRunner:
    """获取 EvaluationRunner 单例，若未初始化则创建"""
    global _eval_runner
    if _eval_runner is not None:
        return _eval_runner

    async with _eval_lock:
        if _eval_runner is not None:
            return _eval_runner

        logger.info("首次初始化 EvaluationRunner ...")
        resource_mgr = EvalResourceManager()
        # load_resources() 无参数调用即可，其内部使用默认的 model_name="qwen-plus"
        resource_mgr.load_resources()
        runner = EvaluationRunner(resource_mgr)
        _eval_runner = runner
        logger.info("EvaluationRunner 单例初始化完成")
        return _eval_runner