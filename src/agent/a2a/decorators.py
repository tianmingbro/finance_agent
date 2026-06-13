# agents/decorators.py
import time
import logging
from functools import wraps
from src.agent.a2a.a2a_types import Task

logger = logging.getLogger(__name__)

def agent_task(name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            # 直接从参数中提取 task（FastAPI 会正确传入 TaskRequest 对象）
            task_request = args[0] if args else kwargs.get('req')
            task = task_request.task
            logger.info("Agent[%s] 开始处理任务 %s", name, task.id)
            try:
                # 调用原函数，原函数仍接收 TaskRequest 对象
                task = await func(task_request)
                if task.status != "failed":
                    task.status = "completed"
            except Exception as e:
                logger.error("Agent[%s] 任务 %s 失败: %s", name, task.id, e)
                task.status = "failed"
                task.artifacts = [{"error": str(e)}]
            elapsed = time.perf_counter() - start
            logger.info("Agent[%s] 任务 %s 完成 (耗时 %.3fs)", name, task.id, elapsed)
            return task
        # 关键：手动复制原函数的 __annotations__ 到 wrapper
        wrapper.__annotations__ = func.__annotations__
        return wrapper
    return decorator