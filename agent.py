"""
agent.py — Day40 金融法规智能体（持久化记忆 + 工具调用日志）
"""
import os
import time
import logging
from typing import Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from tools import financial_qa, evaluate_answer

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")


def build_agent(checkpointer=None, use_sqlite: bool = False, db_path: str = "checkpoints.db"):
    """
    创建金融法规智能体实例。

    Args:
        checkpointer: 可选检查点（用于会话持久化），若不传则自动选择 InMemorySaver 或 SqliteSaver
        use_sqlite: 是否使用 SQLite 持久化（重启后记忆保留）
        db_path: SQLite 数据库文件路径

    Returns:
        编译后的 Agent（CompiledStateGraph）
    """
    if checkpointer is None:
        if use_sqlite:
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(db_path)
            logger.info("使用 SqliteSaver 持久化记忆 (db: %s)", db_path)
        else:
            checkpointer = MemorySaver()
            logger.info("使用 MemorySaver 内存记忆（重启后丢失）")

    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=BASE_URL,
    )

    # 包装工具以添加日志
    logged_tools = [_wrap_with_logging(financial_qa), _wrap_with_logging(evaluate_answer)]

    agent = create_agent(
        model=llm,
        tools=logged_tools,
        system_prompt=(
            "你是专业的金融法规助手。"
            "1. 当用户询问金融法规问题时，你必须优先使用 financial_qa 工具获取准确答案。"
            "若工具调用失败，请诚实说明当前无法获取信息，不要编造任何内容。"
            "2. 当用户要求评测回答质量时，你必须使用 evaluate_answer 工具。"
            "如果用户未给出具体待评测文本，应从对话历史中提取上一轮金融回答；"
            "若无法获取，则请用户提供待评测的回答。"
            "3. 对于涉及违法、绕过监管或危害金融安全的请求，你必须明确拒绝，不调用任何工具。"
            "4. 对于无关闲聊，请直接友好回复，不调用任何工具。"
        ),
        checkpointer=checkpointer,
    )
    return agent


def _wrap_with_logging(tool):
    """为工具函数添加耗时和结果日志（不改变签名）"""
    original_func = tool.func

    def logged_func(*args, **kwargs):
        t0 = time.time()
        try:
            result = original_func(*args, **kwargs)
            elapsed = time.time() - t0
            logger.info("工具 %s 调用成功 (耗时 %.2fs, 返回长度 %d 字符)",
                        tool.name, elapsed, len(str(result)))
            return result
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("工具 %s 调用失败 (耗时 %.2fs): %s", tool.name, elapsed, str(e))
            raise

    tool.func = logged_func
    return tool