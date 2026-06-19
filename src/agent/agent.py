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
from src.tools import financial_qa, evaluate_answer, _get_financial_skill,rag_workflow_query
from src.agent.mcp_client import get_finance_mcp_tools

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")


async def build_agent(checkpointer=None, use_sqlite: bool = False, db_path: str = "checkpoints.db"):
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

    _get_financial_skill()  # 内部会调用 load_resources()，设置全局 LLM 缓存

    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=BASE_URL,
    )
    # llm = ChatOpenAI(
    # model="qwen2.5:7b",                     # Ollama 中的模型名
    # temperature=0,
    # openai_api_key="ollama",                # 任意非空字符串
    # openai_api_base="http://localhost:11434/v1",
    # )
    mcp_tools = await get_finance_mcp_tools()
    if isinstance(mcp_tools, list) and len(mcp_tools) == 1 and isinstance(mcp_tools[0], list):
        mcp_tools = mcp_tools[0]    # 解开外层包装
    agent = create_agent(
        model=llm,
        tools=mcp_tools+[rag_workflow_query],
        system_prompt=(
            "你是专业的金融法规助手。"
            "1. 当用户询问金融法规问题时，你可以使用 search_finance_docs 工具检索相关法规文档，"
            "若工具调用失败，请诚实说明当前无法获取信息，不要编造任何内容。"
            "2. 当用户要求评测回答质量时，你必须使用 evaluate_answer 工具。"
            "如果用户未给出具体待评测文本，应从对话历史中提取上一轮金融回答；"
            "若无法获取，则请用户提供待评测的回答。"
            "3. 对于涉及违法、绕过监管或危害金融安全的请求，你必须明确拒绝，不调用任何工具。"
            "4. 对于无关闲聊，请直接友好回复，不调用任何工具。"
            "5.如果用户说“评测一下刚才那个回答”或类似指代，请从对话历史中找到最近一轮的原始问题和你之前生成的回答，将它们作为参数传递给 evaluate_answer 工具。"

        ),
        checkpointer=checkpointer or MemorySaver(),
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