# agent.py
from langgraph.prebuilt import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import financial_qa, evaluate_answer

def build_agent(model_name="qwen-plus", checkpointer=None):
    """创建金融法规智能体实例"""
    return create_agent(
        model=model_name,
        tools=[financial_qa, evaluate_answer],
        system_prompt=(
            "你是专业的金融法规助手。"
            "当用户询问金融法规问题时，使用 financial_qa 工具获取准确信息并回答。"
            "当用户要求评测某段回答的质量时，使用 evaluate_answer 工具进行评估。"
            "对于无关闲聊，请直接友好回复，不要调用工具。"
        ),
        checkpointer=checkpointer or MemorySaver(),
    )