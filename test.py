"""
test_end_to_end_conversation.py
Day40 交付物：从提问到评测的完整对话流测试
"""
import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from tools import financial_qa, evaluate_answer
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def run_conversation():
    # 1. 创建 Agent（使用 MemorySaver 存储会话）
    checkpointer = MemorySaver()
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=BASE_URL,
    )

    agent = create_agent(
        model=llm,
        tools=[financial_qa, evaluate_answer],
        system_prompt=(
            "你是专业的金融法规助手。"
            "当用户询问金融法规问题时，使用 financial_qa 工具获取准确信息。"
            "当用户要求评测某段回答时，使用 evaluate_answer 工具进行评估。"
            "对于无关闲聊，请直接友好回复。"
        ),
        checkpointer=checkpointer,
    )

    # 配置会话 ID，用于保持同一线程
    config = {"configurable": {"thread_id": "conversation-1"}}

    # 2. 第一轮：用户提问
    print("=" * 60)
    print("👤 用户: 资本充足率的要求是多少？")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "资本充足率的要求是多少？"}]},
        config
    )
    for msg in result1["messages"]:
        print(msg.pretty_repr())  # 显示每条消息的关键信息

    # 3. 第二轮：用户要求评测上一个回答
    #    Agent 会从对话历史中获取上一轮的回答，并调用 evaluate_answer 工具
    print("\n" + "=" * 60)
    print("👤 用户: 请帮我评测一下你刚才给出的回答是否忠实？")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我评测一下你刚才给出的回答是否忠实？"}]},
        config
    )
    for msg in result2["messages"]:
        print(msg.pretty_repr())

    # 4. 第三轮：查看最终回复
    final_messages = result2["messages"]
    if final_messages:
        last_ai = [m for m in final_messages if m.type == "ai"]
        if last_ai:
            print("\n🎯 最终 AI 回复:")
            print(last_ai[-1].content)

if __name__ == "__main__":
    # 确保 API Key 已设置
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("请设置 DASHSCOPE_API_KEY 环境变量")
    run_conversation()