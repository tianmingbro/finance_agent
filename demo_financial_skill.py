"""
demo_financial_skill.py
三层渐进加载交互演示
"""
from financial_rag_skill import FinancialRAGSkill

def main():
    print("=" * 60)
    print("金融 RAG Skill 三层渐进加载 Demo")
    print("=" * 60)

    skill = FinancialRAGSkill()

    test_inputs = [
        "资本充足率是多少？",
        "今天天气如何？",
        "LPR最新报价是多少？",
        "帮我写首诗",
    ]

    for user_input in test_inputs:
        print(f"\n>>> 用户输入: {user_input}")

        # 第一层：触发检查
        triggered = skill.should_trigger(user_input)
        if not triggered:
            print("  [Layer 1] 触发词未匹配，跳过加载指令和资源")
            # 模拟 run 行为的输出，但不调用 run 以免无意中加载资源
            print("  [输出] 我是金融法规助手，请输入您想咨询的金融法规问题。")
            continue

        # 第二层：加载指令
        print("  [Layer 2] 触发词匹配，加载系统指令...")
        instruction = skill.instruction_loader.load_instruction(user_input)
        if instruction:
            print("  [Layer 2] 指令加载完成 (长度: {} 字符)".format(len(instruction)))

        # 判断是否需要检索
        if skill._needs_retrieval(user_input):
            print("  [Layer 3] 需要检索，正在延迟加载向量库和LLM...")
            # 执行完整的 run，内部会 load_resources
            answer = skill.run(user_input)
            print("  [输出] {}".format(answer))
        else:
            print("  [Layer 3] 元问题，无需加载向量库，直接使用LLM回答")
            answer = skill.run(user_input)
            print("  [输出] {}".format(answer))

    print("\n" + "=" * 60)
    print("Demo 完成，请将上述终端输出复制到日志中")
    print("=" * 60)


if __name__ == "__main__":
    main()