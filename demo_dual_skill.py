"""
demo_dual_skill.py
Day31 交互 Demo：金融 RAG Skill + AI 测试 Skill 联动
"""
import os
import sys

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from financial_rag_skill import FinancialRAGSkill
from ai_test_skill import AITestSkill


def create_rag_factory():
    """返回金融 Skill 实例，而非函数"""
    return FinancialRAGSkill()


def main():
    # 初始化两个 Skill
    print("正在初始化系统...")
    # 创建 AI 测试 Skill，并注入金融 Skill 工厂
    test_skill = AITestSkill(rag_skill_factory=create_rag_factory)

    # 演示输入
    user_input = "测试一下LPR的回答质量"

    print("\n" + "=" * 60)
    print("  金融 RAG + AI 测试 Skill 联动 Demo")
    print("=" * 60)
    print(f">>> 用户输入: {user_input}")

    # 由测试 Skill 接管流程，它会自动：
    # 1. 触发金融 Skill 生成回答
    # 2. 执行 DeepEval 评测
    # 3. 格式化输出报告
    output = test_skill.run(user_input)
    print(output)

    print("\n✅ Demo 运行完成。请将上述输出保存至日志。")


if __name__ == "__main__":
    main()