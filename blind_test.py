"""
blind_test.py
对10个全新问题进行盲测，评估更新后向量库的回答质量
"""
import os
import sys
from pathlib import Path

from financial_rag_skill import FinancialRAGSkill

# 盲测问题列表
BLIND_QUESTIONS = [
    "我在两家银行各存了40万，如果两家都倒闭了能全赔吗？",
    "LPR多久调整一次？这个月的LPR是多少？",
    "给孩子汇留学学费超过5万美元怎么办？",
    "怎么判断一个银行是否满足资本充足率要求？",
    "在哪些情况下银行会被认定为系统重要性银行？",
    "朋友借用我的外汇额度购汇，会有什么风险？",
    "如果发现有人利用虚拟货币洗钱，应该向哪里举报？",
    "存款保险的50万限额是终身只有一次吗？",
    "房贷重定价周期可以自己选择吗？",
    "银行资本充足率不达标，储户的钱还安全吗？",
]

def run_blind_test():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    skill = FinancialRAGSkill()
    print("=" * 70)
    print("  金融 RAG 盲测评估（更新后知识库）")
    print("=" * 70)

    for i, q in enumerate(BLIND_QUESTIONS, 1):
        print(f"\n[{i}/10] ❓ {q}")
        try:
            # 使用 run_with_context 获取上下文，便于分析
            result = skill.run_with_context(q)
            answer = result.get("answer", "")
            contexts = result.get("context", [])
            print(f"🤖 回答: {answer}")
            if contexts:
                print(f"📚 检索上下文（前2条）:")
                for j, ctx in enumerate(contexts[:2], 1):
                    snippet = ctx[:150].replace('\n', ' ')
                    print(f"   [{j}] {snippet}...")
            else:
                print("   ⚠️ 未检索到上下文")

            # 简单人工标记（可后续替换为更精确的判断）
            # 这里输出一些提示供人工分析
            if "无法" in answer or "不收录" in answer:
                print("   🚩 可能：拒答/知识缺失")
            elif any(kw in answer for kw in ["5%", "8%", "50万", "5万", "3.0%"]):
                print("   ✅ 可能：包含关键数字")
            else:
                print("   ⚠️ 需人工检查准确性")
        except Exception as e:
            print(f"   ❌ 错误: {e}")

if __name__ == "__main__":
    run_blind_test()