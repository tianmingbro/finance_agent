"""
batch_eval_demo.py (修正版)
从 YAML 加载事实查询，批量评测并输出完整汇总（含 faithfulness, answer_relevancy, contextual_recall）
"""
import yaml
import os
import sys
from typing import List
from ai_test_skill import EvaluationRunner, EvalResourceManager, EvalReport
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

def load_queries_from_yaml(yaml_path: str, category: str = "factual_query") -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    queries = []
    for cat in data["categories"]:
        if cat["category"] == category:
            for entry in cat["entries"]:
                queries.append(entry["query"])
            break
    return queries

def create_mock_rag():
    """模拟 RAG，根据查询关键词返回预设答案（忽略问号差异）"""
    knowledge = [
        {
            "keywords": ["资本充足率", "监管要求"],
            "answer": "核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。",
            "context": ["根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%..."]
        },
        {
            "keywords": ["LPR", "贷款市场报价利率"],
            "answer": "2025年1年期LPR为3.1%，5年期以上LPR为3.6%。",
            "context": ["2025年1年期LPR为3.1%，5年期以上LPR为3.6%。"]
        },
        {
            "keywords": ["购汇", "外汇", "便利化额度"],
            "answer": "个人每年购汇便利化额度为等值5万美元。",
            "context": ["个人每年结汇和购汇的便利化额度为等值5万美元。"]
        }
    ]

    def rag_callable(query: str) -> dict:
        q = query.replace("？", "").replace("?", "").strip()
        for item in knowledge:
            if any(kw in q for kw in item["keywords"]):
                return {"input": query, "answer": item["answer"], "context": item["context"]}
        # fallback
        return {"input": query, "answer": "该问题暂未收录。", "context": []}
    return rag_callable

def batch_evaluate(queries: List[str], runner: EvaluationRunner, rag_func) -> List[EvalReport]:
    reports = []
    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 正在评测: {q}")
        # 使用“全面评测”确保触发三个指标
        user_input = f"全面评测：{q}"
        report = runner.run(user_input, rag_func)
        # 调试输出：检查指标数量
        print(f"  本问题获得 {len(report.metrics)} 个指标")
        for m in report.metrics:
            print(f"    - {m.name}: {m.score:.2f}")
        reports.append(report)
    return reports

def print_summary(reports: List[EvalReport]):
    print("\n" + "=" * 70)
    print("📊 批量评测汇总报告")
    print("=" * 70)
    all_scores = []
    for report in reports:
        print(f"\n❓ 问题: {report.query}")
        print(f"🤖 回答: {report.answer[:80]}")
        scores = {}
        for m in report.metrics:
            scores[m.name] = m.score
            all_scores.append(m.score)
            status = "✅" if m.success else "❌"
            print(f"   {status} {m.name}: {m.score:.2f} (阈值 {m.threshold})")
        if scores:
            avg = sum(scores.values()) / len(scores)
            print(f"   📈 本问题平均分: {avg:.2f} | 信任等级: {report.overall_trust}")

    if all_scores:
        total_avg = sum(all_scores) / len(all_scores)
        print("\n" + "-" * 40)
        print(f"🎯 总体平均分 (所有指标): {total_avg:.2f}")
        print(f"📋 共评测 {len(reports)} 个问题，{len(all_scores)} 项指标")
    else:
        print("⚠️ 未获取到有效指标分数。")

if __name__ == "__main__":
    yaml_file = "eval_dataset.yaml"
    if not os.path.exists(yaml_file):
        print(f"⚠️ 找不到 {yaml_file}，使用内置查询列表。")
        queries = [
            "商业银行的资本充足率监管要求是多少？",
            "2025年1年期LPR是多少？",
            "个人每年购汇便利化额度是多少？"
        ]
    else:
        queries = load_queries_from_yaml(yaml_file, "factual_query")
        print(f"从 YAML 加载了 {len(queries)} 条事实查询。")

    # 初始化资源
    resource_mgr = EvalResourceManager()
    resource_mgr.load_resources()
    runner = EvaluationRunner(resource_mgr)
    rag_func = create_mock_rag()

    reports = batch_evaluate(queries, runner, rag_func)
    print_summary(reports)