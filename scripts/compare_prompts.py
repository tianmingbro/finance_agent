"""
scripts/compare_prompts.py
Day48 辅助脚本：Prompt 优化 A/B 对比
"""
import os
import json
import yaml
import asyncio
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 自定义 LLM 与指标（复用已有的 CustomQwenLLM + DeepEval）
from src.pipeline.eval_components import CustomQwenLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# 检索工具
from src.retriever.tools_mcp import search_finance_docs

# 配置
BASELINE_SYSTEM = "你是一个金融法规专家。"
BASELINE_USER_TMPL = "根据以下信息回答问题：\n{{context}}\n问题：{{query}}\n答案："

EXPERIMENT_CONFIG = {
    "h1_safety": {
        "test_categories": ["adversarial_query"],
        "metric_focus": ["faithfulness", "answer_relevancy"],  # 也可统计拒答率
    },
    "h2_citation": {
        "test_categories": ["factual_query"],
        "metric_focus": ["faithfulness", "answer_relevancy"],
    },
    "h3_reasoning": {
        "test_categories": ["reasoning_query"],
        "metric_focus": ["faithfulness", "answer_relevancy"],
    },
}


def load_test_set(path="data/exp_test_set_v2.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["test_cases"]  # 直接返回列表，每个元素含 category

def select_subset(cases: List[Dict], categories: List[str], max_per_cat: int = 10) -> List[Dict]:
    subset = []
    counter = {c:0 for c in categories}
    for case in cases:
        cat = case.get("category", "")
        if cat in categories and counter[cat] < max_per_cat:
            subset.append(case)
            counter[cat] += 1
    return subset

def generate_answer(query: str, system_prompt: str, user_template: str) -> Tuple[str, List[str]]:
    result_json = asyncio.run(search_finance_docs(query, top_k=4))
    docs_data = json.loads(result_json)
    retrieval_context = [doc["content"] for doc in docs_data.get("documents", [])]
    if not retrieval_context:
        return "未找到相关文档。", []

    context_str = "\n".join([f"文档{i+1}: {c}" for i, c in enumerate(retrieval_context)])
    user_msg = user_template.replace("{{context}}", context_str).replace("{{query}}", query)

    # 使用 CustomQwenLLM 的统一 generate 方法
    llm = CustomQwenLLM()
    full_prompt = f"System: {system_prompt}\n\nUser: {user_msg}\nAssistant:"
    answer = llm.generate(full_prompt)
    return answer, retrieval_context

def evaluate_pair(question: str, answer: str, retrieval_context: List[str]) -> Dict[str, float]:
    """计算忠实度与相关性"""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context
    )
    eval_llm = CustomQwenLLM()
    faith = FaithfulnessMetric(model=eval_llm)
    faith.measure(test_case)
    rel = AnswerRelevancyMetric(model=eval_llm)
    rel.measure(test_case)
    return {"faithfulness": faith.score, "answer_relevancy": rel.score}

def run_experiment(hypothesis: str, template_optimized: dict, categories: List[str]):
    all_cases = load_test_set()
    test_cases = select_subset(all_cases, categories, max_per_cat=10)

    base_answers = []
    opt_answers = []
    results = {"baseline_metrics": [], "optimized_metrics": [], "questions": []}

    for case in test_cases:
        query = case["query"]
        # 基线
        base_answer, base_ctx = generate_answer(query, BASELINE_SYSTEM, BASELINE_USER_TMPL)
        base_metrics = evaluate_pair(query, base_answer, base_ctx)
        # 优化
        opt_sys = template_optimized.get("system", BASELINE_SYSTEM)
        opt_user_tmpl = template_optimized.get("user_template", BASELINE_USER_TMPL)
        opt_answer, opt_ctx = generate_answer(query, opt_sys, opt_user_tmpl)
        opt_metrics = evaluate_pair(query, opt_answer, opt_ctx)

        results["baseline_metrics"].append(base_metrics)
        results["optimized_metrics"].append(opt_metrics)
        base_answers.append(base_answer)
        opt_answers.append(opt_answer)
        results["questions"].append(query)

    # 计算统计指标
    def calc_stats(metric_name):
        base_vals = [m[metric_name] for m in results["baseline_metrics"]]
        opt_vals = [m[metric_name] for m in results["optimized_metrics"]]
        base_mean = statistics.mean(base_vals) if base_vals else 0
        opt_mean = statistics.mean(opt_vals) if opt_vals else 0
        return {
            "baseline": {"mean": base_mean, "std": statistics.stdev(base_vals) if len(base_vals) > 1 else 0},
            "optimized": {"mean": opt_mean, "std": statistics.stdev(opt_vals) if len(opt_vals) > 1 else 0},
            "change": base_mean - opt_mean,
        }

    summary = {
        "faithfulness": calc_stats("faithfulness"),
        "answer_relevancy": calc_stats("answer_relevancy"),
    }

    # 拒答率（仅 H1 需要）
    if hypothesis == "h1_safety":
        base_refuse = sum(1 for a in base_answers if "无法处理" in a)
        opt_refuse = sum(1 for a in opt_answers if "无法处理" in a)
        summary["refusal_rate"] = {
            "baseline": base_refuse / len(base_answers) if base_answers else 0,
            "optimized": opt_refuse / len(opt_answers) if opt_answers else 0,
        }
    return summary, results

if __name__ == "__main__":
    # 加载优化模板
    with open("configs/promptfoo/prompts/rag_qa_optimized_v1.json", "r", encoding="utf-8") as f:
        opt_templates = json.load(f)

    all_summaries = {}
    for hyp in ["h1_safety", "h2_citation", "h3_reasoning"]:
        print(f"\n===== 实验: {hyp} =====")
        tmpl = opt_templates[hyp]
        cats = EXPERIMENT_CONFIG[hyp]["test_categories"]
        summary, _ = run_experiment(hyp, tmpl, cats)
        all_summaries[hyp] = summary
        for metric, data in summary.items():
            if metric == "refusal_rate":
                print(f"  拒答率: 基线 {data['baseline']:.2%} -> 优化 {data['optimized']:.2%}")
            else:
                print(f"  {metric}: 基线 {data['baseline']['mean']:.3f} -> 优化 {data['optimized']['mean']:.3f} "
                      f"(变化 {data['change']:.3f})")

    # 保存结果
    with open("reports/prompt_optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print("\n对比结果已保存至 reports/prompt_optimization_results.json")