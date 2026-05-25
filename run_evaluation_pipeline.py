"""
run_evaluation_pipeline.py
Day34 核心交付物：自动化评测流水线 + 回归检测（第6周完善版）

关键优化：
  1. FinancialRAGSkill 只实例化一次，所有测试用例复用（消除重复加载向量库/LLM）
  2. ThreadPoolExecutor 并行评测，默认 6 并发
  3. 使用 DeepEval 原生 evaluate() 函数，享受内置缓存、异步执行与进度条
  4. 单用例失败不影响整体评测
"""
import os
import sys
import json
import yaml
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dotenv import load_dotenv
load_dotenv()

# 确保 DeepEval 使用 DashScope 的 qwen-plus 模型作为评判器
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LangChain / LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# DeepEval
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, ErrorConfig

from openai import OpenAI

# 现有技能
from financial_rag_skill import FinancialRAGSkill
from integrated_graph import GraphState


# -------------------- Qwen-Plus 评估模型 --------------------
class QwenPlusModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "qwen-turbo"


# -------------------- 配置 --------------------
EVAL_DATASET_PATH = "data/eval_dataset_v2.yaml"
BASELINE_PATH = "data/eval_baseline.json"
REPORT_PATH = "data/eval_report.json"
REGRESSION_THRESHOLD = 0.05

METRIC_THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.8,
    "contextual_recall": 0.7,
}

# 并行评测配置
MAX_CONCURRENT = 1          # 并发度（qwen-plus 限流 200 QPM，6 并发安全）
ENABLE_PARALLEL = True       # 设为 False 可切回串行模式

# 打印锁，防止并发输出混乱
print_lock = Lock()


# -------------------- 数据集加载 --------------------
def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"数据集不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = []
    for category in data.get("categories", []):
        cat_name = category["category"]
        for entry in category.get("entries", []):
            test_cases.append({
                "id": entry.get("id", f"{cat_name}_{len(test_cases)}"),
                "query": entry["query"],
                "expected_answer": entry.get("expected_answer", ""),
                "category": cat_name,
            })
    print(f"📋 加载评测数据集，共 {len(test_cases)} 条测试用例")
    return test_cases


# -------------------- 评测图构建（复用 Skill） --------------------
def build_rag_graph(skill: FinancialRAGSkill):
    """
    构建仅包含金融回答节点的最小 LangGraph 工作流。
    skill 在外部创建一次，所有调用复用同一实例（向量库/LLM 只加载一次）。
    """

    def finance_answer_node(state: GraphState) -> dict:
        query = state["query"]
        # 复用外部 skill 实例，不新建
        if not skill.resource_mgr._loaded:
            skill.resource_mgr.load_resources()
        retriever = skill.resource_mgr.retriever
        llm = skill.resource_mgr.llm

        # 检索
        docs = retriever.invoke(query)
        context_texts = [doc.page_content for doc in docs]

        # 生成回答
        instruction = skill.instruction_loader.load_instruction(query)
        prompt = ChatPromptTemplate.from_messages([
            ("system", instruction + "\n\n上下文:\n{context}"),
            ("human", "{input}"),
        ])
        combine_chain = create_stuff_documents_chain(llm, prompt)
        answer = combine_chain.invoke({"context": docs, "input": query})

        return {"answer": answer, "context": context_texts}

    builder = StateGraph(GraphState)
    builder.add_node("finance_answer", finance_answer_node)
    builder.add_edge(START, "finance_answer")
    builder.add_edge("finance_answer", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# -------------------- 单用例评测（被并行调用） --------------------
# def evaluate_test_case(
#     query: str,
#     answer: str,
#     contexts: List[str],
#     expected_answer: str = "",
#     metrics_cache: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, float]:
#     """
#     对单个测试用例运行三项 DeepEval 指标。
#     支持传入已创建的 metrics 实例来复用。
#     """
#     test_case = LLMTestCase(
#         input=query,
#         actual_output=answer,
#         retrieval_context=contexts,
#         expected_output=expected_answer,
#     )

#     eval_model = QwenPlusModel()
#     scores = {}

#     # 1. Faithfulness
#     metric = FaithfulnessMetric(
#         model=eval_model,
#         threshold=METRIC_THRESHOLDS["faithfulness"],
#         include_reason=False,
#         async_mode=True,  # 启用指标内部异步
#     )
#     metric.measure(test_case)
#     scores["faithfulness"] = metric.score

#     # 2. AnswerRelevancy
#     metric = AnswerRelevancyMetric(
#         model=eval_model,
#         threshold=METRIC_THRESHOLDS["answer_relevancy"],
#         include_reason=False,
#         async_mode=True,
#     )
#     metric.measure(test_case)
#     scores["answer_relevancy"] = metric.score

#     # 3. ContextualRecall（需要 expected_answer）
#     if expected_answer.strip():
#         metric = ContextualRecallMetric(
#             model=eval_model,
#             threshold=METRIC_THRESHOLDS["contextual_recall"],
#             include_reason=False,
#             async_mode=True,
#         )
#         metric.measure(test_case)
#         scores["contextual_recall"] = metric.score
#     else:
#         scores["contextual_recall"] = 0.0

#     return scores

def evaluate_test_case(query, answer, contexts):
    scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "contextual_recall": 0.0}
    for metric_name, MetricCls in [
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
        ("contextual_recall", ContextualRecallMetric),
    ]:
        try:
            metric = MetricCls(model="qwen-plus", threshold=0.7, timeout=15)
            test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts)
            metric.measure(test_case)
            scores[metric_name] = metric.score
        except Exception as e:
            print(f"    ⚠️ {metric_name} 评测失败: {e}")
            scores[metric_name] = None   # 标记为缺失
    return scores

# -------------------- 并行批量评测 --------------------
def run_parallel_evaluation(
    test_cases: List[Dict],
    graph,
    config: Dict,
    max_concurrent: int = 1,
) -> List[Dict]:
    """使用 ThreadPoolExecutor 并行评测所有用例"""
    results = [None] * len(test_cases)  # 保持原始顺序

    def process_single(idx: int, tc: Dict) -> tuple:
        query = tc["query"]
        try:
            state = graph.invoke({"query": query}, config)
            answer = state.get("answer", "")
            contexts = state.get("context", [])
            scores = evaluate_test_case(
                query, answer, contexts,
                expected_answer=tc.get("expected_answer", ""),
            )
            result = {
                "id": tc["id"],
                "query": query,
                "category": tc["category"],
                "answer": answer[:200],
                "context_count": len(contexts),
                "scores": scores,
            }
            with print_lock:
                print(
                    f"[{idx+1}/{len(test_cases)}] ✅ {query[:40]}... "
                    f"F:{scores['faithfulness']:.2f} "
                    f"R:{scores['answer_relevancy']:.2f} "
                    f"C:{scores['contextual_recall']:.2f}"
                )
            return idx, result, None
        except Exception as e:
            with print_lock:
                print(f"[{idx+1}/{len(test_cases)}] ❌ {query[:40]}... 失败: {e}")
            return idx, {
                "id": tc["id"],
                "query": query,
                "category": tc["category"],
                "answer": "",
                "context_count": 0,
                "scores": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "contextual_recall": 0.0,
                },
            }, None

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(process_single, i, tc): i
            for i, tc in enumerate(test_cases)
        }
        for future in as_completed(futures):
            idx, result, _ = future.result()
            results[idx] = result

    return results


# -------------------- 统计与报告（同原版） --------------------
def load_baseline() -> Optional[Dict[str, float]]:
    if not Path(BASELINE_PATH).exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(avg_scores: Dict[str, float]):
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, indent=2)
    print(f"✅ 新基线已保存至 {BASELINE_PATH}")


def check_regression(current_avg: Dict[str, float], baseline: Dict[str, float]) -> List[str]:
    warnings = []
    for metric, cur_val in current_avg.items():
        base_val = baseline.get(metric)
        if base_val is None:
            continue
        drop = base_val - cur_val
        if drop > REGRESSION_THRESHOLD:
            pct = (drop / base_val) * 100
            warnings.append(
                f"⚠️ {metric}: 基线 {base_val:.3f} → 当前 {cur_val:.3f} (下降 {pct:.1f}%)"
            )
    return warnings


def compute_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def generate_report(
    results: List[Dict],
    warnings: List[str],
    baseline: Optional[Dict[str, float]],
) -> Dict:
    metric_scores = {"faithfulness": [], "answer_relevancy": [], "contextual_recall": []}
    for r in results:
        for m in metric_scores:
            if m in r["scores"]:
                metric_scores[m].append(r["scores"][m])

    summary = {}
    for m, vals in metric_scores.items():
        summary[m] = compute_statistics(vals)

    passed = 0
    for r in results:
        if all(
            r["scores"].get(m, 0) >= METRIC_THRESHOLDS[m]
            for m in metric_scores
        ):
            passed += 1
    total = len(results)
    pass_rate = passed / total if total > 0 else 0.0

    return {
        "total_test_cases": total,
        "overall_pass_rate": round(pass_rate, 3),
        "metrics_summary": summary,
        "baseline": baseline,
        "regression_warnings": warnings,
        "detailed_results": results,
    }


# -------------------- 主流程 --------------------
def main():
    print("=" * 60)
    print("  金融 RAG 自动化评测流水线（完善版）")
    print("=" * 60)

    # 备份旧报告
    if Path(REPORT_PATH).exists():
        backup = f"eval_report_{int(time.time())}.json"
        os.rename(REPORT_PATH, backup)
        print(f"📦 旧报告已备份为 {backup}")

    # 1. 加载数据集
    test_cases = load_eval_dataset(EVAL_DATASET_PATH)

    # 2. ★ 关键优化：只创建一次 FinancialRAGSkill，所有用例复用 ★
    print("🔧 初始化金融 RAG Skill（仅一次）...")
    skill = FinancialRAGSkill()
    graph = build_rag_graph(skill)
    config = {"configurable": {"thread_id": "eval_pipeline"}}

    # 3. 评测
    if ENABLE_PARALLEL:
        print(f"⚡ 并行评测模式（{MAX_CONCURRENT} 并发）")
        results = run_parallel_evaluation(
            test_cases, graph, config, max_concurrent=MAX_CONCURRENT
        )
    else:
        print("🐢 串行评测模式")
        results = []
        for i, tc in enumerate(test_cases, 1):
            query = tc["query"]
            print(f"[{i}/{len(test_cases)}] 评测: {query[:50]}...")
            state = graph.invoke({"query": query}, config)
            answer = state.get("answer", "")
            contexts = state.get("context", [])
            try:
                scores = evaluate_test_case(
                    query, answer, contexts,
                    expected_answer=tc.get("expected_answer", ""),
                )
            except Exception as e:
                print(f"  ❌ 指标计算失败: {e}")
                scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "contextual_recall": 0.0}
            results.append({
                "id": tc["id"],
                "query": query,
                "category": tc["category"],
                "answer": answer[:200],
                "context_count": len(contexts),
                "scores": scores,
            })
            print(f"  F:{scores['faithfulness']:.2f} "
                  f"R:{scores['answer_relevancy']:.2f} "
                  f"C:{scores['contextual_recall']:.2f}")

    # 4. 统计
    metric_means = {}
    for m in ["faithfulness", "answer_relevancy", "contextual_recall"]:
        vals = [r["scores"][m] for r in results]
        metric_means[m] = statistics.mean(vals) if vals else 0.0

    # 5. 回归检测
    baseline = load_baseline()
    warnings = []
    if baseline:
        warnings = check_regression(metric_means, baseline)
        if warnings:
            print("\n🚨 回归警告：")
            for w in warnings:
                print("  " + w)
        else:
            print("\n✅ 未检测到显著回归")
    else:
        print("\n📌 无历史基线，保存当前结果为基线")
        save_baseline(metric_means)
        baseline = metric_means

    # 6. 生成报告
    report = generate_report(results, warnings, baseline)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📊 评测报告已保存至 {REPORT_PATH}")
    print(f"总用例: {report['total_test_cases']}, 整体通过率: {report['overall_pass_rate']:.1%}")
    for m, stats in report["metrics_summary"].items():
        print(f"  {m}: 均值 {stats['mean']:.3f} (±{stats['std']:.3f})")


if __name__ == "__main__":
    main()