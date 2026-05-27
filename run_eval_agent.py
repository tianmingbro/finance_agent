"""
run_evaluation_pipeline.py
Day34 核心交付物：自动化评测流水线 + 回归检测（第6周完善版）

关键优化：
  1. FinancialRAGSkill 只实例化一次，所有测试用例复用（消除重复加载向量库/LLM）
  2. ThreadPoolExecutor 并行评测，默认 6 并发
  3. 使用 DeepEval 原生 evaluate() 函数，享受内置缓存、异步执行与进度条
  4. 单用例失败不影响整体评测
"""
from datetime import datetime
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
from deepeval.models import GPTModel

# 现有技能
from financial_rag_skill import FinancialRAGSkill
from integrated_graph import GraphState

os.environ["VECTOR_STORE_BACKEND"] = "pgvector"  # 评测时默认使用 PGVector，确保与生产环境一致  

# 全局评测模型（GPTModel 接受自定义客户端）
EVAL_MODEL = GPTModel(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

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
MAX_CONCURRENT = 6          # 并发度（qwen-plus 限流 200 QPM，6 并发安全）
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

from difflib import SequenceMatcher

# def clean_retrieval_context(contexts: List[str], expected_answer: str, 
#                             similarity_threshold: float = 0.7) -> List[str]:
#     """
#     更保守的清洗策略，避免误删唯一相关证据：
#     - 若 expected_answer 为空，则不清洗
#     - 若检索片段包含 expected_answer 的逐字文本，则认为可能泄露并移除
#     - 否则仅在相似度极高（>0.95）时移除
#     该策略减少因模糊相似判定导致的误删。
#     """
#     if not expected_answer:
#         return contexts

#     cleaned: List[str] = []
#     for ctx in contexts:
#         try:
#             # 逐字包含被视为潜在泄露，需移除
#             if expected_answer.strip() and expected_answer in ctx:
#                 continue
#         except Exception:
#             pass

#         # 仅当相似度极高时（严格门槛）才移除
#         similarity = SequenceMatcher(None, ctx, expected_answer).ratio()
#         if similarity > 0.95:
#             continue
#         cleaned.append(ctx)
#     return cleaned

def evaluate_test_case(query: str, answer: str, contexts: List[str], expected_answer: str = "") -> Dict[str, float]:
    scores = {"faithfulness":None, "answer_relevancy": None, "contextual_recall": None}
        # 创建自定义的 OpenAI 客户端，指向 DashScope

    for metric_name, MetricCls in [
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
        ("contextual_recall", ContextualRecallMetric),
    ]:
        try:
            metric = MetricCls(model=EVAL_MODEL, threshold=0.7,include_reason=False,
                            )
            # 清洗上下文（仅对 faithfulness 和 contextual_recall 有必要）
            cleaned_contexts = contexts
            # if metric_name in ("faithfulness", "contextual_recall") and expected_answer:
            #     cleaned_contexts = clean_retrieval_context(contexts, expected_answer)

            test_case = LLMTestCase(
                input=query, 
                actual_output=answer, 
                retrieval_context=cleaned_contexts
            )
            # test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts)
            if metric_name=="contextual_recall":
                if not expected_answer:
                    raise ValueError("缺少 expected_answer，无法计算 ContextualRecall")
                
                test_case.expected_output = expected_answer
            metric.measure(test_case)
            scores[metric_name] = metric.score if metric.score is not None else 0.0
        except Exception as e:
            print(f"    ⚠️ {metric_name} 评测失败: {e}")
            scores[metric_name] = 0.0   # 标记为缺失
    return scores

# -------------------- 并行批量评测 --------------------
def run_parallel_evaluation(
    test_cases: List[Dict],
    graph,
    max_concurrent: int = 1,          # 移除了 config 参数，因为不再需要统一配置
) -> List[Dict]:
    """使用 ThreadPoolExecutor 并行评测所有用例"""
    results = [None] * len(test_cases)

    def process_single(idx: int, tc: Dict) -> tuple:
        # 每个用例独立的 thread_id，避免状态串扰
        unique_config = {"configurable": {"thread_id": f"eval_{tc['id']}"}}
        query = tc["query"]
        try:
            # ★ 关键修改：使用 unique_config 而不是外部传入的 config
            state = graph.invoke({"query": query}, unique_config)
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
                    "faithfulness": None,        # 建议失败时用 None 而非 0.0，避免拉低统计
                    "answer_relevancy": None,
                    "contextual_recall": None,
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
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "failed": len(values)}
    return {
        "mean": statistics.mean(valid),
        "std": statistics.stdev(valid) if len(valid) > 1 else 0.0,
        "min": min(valid),
        "max": max(valid),
        "failed": len(values) - len(valid)
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
        backup = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
            test_cases, graph, max_concurrent=MAX_CONCURRENT
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
                scores = {"faithfulness": None, "answer_relevancy": None, "contextual_recall": None}
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