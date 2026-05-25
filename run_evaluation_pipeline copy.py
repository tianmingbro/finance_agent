"""
run_evaluation_pipeline.py
Day34 核心交付物：自动化评测流水线 + 回归检测
"""
import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()  # 从 .e
# 确保 DeepEval 使用 DashScope 的 qwen-plus 模型作为评判器
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
import json
import yaml
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain / LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# DeepEval
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 现有技能（确保在同一目录）
from financial_rag_skill import FinancialRAGSkill
from integrated_graph import GraphState  # 复用已有的节点函数

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

class QwenPlusModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def load_model(self):
        pass  # 无需加载本地模型

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "qwen-plus"
    
# -------------------- 配置 --------------------
EVAL_DATASET_PATH = "data/eval_dataset_v2.yaml"     # 扩充后的评测数据集
BASELINE_PATH = "data/eval_baseline.json"                # 历史基线
REPORT_PATH = "data/eval_report.json"                    # 输出报告
REGRESSION_THRESHOLD = 0.05                         # 下降 5% 触发警告

# 评测指标阈值
METRIC_THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.8,
    "contextual_recall": 0.7,
}




def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    """加载评测数据集，返回扁平化的测试用例列表"""
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


def build_rag_graph():
    """构建仅包含金融回答节点的最小 LangGraph 工作流"""
    def finance_answer_node(state: GraphState) -> dict:
        query = state["query"]
        skill = FinancialRAGSkill()
        # 强制加载资源
        skill.resource_mgr.load_resources()
        retriever = skill.resource_mgr.retriever
        llm = skill.resource_mgr.llm

        # 1. 检索
        docs = retriever.invoke(query)
        context_texts = [doc.page_content for doc in docs]

        # 2. 获取指令
        instruction = skill.instruction_loader.load_instruction(query)

        # 3. 生成回答
        prompt = ChatPromptTemplate.from_messages([
            ("system", instruction + "\n\n上下文:\n{context}"),
            ("human", "{input}"),
        ])
        combine_chain = create_stuff_documents_chain(llm, prompt)
        answer = combine_chain.invoke({"context": docs, "input": query})

        return {
            "answer": answer,
            "context": context_texts
        }
    builder = StateGraph(GraphState)
    builder.add_node("finance_answer", finance_answer_node)
    builder.add_edge(START, "finance_answer")
    builder.add_edge("finance_answer", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def evaluate_test_case(query: str, answer: str, contexts: List[str], expected_answer: str = "") -> Dict[str, float]:
    scores = {}
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=expected_answer,   # 用于 ContextualRecall
    )

    eval_model = QwenPlusModel()

    # 1. Faithfulness (无需标准答案)
    metric = FaithfulnessMetric(
        model=eval_model,
        threshold=METRIC_THRESHOLDS["faithfulness"],
        include_reason=False,
    )
    metric.measure(test_case)
    scores["faithfulness"] = metric.score

    # 2. AnswerRelevancy (无需标准答案)
    metric = AnswerRelevancyMetric(
        model=eval_model,
        threshold=METRIC_THRESHOLDS["answer_relevancy"],
        include_reason=False,
    )
    metric.measure(test_case)
    scores["answer_relevancy"] = metric.score

    # 3. ContextualRecall (需要 expected_answer)
    if expected_answer.strip():
        metric = ContextualRecallMetric(
            model=eval_model,
            threshold=METRIC_THRESHOLDS["contextual_recall"],
            include_reason=False,
        )
        metric.measure(test_case)
        scores["contextual_recall"] = metric.score
    else:
        scores["contextual_recall"] = 0.0

    return scores

def load_baseline() -> Optional[Dict[str, float]]:
    """加载历史基线，若不存在返回 None"""
    if not Path(BASELINE_PATH).exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(avg_scores: Dict[str, float]):
    """保存当前指标均值作为新基线"""
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, indent=2)
    print(f"✅ 新基线已保存至 {BASELINE_PATH}")


def check_regression(current_avg: Dict[str, float], baseline: Dict[str, float]) -> List[str]:
    """对比当前均值与基线，返回警告列表"""
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
    """计算均值、标准差、最小值、最大值"""
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
    baseline: Optional[Dict[str, float]]
) -> Dict:
    """组装最终报告"""
    # 收集各指标的所有分数
    metric_scores = {"faithfulness": [], "answer_relevancy": [], "contextual_recall": []}
    for r in results:
        for m in metric_scores:
            if m in r["scores"]:
                metric_scores[m].append(r["scores"][m])

    summary = {}
    for m, vals in metric_scores.items():
        summary[m] = compute_statistics(vals)

    # 整体通过率（所有指标均 >= 阈值）
    passed = 0
    for r in results:
        if all(
            r["scores"].get(m, 0) >= METRIC_THRESHOLDS[m]
            for m in metric_scores
        ):
            passed += 1
    total = len(results)
    pass_rate = passed / total if total > 0 else 0.0

    report = {
        "total_test_cases": total,
        "overall_pass_rate": round(pass_rate, 3),
        "metrics_summary": summary,
        "baseline": baseline,
        "regression_warnings": warnings,
        "detailed_results": results,
    }
    return report


def main():
    print("=" * 60)
    print("  金融 RAG 自动化评测流水线")
    print("=" * 60)

    # 1. 加载数据集
    test_cases = load_eval_dataset(EVAL_DATASET_PATH)

    # 2. 构建 LangGraph 工作流（仅金融回答节点）
    graph = build_rag_graph()
    config = {"configurable": {"thread_id": "eval_pipeline"}}

    # 3. 逐条评测
    results = []
    for i, tc in enumerate(test_cases, 1):
        query = tc["query"]
        print(f"[{i}/{len(test_cases)}] 评测: {query[:50]}...")

        # 调用 LangGraph 获取 answer 和 context
        state = graph.invoke({"query": query}, config)
        answer = state.get("answer", "")
        contexts = state.get("context", [])
        # print(f"  DEBUG context count: {len(contexts)}")   # 加这行
        # 评估指标
        try:
            scores = evaluate_test_case(query, answer, contexts,expected_answer=tc.get("expected_answer", ""))
        except Exception as e:
            print(f"  ❌ 指标计算失败: {e}")
            scores = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "contextual_recall": 0.0,
            }

        results.append({
            "id": tc["id"],
            "query": query,
            "category": tc["category"],
            "answer": answer[:200],   # 截断便于阅读
            "context_count": len(contexts),
            "scores": scores,
        })

        # 打印简要信息
        print(f"  Faithfulness: {scores['faithfulness']:.3f}, "
              f"Relevancy: {scores['answer_relevancy']:.3f}, "
              f"Recall: {scores['contextual_recall']:.3f}")

        time.sleep(0.5)  # 避免 API 限流

    # 4. 计算各指标均值
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
        baseline = metric_means  # 报告中使用当前作为基线

    # 6. 生成并保存报告
    report = generate_report(results, warnings, baseline)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📊 评测报告已保存至 {REPORT_PATH}")
    print(f"总用例: {report['total_test_cases']}, 整体通过率: {report['overall_pass_rate']:.1%}")
    for m, stats in report["metrics_summary"].items():
        print(f"  {m}: 均值 {stats['mean']:.3f} (±{stats['std']:.3f})")


import time
if Path(REPORT_PATH).exists():
    backup = f"eval_report_{int(time.time())}.json"
    os.rename(REPORT_PATH, backup)
    print(f"📦 旧报告已备份为 {backup}")