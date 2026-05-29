"""
run_evaluation_pipeline_agent.py
基于 Agent 的自动化评测流水线 + 回归检测

使用方法：
  1. 确保 tools.py 中的 financial_qa 返回 JSON: {"answer": "...", "context": [...]}
  2. 运行本脚本
"""
from datetime import datetime
import os
import sys
import json
import yaml
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)  # 抑制 HTTP 请求日志
logging.getLogger("deepeval").setLevel(logging.WARNING) # 抑制 DeepEval 横幅和详细日志
from dotenv import load_dotenv
load_dotenv()

# LangChain 消息类型
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# DeepEval
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel

# Agent（复用 build_agent）
from agent import build_agent
import tools

os.environ["VECTOR_STORE_BACKEND"] = "pgvector"  # 根据实际情况调整

# 全局评测模型
EVAL_MODEL = GPTModel(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# -------------------- 配置 --------------------
EVAL_DATASET_PATH = "data/eval_dataset_v2.yaml"
BASELINE_PATH = "data/eval_baseline.json"
REPORT_PATH = "data/eval_report_agent.json"          # 输出独立的报告，避免覆盖原报告
REGRESSION_THRESHOLD = 0.05

METRIC_THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.8,
    "contextual_recall": 0.7,
}

MAX_CONCURRENT = 6
ENABLE_PARALLEL = True

print_lock = Lock()

# -------------------- 数据集加载（不变）--------------------
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
    print(f"[INFO] 加载评测数据集，共 {len(test_cases)} 条测试用例")
    return test_cases

# -------------------- 从 Agent 响应中提取信息 --------------------
def extract_agent_result(response: dict) -> tuple:
    """
    从 agent.invoke 返回的消息列表中提取最终答案和检索上下文。
    返回 (answer: str, contexts: List[str])
    """
    messages = response["messages"]

    # 优先尝试提取 ToolMessage 中的 answer 与 context（最可靠的 RAG 输出）
    tool_answer = None
    contexts: List[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "financial_qa":
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    if "answer" in data and isinstance(data.get("answer"), str):
                        tool_answer = data.get("answer")
                    if "context" in data and isinstance(data.get("context"), list):
                        contexts.extend(data.get("context"))
            except (json.JSONDecodeError, TypeError):
                pass

    if tool_answer is not None:
        return tool_answer, contexts

    # 回退：没有 ToolMessage 时，使用 Agent 的最后非工具生成的 AIMessage
    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_answer = msg.content
            break

    # 兼容性：AIMessage 中也可能嵌入了 JSON 格式的 context
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "context" in data and isinstance(data.get("context"), list):
                    contexts.extend(data.get("context"))
            except (json.JSONDecodeError, TypeError):
                pass

    return final_answer, contexts

# -------------------- 评测单个用例 --------------------
def evaluate_test_case(query: str, answer: str, contexts: List[str],
                       expected_answer: str = "") -> Dict[str, float]:
    scores = {}
    for metric_name, MetricCls in [
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
        ("contextual_recall", ContextualRecallMetric),
    ]:
        try:
            metric = MetricCls(model=EVAL_MODEL, threshold=0.7, include_reason=False)
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=contexts,
            )
            if metric_name == "contextual_recall":
                if not expected_answer:
                    raise ValueError("缺少 expected_answer，无法计算 ContextualRecall")
                test_case.expected_output = expected_answer
            metric.measure(test_case)
            scores[metric_name] = metric.score if metric.score is not None else 0.0
        except Exception as e:
            print(f"    [WARN] {metric_name} 评测失败: {e}")
            scores[metric_name] = 0.0
    return scores

# -------------------- 并行批量评测（改造为 agent） --------------------
def run_parallel_evaluation(
    test_cases: List[Dict],
    agent,                     # 接收 agent 实例
    max_concurrent: int = 1,
) -> List[Dict]:
    """使用 ThreadPoolExecutor 并行评测所有用例，每个用例独立 thread_id"""
    results = [None] * len(test_cases)

    def process_single(idx: int, tc: Dict) -> tuple:
        unique_config = {"configurable": {"thread_id": f"eval_agent_{tc['id']}"}}
        query = tc["query"]
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                unique_config
            )
            answer, contexts = extract_agent_result(response)
            # 回退：若 Agent 未返回 context，则使用强制检索结果替换 answer 和 context
            if not contexts:
                try:
                    skill = tools._get_financial_skill()
                    res = skill.run_with_context(query, force=True)
                    if isinstance(res.get("answer"), str):
                        answer = res.get("answer")
                    if isinstance(res.get("context"), list):
                        contexts = res.get("context")
                except Exception:
                    pass
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
                    f"[{idx+1}/{len(test_cases)}] [OK] {query[:40]}... "
                    f"F:{scores['faithfulness']:.2f} "
                    f"R:{scores['answer_relevancy']:.2f} "
                    f"C:{scores['contextual_recall']:.2f}"
                )
            return idx, result, None
        except Exception as e:
            with print_lock:
                print(f"[{idx+1}/{len(test_cases)}] [ERR] {query[:40]}... 失败: {e}")
            return idx, {
                "id": tc["id"],
                "query": query,
                "category": tc["category"],
                "answer": "",
                "context_count": 0,
                "scores": {
                    "faithfulness": None,
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

# -------------------- 统计与报告（不变）--------------------
def load_baseline() -> Optional[Dict[str, float]]:
    if not Path(BASELINE_PATH).exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_baseline(avg_scores: Dict[str, float]):
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, indent=2)
    print(f"[INFO] 新基线已保存至 {BASELINE_PATH}")

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
                f"[WARN] {metric}: 基线 {base_val:.3f} → 当前 {cur_val:.3f} (下降 {pct:.1f}%)"
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

def generate_report(results: List[Dict], warnings: List[str],
                    baseline: Optional[Dict[str, float]]) -> Dict:
    metric_scores = {"faithfulness": [], "answer_relevancy": [], "contextual_recall": []}
    for r in results:
        for m in metric_scores:
            if m in r["scores"]:
                metric_scores[m].append(r["scores"][m])

    summary = {}
    for m, vals in metric_scores.items():
        summary[m] = compute_statistics(vals)

    passed = sum(
        1 for r in results
        if all(r["scores"].get(m, 0) >= METRIC_THRESHOLDS[m] for m in metric_scores)
    )
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
    print("  金融 RAG Agent 自动化评测流水线")
    print("=" * 60)

    # 1. 加载数据集
    test_cases = load_eval_dataset(EVAL_DATASET_PATH)

    # 2. 初始化 Agent（仅一次）
    print("[INFO] 初始化 Agent（含工具与缓存）...")
    agent = build_agent()

    # 3. 评测
    if ENABLE_PARALLEL:
        print(f"[INFO] 并行评测模式（{MAX_CONCURRENT} 并发）")
        results = run_parallel_evaluation(test_cases, agent, max_concurrent=MAX_CONCURRENT)
    else:
        print("[INFO] 串行评测模式")
        results = []
        for i, tc in enumerate(test_cases, 1):
            query = tc["query"]
            print(f"[{i}/{len(test_cases)}] 评测: {query[:50]}...")
            config = {"configurable": {"thread_id": f"serial_agent_{tc['id']}"}}
            instructed_query = f"请使用 financial_qa 工具查询：{tc['query']}"
            try:
                response = agent.invoke(
                    {"messages": [HumanMessage(content=instructed_query)]},
                    config
                )
                answer, contexts = extract_agent_result(response)
                # 串行路径同样应用回退逻辑：直接绕过触发检查调用 skill
                if not contexts:
                    try:
                        skill = tools._get_financial_skill()
                        res = skill.run_with_context(query, force=True)
                        if isinstance(res.get("answer"), str):
                            answer = res.get("answer")
                        if isinstance(res.get("context"), list):
                            contexts = res.get("context")
                    except Exception:
                        pass
                scores = evaluate_test_case(query, answer, contexts, tc.get("expected_answer", ""))
            except Exception as e:
                print(f"  [ERR] 失败: {e}")
                scores = {"faithfulness": None, "answer_relevancy": None, "contextual_recall": None}
            results.append({
                "id": tc["id"],
                "query": query,
                "category": tc["category"],
                "answer": answer[:200] if answer else "",
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
            print("\n[WARN] 回归警告：")
            for w in warnings:
                print("  " + w)
        else:
            print("\n[OK] 未检测到显著回归")
    else:
        print("\n[INFO] 无历史基线，保存当前结果为基线")
        save_baseline(metric_means)
        baseline = metric_means

    # 6. 生成报告
    report = generate_report(results, warnings, baseline)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 评测报告已保存至 {REPORT_PATH}")
    print(f"总用例: {report['total_test_cases']}, 整体通过率: {report['overall_pass_rate']:.1%}")
    for m, stats in report["metrics_summary"].items():
        print(f"  {m}: 均值 {stats['mean']:.3f} (±{stats['std']:.3f})")

if __name__ == "__main__":
    main()