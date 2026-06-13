#!/usr/bin/env python3
"""
run_evaluation_pipeline.py (生产级)
基于函数式工作流 rag_agent_workflow 的自动化评测管道。
支持并行异步评测、超时控制、基线回归检测。
"""
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import asyncio
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from src.pipeline.eval_components import generate_component_report
import yaml
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel

from src.agent.workflow import rag_agent_workflow

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

# 并发控制
MAX_CONCURRENT = 4                # 同时进行的评测数
PER_CASE_TIMEOUT = 60             # 单个用例最大总时间（秒）

# DeepEval 评判模型
# EVAL_MODEL = GPTModel(
#     model="qwen-plus",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
EVAL_MODEL = GPTModel(
    model="qwen2.5:7b",
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# -------------------- 工具函数 --------------------
def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"数据集不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    test_cases = []
    for category in data.get("categories", []):
        for entry in category.get("entries", []):
            test_cases.append({
                "id": entry.get("id", f"{category['category']}_{len(test_cases)}"),
                "query": entry["query"],
                "expected_answer": entry.get("expected_answer", ""),
                "category": category["category"],
            })
    print(f"📋 加载评测数据集，共 {len(test_cases)} 条测试用例")
    return test_cases


def clean_retrieval_context(contexts: List[str], expected_answer: str,
                            similarity_threshold: float = 0.7) -> List[str]:
    """移除与标准答案高度相似的上下文，避免数据泄露"""
    from difflib import SequenceMatcher
    return [
        ctx for ctx in contexts
        if SequenceMatcher(None, ctx, expected_answer).ratio() < similarity_threshold
    ]


async def evaluate_test_case(
    query: str,
    answer: str,
    contexts: List[str],
    expected_answer: str = "",
) -> Dict[str, Optional[float]]:
    """异步运行三项指标评测，返回分数字典"""
    scores = {"faithfulness": None, "answer_relevancy": None, "contextual_recall": None}
    loop = asyncio.get_running_loop()

    async def _measure(metric_cls, need_expected: bool = False, need_clean: bool = False):
        try:
            metric = metric_cls(model=EVAL_MODEL, threshold=0.7, include_reason=False)
            _contexts = contexts
            if need_clean and expected_answer:
                _contexts = clean_retrieval_context(contexts, expected_answer)
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=_contexts,
            )
            if need_expected:
                if not expected_answer:
                    raise ValueError("缺少 expected_answer")
                test_case.expected_output = expected_answer

            # 在线程池中执行同步 measure
            await loop.run_in_executor(None, metric.measure, test_case)
            return metric.score if metric.score is not None else 0.0
        except Exception as e:
            print(f"    ⚠️ {metric_cls.__name__} 评测失败: {e}")
            return None

    # 并发运行三个指标
    faith, relevancy, recall = await asyncio.gather(
        _measure(FaithfulnessMetric, need_clean=True),
        _measure(AnswerRelevancyMetric),
        _measure(ContextualRecallMetric, need_expected=True),
        return_exceptions=True,
    )

    if isinstance(faith, BaseException):
        faith = None
    if isinstance(relevancy, BaseException):
        relevancy = None
    if isinstance(recall, BaseException):
        recall = None

    return {
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "contextual_recall": recall,
    }


async def process_one_case(idx: int, tc: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个测试用例：调用工作流 + 评测"""
    query = tc["query"]
    start_time = time.time()
    try:
        # 调用异步工作流（设置超时）
        workflow_result = await asyncio.wait_for(
            rag_agent_workflow.ainvoke({"query": query, "need_eval": False}),
            timeout=PER_CASE_TIMEOUT,
        )
        answer = workflow_result.get("answer", "")
        context_list = workflow_result.get("context", [])
    except asyncio.TimeoutError:
        print(f"[{idx+1}] ❌ {query[:50]}... 工作流超时")
        return {
            "id": tc["id"],
            "query": query,
            "category": tc["category"],
            "answer": "",
            "context_count": 0,
            "scores": {"faithfulness": None, "answer_relevancy": None, "contextual_recall": None},
            "error": "Workflow timeout",
        }
    except Exception as e:
        print(f"[{idx+1}] ❌ {query[:50]}... 工作流异常: {e}")
        return {
            "id": tc["id"],
            "query": query,
            "category": tc["category"],
            "answer": "",
            "context_count": 0,
            "scores": {"faithfulness": None, "answer_relevancy": None, "contextual_recall": None},
            "error": str(e),
        }

    # 评测
    try:
        scores = await evaluate_test_case(
            query,
            answer,
            context_list,
            tc.get("expected_answer", ""),
        )
    except Exception as e:
        scores = {"faithfulness": None, "answer_relevancy": None, "contextual_recall": None}

    elapsed = time.time() - start_time
    print(f"[{idx+1}/{total_cases}] {'✅' if scores.get('faithfulness') else '❌'} {query[:40]}... "
          f"F:{scores['faithfulness']} R:{scores['answer_relevancy']} C:{scores['contextual_recall']} "
          f"({elapsed:.1f}s)")

    return {
        "id": tc["id"],
        "query": query,
        "category": tc["category"],
        "answer": answer[:200],
        "context_count": len(context_list),
        "scores": scores,
    }


async def run_parallel_evaluation(test_cases: List[Dict], max_concurrent: int) -> List[Dict]:
    """使用信号量控制并发的异步评测"""
    global total_cases
    total_cases = len(test_cases)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(idx, tc):
        async with semaphore:
            return await process_one_case(idx, tc)

    tasks = [bounded(i, tc) for i, tc in enumerate(test_cases)]
    results = await asyncio.gather(*tasks)
    return list(results)


# -------------------- 基线管理 --------------------
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
            warnings.append(f"⚠️ {metric}: 基线 {base_val:.3f} → 当前 {cur_val:.3f} (下降 {pct:.1f}%)")
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
        "failed": len(values) - len(valid),
    }

def generate_report(results: List[Dict], warnings: List[str],
                    baseline: Optional[Dict[str, float]]) -> Dict:
    metric_scores = {"faithfulness": [], "answer_relevancy": [], "contextual_recall": []}
    for r in results:
        for m in metric_scores:
            metric_scores[m].append(r["scores"].get(m))

    summary = {}
    for m, vals in metric_scores.items():
        summary[m] = compute_statistics(vals)

    passed = 0
    for r in results:
        if all(
            r["scores"].get(m) is not None and r["scores"].get(m, 0) >= METRIC_THRESHOLDS[m]
            for m in metric_scores
        ):
            passed += 1
    pass_rate = passed / len(results) if results else 0.0

    return {
        "total_test_cases": len(results),
        "overall_pass_rate": round(pass_rate, 3),
        "metrics_summary": summary,
        "baseline": baseline,
        "regression_warnings": warnings,
        "detailed_results": results,
    }


# -------------------- 主入口 --------------------
async def main():
    print("=" * 60)
    print("  金融 RAG 自动化评测流水线（生产级异步版）")
    print("=" * 60)
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", action="store_true",
                        help="运行组件级评估而非整体评估")
    args = parser.parse_args()

    if args.component:
        # 加载组件测试数据
        data_path = "data/component_eval_data.yaml"
        if not Path(data_path).exists():
            print(f"组件测试数据文件缺失: {data_path}")
            sys.exit(1)
        with open(data_path, "r", encoding="utf-8") as f:
            test_data = yaml.safe_load(f)["test_cases"]
        report = generate_component_report(test_data)
        output = "component_eval_report.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"组件评估报告已保存至 {output}")
    else:
        # 备份旧报告
        if Path(REPORT_PATH).exists():
            backup = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.rename(REPORT_PATH, backup)
            print(f"📦 旧报告已备份为 {backup}")

        test_cases = load_eval_dataset(EVAL_DATASET_PATH)
        if not test_cases:
            print("❌ 无测试用例，退出")
            return

        print(f"⚡ 开始异步并行评测（最多 {MAX_CONCURRENT} 并发，单用例超时 {PER_CASE_TIMEOUT}s）")
        start_all = time.time()
        results = await run_parallel_evaluation(test_cases, MAX_CONCURRENT)
        elapsed_all = time.time() - start_all

        # 计算指标均值
        metric_means = {}
        for m in ["faithfulness", "answer_relevancy", "contextual_recall"]:
            vals = [r["scores"].get(m) for r in results if r["scores"].get(m) is not None]
            metric_means[m] = statistics.mean(vals) if vals else 0.0

        # 回归检测
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

        report = generate_report(results, warnings, baseline)
        report["total_time_seconds"] = round(elapsed_all, 1)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📊 评测报告已保存至 {REPORT_PATH}")
        print(f"总用例: {report['total_test_cases']}, 整体通过率: {report['overall_pass_rate']:.1%}")
        print(f"总耗时: {elapsed_all:.1f}s")
        for m, stats in report["metrics_summary"].items():
            print(f"  {m}: 均值 {stats['mean']:.3f} (±{stats['std']:.3f}, 失败 {stats['failed']})")


if __name__ == "__main__":
    asyncio.run(main())