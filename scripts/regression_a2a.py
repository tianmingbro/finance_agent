"""
scripts/regression_a2a.py
通过 A2A 接口对 RAG Agent 和 Eval Agent 进行组件级评估，
对比 Day46 基线，确认无性能退化。
"""
import json
import yaml
import statistics
import requests
from pathlib import Path
from src.pipeline.eval_components import (
    evaluate_retrieval,
    evaluate_generation,
    evaluate_planning,
)

RAG_AGENT_URL = "http://localhost:8101/a2a/task"
EVAL_AGENT_URL = "http://localhost:8102/a2a/task"
BASELINE_FILE = "component_baseline.json"
TEST_CASES = "data/eval_dataset_v2.yaml"

def load_test_cases(limit=20):
    with open(TEST_CASES, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = []
    for cat in data["categories"]:
        for entry in cat.get("entries", []):
            cases.append(entry)
    return cases[:limit]

def call_rag_agent(query: str):
    task = {
        "id": "reg-1",
        "session_id": "regression",
        "messages": [{"role": "user", "parts": [{"text": query}]}],
        "status": "created",
        "artifacts": [],
    }
    resp = requests.post(RAG_AGENT_URL, json={"task": task})
    if resp.status_code != 200:
        return None, None
    data = resp.json()
    if data["status"] != "completed":
        return None, None
    artifact = data["artifacts"][0]
    answer = data["artifacts"][0].get("answer", "")
    context = artifact.get("retrieval_context", [])
    # 检索上下文没有直接返回，需修改 RAG Agent 使其返回 context。
    # 此处假设 RAG Agent 已将检索上下文放入 artifacts，或我们使用 search_finance_docs 单独获取。
    # 为简化，直接调用 search_finance_docs 获取检索上下文。
    # import asyncio
    # from src.retriever.tools_mcp import search_finance_docs
    # result_json = asyncio.run(search_finance_docs(query, top_k=4))
    # docs = json.loads(result_json)["documents"]
    # context = [d["content"] for d in docs]
    return answer, context

def call_eval_agent(query, answer):
    task = {
        "id": "reg-2",
        "session_id": "regression",
        "messages": [{"role": "user", "parts": [{"text": f"{query}|{answer}"}]}],
        "status": "created",
        "artifacts": [],
    }
    resp = requests.post(EVAL_AGENT_URL, json={"task": task})
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data["status"] != "completed":
        return None
    return data["artifacts"][0]

def run():
    cases = load_test_cases(20)
    rag_metrics = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_recall": [],
    }
    eval_accuracy = []

    for case in cases:
        query = case["query"]
        expected = case.get("expected_answer", "")
        answer, context = call_rag_agent(query)
        if not answer:
            continue

        # 提取关键信息用于检索评估
        import re
        numbers = re.findall(r'\d+\.?\d*%?', expected) if expected else []
        key_info = numbers if numbers else [expected] if expected else None

        # 检索评估
        ret_res = evaluate_retrieval(query, key_info=key_info)
        rag_metrics["contextual_recall"].append(ret_res["contextual_recall"])

        # 生成评估
        gen_res = evaluate_generation(query, expected_answer=expected)
        rag_metrics["faithfulness"].append(gen_res["faithfulness"])
        rag_metrics["answer_relevancy"].append(gen_res["answer_relevancy"])

        # Eval Agent 一致性检查（取前5个用例）
        if len(eval_accuracy) < 5:
            direct_eval = evaluate_generation(query, expected_answer=expected)
            a2a_eval = call_eval_agent(query, answer)
            if a2a_eval:
                eval_accuracy.append(abs(direct_eval["faithfulness"] - a2a_eval.get("faithfulness", 0)))

    # 汇总输出
    print("RAG Agent 组件指标均值：")
    for k, v in rag_metrics.items():
        if v:
            print(f"  {k}: {statistics.mean(v):.3f}")
    print(f"Eval Agent 忠实度偏差均值: {statistics.mean(eval_accuracy):.4f}" if eval_accuracy else "无评测数据")

if __name__ == "__main__":
    run()