#!/usr/bin/env python3
"""
benchmark_complex.py
性能基准：模拟 10 个并发用户发送复合请求，测量 P50/P99 延迟与子任务成功率。
"""
import asyncio
import time
import statistics
import httpx
import json
import sys

ORCHESTRATOR_URL = "http://localhost:8100/a2a/task"

# 测试指令（复合意图）
COMPLEX_INSTRUCTION = "查一下资本充足率并评估答案是否准确"

# 简单任务指令（仅问答）
SIMPLE_INSTRUCTION = "资本充足率是多少？"

CONCURRENT_USERS = 10

def make_payload(instruction, session_id):
    return {
        "task": {
            "id": f"bench-{int(time.time()*1000)}",
            "session_id": session_id,
            "messages": [{"role": "user", "parts": [{"text": instruction}]}],
            "context": {},
            "status": "created",
            "artifacts": []
        }
    }

async def send_request(client, instruction, session_id):
    """发送一次请求，返回 (延迟秒, 是否成功, 是否复合, 子任务错误数)"""
    payload = make_payload(instruction, session_id)
    start = time.perf_counter()
    try:
        resp = await client.post(ORCHESTRATOR_URL, json=payload, timeout=120.0)
        elapsed = time.perf_counter() - start
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            artifacts = data.get("artifacts", [])
            # 判断是否成功：状态为 completed 且 artifacts 不包含 error
            success = (status == "completed") and not any("error" in a for a in artifacts)
            # 统计子任务错误数（用于复合任务）
            sub_errors = sum(1 for a in artifacts if "error" in a)
            return elapsed, success, True, sub_errors
        else:
            return elapsed, False, True, 1
    except Exception as e:
        elapsed = time.perf_counter() - start
        return elapsed, False, True, 1

async def run_benchmark(instruction, label):
    print(f"\n===== {label} =====")
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(CONCURRENT_USERS):
            session_id = f"bench-{label}-{i}"
            tasks.append(send_request(client, instruction, session_id))
        results = await asyncio.gather(*tasks)

    latencies = [r[0] for r in results]
    successes = sum(1 for r in results if r[1])
    total = len(results)
    success_rate = successes / total * 100

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.5)]
    p99 = latencies[int(len(latencies) * 0.99) - 1] if len(latencies) > 1 else latencies[0]

    print(f"并发数: {CONCURRENT_USERS}")
    print(f"成功率: {successes}/{total} ({success_rate:.1f}%)")
    print(f"延迟 P50: {p50:.3f}s, P99: {p99:.3f}s, 均值: {statistics.mean(latencies):.3f}s")
    # 对于复合任务，统计子任务错误
    if label == "复合任务":
        total_sub_errors = sum(r[3] for r in results)
        print(f"子任务错误总数: {total_sub_errors}")
    return latencies, success_rate

async def main():
    # 检查 Orchestrator 是否可达
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:8100/.well-known/agent-card", timeout=5)
            if resp.status_code != 200:
                print("Orchestrator 未就绪，请先启动 agents/orchestrator.py")
                sys.exit(1)
        except Exception:
            print("无法连接 Orchestrator，请确保服务已启动。")
            sys.exit(1)

    # 先执行简单任务基线
    simple_latencies, simple_success = await run_benchmark(SIMPLE_INSTRUCTION, "简单任务")

    # 再执行复合任务
    complex_latencies, complex_success = await run_benchmark(COMPLEX_INSTRUCTION, "复合任务")

    print("\n===== 对比摘要 =====")
    print(f"简单任务平均延迟: {statistics.mean(simple_latencies):.3f}s, 成功率: {simple_success:.1f}%")
    print(f"复合任务平均延迟: {statistics.mean(complex_latencies):.3f}s, 成功率: {complex_success:.1f}%")
    overhead = statistics.mean(complex_latencies) - statistics.mean(simple_latencies)
    print(f"拆分开销 (额外延迟): {overhead:.3f}s")
    # 保存基线
    with open("benchmark_baseline.json", "w") as f:
        json.dump({
            "simple": {
                "p50": statistics.median(simple_latencies),
                "p99": statistics.quantiles(simple_latencies, n=100)[98] if len(simple_latencies) > 1 else simple_latencies[0],
                "mean": statistics.mean(simple_latencies),
                "success_rate": simple_success
            },
            "complex": {
                "p50": statistics.median(complex_latencies),
                "p99": statistics.quantiles(complex_latencies, n=100)[98] if len(complex_latencies) > 1 else complex_latencies[0],
                "mean": statistics.mean(complex_latencies),
                "success_rate": complex_success
            }
        }, f, indent=2)
    print("基线已保存至 benchmark_baseline.json")

if __name__ == "__main__":
    asyncio.run(main())