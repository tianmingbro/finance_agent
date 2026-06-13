"""
benchmark_cache.py
Day41 性能基准：比较无缓存、精确缓存、语义缓存三种模式的延迟和成本
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pydantic import PrivateAttr
from langchain_core.globals import set_llm_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from src.cache.caching_manager import CachingManager
from src.config import get_embedding_model_path
MODEL_PATH = get_embedding_model_path()
# -------------------- 配置 --------------------
REDIS_URL = "redis://localhost:6379"
EMBEDDING_MODEL_NAME = MODEL_PATH
LLM_MODEL = "qwen-plus"
TTL = 600  # 10 分钟，足够测试完成
COST_PER_CALL = 0.002  # 估算每次 API 调用费用（美元）
DEFAULT_ATTEMPTS = 2  # 一次命中一次未命中，足够比较缓存效果
DEFAULT_QUERY_COUNT = 5  # 默认仅取前 5 个问题，避免过多远程调用

# 10 个金融法规高频问题
TEST_QUERIES = [
    "商业银行的核心一级资本充足率要求是多少？",
    "存款保险最高偿付限额是多少？",
    "个人每年便利化购汇额度是多少？",
    "2025年1年期LPR最新报价是多少？",
    "什么是贷款市场报价利率（LPR）？",
    "资本充足率不达标会有什么后果？",
    "反洗钱法规定金融机构应履行哪些义务？",
    "系统重要性银行有哪些附加资本要求？",
    "存款保险条例的立法目的是什么？",
    "个人外汇业务中经常项目与资本项目的区别是什么？",
]


# -------------------- 计数 LLM --------------------
class CountedChatOpenAI(ChatOpenAI):
    """带调用计数的 ChatOpenAI，仅统计实际 API 调用（缓存命中不计数）"""
    _call_count: int = PrivateAttr(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# -------------------- 基准测试函数 --------------------
def run_benchmark(
    mode: str,
    queries: List[str],
    attempts: int = DEFAULT_ATTEMPTS,
) -> Dict[str, Any]:
    """
    执行某一缓存模式下的基准测试。

    Args:
        mode: "none", "exact", "semantic"
        queries: 待测试问题列表
        attempts: 每个问题的重复调用次数，用于检测缓存命中率

    Returns:
        包含运行详情、总耗时、LLM 调用次数和估算费用的字典
    """
    set_llm_cache(None)

    manager = None
    if mode == "semantic":
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        manager = CachingManager(
            redis_url=REDIS_URL,
            embedding_model=embedding,
            mode=mode,
            ttl=TTL,
        )
        manager.enable_llm_cache()
        manager.clear_cache()
    elif mode == "exact":
        manager = CachingManager(
            redis_url=REDIS_URL,
            mode=mode,
            ttl=TTL,
        )
        manager.enable_llm_cache()
        manager.clear_cache()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 OpenAI API Key，请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量。"
        )

    llm = CountedChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    runs: List[Dict[str, Any]] = []
    total_time = 0.0

    for query in queries:
        for attempt in range(1, attempts + 1):
            count_before = llm.call_count
            start = time.time()
            llm.invoke(query)
            elapsed = time.time() - start
            runs.append({
                "query": query,
                "attempt": attempt,
                "time_seconds": round(elapsed, 4),
                "cache_hit": llm.call_count == count_before,
            })
            total_time += elapsed

    total_llm_calls = llm.call_count
    estimated_cost = round(total_llm_calls * COST_PER_CALL, 4)

    return {
        "runs": runs,
        "total_time_seconds": round(total_time, 3),
        "total_llm_calls": total_llm_calls,
        "estimated_cost_usd": estimated_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="缓存基准测试脚本")
    parser.add_argument("--mode", choices=["none", "exact", "semantic", "all"], default="all",
                        help="运行哪种模式，默认为 all")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERY_COUNT,
                        help="待测问题数量，默认 10")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                        help="每个问题的调用次数，默认 2")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：少量问题、较少尝试次数")
    args = parser.parse_args()

    if args.quick:
        args.queries = min(args.queries, 5)
        args.attempts = min(args.attempts, 2)

    # 检查 Redis 是否可用（简单检查）
    import redis
    try:
        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
    except Exception:
        print("⚠️ Redis 未启动，请先执行 docker run -d -p 6379:6379 redis:latest")
        return

    selected_queries = TEST_QUERIES[: args.queries]
    print(f"🚀 开始缓存基准测试: mode={args.mode}, queries={len(selected_queries)}, attempts={args.attempts}")

    modes = [args.mode] if args.mode != "all" else ["none", "exact", "semantic"]
    results: Dict[str, Any] = {}
    for mode in modes:
        print(f"  测试模式: {mode}")
        results[mode] = run_benchmark(mode, selected_queries, attempts=args.attempts)

    if "none" not in results:
        print("⚠️ 基准对比需要 none 模式结果，建议同时执行 all 或包含 none")

    none_time = results["none"]["total_time_seconds"] if "none" in results else 0
    none_cost = results["none"]["estimated_cost_usd"] if "none" in results else 0

    def compare(data: Dict[str, Any]) -> Dict[str, Any]:
        time_saved = none_time - data["total_time_seconds"]
        time_percent = (time_saved / none_time) * 100 if none_time > 0 else 0
        cost_saved = none_cost - data["estimated_cost_usd"]
        return {
            "time_saved_seconds": round(time_saved, 3),
            "time_saved_percent": round(time_percent, 1),
            "cost_saved_usd": round(cost_saved, 4),
            "total_llm_calls": data["total_llm_calls"],
        }

    comparison = {
        "exact_vs_none": compare(results["exact"]) if "exact" in results and "none" in results else None,
        "semantic_vs_none": compare(results["semantic"]) if "semantic" in results and "none" in results else None,
    }

    report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_queries": len(selected_queries),
        "attempts_per_query": args.attempts,
        "test_queries": selected_queries,
        "modes": results,
        "comparison": comparison,
    }

    with open("cache_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("✅ 基准报告已保存至 cache_benchmark.json")
    if "none" in results:
        print(f"无缓存: 总耗时 {none_time:.2f}s, 调用 {results['none']['total_llm_calls']} 次")
    if "exact" in results:
        exact = results["exact"]
        exact_cmp = comparison["exact_vs_none"]
        print(f"精确缓存: 耗时 {exact['total_time_seconds']:.2f}s, "
              f"调用 {exact['total_llm_calls']} 次, "
              f"节省时间 {exact_cmp['time_saved_percent'] if exact_cmp else 'N/A'}%")
    if "semantic" in results:
        semantic = results["semantic"]
        semantic_cmp = comparison["semantic_vs_none"]
        print(f"语义缓存: 耗时 {semantic['total_time_seconds']:.2f}s, "
              f"调用 {semantic['total_llm_calls']} 次, "
              f"节省时间 {semantic_cmp['time_saved_percent'] if semantic_cmp else 'N/A'}%")


if __name__ == "__main__":
    main()