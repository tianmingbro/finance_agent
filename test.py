import json

def load_report(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

base = load_report("data/eval_report_nocache.json")
cached = load_report("data/eval_report_cache.json")

print("=" * 50)
print("        引入缓存前后对比")
print("=" * 50)

# 1. 核心质量指标（不应有明显变化）
for m in ["faithfulness", "answer_relevancy", "contextual_recall"]:
    base_mean = base["metrics_summary"][m]["mean"]
    cache_mean = cached["metrics_summary"][m]["mean"]
    diff = (cache_mean - base_mean) * 100
    print(f"{m:25s}: {base_mean:.4f} → {cache_mean:.4f}  ({diff:+.1f}%)")

# 2. API 调用次数（需在评测脚本中嵌入计数器，这里假设已记录）
base_calls = base.get("total_llm_calls", "未统计")
cache_calls = cached.get("total_llm_calls", "未统计")
print(f"\nLLM API 调用次数: 基线 {base_calls} → 缓存 {cache_calls}")

# 3. 端到端耗时（需在评测脚本中记录，此处仅示例）
base_time = base.get("total_time_seconds", "未统计")
cache_time = cached.get("total_time_seconds", "未统计")
print(f"总耗时(秒): 基线 {base_time} → 缓存 {cache_time}")