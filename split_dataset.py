"""
split_dataset.py
Day34 数据准备：按 80/20 分层抽样切分评测数据集
"""
import yaml
import random
import math
from pathlib import Path
from typing import Dict, List, Any

# -------------------- 配置 --------------------
INPUT_DATASET = "week5_finance/data/eval_dataset_v2.yaml"   # 扩充后的完整数据集
OUTPUT_EVAL = "week5_finance/data/eval_dataset.yaml"        # 80% 公开评测集（覆盖旧文件）
OUTPUT_HOLDOUT = "week5_finance/data/holdout_dataset.yaml"  # 20% holdout 集
RANDOM_SEED = 42                              # 固定种子，保证可复现
HOLDOUT_RATIO = 0.2                           # holdout 比例

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """加载 YAML 数据集，返回用例列表（保留类别信息）"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_cases = []
    for category in data.get("categories", []):
        cat_name = category["category"]
        for entry in category.get("entries", []):
            entry["_category"] = cat_name  # 临时标记类别
            all_cases.append(entry)
    return all_cases

def group_by_category(cases: List[Dict]) -> Dict[str, List[Dict]]:
    """按类别分组"""
    groups = {}
    for case in cases:
        cat = case.pop("_category", "unknown")
        groups.setdefault(cat, []).append(case)
    return groups

def stratified_split(groups: Dict[str, List[Dict]], holdout_ratio: float) -> tuple:
    """分层抽样：每个类别独立切分"""
    random.seed(RANDOM_SEED)

    eval_groups = {}
    holdout_groups = {}

    for cat, cases in groups.items():
        shuffled = cases[:]  # 复制
        random.shuffle(shuffled)

        # 确保 holdout 至少有 1 条
        holdout_count = max(1, math.floor(len(shuffled) * holdout_ratio))
        
        holdout_groups[cat] = shuffled[:holdout_count]
        eval_groups[cat] = shuffled[holdout_count:]

        print(f"  {cat}: 总计 {len(cases)} → 评测集 {len(eval_groups[cat])} / holdout {len(holdout_groups[cat])}")

    return eval_groups, holdout_groups

def save_dataset(groups: Dict[str, List[Dict]], output_path: str):
    """保存为与原始格式一致的 YAML"""
    categories = []
    for cat_name, entries in groups.items():
        # 根据类别名推断描述
        descriptions = {
            "factual_query": "事实查询：答案直接存在于知识库中",
            "reasoning_query": "推理查询：需综合多条信息或简单推理",
            "ambiguous_query": "模糊查询：指代不明或存在歧义",
            "adversarial_query": "对抗查询：注入指令或恶意请求",
        }
        categories.append({
            "category": cat_name,
            "description": descriptions.get(cat_name, ""),
            "entries": entries,
        })

    output = {"categories": categories}

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, allow_unicode=True, sort_keys=False)

    total = sum(len(c["entries"]) for c in categories)
    print(f"✅ 已保存 {total} 条至 {output_path}")

def main():
    print("=" * 60)
    print("  数据集分层抽样 (80/20)")
    print("=" * 60)

    # 1. 加载完整数据集
    cases = load_dataset(INPUT_DATASET)
    print(f"📋 加载完整数据集，共 {len(cases)} 条")

    # 2. 按类别分组
    groups = group_by_category(cases)
    print(f"📂 共 {len(groups)} 个类别: {list(groups.keys())}")

    # 3. 分层抽样
    print("\n🔀 执行分层抽样...")
    eval_groups, holdout_groups = stratified_split(groups, HOLDOUT_RATIO)

    # 4. 保存结果
    save_dataset(eval_groups, OUTPUT_EVAL)
    save_dataset(holdout_groups, OUTPUT_HOLDOUT)

    # 5. 统计
    eval_total = sum(len(v) for v in eval_groups.values())
    holdout_total = sum(len(v) for v in holdout_groups.values())
    print(f"\n📊 切分结果: 评测集 {eval_total} 条 / holdout {holdout_total} 条 "
          f"({holdout_total/(eval_total+holdout_total)*100:.1f}%)")

if __name__ == "__main__":
    main()