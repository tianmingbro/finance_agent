import yaml

FIXES = {
    "new_035": "reasoning_query", "new_033": "reasoning_query",
    "new_045": "reasoning_query", "new_040": "reasoning_query",
    "new_044": "reasoning_query", "new_043": "reasoning_query",
    "new_054": "reasoning_query", "new_058": "reasoning_query",
    "new_049": "reasoning_query", "new_055": "reasoning_query",
    "new_039": "reasoning_query", "new_023": "reasoning_query",
    "new_022": "adversarial_query",
    "new_013": "ambiguous_query", "new_029": "ambiguous_query", "new_021": "ambiguous_query",
}

with open("data/exp_test_set.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

for case in data["test_cases"]:
    if case["id"] in FIXES:
        old = case["category"]
        case["category"] = FIXES[case["id"]]
        print(f"  {case['id']}: {old} → {case['category']}")

with open("data/exp_test_set_v2.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False)
print("修正完成，保存至 data/exp_test_set_v2.yaml")