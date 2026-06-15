#!/usr/bin/env python3
"""直接运行意图解析测试，无需 pytest"""

from src.agent.a2a.orchestrator_agent import parse_complex_intent

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")

# 测试1：串行链
subtasks = parse_complex_intent("查一下资本充足率并评测回答是否准确")
test("识别串行链子任务数量", len(subtasks) == 2)
if len(subtasks) >= 2:
    test("第一个子任务是 rag", subtasks[0].type == "rag")
    test("rag 任务包含资本充足率", "资本充足率" in subtasks[0].query)
    test("第二个子任务是 eval", subtasks[1].type == "eval")
    test("eval 依赖第一个任务", subtasks[1].depends_on == [0])

# 测试2：并行双问
subtasks = parse_complex_intent("资本充足率和存款保险上限各是多少？")
test("识别并行双问子任务数量", len(subtasks) == 2)
for sub in subtasks:
    test(f"子任务 {sub.type} 是 rag", sub.type == "rag")
    test(f"子任务 {sub.type} 无依赖", sub.depends_on == [])

# 测试3：条件汇总
subtasks = parse_complex_intent("评测这个回答：核心一级资本充足率是8%，如果分数低于0.7就重新回答")
test("识别条件汇总子任务数量", len(subtasks) == 2)
if len(subtasks) >= 2:
    test("第一个子任务是 eval", subtasks[0].type == "eval")
    test("第二个子任务是 rag", subtasks[1].type == "rag")
    test("第二个子任务包含条件", subtasks[1].condition is not None)

# 测试4：普通问题不触发复合意图
subtasks = parse_complex_intent("资本充足率是多少？")
test("普通问题不解析为复合意图", subtasks == [])

# 测试5：闲聊不误解析
subtasks = parse_complex_intent("你好，今天天气怎么样？")
test("闲聊不误解析", subtasks == [])

print(f"\n总计: {passed} 通过, {failed} 失败")
if failed > 0:
    exit(1)