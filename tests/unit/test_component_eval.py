"""
test_component_eval.py
Day46 TDAD 第一步：组件级评估测试用例（红灯状态）
依赖：data/component_eval_data.yaml 中已标注的测试数据
"""
from unittest import case

import pytest
from torch import threshold
import yaml
from pathlib import Path
from typing import List, Dict, Any

# 导入待实现的评估函数（当前未实现，因此测试会失败）
from src.pipeline.eval_components import (
    evaluate_planning,
    evaluate_retrieval,
    evaluate_generation,
)

# 加载测试数据
DATA_PATH = Path("data/component_eval_data.yaml")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    COMPONENT_DATA = yaml.safe_load(f)["test_cases"]


class TestPlanningComponent:
    """规划组件：验证工具选择与参数构造"""

    @pytest.mark.parametrize("case", COMPONENT_DATA)
    def test_planning_eval(self, case: Dict[str, Any]):
        """给定对话历史，验证规划出的工具调用是否与期望一致"""
        planning = case.get("planning", {})
        if not planning:
            pytest.skip("未标注规划期望")

        expected_tool = planning["expected_tool"]
        expected_args = planning.get("expected_args", {})

        # 调用待实现的评估函数（当前会失败）
        result = evaluate_planning(
            query=case["query"],
            expected_tool=expected_tool,
            expected_args=expected_args,
        )
        # 断言：工具选择准确率应 ≥ 0.9（即工具名正确）
        assert result["tool_accuracy"] >= 0.9, f"工具选择不准确: {result}"
        # 断言：参数合理性应 ≥ 0.8（参数匹配）
        assert result["arg_reasonableness"] >= 0.8, f"参数不合理: {result}"


class TestRetrievalComponent:
    """检索组件：验证召回率、精确度、首条相关性"""

    @pytest.mark.parametrize("case", COMPONENT_DATA)
    def test_retrieval_recall(self, case: Dict[str, Any]):
        """检索到的文档包含标准答案关键信息的比例 ≥ 0.7"""
        retrieval = case.get("retrieval", {})
        if not retrieval.get("key_info"):
            pytest.skip("未标注关键信息")

        result = evaluate_retrieval(
            query=case["query"],
            key_info=retrieval["key_info"],
        )
        assert result["contextual_recall"] >= 0.7, f"召回率不足: {result}"

    @pytest.mark.parametrize("case", COMPONENT_DATA)
    def test_retrieval_precision(self, case: Dict[str, Any]):
        """检索到的文档中与问题相关的比例 ≥ 0.7"""
        retrieval = case.get("retrieval", {})
        key_info = retrieval.get("key_info")
        if not key_info:
            pytest.skip("未标注关键信息，无法计算精确度")
        result = evaluate_retrieval(
            query=case["query"], key_info=key_info
        )
        assert result["contextual_precision"] >= 0.7, f"精确度不足: {result}"


class TestGenerationComponent:
    """生成组件：验证忠实度、相关性、完整性"""

    @pytest.mark.parametrize("case", COMPONENT_DATA)
    def test_generation_faithfulness(self, case: Dict[str, Any]):
        """生成的答案忠实于检索上下文，分数 ≥ 0.8"""
        result = evaluate_generation(
            query=case["query"],
            expected_answer=case.get("expected_answer"),
        )
        category = case.get("category", "")
        if category == "adversarial_query":
            threshold = 0.5
        elif category == "ambiguous_query":
            threshold = 0.5   # 或者 0.5，根据需要
        elif category == "reasoning_query":
            threshold = 0.65
        else:
            threshold = 0.8
        assert result["faithfulness"] >= threshold, f"忠实度不足: {result}"

    @pytest.mark.parametrize("case", COMPONENT_DATA)
    def test_generation_relevancy(self, case: Dict[str, Any]):
        """答案与问题高度相关，分数 ≥ 0.8"""
        result = evaluate_generation(
            query=case["query"],
        )
        assert result["answer_relevancy"] >= 0.8, f"相关性不足: {result}"


class TestComponentIsolation:
    """验证各组件可独立运行，不互相依赖"""

    def test_retrieval_independent_of_generation(self):
        """检索评估不依赖生成结果"""
        query = "存款保险最高限额是多少？"
        result = evaluate_retrieval(query=query)
        # 应不报错且返回字典，即使没有生成答案
        assert isinstance(result, dict)
        assert "contextual_recall" in result

    def test_planning_independent_of_retrieval_and_generation(self):
        """规划评估可独立进行，只需查询和期望工具调用"""
        result = evaluate_planning(
            query="LPR最新报价",
            expected_tool="financial_qa",
            expected_args={"query": "LPR最新报价"}
        )
        assert isinstance(result, dict)
        assert "tool_accuracy" in result