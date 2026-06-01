"""
test_financial_skill.py
Day30 测试交付物：触发精度测试 + 延迟加载测试
对应 Day29 AI 测试 Skill 映射：
  - 测试点1：触发精度（该触发时触发，不该触发时不触发）
  - 测试点2：延迟加载正确性（资源未加载前不应访问向量库）
"""
import pytest
from src.skill.financial_rag_skill import FinancialRAGSkill  # 确保 skill 文件在同一目录


class TestTriggerPrecision:
    """测试 should_trigger 的精准性"""

    def setup_method(self):
        self.skill = FinancialRAGSkill()

    # 正向触发用例
    @pytest.mark.parametrize("query", [
        "资本充足率是多少？",
        "最新的LPR报价",
        "个人购汇额度有限制吗？",
        "房贷利率和LPR的关系",
        "外汇管理的政策有哪些？",
        "商业银行监管要求",
    ])
    def test_should_trigger_on_finance_terms(self, query):
        """包含金融触发词的问题必须触发"""
        assert self.skill.should_trigger(query), f"未能触发：{query}"

    # 负向触发用例
    @pytest.mark.parametrize("query", [
        "今天天气真好",
        "帮我写一首诗",
        "Python的列表如何排序？",
        "你好吗",
        "推荐一本小说",
    ])
    def test_should_not_trigger_on_irrelevant(self, query):
        """不包含任何金融触发词的问题不应触发"""
        assert not self.skill.should_trigger(query), f"错误触发：{query}"

    # 边界情况：触发词出现在否定语境中（当前实现仅依赖关键词，因此仍会触发，但测试应记录预期行为）
    def test_trigger_with_negation_context(self):
        """否定语境中的触发词仍会触发（已知行为，后续可优化）"""
        # 当前版本基于关键词匹配，未做语义分析
        assert self.skill.should_trigger("我不关心资本充足率")


class TestLazyLoading:
    """测试资源延迟加载机制"""

    def setup_method(self):
        self.skill = FinancialRAGSkill()

    def test_resource_not_loaded_initially(self):
        """技能初始化后，资源管理器未加载重资源"""
        assert not self.skill.resource_mgr._loaded

    def test_access_retriever_before_load_raises(self):
        """在调用 load_resources 前访问检索器应抛出 RuntimeError"""
        with pytest.raises(RuntimeError, match="资源尚未加载"):
            _ = self.skill.resource_mgr.retriever

    def test_access_llm_before_load_raises(self):
        """在调用 load_resources 前访问 LLM 应抛出 RuntimeError"""
        with pytest.raises(RuntimeError, match="资源尚未加载"):
            _ = self.skill.resource_mgr.llm

    def test_load_resources_makes_retriever_available(self):
        """调用 load_resources 后检索器可正常访问"""
        self.skill.resource_mgr.load_resources()
        # 不抛异常即为通过
        retriever = self.skill.resource_mgr.retriever
        assert retriever is not None

    def test_load_resources_is_idempotent(self):
        """重复调用 load_resources 不会出错"""
        self.skill.resource_mgr.load_resources()
        first_retriever = self.skill.resource_mgr.retriever
        self.skill.resource_mgr.load_resources()
        second_retriever = self.skill.resource_mgr.retriever
        assert first_retriever is second_retriever

    def test_run_triggers_lazy_loading_when_needed(self):
        """run 方法在需要检索时自动触发延迟加载"""
        query = "资本充足率是多少？"
        # 确保运行前资源未加载
        assert not self.skill.resource_mgr._loaded
        answer = self.skill.run(query)
        # 运行后资源应已加载
        assert self.skill.resource_mgr._loaded
        # 答案应包含金融信息（不要求精确，但不应是错误信息）
        assert len(answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])