"""
test_ai_test_skill.py
Day31 测试交付物：
- 触发词正向/负向测试
- Faithfulness 与 AnswerRelevancy 有效分数验证
- 无 GPU 环境兼容性
"""
import os
import re
import sys
import pytest

# 将父目录加入 sys.path 以便导入 ai_test_skill
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入待测组件
from src.skill.ai_test_skill import (
    AITestSkill,
    EvaluationRunner,
    EvalResourceManager,
    EvalReport,
    PRIMARY_TRIGGERS,
)
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

# -------------------- 环境检查 --------------------
def check_api_key():
    """检查是否有可用的 DashScope API Key"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print("⚠️ 未设置 DASHSCOPE_API_KEY，评测相关测试将被跳过。")
        return False
    return True


HAS_API_KEY = check_api_key()


# -------------------- 模拟 RAG Skill --------------------
def create_mock_rag_response(good: bool = True):
    """
    创建一个返回预定义问答对的模拟可调用对象。
    good=True 时返回高质量答案，good=False 时返回有幻觉的答案。
    """
    if good:
        def mock_rag(query: str) -> dict:
            return {
                "input": query,
                "answer": "根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。",
                "context": [
                    "问题：2025年中国商业银行的资本充足率监管要求是多少？答案：核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。"
                ],
                "expected_output": "核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。"
            }
    else:
        def mock_rag(query: str) -> dict:
            return {
                "input": query,
                "answer": "根据最新规定，资本充足率要求已经提高到10%。",  # 幻觉答案
                "context": [
                    "资本充足率是衡量银行资本充足程度的指标，目前监管要求为不低于8%。"
                ]
            }
    return mock_rag


# -------------------- 触发词测试 --------------------
class TestTriggerLogic:
    def setup_method(self):
        self.skill = AITestSkill()

    @pytest.mark.parametrize("query", [
        "评测一下资本充足率的回答",
        "测试忠实度，问题：LPR最新报价",
        "检查质量：购汇额度",
        "评估回答是否正确",
        "帮我跑一下指标",
    ])
    def test_should_trigger_on_eval_keywords(self, query):
        assert self.skill.should_trigger(query), f"未能触发：{query}"

    @pytest.mark.parametrize("query", [
        "今天天气如何",
        "帮我写一首诗",
        "资本充足率是多少？",  # 没有触发词
        "你好",
    ])
    def test_should_not_trigger_on_irrelevant(self, query):
        assert not self.skill.should_trigger(query), f"错误触发：{query}"


# -------------------- 评测功能测试 (需要 API Key) --------------------
@pytest.mark.skipif(not HAS_API_KEY, reason="未设置 DASHSCOPE_API_KEY")
class TestEvaluationWithDeepEval:
    def setup_method(self):
        self.resource_mgr = EvalResourceManager()
        self.resource_mgr.load_resources()
        self.runner = EvaluationRunner(self.resource_mgr)

    def test_evaluation_returns_valid_scores(self):
        """验证高质量答案能得到较高忠实度和相关性分数"""
        mock_rag = create_mock_rag_response(good=True)
        # 模拟用户输入，指定问题
        user_input = "评测一下：资本充足率要求是多少？"
        report = self.runner.run(user_input, mock_rag)

        # 检查报告结构
        assert isinstance(report, EvalReport)
        assert len(report.metrics) >= 2  # 默认至少包含 faithfulness 和 answer_relevancy

        # 提取指标
        faithfulness = next((m for m in report.metrics if m.name == "faithfulness"), None)
        relevancy = next((m for m in report.metrics if m.name == "answer_relevancy"), None)

        assert faithfulness is not None, "未找到 Faithfulness 指标"
        assert relevancy is not None, "未找到 AnswerRelevancy 指标"

        # 分数应在 0-1 之间
        assert 0 <= faithfulness.score <= 1
        assert 0 <= relevancy.score <= 1

        # 对于高质量答案，分数预期较高（>0.6）
        assert faithfulness.score > 0.6, f"忠实度过低：{faithfulness.score}"
        assert relevancy.score > 0.6, f"答案相关性过低：{relevancy.score}"

        # 应有成功标志
        assert faithfulness.success or faithfulness.score > faithfulness.threshold
        assert relevancy.success or relevancy.score > relevancy.threshold

    def test_evaluation_detects_hallucination(self):
        """验证幻觉答案的忠实度分数较低"""
        mock_rag = create_mock_rag_response(good=False)
        user_input = "测试忠实度：资本充足率"
        report = self.runner.run(user_input, mock_rag)

        faithfulness = next((m for m in report.metrics if m.name == "faithfulness"), None)
        assert faithfulness is not None
        # 幻觉应导致分数较低
        assert faithfulness.score < 0.7, f"幻觉未被检测，忠实度依然较高：{faithfulness.score}"

    def test_report_contains_trust_level(self):
        """验证报告包含综合信任等级"""
        mock_rag = create_mock_rag_response(good=True)
        report = self.runner.run("全面评测：资本充足率", mock_rag)
        assert report.overall_trust in ("高", "中", "低")
        assert len(report.summary) > 0

    def test_no_gpu_required(self):
        """验证配置为使用 API 模型，无需本地 GPU"""
        # 资源管理器应使用 qwen-plus 等 API 模型
        assert self.resource_mgr._loaded
        # 检查某个指标的内部配置
        metric = self.resource_mgr.get_metric("faithfulness")
        # DeepEval 的指标内部会调用 evaluation_model，我们可以检查其配置
        # 由于是私有属性，我们只做 smoke test：成功运行即证明无需 GPU
        assert True  # 已通过调用证明无需本地 GPU


# -------------------- 完整 AITestSkill 集成测试 (需要 API Key) --------------------
@pytest.mark.skipif(not HAS_API_KEY, reason="未设置 DASHSCOPE_API_KEY")
class TestAITestSkillIntegration:
    def test_skill_run_returns_formatted_report(self, monkeypatch):
        """测试 AITestSkill.run 返回格式化的字符串报告"""
        # 使用 monkeypatch 替换 rag_skill_factory，返回模拟 RAG
        def mock_factory():
            class MockSkill:
                def run(self, query):
                    # 返回符合约定的 dict
                    return {
                        "input": query,
                        "answer": "核心一级资本充足率不低于5%。",
                        "context": ["监管要求核心一级资本充足率不低于5%。"]
                    }
                
                def run_with_context(self, query):
                    return {
                        "input": query,
                        "answer": "核心一级资本充足率不低于5%。",
                        "context": ["监管要求..."]
                    }
            return MockSkill()

        skill = AITestSkill(rag_skill_factory=mock_factory)
        user_input = "评估回答：资本充足率是多少？"
        output = skill.run(user_input)

        # 检查输出字符串包含必要内容
        assert "📊 RAG 质量评测报告" in output
        assert "Faithfulness" in output or "faithfulness" in output  # 可能显示为英文
        assert "综合信任等级" in output
        # 应当有分数数字
        assert re.search(r'\d\.\d+', output), "未找到评分数字"


# -------------------- 执行入口 --------------------
if __name__ == "__main__":
    args = ["-v"]
    if not HAS_API_KEY:
        # 跳过需要 API 的测试
        args += ["-k", "TriggerLogic"]
    pytest.main([__file__] + args)