"""
AI 测试 Skill - 三层渐进加载封装
Day31 核心交付物
"""
import logging
import re
import os
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

# DeepEval 相关
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.openai_model import GPTModel  # 确保导入

# 用于延迟导入 FinancialRAGSkill（避免循环引用）
import importlib
logger = logging.getLogger(__name__)

# -------------------- 1. 元数据层 --------------------
# 触发词定义
PRIMARY_TRIGGERS = ["评测", "测试", "检查质量", "评估回答"]
AUXILIARY_TRIGGERS = ["跑一下指标", "忠实度", "召回率", "幻觉检测",
                      "答案相关性", "正确性", "指标", "打分", "评价"]
ALL_EVAL_KEYWORDS = PRIMARY_TRIGGERS + AUXILIARY_TRIGGERS



@dataclass
class EvalSkillMetadata:
    """AI 测试 Skill 元数据"""
    name: str = "rag_evaluator"
    version: str = "0.1.0"
    description: str = "对 RAG 应用的回答进行自动化质量评估，支持忠实度、答案相关性、召回率等指标"
    author: str = "AI转型训练营"
    dependencies: List[str] = field(default_factory=lambda: [
        "deepeval>=0.21.0",
        "financial_rag_skill (内部 Skill)"
    ])
    trigger_keywords: List[str] = field(default_factory=lambda: ALL_EVAL_KEYWORDS)
    performance_baseline: Dict[str, float] = field(default_factory=lambda: {
        "single_eval_seconds": 5.0,
        "batch_eval_per_query_seconds": 3.0
    })


# -------------------- 2. 指令层 --------------------
class EvalInstructionLoader:
    """负责加载评测系统指令"""

    def __init__(self, metadata: EvalSkillMetadata):
        self.metadata = metadata
        # 编译触发词正则
        escaped = [re.escape(kw) for kw in self.metadata.trigger_keywords]
        self._pattern = re.compile("|".join(escaped), re.IGNORECASE)

        # 默认系统指令
        self.system_prompt = (
            "你是一个专业的 RAG 质量评估专家，使用 DeepEval 框架对问答结果进行量化分析。\n\n"
            "**可用指标**：\n"
            "- 忠实度 (Faithfulness)：回答中的信息是否能从检索上下文中推导，防止幻觉\n"
            "- 答案相关性 (Answer Relevancy)：回答与问题的关联程度\n"
            "- 上下文召回率 (Contextual Recall)：检索上下文覆盖标准答案的程度\n"
            "- 上下文精确度 (Contextual Precision)：检索上下文的噪声比例\n\n"
            "**输出格式要求**：\n"
            "1. 对每个指标给出 0~1 之间的分数\n"
            "2. 用一句话解释分数含义\n"
            "3. 如果分数低于 0.6，必须给出具体改进建议\n"
            "4. 最后提供一个综合信任等级（高/中/低）\n\n"
            "**约束**：\n"
            "- 只基于提供的问答对和检索上下文进行评估，不引入外部知识\n"
            "- 不要因为回答风格而扣分，只关注事实正确性与信息完整性\n"
            "- 如果无法计算某个指标，请明确说明原因并跳过\n"
        )

    def should_trigger(self, user_input: str) -> bool:
        # """检查是否包含主触发词（任一）"""
        # # 主触发词必须出现至少一个
        # for kw in PRIMARY_TRIGGERS:
        #     if kw in user_input:
        #         return True
        # return False
        return bool(self._pattern.search(user_input))

    def load_instruction(self, user_input: str) -> str:
        """条件加载：匹配触发词则返回指令，否则空字符串"""
        if not self.should_trigger(user_input):
            return ""
        return self.system_prompt


# -------------------- 3. 资源管理器 (延迟加载) --------------------
class EvalResourceManager:
    """管理 DeepEval 模型的延迟加载和缓存"""

    def __init__(self):
        self._loaded = False
        self._metrics_cache = {}  # 缓存指标实例

    def load_resources(self, model_name: str = "qwen-plus"):
        """首次调用时初始化 DeepEval 模型，后续复用"""
        if self._loaded:
            return
        logger.info("延迟加载 正在初始化 DeepEval 评估模型...")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY，无法初始化评测模型")
        # 创建指向 DashScope 兼容接口的 GPTModel
        # self._custom_model = GPTModel(
        #     model=model_name,
        #     api_key=api_key,
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        self._custom_model = GPTModel(
            model="qwen2.5:7b",                          # Ollama 中的模型名
            api_key="ollama",                            # 任意非空字符串
            base_url="http://localhost:11434/v1",        # Ollama 的 OpenAI 兼容端点
        )
        self._loaded = True
        logger.info("DeepEval 评估资源就绪")

    def get_metric(self, metric_name: str, model: str = "qwen-plus",
                   threshold: float = 0.7) -> object:
        if not self._loaded:
            raise RuntimeError("请先调用 load_resources()")
        """获取缓存的指标实例（懒实例化）"""
        if metric_name not in self._metrics_cache:
            if metric_name == "faithfulness":
                metric = FaithfulnessMetric(model=self._custom_model, threshold=threshold,
                                            include_reason=True)
            elif metric_name == "answer_relevancy":
                metric = AnswerRelevancyMetric(model=self._custom_model, threshold=threshold,
                                               include_reason=True)
            elif metric_name == "contextual_recall":
                metric = ContextualRecallMetric(model=self._custom_model, threshold=threshold,
                                                include_reason=True)
            else:
                raise ValueError(f"不支持的指标: {metric_name}")
            self._metrics_cache[metric_name] = metric
        return self._metrics_cache[metric_name]


# -------------------- 评测报告数据结构 --------------------
@dataclass
class MetricResult:
    name: str
    score: float
    threshold: float
    success: bool
    reason: Optional[str] = None


@dataclass
class EvalReport:
    query: str
    answer: str
    retrieval_context: List[str]
    metrics: List[MetricResult]
    overall_trust: str
    summary: str
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieval_context": self.retrieval_context,
            "metrics": [{"name": m.name, "score": m.score, "threshold": m.threshold,
                         "success": m.success, "reason": m.reason} for m in self.metrics],
            "overall_trust": self.overall_trust,
            "summary": self.summary,
            "improvements": self.improvements,
        }


# -------------------- 评测引擎 (EvaluationRunner) --------------------
class EvaluationRunner:
    """封装 DeepEval 指标计算，调用目标 RAG Skill"""

    def __init__(self, resource_mgr: EvalResourceManager):
        self.resource_mgr = resource_mgr

    def resolve_metrics(self, user_input: str) -> List[str]:
        """从用户输入中解析需要运行的指标名"""
        selected = []
        lower = user_input.lower()
        # 默认全部跑
        if any(kw in lower for kw in ["全面评测", "全部指标", "所有指标"]):
            return ["faithfulness", "answer_relevancy", "contextual_recall"]
        if "忠实度" in lower or "faithfulness" in lower:
            selected.append("faithfulness")
        if "相关性" in lower or "relevancy" in lower:
            selected.append("answer_relevancy")
        if "召回" in lower or "recall" in lower:
            selected.append("contextual_recall")
        if "幻觉" in lower:
            selected.append("faithfulness")
        return selected or ["faithfulness", "answer_relevancy", "contextual_recall"]

    def extract_query(self, user_input: str) -> Optional[str]:
        """从指令中提取待测问题"""
        import re
        # 匹配引号或冒号后的内容
        patterns = [
            r'["\u201c](.+?)["\u201d]',
            r'[\u2018](.+?)[\u2019]',
            r'：(.+?)[。！？\n]',
            r':\s*(.+?)[.!\n]',
        ]
        for pat in patterns:
            match = re.search(pat, user_input)
            if match:
                return match.group(1).strip()
        return None

    def run(self, user_input: str,
        rag_skill: Callable[[str], dict]) -> EvalReport:
        # Step 1: 提取问题
        query = self.extract_query(user_input)
        if not query:
            query = "商业银行的资本充足率要求是多少？"

        # Step 2: 调用 RAG Skill
        logger.info("正在调用金融 RAG Skill 回答: %s", query)
        rag_response = rag_skill(query)

        # Step 3: 确定指标
        metric_names = self.resolve_metrics(user_input)

        # 如果缺少 expected_output，移除需要该字段的指标（如 ContextualRecall）
        if "expected_output" not in rag_response or not rag_response.get("expected_output"):
            metric_names = [m for m in metric_names if m != "contextual_recall"]

        # Step 4: 逐指标评测
        results = []
        for mname in metric_names:
            metric = self.resource_mgr.get_metric(mname)
            test_case = LLMTestCase(
                input=rag_response["input"],
                actual_output=rag_response["answer"],
                retrieval_context=rag_response.get("context", []),
                expected_output=rag_response.get("expected_output"),  # 可选
            )
            logger.info("正在计算 %s ...", mname)
            metric.measure(test_case)
            results.append(MetricResult(
                name=mname,
                score=metric.score,
                threshold=0.7,
                success=metric.success,
                reason=metric.reason
            ))

        # Step 5: 生成总结
        trust, summary, improvements = self._generate_summary(results)

        return EvalReport(
            query=rag_response["input"],
            answer=rag_response["answer"],
            retrieval_context=rag_response.get("context", []),
            metrics=results,
            overall_trust=trust,
            summary=summary,
            improvements=improvements,
        )

    def _generate_summary(self, metrics: List[MetricResult]) -> tuple:
        avg = sum(m.score for m in metrics) / len(metrics) if metrics else 0
        failed = [m for m in metrics if not m.success]
        if not failed:
            trust = "高" if avg >= 0.8 else "中"
            summary = f"回答在所有 {len(metrics)} 项指标上均通过（平均 {avg:.0%}）。总体可信。"
        else:
            trust = "低" if avg < 0.6 else "中"
            failed_names = "、".join(m.name for m in failed)
            summary = f"回答在 {failed_names} 指标上未通过阈值，平均 {avg:.0%}。建议重点关注。"
        improvements = [f"[{m.name}] {m.reason}" for m in failed]
        # 对低分但通过的也给出建议
        for m in metrics:
            if 0.6 <= m.score < 0.75 and m.success:
                improvements.append(f"[{m.name}] 得分{m.score:.0%}，可优化。")
        return trust, summary, improvements


# -------------------- 4. AI 测试 Skill 主类 --------------------
class AITestSkill:
    """AI 测试 Skill，封装触发、评测、报告输出"""

    def __init__(self, rag_skill_factory: Callable = None):
        """
        Args:
            rag_skill_factory: 返回 RAG Skill 实例的可调用对象。
                               若未提供，则默认尝试加载 FinancialRAGSkill。
        """
        # 第一层：元数据
        self.metadata = EvalSkillMetadata()
        # 第二层：指令加载器
        self.instruction_loader = EvalInstructionLoader(self.metadata)
        # 第三层：资源管理器（延迟加载）
        self.resource_mgr = EvalResourceManager()
        # 评测引擎
        self.eval_runner = EvaluationRunner(self.resource_mgr)
        # 目标 RAG Skill 工厂
        self._rag_factory = rag_skill_factory or self._default_rag_factory

    @staticmethod
    def _default_rag_factory():
        """默认工厂：延迟导入 FinancialRAGSkill 避免循环依赖"""
        try:
            from src.skill.financial_rag_skill import FinancialRAGSkill
            return FinancialRAGSkill()
        except ImportError:
            raise RuntimeError("未找到 FinancialRAGSkill，请提供 rag_skill_factory 参数")

    def should_trigger(self, user_input: str) -> bool:
        return self.instruction_loader.should_trigger(user_input)

    def run(self, user_input: str) -> str:
        """技能入口：返回格式化的评测报告（字符串）"""
        if not self.should_trigger(user_input):
            return "我是评测助手，请说'评测'或'测试'启动质量评估，例如：'评测一下资本充足率的回答'。"

        # 确保资源加载
        self.resource_mgr.load_resources()

        # 获取 RAG Skill 实例
        rag_skill = self._rag_factory()
        rag_callable = rag_skill.run_with_context   # 实例方法，签名符合 (str) -> dict
        # 需要定义 rag_skill 的调用接口：期望 rag_skill.run(query) 返回 dict
        """def rag_callable(query: str) -> dict:
            response = rag_skill.run(query)
            if isinstance(response, dict):
                # 如果 Skill 已经返回协议格式，直接使用
                # 确保 answer 是字符串（可能需要从 response 中获取）
                answer = response.get("answer", response.get("output", ""))
                return {
                    "input": response.get("input", query),
                    "answer": answer,
                    "context": response.get("context", []),
                    "expected_output": response.get("expected_output"),  # 可选
                }
            else:
                # 兼容只返回字符串的旧版 Skill
                logger.warning("当前 RAG Skill 仅返回文本，未提供检索上下文，评测可能不完整。")
                return {
                    "input": query,
                    "answer": str(response),
                    "context": [],
                }
"""        # 执行评测
        report = self.eval_runner.run(user_input, rag_callable)

        # 格式化输出
        output = self._format_report(report)
        return output

    def _format_report(self, report: EvalReport) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("📊 RAG 质量评测报告")
        lines.append("=" * 60)
        lines.append(f"❓ 评测问题: {report.query}")
        lines.append(f"🤖 系统回答: {report.answer}")
        lines.append("")
        lines.append("📈 指标结果:")
        for m in report.metrics:
            status = "✅" if m.success else "❌"
            lines.append(f"  {status} {m.name:20s} 得分: {m.score:.2f}  (阈值: {m.threshold:.2f}) — {'通过' if m.success else '未通过'}")
        lines.append("")
        lines.append(f"🔒 综合信任等级: {report.overall_trust}")
        lines.append(f"📝 总结: {report.summary}")
        if report.improvements:
            lines.append("💡 改进建议:")
            for imp in report.improvements:
                lines.append(f"  - {imp}")
        if report.retrieval_context:
            lines.append("📚 检索上下文:")
            for i, ctx in enumerate(report.retrieval_context, 1):
                lines.append(f"  [{i}] {ctx[:100]}...")
        lines.append("=" * 60)
        return "\n".join(lines)


# -------------------- 5. 演示入口 --------------------
if __name__ == "__main__":
    # 使用前确保已设置 DASHSCOPE_API_KEY
    print("初始化 AI 测试 Skill...")
    test_skill = AITestSkill()
    print("\n" + "=" * 60)
    print("AI 测试 Skill 三层渐进加载 Demo")
    print("=" * 60)

    # 测试一：触发测试
    query = "评测一下：资本充足率要求是多少？"
    print(f"\n>>> 用户输入: {query}")
    if test_skill.should_trigger(query):
        print("  [Layer 2] 触发词匹配，加载评测指令...")
        # 实际 run 会完整执行
        report = test_skill.run(query)
        print(report)
    else:
        print("  [Layer 1] 触发词未匹配，跳过")

    # 测试二：非触发
    query = "今天天气如何"
    print(f"\n>>> 用户输入: {query}")
    print(test_skill.run(query))