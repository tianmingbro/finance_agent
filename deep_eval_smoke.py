"""
deep_eval_smoke.py
Day31 冒烟验证：用硬编码的金融问答对验证 DeepEval 基本用法
评估模型：qwen-plus (阿里云 DashScope)
"""
import os

# ============================================================
# Step 1: 定义 Qwen 自定义评估模型 (继承 DeepEvalBaseLLM)
# 参考: deepeval.com/guides/guides-using-custom-llms
# ============================================================
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

class QwenEvaluationModel(DeepEvalBaseLLM):
    """用于 DeepEval 评估的通义千问自定义模型"""

    def __init__(self, model_name: str = "qwen-plus", api_key: str = None):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self._client = None

    def load_model(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️ 调用评估模型失败: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"qwen-plus (DashScope)"


# ============================================================
# Step 2: 准备硬编码的金融问答测试数据
# 来源: Day29 构建的金融 RAG MVP 回答
# ============================================================
# 模拟 RAG 管道的输出
test_data = {
    "query": "商业银行的资本充足率监管要求是多少？",
    "actual_output": "根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。",
    "retrieval_context": [
        "根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。",
    ],
    "expected_output": "核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，资本充足率不得低于8%。",
}


# ============================================================
# Step 3: 构建 LLMTestCase 并运行 DeepEval 指标
# ============================================================
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric


def run_smoke_test():
    print("=" * 60)
    print("🧪 DeepEval 冒烟验证 - 金融 RAG 问答评测")
    print(f"   评估模型: qwen-plus")
    print(f"   测试问题: {test_data['query']}")
    print("=" * 60)

    # 初始化自定义评估模型
    eval_model = QwenEvaluationModel(
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # 构建测试用例
    test_case = LLMTestCase(
        input=test_data["query"],
        actual_output=test_data["actual_output"],
        retrieval_context=test_data["retrieval_context"],
        expected_output=test_data["expected_output"],
    )

    # 定义评测指标
    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model=eval_model,
        include_reason=True,
        verbose_mode=True,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model=eval_model,
        include_reason=True,
    )

    # 逐个评测并打印结果
    print("\n📊 指标 1: Faithfulness (忠实度)")
    print("-" * 40)
    faithfulness.measure(test_case)
    print(f"   得分: {faithfulness.score:.4f}")
    print(f"   是否通过: {'✅ 通过' if faithfulness.is_successful() else '❌ 未通过'}")
    if faithfulness.reason:
        print(f"   判定理由: {faithfulness.reason}")

    print("\n📊 指标 2: Answer Relevancy (答案相关性)")
    print("-" * 40)
    answer_relevancy.measure(test_case)
    print(f"   得分: {answer_relevancy.score:.4f}")
    print(f"   是否通过: {'✅ 通过' if answer_relevancy.is_successful() else '❌ 未通过'}")
    if answer_relevancy.reason:
        print(f"   判定理由: {answer_relevancy.reason}")

    # 汇总
    print("\n" + "=" * 60)
    print("📋 评测汇总")
    print(f"   Faithfulness:      {faithfulness.score:.4f} "
          f"({'✅' if faithfulness.is_successful() else '❌'})")
    print(f"   Answer Relevancy:  {answer_relevancy.score:.4f} "
          f"({'✅' if answer_relevancy.is_successful() else '❌'})")
    print("=" * 60)

    return faithfulness, answer_relevancy


# ============================================================
# Step 4: 也演示使用 evaluate() 批量运行 (与 Day31 的 batch_evaluate 对应)
# ============================================================
def run_batch_demo():
    """演示批量 evaluate() 语法"""
    print("\n\n📦 批量评测 Demo (evaluate 语法)")
    print("=" * 60)

    eval_model = QwenEvaluationModel(
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    test_case = LLMTestCase(
        input=test_data["query"],
        actual_output=test_data["actual_output"],
        retrieval_context=test_data["retrieval_context"],
    )

    evaluate(
        test_cases=[test_case],
        metrics=[
            FaithfulnessMetric(model=eval_model, threshold=0.5),
            AnswerRelevancyMetric(model=eval_model, threshold=0.5),
        ],
    )


if __name__ == "__main__":
    run_smoke_test()
    run_batch_demo()