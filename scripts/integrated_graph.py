"""
integrated_graph.py
Day32 核心交付物：将金融 RAG Skill 和 AI 测试 Skill 封装为 LangGraph 条件图，
并运行 3 个端到端测试用例。
"""
from importlib import util
import os
import sys
from typing import TypedDict, Optional, List, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 导入现有 Skill（确保在同一目录下）
from src.skill.financial_rag_skill import BASE_DIR, FinancialRAGSkill
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

# ==================== 1. 图状态定义 ====================
class GraphState(TypedDict, total=False):
    query: str
    answer: Optional[str]
    context: Optional[List[str]]
    eval_report: Optional[str]


# ==================== 2. 节点函数封装 ====================
def finance_answer_node(state: GraphState) -> dict:
    """
    金融 RAG 节点：调用 FinancialRAGSkill.run_with_context。
    输入 state['query']，返回 answer 和 context。
    """
    query = state["query"]
    print(f"🏦 [金融节点] 处理问题: {query}")

    skill = FinancialRAGSkill()
    try:
        result = skill.run_with_context(query)
    except Exception as e:
        print(f"❌ 金融 Skill 执行失败: {e}")
        return {"answer": "系统暂时无法回答该问题，请稍后重试。", "context": []}

    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", []),
    }


def eval_report_node(state: GraphState) -> dict:
    """
    评测节点：从 state 读取 answer 和 context，调用 EvaluationRunner 生成报告。
    返回 eval_report 字符串。
    """
    print("🧪 [评测节点] 开始评测回答质量...")

    answer = state.get("answer", "")
    context = state.get("context", [])
    query = state.get("query", "")

    if not answer:
        return {"eval_report": "无法评测：金融 Skill 未生成有效回答。"}

    # 初始化 DeepEval 资源（延迟加载）
    resource_mgr = EvalResourceManager()
    resource_mgr.load_resources()
    runner = EvaluationRunner(resource_mgr)

    # 构造符合接口的 rag 可调用对象（从当前状态提供数据）
    def rag_callable(q: str) -> dict:
        return {
            "input": query,
            "answer": answer,
            "context": context,
        }

    try:
        report = runner.run(f"测试忠实度：{query}", rag_callable)
    except Exception as e:
        print(f"❌ 评测执行失败: {e}")
        return {"eval_report": "评测过程出错，无法生成报告。"}

    # 格式化简要报告
    lines = []
    for m in report.metrics:
        status = "✅" if m.success else "❌"
        lines.append(f"{m.name}: {m.score:.2f} ({status})")
    lines.append(f"综合信任等级: {report.overall_trust}")
    eval_summary = "\n".join(lines)

    print(f"   评测结果:\n{eval_summary}")
    return {"eval_report": eval_summary}


# ==================== 3. 路由函数 ====================
def route_after_answer(state: GraphState) -> str:
    """条件边：检查 query 是否包含“评测”关键词"""
    if "评测" in state.get("query", ""):
        print("🔀 检测到触发词“评测”，进入评测分支")
        return "eval_report"
    else:
        print("🔀 无触发词，直接结束")
        return END


# ==================== 4. 构建条件图 ====================
def build_graph():
    builder = StateGraph(GraphState)

    # 注册节点
    builder.add_node("finance_answer", finance_answer_node)
    builder.add_node("eval_report", eval_report_node)

    # 固定边：START -> finance_answer
    builder.add_edge(START, "finance_answer")

    # 条件边：finance_answer 之后判断是否评测
    builder.add_conditional_edges(
        "finance_answer",
        route_after_answer,
        {
            "eval_report": "eval_report",
            END: END,
        }
    )

    # 评测节点结束
    builder.add_edge("eval_report", END)

    # 编译图（使用内存检查点）
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ==================== 5. 端到端测试用例 ====================
def run_test_cases():
    # 前置检查
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️ 未设置 DASHSCOPE_API_KEY，请先设置。")
        sys.exit(1)

    graph = build_graph()

    test_cases = [
        {
            "name": "测试1: 纯金融问题（无触发词）→ 仅回答",
            "query": "资本充足率是多少",
            "config": {"configurable": {"thread_id": "test1"}},
            "expect_eval": False,  # 不期望包含评测报告
        },
        {
            "name": "测试2: 带评测指令 → 回答 + 评测报告",
            "query": "评测资本充足率",
            "config": {"configurable": {"thread_id": "test2"}},
            "expect_eval": True,
        },
        {
            "name": "测试3: 金融问题（LPR）带评测 → 回答 + 评测报告",
            "query": "评测 LPR最新报价",
            "config": {"configurable": {"thread_id": "test3"}},
            "expect_eval": True,
        },
    ]

    for case in test_cases:
        print("\n" + "=" * 70)
        print(f"  {case['name']}")
        print("=" * 70)

        initial_state = {"query": case["query"]}
        result = graph.invoke(initial_state, case["config"])

        # 输出
        print(f"🤖 回答: {result.get('answer', 'N/A')[:100]}...")
        eval_report = result.get("eval_report")
        if eval_report:
            print(f"📈 评测报告:\n{eval_report}")
        else:
            print("📈 评测报告: (无)")

        # 断言检查
        if case["expect_eval"]:
            assert eval_report is not None, "应包含评测报告，但未生成！"
            print("✅ 通过：包含评测报告")
        else:
            assert eval_report is None, "不应包含评测报告，但实际生成了！"
            print("✅ 通过：无评测报告")

    print("\n🎉 全部测试用例通过！")


# 在 integrated_graph.py 末尾增加验证函数
def verify_retrieval_accuracy():
    """跨后端检索正确性验证"""
    from pathlib import Path
    # 获取当前脚本所在的绝对路径
    BASE_DIR = Path(__file__).resolve().parent
    # 使用 / 运算符直接拼接路径，清晰直观
    MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"
    from sentence_transformers import SentenceTransformer, util

    # 直接加载同一个 embedding 模型，与向量库使用的完全一致
    print("正在加载语义相似度模型...")
    sim_model = SentenceTransformer(MODEL_PATH)
    test_queries = [
        "资本充足率是多少",
        "LPR最新报价",
        "个人购汇额度",
    ]
    expected_phrases = {
        "资本充足率是多少": ["核心一级资本充足率不得低于5%", "资本充足率不得低于8%"],
        "LPR最新报价": ["3.1%", "贷款市场报价利率"],
        "个人购汇额度": ["5万美元", "便利化额度"],
    }

    backends = ["chroma", "pgvector"]
    results = {}

    for backend in backends:
        os.environ["VECTOR_STORE_BACKEND"] = backend
        print(f"\n=== 当前后端: {backend} ===")
        graph = build_graph()
        config = {"configurable": {"thread_id": f"verify-{backend}"}}

        for query in test_queries:
            state = graph.invoke({"query": query}, config)
            answer = state.get("answer", "")
            # 检查关键实体
            phrases = expected_phrases.get(query, [])
            found = any(p in answer for p in phrases)
            print(f"Query: {query}")
            print(f"Answer: {answer[:100]}...")
            print(f"关键实体匹配: {'✅' if found else '❌'}")
            assert found, f"后端 {backend} 的答案未包含预期关键信息: {query}"
            results.setdefault(query, {})[backend] = answer

    # 跨后端一致性检查
    print("\n=== 跨后端一致性检查 ===")
    for query in test_queries:
        ans1 = results[query]["chroma"]
        ans2 = results[query]["pgvector"]
        emb1 = sim_model.encode(ans1, convert_to_tensor=True)
        emb2 = sim_model.encode(ans2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        print(f"Query: {query} → 语义相似度: {similarity:.4f}")
        assert similarity >= 0.95, f"两个后端答案差异过大 ({similarity:.2f})"
    print("✅ 所有检索正确性验证通过！")

if __name__ == "__main__":
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    print(f"当前向量库后端: {backend}")
    run_test_cases()
    verify_retrieval_accuracy()