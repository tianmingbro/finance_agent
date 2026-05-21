"""
LangGraph 极简工作流：金融 RAG Skill → AI 测试 Skill
兼容：langchain v1.2.6 + langgraph >= 1.0.2
"""
import os
import sys
from typing import TypedDict, Optional, List, Literal
# LangGraph 核心
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 现有 Skill（确保在同一目录下）
from financial_rag_skill import FinancialRAGSkill
from ai_test_skill import EvaluationRunner, EvalResourceManager
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

# ==================== 1. 定义共享状态 ====================
class GraphState(TypedDict, total=False):
    query: str
    answer: Optional[str]
    context: Optional[List[str]]
    eval_report: Optional[str]

# ==================== 2. 定义节点函数 ====================
def finance_answer_node(state: GraphState) -> dict:
    """
    节点 1：调用金融 RAG Skill，生成回答。
    读取 state['query']，更新 state['answer'] 和 state['context']。
    """
    print(f"🏦 [金融节点] 正在处理: {state['query']}")
    skill = FinancialRAGSkill()
    result = skill.run_with_context(state["query"])
    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", []),
    }

def eval_report_node(state: GraphState) -> dict:
    """
    节点 2：调用 AI 测试 Skill，评测回答质量。
    读取 state['query'], state['answer'], state['context']，更新 state['eval_report']。
    """
    print(f"🧪 [评测节点] 正在评测回答...")

    # 初始化资源管理器（延迟加载 DeepEval 模型）
    resource_mgr = EvalResourceManager()
    resource_mgr.load_resources()

    runner = EvaluationRunner(resource_mgr)

    # 模拟 EvaluationRunner.run 需要的 rag_callable
    def mock_rag(query: str) -> dict:
        return {
            "input": state["query"],
            "answer": state["answer"],
            "context": state["context"],
        }

    # 跑一个简单的评测（仅忠实度指标）
    report = runner.run(f"测试忠实度：{state['query']}", mock_rag)

    # 格式化简短的评测结果
    eval_summary = f"忠实度: {report.metrics[0].score:.2f} (通过: {'✅' if report.metrics[0].success else '❌'})"
    if len(report.metrics) > 1:
        eval_summary += f", 答案相关性: {report.metrics[1].score:.2f} (通过: {'✅' if report.metrics[1].success else '❌'})"
    eval_summary += f"\n综合信任等级: {report.overall_trust}"

    print(f"   {eval_summary}")
    return {"eval_report": eval_summary}

# ==================== 3. 路由函数 ====================
def route_after_answer(state: GraphState) -> Literal["eval_report", END]:
    """根据用户输入判断是否需要评测"""
    query = state.get("query", "")
    if "评测" in query:
        print("🔀 检测到触发词“评测”，进入评测分支")
        return "eval_report"
    else:
        print("🔀 无触发词，直接结束")
        return END
    
# ==================== 4. 构建图 ====================
def build_conditional_graph():
    builder = StateGraph(GraphState)

    # 添加节点
    builder.add_node("finance_answer", finance_answer_node)
    builder.add_node("eval_report", eval_report_node)

    # 固定边：START → finance_answer
    builder.add_edge(START, "finance_answer")

    # 条件边：finance_answer 之后根据路由函数分发
    builder.add_conditional_edges(
        "finance_answer",
        route_after_answer,
        {
            "eval_report": "eval_report",  # 路由到评测节点
            END: END,                      # 直接结束
        }
    )

    # 评测节点结束后退出
    builder.add_edge("eval_report", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ==================== 5. 测试验证 ====================
if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️ 请先设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    graph = build_conditional_graph()

    # ----- 测试 1：不包含“评测” → 只输出回答 -----
    print("\n" + "=" * 60)
    print("  测试 1: 无触发词（仅回答）")
    print("=" * 60)
    state1 = {"query": "资本充足率是多少"}
    config1 = {"configurable": {"thread_id": "test1"}}
    result1 = graph.invoke(state1, config1)

    print("\n📊 最终输出:")
    print(f"🤖 金融回答: {result1.get('answer', 'N/A')}")
    print(f"📈 评测报告: {result1.get('eval_report', '无（未触发评测）')}")
    # 断言：eval_report 应为 None
    assert result1.get("eval_report") is None, "不应包含评测报告"
    print("✅ 测试 1 通过（未触发评测）")

    # ----- 测试 2：包含“评测” → 回答 + 评测 -----
    print("\n" + "=" * 60)
    print("  测试 2: 有触发词“评测”（回答 + 评测）")
    print("=" * 60)
    state2 = {"query": "评测资本充足率"}
    config2 = {"configurable": {"thread_id": "test2"}}
    result2 = graph.invoke(state2, config2)

    print("\n📊 最终输出:")
    print(f"🤖 金融回答: {result2.get('answer', 'N/A')}")
    print(f"📈 评测报告: {result2.get('eval_report', '无')}")
    # 断言：eval_report 应存在
    assert result2.get("eval_report") is not None, "应包含评测报告"
    print("✅ 测试 2 通过（成功触发评测）")