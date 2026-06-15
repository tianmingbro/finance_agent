"""
使用 LangGraph StateGraph 执行复合子任务
"""
import logging
import asyncio
import httpx
from typing import TypedDict, List, Optional, Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

from src.agent.a2a.a2a_types import Task, Message, TextPart
from src.agent.a2a.orchestrator_agent import (
    RAG_AGENT_URL,
    EVAL_AGENT_URL,
    get_session,
    set_session,
    save_subtask_result,
    Subtask,
)

logger = logging.getLogger(__name__)

# ---------- 状态定义 ----------
class GraphState(TypedDict, total=False):
    subtasks: List[Subtask]          # 输入：待执行的子任务列表
    session_id: str                  # 会话 ID
    task_id: str                     # 主任务 ID
    results: Dict[int, Any]          # 子任务结果，键为索引
    error: Optional[str]             # 全局错误
    final_artifacts: List[Any]       # 汇总的 artifacts
    final_status: str                # 最终状态

# ---------- 子任务执行节点 ----------
async def execute_rag_node(state: GraphState) -> GraphState:
    index = state.get("current_subtask_index", 0)  # 由调用者设置，或通过节点名称判断
    # 实际通过节点名称区分，这里简化：每个节点对应一个特定子任务索引
    return state

# 更好的方式：为每个子任务动态创建专门的节点函数
def make_rag_node(index: int, subtask: Subtask, session_id: str, task_id: str):
    """创建针对特定子任务的 RAG 节点"""
    async def node(state: GraphState) -> GraphState:
        # 断点续查
        subtask_id = f"{task_id}-{index}"
        saved = get_session(session_id).get("subtasks", {}).get(subtask_id)
        if saved and saved.get("status") == "completed":
            state["results"][index] = saved
            return state

        # 调用 RAG Agent
        payload = Task(
            id=subtask_id,
            session_id=session_id,
            messages=[Message(role="user", parts=[TextPart(text=subtask.query)])],
            context={}
        )
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{RAG_AGENT_URL}/a2a/task", json={"task": payload.__dict__})
                resp.raise_for_status()
                data = resp.json()
            state["results"][index] = data
            save_subtask_result(session_id, subtask_id, data)
        except Exception as e:
            logger.error("RAG 子任务 %d 失败: %s", index, e)
            state["results"][index] = {"status": "failed", "error": str(e)}
            save_subtask_result(session_id, subtask_id, state["results"][index])
        return state
    return node

def make_eval_node(index: int, subtask: Subtask, session_id: str, task_id: str, rag_result: Optional[dict] = None):
    """创建针对特定子任务的 Eval 节点"""
    async def node(state: GraphState) -> GraphState:
        subtask_id = f"{task_id}-{index}"
        saved = get_session(session_id).get("subtasks", {}).get(subtask_id)
        if saved and saved.get("status") == "completed":
            state["results"][index] = saved
            return state

        # 从依赖的 RAG 结果中提取答案（如果未提供）
        query = subtask.query
        answer = subtask.answer
        if not answer and subtask.depends_on:
            dep_index = subtask.depends_on[0]
            rag_data = state["results"].get(dep_index, {})
            answer = rag_data.get("artifacts", [{}])[0].get("answer", "")
            query = rag_data.get("context", {}).get("query", query)

        ctx = {"query": query, "answer": answer}
        payload = Task(
            id=subtask_id,
            session_id=session_id,
            messages=[Message(role="user", parts=[TextPart(text=f"{query}|{answer}")])],
            context=ctx
        )
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{EVAL_AGENT_URL}/a2a/task", json={"task": payload.__dict__})
                resp.raise_for_status()
                data = resp.json()
            state["results"][index] = data
            save_subtask_result(session_id, subtask_id, data)
        except Exception as e:
            logger.error("Eval 子任务 %d 失败: %s", index, e)
            state["results"][index] = {"status": "failed", "error": str(e)}
            save_subtask_result(session_id, subtask_id, state["results"][index])
        return state
    return node

# ---------- 图构建器 ----------
def build_workflow(subtasks: List[Subtask], session_id: str, task_id: str) -> CompiledStateGraph:
    """
    根据子任务列表动态构建 LangGraph 工作流。
    返回编译后的图（可执行）。
    """
    builder = StateGraph(GraphState)

    # 添加节点（动态）
    for i, sub in enumerate(subtasks):
        if sub.type == "rag":
            node_fn = make_rag_node(i, sub, session_id, task_id)
        else:
            node_fn = make_eval_node(i, sub, session_id, task_id)
        # 配置重试策略：最多重试 2 次，每次间隔 1 秒
        retry = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2)
        builder.add_node(f"subtask_{i}", node_fn, retry=retry)

    # 添加边：根据依赖关系
    has_start_edge = set()
    for i, sub in enumerate(subtasks):
        if not sub.depends_on:
            builder.add_edge(START, f"subtask_{i}")
            has_start_edge.add(i)
        else:
            for dep in sub.depends_on:
                builder.add_edge(f"subtask_{dep}", f"subtask_{i}")

    # 所有子任务完成后进入汇总节点
    def aggregate(state: GraphState) -> GraphState:
        artifacts = []
        any_failed = False
        for i, res in state["results"].items():
            if isinstance(res, dict):
                if res.get("status") == "failed":
                    artifacts.append({"error": res.get("error", "子任务失败"), "subtask_id": f"{task_id}-{i}"})
                    any_failed = True
                else:
                    artifacts.extend(res.get("artifacts", []))
        state["final_artifacts"] = artifacts
        state["final_status"] = "partial" if any_failed else "completed"
        return state

    builder.add_node("aggregate", aggregate)
    # 所有子任务节点完成后进入 aggregate
    for i in range(len(subtasks)):
        builder.add_edge(f"subtask_{i}", "aggregate")
    builder.add_edge("aggregate", END)

    # 编译（使用内存检查点，便于重试和恢复）
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

# ---------- 在 Orchestrator 端点中集成 ----------
async def execute_with_graph(subtasks: List[Subtask], session_id: str, task_id: str) -> Task:
    """使用 LangGraph 工作流执行子任务"""
    graph = build_workflow(subtasks, session_id, task_id)
    initial_state = {
        "subtasks": subtasks,
        "session_id": session_id,
        "task_id": task_id,
        "results": {},
        "error": None,
        "final_artifacts": [],
        "final_status": "running",
    }
    # 使用相同的 thread_id 保证同一会话的状态持久化
    config = {"configurable": {"thread_id": session_id}}
    final_state = await graph.ainvoke(initial_state, config)

    # 构造最终 Task
    task = Task(id=task_id, session_id=session_id)
    task.artifacts = final_state.get("final_artifacts", [])
    task.status = final_state.get("final_status", "failed")
    return task