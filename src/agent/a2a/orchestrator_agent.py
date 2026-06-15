import time
import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.a2a.a2a_types import Task, Message, TextPart, AgentCard, AgentSkill
import uvicorn

app = FastAPI()
logger = logging.getLogger(__name__)

RAG_AGENT_URL = "http://localhost:8101"
EVAL_AGENT_URL = "http://localhost:8102"

# ---- 会话存储 (与 Day51 相同) ----
session_store: dict = {}
SESSION_TTL = 300

def get_session(session_id: str) -> dict:
    if session_id in session_store:
        entry = session_store[session_id]
        if time.time() - entry["timestamp"] < SESSION_TTL:
            return entry["data"]
        del session_store[session_id]
    return {}

def set_session(session_id: str, data: dict):
    session_store[session_id] = {"data": data, "timestamp": time.time()}

# ---- 子任务数据结构 ----
class Subtask:
    def __init__(self, type: str, query: str="", answer: str = "",
                 depends_on: List[int] = None, condition: str = None):
        self.type = type          # "rag" 或 "eval"
        self.query = query
        self.answer = answer
        self.depends_on = depends_on or []   # 依赖的子任务索引列表
        self.condition = condition           # 条件表达式，如 "eval.score < 0.7"

# ---- 意图解析器 ----
def parse_complex_intent(text: str) -> List[Subtask]:
    """解析复合意图，返回子任务列表"""
    text = text.strip()
    subtasks = []

    # 定义检测词
    query_keywords = ["查", "问", "回答", "提问", "检索", "搜索"]
    eval_keywords = ["评测", "评估", "检查", "验证", "打分"]

    has_query = any(kw in text for kw in query_keywords)
    has_eval = any(kw in text for kw in eval_keywords)

    # 模式1：串行链 —— 同时提到查询和评测/评估
    if has_query and has_eval:
        import re
        # 用常见的连接词分割，保留左侧作为问题
        split_pattern = r'并评测|并评估|然后评测|然后评估|并检查|然后检查|并验证|再评测|再评估|以及评测|与评测'
        parts = re.split(split_pattern, text, maxsplit=1)
        question_part = parts[0].strip()

        # 去掉问题中的前缀动词
        for prefix in ["查一下", "帮我查", "请查", "查询", "查", "问一下", "帮我问", "搜索"]:
            if question_part.startswith(prefix):
                question_part = question_part[len(prefix):].strip()
                break

        # 去掉尾部标点
        question_part = question_part.rstrip('。，,！!？? ')

        if question_part:
            subtasks.append(Subtask(type="rag", query=question_part))
            subtasks.append(Subtask(type="eval", depends_on=[0]))
            return subtasks

    # 模式2：并行双问
    if "各是多少" in text or "分别" in text or ("和" in text and "多少" in text):
        import re
        # 按“和”、“与”、“及”、“以及”分割
        parts = re.split(r'和|与|及|以及', text)
        queries = []
        for p in parts:
            p = p.strip().rstrip('?？').rstrip('各是多少').strip()
            if p:
                queries.append(p)
        if len(queries) >= 2:
            for q in queries:
                subtasks.append(Subtask(type="rag", query=q))
            return subtasks

    # 模式3：条件汇总
    if has_eval and ("如果" in text or "若是" in text):
        import re
        match = re.search(r'评测这个回答[：:]\s*(.+?)\s*如果(?:分数)?低于([\d.]+)就重新回答', text)
        if match:
            answer_to_eval = match.group(1).strip()
            threshold = float(match.group(2))
            subtasks.append(Subtask(type="eval", query="", answer=answer_to_eval))
            subtasks.append(Subtask(type="rag", query="", depends_on=[0],
                                    condition=f"eval_score < {threshold}"))
            return subtasks

    # 模式4：纯评测（类似“评测一下”），留给 simple handler 处理，但也可生成一个 eval 任务并回退
    # 此处不主动生成，返回空列表，交由简单路由利用 session 上下文

    return subtasks
# 子任务重试配置
MAX_RETRIES = 2
RETRY_DELAY = 1.0      # 秒
SUB_TIMEOUT = 30       # 单次调用超时

async def call_agent(type: str, query: str, answer: str, context: dict,
                     session_id: str, task_id: str) -> dict:
    url = RAG_AGENT_URL if type == "rag" else EVAL_AGENT_URL
    payload = Task(
        id=f"{task_id}-sub",
        session_id=session_id,
        messages=[Message(role="user", parts=[TextPart(
            text=query if type == "rag" else f"{query}|{answer}"
        )])],
        context=context
    )
    async with httpx.AsyncClient(timeout=SUB_TIMEOUT) as client:
        resp = await client.post(f"{url}/a2a/task", json={"task": payload.to_dict()})
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Agent {type} 返回 {resp.status_code}")
        
# ---- 子任务执行器 ----
async def execute_subtasks(subtasks: List[Subtask], session_id: str, task_id: str) -> Task:
    """
    按依赖关系执行子任务，返回聚合的 Task。
    """
    # 结果存储：索引对应子任务
    results = [None] * len(subtasks)
    # 记录哪些子任务已完成
    completed = set()
    # 从 session 中获取已有的子任务结果，用于断点续查
    session_data = get_session(session_id)
    saved_subtasks = session_data.get("subtasks", {}) if session_data else {}

    
    async def run_subtask(index: int):
        sub = subtasks[index]
        subtask_id = f"{task_id}-{index}"
        print(f"🔍 子任务 {index} 开始, 类型={sub.type}, query={sub.query}")   # ← 添加
         # 断点续查：如果已成功完成，直接使用缓存
        if subtask_id in saved_subtasks and saved_subtasks[subtask_id].get("status") == "completed":
            print(f"🔍 子任务 {index} 从缓存获取")   # ← 添加
            results[index] = saved_subtasks[subtask_id]
            completed.add(index)
            return
        
        # 等待依赖完成
        for dep in sub.depends_on:
            print(f"🔍 子任务 {index} 等待依赖 {dep}")   # ← 添加
            while dep not in completed:
                await asyncio.sleep(0.1)
        print(f"🔍 子任务 {index} 依赖满足，开始调用 Agent")   # ← 添加
        # 构建上下文：汇总依赖结果
        ctx = {}
        for dep in sub.depends_on:
            if results[dep]:
                ctx.update(results[dep].get("context", {}))
                # 如果依赖是 eval，且当前需要条件判断，可以检查分数
                if sub.condition and "faithfulness" in results[dep].get("artifacts", [{}])[0]:
                    score = results[dep]["artifacts"][0]["faithfulness"]
                    # 动态执行条件表达式（简单替换）
                    condition_met = eval(sub.condition.replace("eval_score", str(score)))
                    if not condition_met:
                        logger.info("子任务 %d 条件不满足，跳过", index)
                        results[index] = {"status": "skipped"}
                        return
                # 将依赖的 artifacts 中的 answer 带入 context
                if sub.type == "eval" and not sub.answer:
                    # 自动从依赖的 rag 结果中提取 answer
                    rag_result = results[dep]
                    if rag_result and rag_result.get("artifacts"):
                        sub.answer = rag_result["artifacts"][0].get("answer", "")
                        ctx["answer"] = sub.answer
                        ctx["query"] = rag_result["context"].get("query", sub.query)

        # 补充 query（如果为空）
        if not sub.query:
            # 从 session 或 context 获取
            sub.query = ctx.get("query", get_session(session_id).get("query", ""))

        # 带重试的调用
        last_exception = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info("子任务 %s 第 %d 次尝试", subtask_id, attempt + 1)
                response = await asyncio.wait_for(
                    call_agent(sub.type, sub.query, sub.answer, ctx,session_id, task_id),
                    timeout=SUB_TIMEOUT
                )
                results[index] = response
                # 存入 session 的子任务记录
                save_subtask_result(session_id, subtask_id, response)
                completed.add(index)
                return
            except asyncio.TimeoutError:
                last_exception = TimeoutError("子任务超时")
                logger.warning("子任务 %s 超时", subtask_id)
            except Exception as e:
                last_exception = e
                logger.warning("子任务 %s 失败: %s", subtask_id, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

        # 所有重试失败
        results[index] = {
            "status": "failed",
            "error": str(last_exception),
            "subtask_id": subtask_id
        }
        # 保存失败状态
        save_subtask_result(session_id, subtask_id, results[index])
        completed.add(index)

    # 并发启动所有无依赖的子任务
     # 顺序执行，保证依赖关系和日志清晰
    for i in range(len(subtasks)):
        await run_subtask(i)

    # 汇总结果到 Task
    final_task = Task(id=task_id, session_id=session_id)
    final_task.artifacts = []
    any_failed = False          # ← 确保这一行存在
    for i, res in enumerate(results):
        if res.get("status") == "failed":
            final_task.artifacts.append({
                "error": res.get("error", "子任务失败"),
                "subtask_id": res.get("subtask_id", f"{task_id}-{i}")
            })
            any_failed = True
        elif res.get("status") == "skipped":
            pass
        else:
            final_task.artifacts.extend(res.get("artifacts", []))
    final_task.status = "partial" if any_failed else "completed"
    return final_task

def save_subtask_result(session_id: str, subtask_id: str, result: dict):
    """将子任务结果存入 session_store"""
    session_data = get_session(session_id)
    if not session_data:
        session_data = {}
    if "subtasks" not in session_data:
        session_data["subtasks"] = {}
    session_data["subtasks"][subtask_id] = result
    set_session(session_id, session_data)

# ---- 修改后的主端点 ----
class TaskRequest(BaseModel):
    task: Task

@app.post("/a2a/task")
async def handle_task(req: TaskRequest):
    task = req.task
    user_text = task.messages[-1].parts[0].text if task.messages else ""
    print(f"\n🔍 收到请求: session={task.session_id}, text='{user_text}'")  # 添加这行
    # 尝试解析复合意图
    subtasks = parse_complex_intent(user_text)
    print(f"🔍 解析结果: {subtasks}")  # 添加这行
    if subtasks:
        logger.info("识别为复合意图，子任务数: %d", len(subtasks))
        # 注意：可能还需要从指令中提取 query/answer 等，parse_complex_intent 已经基本处理
        result_task = await execute_subtasks(subtasks, task.session_id, task.id)
        return result_task
    else:
        # 回退到简单路由（Day51 的逻辑）
        # 此处为简化，调用原有的单任务处理函数
        return await handle_simple_task(task)

# ---- 简单任务处理（保留 Day51 的逻辑） ----
async def handle_simple_task(task: Task) -> Task:
    user_text = task.messages[-1].parts[0].text if task.messages else ""

    if "评测" in user_text:
        content = user_text.replace("评测", "", 1).strip()
        parts = content.split("|")
        if len(parts) == 2:
            query, answer = parts[0].strip(), parts[1].strip()
        else:
            session_data = get_session(task.session_id)
            query = session_data.get("query", "")
            answer = session_data.get("answer", "")
            if not query or not answer:
                task.status = "failed"
                task.artifacts = [{"error": "没有找到上一次对话，请先提问或手动输入 评测 问题|答案"}]
                return task
        context = {"query": query, "answer": answer}
        payload = Task(
            id=task.id,
            session_id=task.session_id,
            messages=[Message(role="user", parts=[TextPart(text=f"{query}|{answer}")])],
            context=context,
        )
        target_url = EVAL_AGENT_URL
    else:
        payload = Task(
            id=task.id,
            session_id=task.session_id,
            messages=[Message(role="user", parts=[TextPart(text=user_text)])],
            context=task.context
        )
        target_url = RAG_AGENT_URL

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{target_url}/a2a/task", json={"task": payload.to_dict()})
            resp.raise_for_status()
            task = Task(**resp.json())
        logger.info("调用 %s 成功，耗时 %.3fs", target_url, time.perf_counter()-start)
    except Exception as e:
        logger.error("调用 %s 失败: %s", target_url, e)
        task.status = "failed"
        task.artifacts = [{"error": f"服务不可用: {str(e)}"}]
        return task

    if target_url == RAG_AGENT_URL and task.status == "completed":
        ctx = task.context if task.context else {}
        set_session(task.session_id, {
            "query": ctx.get("query", user_text),
            "answer": ctx.get("answer", ""),
            "sources": ctx.get("sources", [])
        })
    return task

# ---- 其余 Agent Card 与启动 ----
@app.get("/.well-known/agent-card")
def agent_card():
    return AgentCard(
        name="orchestrator",
        description="主控代理，支持复合意图解析与子任务编排",
        url="http://localhost:8100",
        skills=[AgentSkill(id="dispatch", name="任务分发")],
        capabilities={"streaming": False}
    ).to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)