"""
api/main.py
Day45 最终版：全局异常处理 + 请求耗时日志 + 测试端点
（已修正：ainvoke 调用、eval 端点、config 处理）
"""
import asyncio
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import sys, os, json

from eval_components import evaluate_generation, evaluate_planning, evaluate_retrieval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import asynccontextmanager
from src.agent.workflow import rag_agent_workflow
from src.skill.ai_test_skill import EvaluationRunner, EvalResourceManager
from src.api.dependencies import get_financial_skill, get_eval_runner

logger = logging.getLogger(__name__)

# ── 应用生命周期：预热 ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预热单例资源...")
    async def warmup():
        await get_financial_skill()
        await get_eval_runner()
        logger.info("预热完成")
    asyncio.create_task(warmup())
    yield

app = FastAPI(
    title="金融 RAG Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Pydantic 模型 ──────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    need_eval: bool = Field(False)

class EvalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: list
    evaluation: Optional[str] = None
    elapsed_seconds: float

class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    trust: str

# ── 辅助函数 ────────────────────────────────────
def _extract_documents(retrieved_docs) -> list:
    if not retrieved_docs:
        return []
    if isinstance(retrieved_docs, dict):
        return retrieved_docs.get("documents", [])
    if isinstance(retrieved_docs, list):
        return retrieved_docs
    return []

def _run_evaluation(query: str, answer: str) -> dict:
    """同步执行评测，供 asyncio.to_thread 调用"""
    resource_mgr = EvalResourceManager()
    resource_mgr.load_resources()
    runner = EvaluationRunner(resource_mgr)

    def rag_callable(q: str) -> dict:
        return {"input": query, "answer": answer, "context": []}

    report = runner.run(f"评测忠实度：{query}", rag_callable)
    faith = next((m.score for m in report.metrics if m.name == "faithfulness"), 0.0)
    relevancy = next((m.score for m in report.metrics if m.name == "answer_relevancy"), 0.0)
    return {
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "trust": report.overall_trust,
    }

# ── 全局异常处理器 ──────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": f"请求参数错误: {exc}"})

@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    return JSONResponse(status_code=502, content={"detail": "上游服务连接失败，请稍后重试。"})

# ── 请求耗时中间件 ──────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = str(elapsed)
    logger.info("%s %s [%d] %.2fms",
                request.method, request.url.path,
                response.status_code, elapsed * 1000)
    return response

# ── 端点实现 ────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest):
    start = time.time()
    # ✅ 异步入口必须用 ainvoke，并传入空 config（或构造 RunnableConfig）
    try:
        result = await rag_agent_workflow.ainvoke(
            {"query": req.query, "need_eval": req.need_eval},
            config={"configurable": {}}       # 必须提供 config，即使为空
        )
    except Exception as e:
        logger.exception("RAG 工作流失败")
        raise HTTPException(500, str(e))

    elapsed = time.time() - start
    return QueryResponse(
        answer=result.get("answer", ""),
        retrieved_docs=_extract_documents(result.get("retrieved_docs")),
        evaluation=result.get("evaluation"),
        elapsed_seconds=elapsed,
    )

@app.post("/rag/query/stream")
async def rag_query_stream(req: QueryRequest):
    async def generate():
        result = await rag_agent_workflow.ainvoke(
            {"query": req.query, "need_eval": req.need_eval},
            config={"configurable": {}},
        )
        answer = result.get("answer", "")
        for i in range(0, len(answer), 10):
            yield answer[i:i+10]
            await asyncio.sleep(0.01)
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/eval/score", response_model=EvalResponse)
async def eval_score(req: EvalRequest):
    try:
        # ✅ 将同步评测放入线程池，传递 query 和 answer 两个参数
        eval_data = await asyncio.to_thread(
            _run_evaluation, req.query, req.answer
        )
        return EvalResponse(**eval_data)
    except Exception as e:
        logger.exception("评测失败")
        raise HTTPException(500, str(e))
    
# ── 测试专用端点（仅调试用，生产环境应移除）────
@app.get("/debug/raise-error")
async def raise_error(error_type: str = "value_error"):
    """
    用于验证全局异常处理器的端点。
    """
    if error_type == "value_error":
        raise ValueError("测试: 无效的查询参数")
    elif error_type == "connection_error":
        raise ConnectionError("测试: 无法连接数据库")
    else:
        raise HTTPException(400, f"未知错误类型: {error_type}")
    
# api/main.py 新增内容

class ComponentEvalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    expected_tool: Optional[str] = None
    expected_args: Optional[dict] = None
    key_info: Optional[List[str]] = None
    expected_answer: Optional[str] = None

class ComponentEvalResponse(BaseModel):
    planning: dict
    retrieval: dict
    generation: dict

@app.post("/eval/component", response_model=ComponentEvalResponse)
async def evaluate_component(req: ComponentEvalRequest):
    """执行组件级评估：规划、检索、生成"""
    planning = evaluate_planning(
        query=req.query,
        expected_tool=req.expected_tool or "financial_qa",
        expected_args=req.expected_args or {"query": req.query}
    )
    retrieval = await asyncio.to_thread(
        evaluate_retrieval,
        query=req.query,
        key_info=req.key_info
    )
    generation = await asyncio.to_thread(
        evaluate_generation,
        query=req.query,
        expected_answer=req.expected_answer
    )
    return ComponentEvalResponse(
        planning=planning,
        retrieval=retrieval,
        generation=generation
    )