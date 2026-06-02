"""
tools_mcp.py
将 HybridRetriever 封装为可供 MCP 调用的异步工具函数。
"""
import json
import sys
import platform
if platform.system() == "Windows":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import asyncio
import sys
from typing import List
from langchain_core.documents import Document
from src.retriever.hybrid_retriever import HybridRetriever, create_bm25_retriever
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from config import get_embedding_model_path
modelname = get_embedding_model_path()
# 全局单例，避免每次调用都重新加载模型和向量库
_hybrid_retriever = None

# tools_mcp.py 顶部添加
def warmup():
    """预热：提前加载模型和向量库，防止首次调用超时"""
    _get_hybrid_retriever()

def _get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=modelname,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
def _get_vector_retriever(embedding, search_k=20):
    """根据环境变量创建向量检索器"""
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
    try:
        if backend == "pgvector":
            from langchain_postgres import PGEngine, PGVectorStore
            conn_str = os.getenv(
                "PGVECTOR_CONNECTION_STRING",
                "postgresql+psycopg://pgvector:pgvector@localhost:5433/ai_rag"
            )
            engine = PGEngine.from_connection_string(url=conn_str)
            table_name = os.getenv("PG_TABLE", "finance_knowledge")
            vectorstore = PGVectorStore.create_sync(
                engine=engine,
                table_name=table_name,
                embedding_service=embedding,
            )
        else:  # 默认 chroma
            from langchain_chroma import Chroma
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embedding,
                collection_name="finance_qa",
            )
    except Exception as e:
        raise RuntimeError(f"向量库连接失败 (backend={backend}): {e}") from e

    return vectorstore.as_retriever(search_kwargs={"k": search_k})

def _get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is not None:
        return _hybrid_retriever

    embedding = _get_embedding_model()
    # 1. 向量检索器（根据环境变量选择后端）
    vector_retriever = _get_vector_retriever(embedding, search_k=20)

    # 2. BM25 检索器（从 final_qa_dataset.yaml 构建索引）
    import yaml
    from pathlib import Path
    from langchain_core.documents import Document as LCDocument

    bm25_docs = []
    qa_path = Path("data/final_qa_dataset.yaml")
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        qa_list = data if isinstance(data, list) else data.get("final_qa_dataset", [])
        for item in qa_list:
            q = item.get("query") or item.get("question", "")
            a = item.get("answer", "")
            if q and a:
                bm25_docs.append(LCDocument(
                    page_content=f"问题：{q}\n答案：{a}",
                    metadata={"source": "qa_dataset"}
                ))
    bm25_retriever = create_bm25_retriever(bm25_docs, k=20)

    _hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fusion_strategy="rrf",
        k=4,
        fetch_k=20,
    )
    return _hybrid_retriever


async def search_finance_docs(query: str, top_k: int = 4) -> str:
    """
    检索金融法规相关文档，返回结构化 JSON。
    输入：
      - query: 查询文本
      - top_k: 返回的文档数量，默认 4
    输出：JSON 字符串，包含 documents 列表和 sources 汇总。
    """
    print(f"[Tool] 调用 search_finance_docs: {query}", file=sys.stderr)
    if top_k <= 0:
        return json.dumps({"error": "参数 top_k 必须大于 0"}, ensure_ascii=False)

    retriever = _get_hybrid_retriever()
    docs: List[Document] = await asyncio.to_thread(
        retriever.get_relevant_documents, query
    )
    print(f"[Tool] 检索完成，得到 {len(docs)} 篇文档", file=sys.stderr)
    if not docs:
        return json.dumps({"documents": [], "sources": []}, ensure_ascii=False)

    # 构建结构化结果
    documents = []
    sources_set = set()
    for i, doc in enumerate(docs[:top_k]):
        content = doc.page_content.replace("\n", " ").strip()[:300]  # 适当截断
        source = doc.metadata.get("source", "未知")
        documents.append({
            "index": i + 1,
            "content": content,
            "source": source,
            # 可选：如果有分数可添加 "score": doc.metadata.get("score", None)
        })
        sources_set.add(source)

    result = {
        "documents": documents,
        "sources": list(sources_set)
    }
    return json.dumps(result, ensure_ascii=False)

async def list_available_topics() -> str:
    """
    列出当前金融知识库覆盖的主题。
    输出：JSON 字符串，包含 topics 数组。
    """
    import yaml
    from pathlib import Path
    topics = set()
    # 从 QA 数据集中提取 category
    qa_path = Path("data/final_qa_dataset.yaml")
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        qa_list = data if isinstance(data, list) else data.get("final_qa_dataset", [])
        for item in qa_list:
            cat = item.get("category")
            if cat:
                topics.add(cat)
    # 补充预定义主题（如果数据集为空）
    if not topics:
        topics = {"资本充足率", "存款保险", "外汇管理", "LPR", "反洗钱", "金融法规"}

    result = {"topics": sorted(list(topics))}
    return json.dumps(result, ensure_ascii=False)