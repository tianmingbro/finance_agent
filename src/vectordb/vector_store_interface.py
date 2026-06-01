"""
vector_store_interface.py
向量库抽象层：定义统一接口 + 工厂方法
"""
from typing import Protocol, runtime_checkable, List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


@runtime_checkable
class VectorStoreInterface(Protocol):
    """
    向量库抽象接口（Python Protocol，非 LangChain 内部 VectorStore）
    任何实现此接口的类都可以被 ResourceManager 使用，无需继承特定基类。
    """

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """语义相似度搜索，返回最相关的 k 个文档"""
        ...

    def add_documents(
        self, documents: List[Document], **kwargs
    ) -> List[str]:
        """批量插入文档及其嵌入向量，返回文档 ID 列表"""
        ...

    def as_retriever(
        self, **kwargs
    ) -> VectorStoreRetriever:
        """将向量库包装为 LangChain Retriever，供 RAG 链使用"""
        ...

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """删除指定 ID 的文档"""
        ...


# ─── 可选的抽象层工具函数 ───────────────────────────────

def validate_store(store) -> VectorStoreInterface:
    """
    运行时验证：确保传入的对象实现了 VectorStoreInterface。
    若不符合，抛出 TypeError 并给出明确的字段级诊断。
    """
    if not isinstance(store, VectorStoreInterface):
        missing = []
        for attr in ["similarity_search", "add_documents", "as_retriever"]:
            if not hasattr(store, attr):
                missing.append(attr)
        raise TypeError(
            f"向量库 {type(store).__name__} 不符合 VectorStoreInterface："
            f"缺少方法 {missing}"
        )
    return store

import os

def create_vector_store(
    embedding_model,
    config: Optional[Dict[str, Any]] = None,
) -> VectorStoreInterface:
    """
    工厂方法：根据环境变量或配置选择向量库后端。
    优先级：环境变量 > 显式 config > 默认 Chroma

    环境变量:
      VECTOR_STORE_BACKEND: "chroma"（默认）| "pgvector"
      PGVECTOR_CONNECTION_STRING: PGVectorStore 连接串
    """
    config = config or {}
    backend = (
        config.get("backend") or
        os.getenv("VECTOR_STORE_BACKEND") or
        "chroma"
    ).lower()

    if backend == "chroma":
        # ─── Chroma（默认开发环境）───────────────
        from langchain_chroma import Chroma
        persist_dir = config.get("persist_directory", "./chroma_db")
        collection = config.get("collection_name", "finance_qa")

        store = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_dir,
            collection_name=collection,
        )
        print(f"📦 使用向量库后端: Chroma (persist: {persist_dir})")

    elif backend == "pgvector":
        # ─── PGVectorStore（生产预留）─────────────
        from langchain_postgres import PGEngine, PGVectorStore

        conn_str = (
            config.get("connection_string") or
            os.getenv("PGVECTOR_CONNECTION_STRING") or
            "postgresql+psycopg://pgvector:pgvector@localhost:5432/ai_rag"
        )
        table_name = config.get("table_name", "finance_knowledge")
        vector_size = config.get("vector_size", 384)

        engine = PGEngine.from_connection_string(url=conn_str)

        # 首次使用时建表（幂等操作）
        try:
            engine.init_vectorstore_table(
                table_name=table_name,
                vector_size=vector_size,
            )
        except Exception:
            pass  # 表已存在

        store = PGVectorStore.create_sync(
            engine=engine,
            table_name=table_name,
            embedding_service=embedding_model,
        )
        print(f"📦 使用向量库后端: PGVectorStore (table: {table_name})")

    else:
        raise ValueError(f"不支持的向量库后端: {backend}")

    return validate_store(store)