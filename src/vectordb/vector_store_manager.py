"""
vector_store_manager.py
Day38 核心交付物：向量库管理器，支持 Chroma / PGVector 统一接口
兼容：langchain v1.2 + langchain-chroma / langchain-postgres
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
import time
from typing import List, Optional, Dict, Any

# Windows 兼容性：psycopg 在 Windows 上需要 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import DistanceStrategy

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        connection_string: str,
        embedding_model,
        table_name: str = "finance_knowledge",
        vector_size: int = 768,
    ):
        if not connection_string:
            raise ValueError("connection_string 不能为空")
        if vector_size <= 0:
            raise ValueError("vector_size 必须大于 0")

        self.connection_string = connection_string
        self.embedding_model = embedding_model
        self.table_name = table_name
        self.vector_size = vector_size
        # # 从 embedding 模型获取实际维度
        # test_embedding = self.embedding_model.embed_query("dimension test")
        # actual_dim = len(test_embedding)
        # self.vector_size = actual_dim
        # # 如果传入的 vector_size 与实际维度不一致，使用实际维度
        # if vector_size != actual_dim:
        #     logger.warning("传入的 vector_size (%d) 与实际维度 (%d) 不一致", vector_size, actual_dim)

        self._engine: Optional[PGEngine] = None
        self._store: Optional[PGVectorStore] = None

    # ── 上下文管理器 ─────────────────────
    def __enter__(self):
        self.create_store()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._engine:
            self._engine.close()
            logger.info("已关闭 PGEngine 连接")
        self._store = None
        return False  # 不抑制异常

    # ── 核心方法（带日志与耗时） ─────────
    def create_store(self):
        if self._store is not None:
            return self._store

        logger.info("创建 PGEngine 并初始化向量表 '%s'", self.table_name)
        t0 = time.time()
        self._engine = PGEngine.from_connection_string(url=self.connection_string)
        try:
            self._engine.init_vectorstore_table(
                table_name=self.table_name,
                vector_size=self.vector_size,
            )
        except Exception as e:
            logger.warning("init_vectorstore_table 异常（已忽略）: %s", e)

        self._store = PGVectorStore.create_sync(
            engine=self._engine,
            table_name=self.table_name,
            embedding_service=self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,  # 显式指定余弦相似度
        )
        logger.info("向量库就绪 (耗时 %.2fs)", time.time() - t0)
        return self._store

    def add_documents(self, docs: List[Document]) -> List[str]:
        store = self.create_store()
        t0 = time.time()
        ids = store.add_documents(docs)
        logger.info("插入 %d 条文档 (耗时 %.2fs)", len(docs), time.time() - t0)
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        store = self.create_store()
        t0 = time.time()
        results = store.similarity_search(query, k=k)
        logger.info("搜索 '%s' 返回 %d 条结果 (耗时 %.2fs)", query, len(results), time.time() - t0)
        return results

    def delete_collection(self):
        if self._engine:
            from sqlalchemy import text as _text
            async def _drop():
                async with self._engine._pool.connect() as conn:
                    await conn.execute(_text(f"DROP TABLE IF EXISTS {self.table_name}"))
                    await conn.commit()
                await self._engine._pool.dispose()
            self._engine._run_as_sync(_drop())
            logger.info("已删除表 '%s'", self.table_name)
        self._store = None
        self._engine = None

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        return self._store.as_retriever(**kwargs)

    def get_store(self):
        return self._store