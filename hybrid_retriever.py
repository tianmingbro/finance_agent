"""
hybrid_retriever.py (重构版)
混合检索器：向量语义 + BM25 关键词
使用内嵌轻量 BM25，完全兼容 Pydantic v2 且不绕过验证
"""
import logging
import time
from typing import List, Tuple, Optional, Any
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ── 内嵌 BM25 检索器 ──────────────────────────────────
class SimpleBM25Retriever(BaseRetriever):
    """轻量级 BM25 检索器，基于 rank_bm25，完全兼容 LangChain"""
    documents: List[Document] = Field(default_factory=list)
    k: int = 4

    _vectorizer: Any = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, documents: List[Document], k: int = 4, **kwargs):
        super().__init__(documents=documents, k=k, **kwargs)
        self._build_index()

    def _build_index(self):
        if not self.documents:
            return
        tokenized = [doc.page_content.split() for doc in self.documents]
        self._vectorizer = BM25Okapi(tokenized)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        if not self.documents or self._vectorizer is None:
            return []
        tokenized_query = query.split()
        scores = self._vectorizer.get_scores(tokenized_query)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:self.k]
        return [self.documents[i] for i, _ in indexed]

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 显式暴露以兼容旧版调用
        return self._get_relevant_documents(query)


def create_bm25_retriever(documents: List[Document], k: int = 4) -> SimpleBM25Retriever:
    return SimpleBM25Retriever(documents=documents, k=k)


# ── 混合检索器 ────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    """融合向量检索与 BM25 检索，支持 RRF 和加权融合"""

    # 用 PrivateAttr 存储检索器，完全避免 Pydantic 字段验证问题
    _vector_retriever: Any = PrivateAttr(default=None)
    _bm25_retriever: Any = PrivateAttr(default=None)

    fusion_strategy: str = Field(default="rrf")
    vector_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    k: int = Field(default=4, ge=1)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        # 提取检索器
        vector_retriever = kwargs.pop("vector_retriever", None)
        bm25_retriever = kwargs.pop("bm25_retriever", None)
        # 兼容旧版权重字典
        weights = kwargs.pop("weights", None)
        if weights:
            kwargs.setdefault("vector_weight", weights.get("vector", 0.5))
            kwargs.setdefault("bm25_weight", weights.get("bm25", 0.5))

        super().__init__(**kwargs)

        # 安全设置私有属性
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever

    @property
    def vector_retriever(self):
        return self._vector_retriever

    @property
    def bm25_retriever(self):
        return self._bm25_retriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        if not query.strip():
            return []
        start_time = time.time()
        vector_docs = self._retrieve(self._vector_retriever, query)
        bm25_docs = self._retrieve(self._bm25_retriever, query)

        logger.info("检索完成 - 向量: %d 篇, BM25: %d 篇", len(vector_docs), len(bm25_docs))
        if not vector_docs and not bm25_docs:
            return []

        if self.fusion_strategy == "rrf":
            merged = self._rrf_fusion([vector_docs, bm25_docs])
        else:
            merged = self._weighted_fusion(vector_docs, bm25_docs)

        final_docs = merged[: self.k]
        elapsed = time.time() - start_time
        logger.info(
            "融合完成 - 策略: %s, 融合后: %d 篇, 最终返回: %d 篇, 耗时: %.3fs",
            self.fusion_strategy, len(merged), len(final_docs), elapsed
        )
        return final_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    def _retrieve(self, retriever, query: str) -> List[Document]:
        if retriever is None:
            return []
        try:
            return retriever.invoke(query)
        except AttributeError:
            try:
                return retriever.get_relevant_documents(query)
            except AttributeError:
                logger.warning("检索器未实现 invoke 或 get_relevant_documents")
                return []

    def _rrf_fusion(self, doc_lists: List[List[Document]], rrf_k: int = 60) -> List[Document]:
        scores = defaultdict(float)
        doc_map = {}
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                key = doc.page_content[:200]
                scores[key] += 1.0 / (rank + rrf_k)
                if key not in doc_map:
                    doc_map[key] = doc
        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[key] for key in sorted_keys]

    def _weighted_fusion(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        scores = defaultdict(float)
        doc_map = {}
        v_docs = self._attach_scores(vector_docs, self.vector_weight)
        b_docs = self._attach_scores(bm25_docs, self.bm25_weight)
        for doc, score in v_docs + b_docs:
            key = doc.page_content[:200]
            scores[key] += score
            if key not in doc_map:
                doc_map[key] = doc
        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[key] for key in sorted_keys]

    def _attach_scores(self, docs: List[Document], weight: float) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        has_score = any("score" in doc.metadata for doc in docs)
        if has_score:
            raw = [(doc, float(doc.metadata["score"])) for doc in docs]
            max_s = max(s[1] for s in raw)
            min_s = min(s[1] for s in raw)
            if max_s > min_s:
                return [(doc, weight * (s - min_s) / (max_s - min_s)) for doc, s in raw]
            else:
                return [(doc, weight) for doc, _ in raw]
        else:
            n = len(docs)
            return [(doc, weight * (1.0 - i / n)) for i, doc in enumerate(docs)]

    def update_weights(self, vector_weight: float, bm25_weight: float):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        logger.info("权重已更新: vector=%.2f, bm25=%.2f", vector_weight, bm25_weight)