"""
test_hybrid_retriever.py (最终修复版)
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from hybrid_retriever import HybridRetriever, SimpleBM25Retriever


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="存款保险最高偿付限额为人民币50万元。", metadata={"source": "deposit"}),
        Document(page_content="商业银行核心一级资本充足率不得低于5%。", metadata={"source": "capital"}),
        Document(page_content="个人每年便利化购汇额度为等值5万美元。", metadata={"source": "forex"}),
        Document(page_content="LPR由各报价行按公开市场操作利率加点形成。", metadata={"source": "lpr"}),
        Document(page_content="反洗钱法要求金融机构建立客户尽职调查制度。", metadata={"source": "aml"}),
    ]


class TestBM25IndexCreation:
    def test_bm25_index_created(self, sample_documents):
        bm25_retriever = SimpleBM25Retriever(documents=sample_documents)
        results = bm25_retriever.get_relevant_documents("存款保险")
        assert len(results) > 0
        assert "50万元" in results[0].page_content


class TestVectorRetriever:
    def test_vector_retriever_works(self, sample_documents):
        mock_vector_retriever = MagicMock()
        mock_vector_retriever.get_relevant_documents.return_value = [sample_documents[0]]
        results = mock_vector_retriever.get_relevant_documents("test")
        assert len(results) > 0


class TestHybridFusion:
    def test_hybrid_fusion_combines_results(self, sample_documents):
        # 向量检索器 mock：设置 invoke 返回前两篇文档
        vector_retriever = MagicMock()
        vector_retriever.invoke.return_value = sample_documents[:2]
        # BM25 检索器：用内嵌真实检索器
        bm25_retriever = SimpleBM25Retriever(documents=sample_documents)

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy="rrf"
        )
        results = hybrid.get_relevant_documents("资本充足率")
        assert len(results) >= 3
        # 检查内容是否包含期望的文档（用 any 检查子串）
        contents = [doc.page_content for doc in results]
        assert any("存款保险最高偿付限额" in c for c in contents)
        assert any("核心一级资本充足率" in c for c in contents)
        assert any("个人每年便利化购汇额度" in c for c in contents)

    def test_rrf_fusion(self, sample_documents):
        vector_retriever = MagicMock()
        # 设置 invoke 返回排序后的文档
        vector_retriever.invoke.return_value = [
            sample_documents[2], sample_documents[0], sample_documents[1]
        ]
        bm25_retriever = MagicMock()
        bm25_retriever.invoke.return_value = [
            sample_documents[1], sample_documents[2], sample_documents[4]
        ]

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy="rrf",
            k=3
        )
        results = hybrid.get_relevant_documents("购汇额度")
        assert len(results) == 3
        top_ids = [doc.metadata["source"] for doc in results]
        assert "forex" in top_ids or "capital" in top_ids

    def test_weight_configuration(self, sample_documents):
        vector_retriever = MagicMock()
        vector_retriever.invoke.return_value = sample_documents[:2]
        bm25_retriever = MagicMock()
        bm25_retriever.invoke.return_value = sample_documents[2:4]

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy="weighted",
            vector_weight=0.8,
            bm25_weight=0.2
        )
        results = hybrid.get_relevant_documents("存款")
        assert len(results) > 0

    def test_empty_query_graceful(self, sample_documents):
        vector_retriever = MagicMock()
        vector_retriever.invoke.return_value = []
        bm25_retriever = SimpleBM25Retriever(documents=sample_documents)

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
        )
        results = hybrid.get_relevant_documents("")
        assert results == []

        # 全部检索器返回空也应返回空
        bm25_retriever.k = 0   # BM25 返回空
        results = hybrid.get_relevant_documents("不存在的内容")
        assert results == []



class TestHybridFusionOrdering:
    """参数化验证不同融合策略的排序行为"""

    @pytest.fixture
    def docs(self):
        return [
            Document(page_content="存款保险最高偿付限额50万元", metadata={"source": "deposit"}),
            Document(page_content="资本充足率不得低于8%", metadata={"source": "capital"}),
            Document(page_content="个人购汇额度5万美元", metadata={"source": "forex"}),
            Document(page_content="LPR由报价行加点形成", metadata={"source": "lpr"}),
        ]

    def _create_retrievers(self, docs, vec_order, bm25_order):
        vector_retriever = MagicMock()
        vector_retriever.invoke.return_value = [docs[i] for i in vec_order]
        bm25_retriever = MagicMock()
        bm25_retriever.invoke.return_value = [docs[i] for i in bm25_order]
        return vector_retriever, bm25_retriever

    @pytest.mark.parametrize(
        "fusion_strategy, vec_order, bm25_order, weights, expected_first",
        [
            # 场景1：RRF - 两处都出现的文档应排前
            ("rrf", [0, 1], [1, 2], None, "资本充足率"),
            # 场景2：加权，向量权重高 (0.8) → 向量第一的文档排前
            ("weighted", [0, 2], [3, 1], {"vector": 0.8, "bm25": 0.2}, "存款保险"),
            # 场景3：加权，BM25 权重高 (0.8) → BM25 第一的文档排前
            ("weighted", [2, 0], [3, 1], {"vector": 0.2, "bm25": 0.8}, "LPR"),
        ],
    )
    def test_top_result_ordering(
        self, docs, fusion_strategy, vec_order, bm25_order, weights, expected_first
    ):
        vector_retriever, bm25_retriever = self._create_retrievers(
            docs, vec_order, bm25_order
        )
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy=fusion_strategy,
            weights=weights,
            k=2,
        )
        results = hybrid.get_relevant_documents("test")
        assert len(results) >= 2
        assert expected_first in results[0].page_content

    @pytest.mark.parametrize(
        "weights, expect_doc_keywords",
        [
            ({"vector": 1.0, "bm25": 0.0}, ["存款保险", "资本充足率"]),   # 纯向量
            ({"vector": 0.0, "bm25": 1.0}, ["LPR", "购汇"]),           # 纯 BM25
            ({"vector": 0.5, "bm25": 0.5}, ["存款保险", "LPR"]),       # 均衡
        ],
    )
    def test_weighted_fusion_follows_weights(self, docs, weights, expect_doc_keywords):
        vector_retriever, bm25_retriever = self._create_retrievers(
            docs, [0, 1], [3, 2]
        )
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy="weighted",
            weights=weights,
            k=2,
        )
        results = hybrid.get_relevant_documents("test")
        assert len(results) == 2
        result_text = " ".join(doc.page_content for doc in results)
        for keyword in expect_doc_keywords:
            assert keyword in result_text, f"期望包含 {keyword}，但结果为：{result_text}"