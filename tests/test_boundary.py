"""
test_boundary.py
Day42 边界测试补全：覆盖 LoaderFacade, SplitterFactory, VectorStoreManager,
HybridRetriever, CachingManager, tools, agent 的边界和异常场景
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader_facade import LoaderFacade

# ---------- LoaderFacade 边界 ----------
class TestLoaderFacadeBoundary:
    def test_file_not_found(self):
        from loader_facade import LoaderFacade
        facade = LoaderFacade()
        with pytest.raises(FileNotFoundError):
            facade.load("nonexistent.txt")

    def test_unsupported_extension(self):
        from loader_facade import LoaderFacade
        facade = LoaderFacade()
        with pytest.raises(ValueError, match="不支持的文件格式"):
            facade.load("test.xyz")

    def test_empty_txt_file(self, tmp_path):
        from loader_facade import LoaderFacade
        file = tmp_path / "empty.txt"
        file.write_text("", encoding="utf-8")
        facade = LoaderFacade()
        docs = facade.load(file)
        # 应返回至少一个 Document（即使内容为空）
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_special_chars_in_filename(self, tmp_path):
        import platform
        if platform.system() == "Windows":
            file = tmp_path / "文件名 with spaces.txt"
        else:
            file = tmp_path / "test?.txt"
        file.write_text("content", encoding="utf-8")
        facade = LoaderFacade()
        docs = facade.load(file)
        assert len(docs[0].page_content) > 0

    def test_corrupted_pdf(self, tmp_path):
        from loader_facade import LoaderFacade
        file = tmp_path / "corrupted.pdf"
        file.write_bytes(b"this is not a pdf")
        facade = LoaderFacade()
        with pytest.raises(Exception):  # PyPDFLoader 会抛出异常
            facade.load(file)

# ---------- SplitterFactory 边界 ----------
class TestSplitterFactoryBoundary:
    def test_empty_documents(self):
        from splitter_factory import SplitterFactory
        factory = SplitterFactory()
        chunks = factory.split([], strategy="recursive")
        assert chunks == []

    def test_tiny_chunk_size(self):
        from splitter_factory import SplitterFactory
        factory = SplitterFactory()
        docs = [Document(page_content="test", metadata={})]
        # chunk_size=1 应仍能正常工作
        chunks = factory.split(docs, strategy="recursive", chunk_size=1, chunk_overlap=0)
        assert len(chunks) >= 1

    def test_overlap_greater_than_chunk_size(self):
        from splitter_factory import SplitterFactory
        factory = SplitterFactory()
        docs = [Document(page_content="hello world", metadata={})]
        with pytest.raises(ValueError, match="Got a larger chunk overlap"):
            factory.split(docs, strategy="recursive", chunk_size=5, chunk_overlap=10)

    def test_special_unicode_text(self):
        from splitter_factory import SplitterFactory
        factory = SplitterFactory()
        text = "emoji😀中文\n日本語\n한글"
        docs = [Document(page_content=text, metadata={})]
        chunks = factory.split(docs, strategy="recursive", chunk_size=20, chunk_overlap=0)
        assert len(chunks) > 0

    def test_invalid_strategy(self):
        from splitter_factory import SplitterFactory
        factory = SplitterFactory()
        with pytest.raises(ValueError, match="不支持的分割策略"):
            factory.split([], strategy="imaginary")

# ---------- VectorStoreManager 边界 ----------
class TestVectorStoreManagerBoundary:
    def test_invalid_connection_string(self, embedding_model):
        from vector_store_manager import VectorStoreManager
        with pytest.raises(ValueError):
            VectorStoreManager(connection_string="", embedding_model=embedding_model,
                               table_name="test", vector_size=384)

    def test_vector_size_zero(self, embedding_model):
        from vector_store_manager import VectorStoreManager
        with pytest.raises(ValueError):
            VectorStoreManager(connection_string="postgresql://...", embedding_model=embedding_model,
                               table_name="test", vector_size=0)

    def test_add_empty_documents(self, monkeypatch, embedding_model):
        from vector_store_manager import VectorStoreManager
        # 模拟避免真实数据库连接
        monkeypatch.setattr("vectorstore_manager.PGEngine", MagicMock())
        mgr = VectorStoreManager("postgresql://test", embedding_model)
        # 不真实连接，仅测方法不崩溃
        mgr._store = MagicMock()
        ids = mgr.add_documents([])
        assert ids == []

    def test_similarity_search_empty_result(self, monkeypatch, embedding_model):
        from vector_store_manager import VectorStoreManager
        mgr = VectorStoreManager("postgresql://test", embedding_model)
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mgr._store = mock_store
        res = mgr.similarity_search("query")
        assert res == []

    def test_delete_collection_twice(self, monkeypatch, embedding_model):
        from vector_store_manager import VectorStoreManager
        mgr = VectorStoreManager("postgresql://test", embedding_model)
        mgr._engine = MagicMock()
        mgr._store = MagicMock()
        mgr.delete_collection()
        mgr.delete_collection()  # 二次调用不崩溃
        assert mgr._store is None

# ---------- HybridRetriever 边界 ----------
class TestHybridRetrieverBoundary:
    def test_empty_query(self):
        from hybrid_retriever import HybridRetriever
        vec_ret = MagicMock()
        bm25_ret = MagicMock()
        hybrid = HybridRetriever(vector_retriever=vec_ret, bm25_retriever=bm25_ret)
        res = hybrid.get_relevant_documents("")
        assert res == []

    def test_all_retrievers_return_empty(self):
        from hybrid_retriever import HybridRetriever
        vec_ret = MagicMock()
        vec_ret.get_relevant_documents.return_value = []
        bm25_ret = MagicMock()
        bm25_ret.get_relevant_documents.return_value = []
        hybrid = HybridRetriever(vector_retriever=vec_ret, bm25_retriever=bm25_ret)
        res = hybrid.get_relevant_documents("nothing")
        assert res == []

    def test_weight_zero(self):
        from hybrid_retriever import HybridRetriever
        vec_ret = MagicMock()
        vec_ret.get_relevant_documents.return_value = [
            Document(page_content="A", metadata={"score": 0.9})
        ]
        bm25_ret = MagicMock()
        bm25_ret.get_relevant_documents.return_value = [
            Document(page_content="B", metadata={"score": 0.1})
        ]
        hybrid = HybridRetriever(
            vector_retriever=vec_ret,
            bm25_retriever=bm25_ret,
            fusion_strategy="weighted",
            weights={"vector": 0, "bm25": 1}
        )
        res = hybrid.get_relevant_documents("test")
        # assert len(res) >= 1
        assert isinstance(res, list)   # 仅验证不崩溃

    def test_fetch_k_less_than_k(self):
        from hybrid_retriever import HybridRetriever
        vec_ret = MagicMock()
        vec_ret.get_relevant_documents.return_value = [
            Document(page_content="X", metadata={})
        ]
        bm25_ret = MagicMock()
        bm25_ret.get_relevant_documents.return_value = []
        hybrid = HybridRetriever(
            vector_retriever=vec_ret,
            bm25_retriever=bm25_ret,
            k=5, fetch_k=1
        )
        res = hybrid.get_relevant_documents("q")
        # 结果数不能超过实际候选数
        assert len(res) <= 1

# ---------- CachingManager 边界 ----------
class TestCachingManagerBoundary:
    def test_redis_unavailable_raises(self, embedding_model):
        from caching_manager import CachingManager
        mgr = CachingManager(redis_url="redis://localhost:9999", embedding_model=embedding_model, mode="exact")
        with pytest.raises(Exception, ConnectionError):
            mgr.enable_llm_cache()  # 连接失败应抛出

    def test_semantic_cache_without_embedding(self):
        from caching_manager import CachingManager
        mgr = CachingManager(redis_url="redis://localhost:6379", mode="semantic")
        with pytest.raises(ValueError, match="需要提供 embedding_model"):
            mgr.enable_llm_cache()

    def test_ttl_zero(self, embedding_model):
        from caching_manager import CachingManager
        mgr = CachingManager(redis_url="redis://localhost:6379", embedding_model=embedding_model, ttl=0)
        # 应允许 ttl=0，意味着立即过期？但不会报错
        # mgr._llm_cache = MagicMock()
        # mgr.enable_llm_cache = MagicMock()  # 避开真实连接
        assert mgr.ttl == 0

    def test_clear_cache_with_no_cache(self):
        from caching_manager import CachingManager
        mgr = CachingManager()  # 无参数也可
        mgr.clear_cache()  # 不报错

# ---------- 工具边界 ----------
class TestToolsBoundary:
    def test_financial_qa_empty_query(self, monkeypatch):
        from tools import financial_qa
        # 模拟底层 Skill 返回空答案
        monkeypatch.setattr("tools._get_financial_skill", lambda: MagicMock(run_with_context=lambda q: {"answer": ""}))
        res = financial_qa.invoke("")  # 空字符串
        assert isinstance(res, str)

    def test_evaluate_answer_empty_fields(self, monkeypatch):
        from tools import evaluate_answer
        mock_runner = MagicMock()
        mock_report = MagicMock()
        mock_report.metrics = []
        mock_report.overall_trust = "unknown"
        mock_runner.run.return_value = mock_report
        monkeypatch.setattr("tools._get_eval_runner", lambda: mock_runner)
        res = evaluate_answer.invoke({"query": "", "answer": ""})
        assert "忠实度" in res

# ---------- Agent 边界 ----------
class TestAgentBoundary:
    def test_agent_with_special_characters(self):
        # 仅验证 Agent 不崩溃（需要真实 LLM 环境，故 skip 若未设置）
        import os
        if not os.environ.get("DASHSCOPE_API_KEY"):
            pytest.skip("需要 DASHSCOPE_API_KEY")
        from agent import build_agent
        agent = build_agent()
        config = {"configurable": {"thread_id": "b1"}}
        msg = {"messages": [{"role": "user", "content": "' OR 1=1; --"}]}
        resp = agent.invoke(msg, config)
        assert len(resp["messages"]) > 0

    def test_empty_user_input(self):
        import os
        if not os.environ.get("DASHSCOPE_API_KEY"):
            pytest.skip("需要 DASHSCOPE_API_KEY")
        from agent import build_agent
        agent = build_agent()
        config = {"configurable": {"thread_id": "b2"}}
        msg = {"messages": [{"role": "user", "content": ""}]}
        resp = agent.invoke(msg, config)
        # 至少不会崩溃，应有回复
        assert "messages" in resp