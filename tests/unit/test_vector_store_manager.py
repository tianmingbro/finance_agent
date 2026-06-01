"""
test_vector_store_manager.py
Day38 TDAD 第一步：VectorStoreManager 测试用例
"""
import os

import pytest
import socket
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 待实现的 VectorStoreManager
from src.vectordb.vector_store_manager import VectorStoreManager
from config import get_embedding_model_path

EMBEDDING_MODEL =  get_embedding_model_path()

# -------------------- 测试环境检查 --------------------
def is_pgvector_available(host="localhost", port=5433):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False

# 若 pgvector 容器未运行，跳过所有数据库相关测试
skip_if_no_pgvector = pytest.mark.skipif(
    not is_pgvector_available(),
    reason="pgvector 容器未运行，请执行: docker run -d --name pgvector-test ..."
)


# -------------------- Fixture --------------------
@pytest.fixture(scope="session")
def embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@pytest.fixture
def connection_string():
    return "postgresql+psycopg://pgvector:pgvector@localhost:5433/test_db"

@pytest.fixture
def manager(connection_string, embedding_model):
    return VectorStoreManager(
        connection_string=connection_string,
        embedding_model=embedding_model,
        table_name="test_finance",
        vector_size=768,
    )
@pytest.fixture
def manager_with_data(manager):
    """预填充文档的 manager"""
    docs = [
        Document(page_content="存款保险最高偿付限额为人民币50万元。", metadata={"id": "doc1"}),
        Document(page_content="商业银行核心一级资本充足率不得低于5%。", metadata={"id": "doc2"}),
        Document(page_content="个人每年便利化购汇额度为等值5万美元。", metadata={"id": "doc3"}),
        Document(page_content="LPR由各报价行按公开市场操作利率加点形成。", metadata={"id": "doc4"}),
    ]
    manager.add_documents(docs)
    return manager

@skip_if_no_pgvector
@pytest.mark.parametrize("query,expected_keyword,min_results", [
    ("存款保险最多赔多少", "50万元", 1),
    ("银行资本要求", "不低于5%", 1),
    ("外汇额度", "5万美元", 1),
    ("利率怎么定的", "LPR", 1),
    ("比特币风险", "……", 0),  # 预期无相关结果
])
def test_search_relevance(manager_with_data, query, expected_keyword, min_results):
    results = manager_with_data.similarity_search(query, k=2)
    assert len(results) >= min_results
    if min_results > 0:
        # 检查 top-1 结果是否包含预期关键词
        assert expected_keyword in results[0].page_content
# -------------------- 测试类 --------------------
@skip_if_no_pgvector
class TestVectorStoreManager:
    """核心功能测试（需要真实 pgvector 数据库）"""

    def test_create_store(self, manager):
        """验证能成功创建 PGVectorStore 并初始化表结构"""
        store = manager.create_store()
        assert store is not None, "create_store 应返回非空实例"
        # 再次调用应返回同一实例（单例）
        store2 = manager.create_store()
        assert store is store2, "应返回同一个 store 实例"

    def test_add_documents_and_search(self, manager):
        """插入文档后能通过相似搜索找到"""
        docs = [
            Document(
                page_content="存款保险最高偿付限额为人民币50万元。",
                metadata={"source": "test.txt"}
            ),
            Document(
                page_content="商业银行资本充足率不得低于8%。",
                metadata={"source": "test.txt"}
            ),
        ]
        # 插入文档
        manager.add_documents(docs)

        # 搜索相关文档
        results = manager.similarity_search("存款保险赔付上限", k=1)
        assert len(results) > 0
        assert "50万元" in results[0].page_content

    def test_delete_collection(self, manager):
        """删除集合后再搜索应返回空"""
        # 先插入一些文档
        docs = [Document(page_content="测试文档", metadata={})]
        manager.add_documents(docs)

        # 删除整张表
        manager.delete_collection()

        # 重建 store（因为表已删除）
        store = manager.create_store()

        # 搜索应返回空
        results = store.similarity_search("测试", k=2)
        assert len(results) == 0

    def test_singleton_behavior(self, manager):
        """连续两次 create_store 返回同一实例，且内部引擎复用"""
        store1 = manager.create_store()
        store2 = manager.create_store()
        assert store1 is store2

        # 验证内部引擎也是同一个
        assert manager._engine is not None
        assert manager._store is not None


class TestInputValidation:
    """输入校验测试（不需要真实数据库）"""

    def test_missing_connection_string_raises(self, embedding_model):
        with pytest.raises(ValueError, match="connection_string 不能为空"):
            VectorStoreManager(
                connection_string="",
                embedding_model=embedding_model,
                table_name="test",
                vector_size=768,
            )

    def test_invalid_vector_size_raises(self, connection_string, embedding_model):
        with pytest.raises(ValueError, match="vector_size 必须大于 0"):
            VectorStoreManager(
                connection_string=connection_string,
                embedding_model=embedding_model,
                table_name="test",
                vector_size=0,
            )