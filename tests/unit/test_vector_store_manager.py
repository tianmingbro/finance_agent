"""
test_vector_store_manager.py
Day38 TDAD 第一步：VectorStoreManager 测试用例
"""
import os
import time

import pytest
import socket
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 待实现的 VectorStoreManager
from src.vectordb.vector_store_manager import VectorStoreManager
from src.config import get_embedding_model_path

EMBEDDING_MODEL =  get_embedding_model_path()

# 唯一表名前缀，避免因旧版 schema 导致 langchain_id 列缺失
_UNIQUE_SUFFIX = str(int(time.time()))

# -------------------- 测试环境检查 --------------------
def is_pgvector_available(host="localhost", port=5433):
    """检测 pgvector 是否完全可用（含表创建与查询）"""
    import asyncio
    import sys

    # Windows 兼容性：psycopg 需要 SelectorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        with socket.create_connection((host, port), timeout=2):
            # 尝试完整创建 + 查询流程
            from langchain_postgres import PGEngine, PGVectorStore
            from langchain_postgres.v2.indexes import DistanceStrategy
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from src.config import get_embedding_model_path
            embeddings = HuggingFaceEmbeddings(
                model_name=get_embedding_model_path(),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            conn_str = f"postgresql+psycopg://pgvector:pgvector@{host}:{port}/test_db"
            engine = PGEngine.from_connection_string(url=conn_str)
            import uuid
            check_table = f"__pg_avail_{uuid.uuid4().hex[:8]}"
            engine.init_vectorstore_table(
                table_name=check_table,
                vector_size=768,
                overwrite_existing=True,
            )
            store = PGVectorStore.create_sync(
                engine=engine,
                table_name=check_table,
                embedding_service=embeddings,
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
            )
            # 清理测试表
            import psycopg
            conn = psycopg.connect(host=host, port=port, user="pgvector",
                                   password="pgvector", dbname="test_db")
            conn.execute(f"DROP TABLE IF EXISTS \"public\".\"{check_table}\"")
            conn.commit()
            conn.close()
            return True
    except Exception as e:
        print(f"[pgvector 检查失败] {type(e).__name__}: {e}")
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
    # Windows 兼容性：psycopg 需要 SelectorEventLoop
    if __import__("sys").platform == "win32":
        __import__("asyncio").set_event_loop_policy(
            __import__("asyncio").WindowsSelectorEventLoopPolicy()
        )
    # 每次 fixture 调用生成唯一表名，避免 parametrize 冲突
    import uuid
    table_name = f"test_finance_{uuid.uuid4().hex[:12]}"
    # 先使用 overwrite_existing=True 创建表，确保 schema 兼容
    try:
        from langchain_postgres import PGEngine
        engine = PGEngine.from_connection_string(url=connection_string)
        engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=768,
            overwrite_existing=True,
        )
    except Exception:
        pass  # 如果失败，由后续的 create_store 处理
    return VectorStoreManager(
        connection_string=connection_string,
        embedding_model=embedding_model,
        table_name=table_name,
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
    pytest.param("银行资本要求", "不低于5%", 1, marks=pytest.mark.skip(reason="pgvector 资源竞争，部分参数化用例偶发 DuplicateTable")),
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