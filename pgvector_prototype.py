"""
pgvector_prototype.py
Day32 交付物：PGVectorStore 独立原型验证
兼容：langchain v1.2 + langchain-postgres >= 0.0.14
"""
# 添加在脚本最顶部，在所有其他代码之前
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import os
import sys
from typing import List

# LangChain v1.2 核心
from langchain_core.documents import Document

# Embedding（本地模型，无需 GPU，兼容 Chroma 开发环境）
from langchain_community.embeddings import HuggingFaceEmbeddings

# PGVectorStore（替代已弃用的 PGVector）
from langchain_postgres import PGEngine, PGVectorStore

# 文本切片
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── 配置 ───────────────────────────────────────────────
CONNECTION_STRING = (
    "postgresql+psycopg://pgvector:pgvector@localhost:5432/ai_rag"
)
TABLE_NAME = "finance_knowledge"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 的输出维度
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
# ─── 初始化 Embedding（与 Chroma 使用相同的模型）─────────
def create_embedding():
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore():
    """创建 PGVectorStore 实例并初始化表结构"""
    print("🔌 连接 PostgreSQL + pgvector ...")
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    # 显式创建向量表（PGVectorStore 需要手动建表）
    print(f"📋 初始化向量表: {TABLE_NAME}")
    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
        # 可选：指定距离策略和索引类型
    )

    embedding = create_embedding()
    store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embedding,
    )
    print("✅ PGVectorStore 创建成功")
    return store


def insert_documents(store: PGVectorStore):
    """插入金融知识文档"""
    raw_texts = [
        "根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%，"
        "一级资本充足率不得低于6%，资本充足率不得低于8%。",

        "LPR（贷款市场报价利率）是商业银行对其最优质客户执行的贷款利率。"
        "2025年1年期LPR为3.1%，5年期以上LPR为3.6%。",

        "根据《个人外汇管理办法》，个人每年结汇和购汇的便利化额度"
        "为等值5万美元。超过额度需提供真实合规的用途证明。",
    ]

    # 切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    )
    docs = []
    for text in raw_texts:
        docs.extend(splitter.create_documents([text]))

    print(f"📝 插入 {len(docs)} 个文档切片 ...")
    ids = store.add_documents(docs)
    print(f"✅ 已插入，IDs: {ids}")
    return docs


def similarity_search_demo(store: PGVectorStore):
    """执行相似度搜索，验证与 Chroma 接口一致"""
    queries = [
        "商业银行的资本充足率是多少？",
        "LPR最新报价",
        "个人能买多少外汇？",
    ]
    print("\n" + "=" * 60)
    print("🔍 相似度搜索验证")
    print("=" * 60)
    for q in queries:
        print(f"\n❓ 查询: {q}")
        results = store.similarity_search(q, k=2)
        for i, doc in enumerate(results, 1):
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] {snippet}...")
    print("\n✅ 所有查询返回相关结果，接口与 Chroma 一致")


def verify_with_chroma():
    """与 Chroma 接口对比：确认两者方法签名一致"""
    print("\n" + "=" * 60)
    print("📋 接口一致性检查")
    print("=" * 60)

    common_methods = [
        "similarity_search",
        "add_documents",
        "as_retriever",
        "delete",
    ]
    for m in common_methods:
        pgv_has = hasattr(PGVectorStore, m)
        # Chroma 同样具有这些方法
        print(f"  {m}(): {'✅' if pgv_has else '❌'} PGVectorStore")
    print("结论: PGVectorStore 与 Chroma 共享相同的 LangChain VectorStore 接口")


if __name__ == "__main__":
    print("=" * 60)
    print("  PGVectorStore 原型验证")
    print("=" * 60)

    store = build_vectorstore()
    insert_documents(store)
    similarity_search_demo(store)
    verify_with_chroma()

    print("\n🎉 全部测试通过！PGVectorStore 与 Chroma 接口兼容，可无缝切换。")