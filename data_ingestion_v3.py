"""
ingest_data.py
统一数据导入：从 final_qa_dataset.yaml 切片后写入向量库。
通过环境变量 VECTOR_BACKEND 选择 chroma（默认）或 pgvector。
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from loader_facade import LoaderFacade        # 可选，这里未直接使用
from splitter_factory import SplitterFactory
from vector_store_manager import VectorStoreManager
from config import get_embedding_model_path, get_vector_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 配置 --------------------
FINAL_QA_FILE = "data/final_qa_dataset.yaml"
CHROMA_PERSIST_DIR = "./chroma_db"
PG_CONNECTION_STRING_ENV = "PGVECTOR_CONNECTION_STRING"
DEFAULT_PG_CONNECTION = "postgresql+psycopg://pgvector:pgvector@localhost:5433/ai_rag"
TABLE_NAME = "finance_knowledge"
VECTOR_SIZE = get_vector_size()  # 从 config.py 获取向量维度
EMBEDDING_MODEL_NAME = get_embedding_model_path()  # 从 config.py 获取模型路径

def load_qa_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载 final_qa_dataset.yaml，返回 QA 列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 兼容不同的根键
    if isinstance(data, list):
        return data
    for key in ["final_qa_dataset", "reviewed_qa", "candidate_qa"]:
        if key in data:
            return data[key]
    return []

def qa_to_documents(qa_list: List[Dict]) -> List[Document]:
    """将 QA 对转换为 Document 列表"""
    docs = []
    for item in qa_list:
        query = item.get("query") or item.get("question", "")
        answer = item.get("answer", "")
        if not query or not answer:
            continue
        content = f"问题：{query}\n答案：{answer}"
        metadata = {
            "source": item.get("source", "final_qa_dataset"),
            "category": item.get("category", "unknown"),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    """使用 SplitterFactory 切分文档"""
    factory = SplitterFactory()
    return factory.split(docs, strategy="recursive", chunk_size=500, chunk_overlap=100)

def get_embedding_model():
    """获取统一的 Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def ingest_to_chroma(chunks: List[Document]):
    """写入 Chroma 向量库"""
    embedding = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding,
        collection_name="finance_qa",
    )
    vectorstore.add_documents(chunks)
    logger.info("✅ 已写入 Chroma (%s)，共 %d 个片段", CHROMA_PERSIST_DIR, len(chunks))
    return vectorstore

def ingest_to_pgvector(chunks: List[Document]):
    """通过 VectorStoreManager 写入 PGVector"""
    conn_str = os.getenv(PG_CONNECTION_STRING_ENV, DEFAULT_PG_CONNECTION)
    embedding = get_embedding_model()
    with VectorStoreManager(
        connection_string=conn_str,
        embedding_model=embedding,
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
    ) as mgr:
        mgr.add_documents(chunks)
        logger.info("✅ 已写入 PGVector (表 %s)，共 %d 个片段", TABLE_NAME, len(chunks))
        return mgr  # 注意：退出 with 会关闭引擎，但此处仅为写入，后续验证会重新连接

def verify_retrieval(backend: str):
    """执行几个查询，确认数据可被检索"""
    embedding = get_embedding_model()
    if backend == "chroma":
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding,
            collection_name="finance_qa",
        )
    else:  # pgvector
        conn_str = os.getenv(PG_CONNECTION_STRING_ENV, DEFAULT_PG_CONNECTION)
        mgr = VectorStoreManager(
            connection_string=conn_str,
            embedding_model=embedding,
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE,
        )
        vectorstore = mgr.create_store()

    test_queries = [
        "存款保险最高偿付限额是多少？",
        "个人外汇便利化额度是多少？",
        "商业银行的核心一级资本充足率要求？",
    ]
    print("\n🔍 检索验证 (后端: {})".format(backend))
    for q in test_queries:
        docs = vectorstore.similarity_search(q, k=2)
        print(f"\n❓ {q}")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc.page_content[:100]}...")

def main():
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
    if backend not in ("chroma", "pgvector"):
        raise ValueError(f"不支持的向量库后端: {backend}")

    # 1. 加载 QA 数据集
    if not Path(FINAL_QA_FILE).exists():
        print(f"❌ 数据集文件不存在: {FINAL_QA_FILE}")
        sys.exit(1)
    qa_list = load_qa_dataset(FINAL_QA_FILE)
    print(f"📋 加载 {len(qa_list)} 条 QA 对")

    # 2. 转换为 Document 并切片
    docs = qa_to_documents(qa_list)
    chunks = split_documents(docs)
    print(f"🔪 切片得到 {len(chunks)} 个片段")

    # 3. 写入向量库
    if backend == "chroma":
        ingest_to_chroma(chunks)
    else:
        ingest_to_pgvector(chunks)

    # 4. 检索验证
    verify_retrieval(backend)

    print("\n✅ 导入流程完成")

if __name__ == "__main__":
    main()