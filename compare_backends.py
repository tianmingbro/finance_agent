"""
compare_backends.py
对比 Chroma 与 PGVector 在相同查询集上的 top-k 检索结果（兼容 Windows）
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import sys
import json
import asyncio
from typing import List, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import get_embedding_model_path, get_vector_size

# ── Windows 事件循环兼容（必须在导入 psycopg 前设置）──
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Embedding 导入（优先使用新版 langchain-huggingface）──
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # 若未安装，回退到旧版（会有弃用警告但不影响运行）
    from langchain_community.embeddings import HuggingFaceEmbeddings

# PGVector 管理器
from vector_store_manager import VectorStoreManager

# -------------------- 配置 --------------------
CHROMA_DIR = "./chroma_db"
PG_TABLE = "finance_knowledge"
PG_CONN = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql+psycopg://pgvector:pgvector@127.0.0.1:5433/ai_rag"
)
EMBEDDING_MODEL = get_embedding_model_path()    
VECTOR_SIZE = get_vector_size()  # 从 config.py 获取向量维度
K = 2

# 测试查询集
TEST_QUERIES = [
    "存款保险最高偿付限额是多少？",
    "个人外汇便利化额度是多少？",
    "商业银行的核心一级资本充足率要求？",
    "LPR最新报价是多少？",
    "资本充足率不达标会有什么后果？",
    "反洗钱法规定了哪些义务？",
]

def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_chroma_store():
    embedding = get_embedding()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
        collection_name="finance_qa",
    )

def get_pgvector_store():
    embedding = get_embedding()
    mgr = VectorStoreManager(
        connection_string=PG_CONN,
        embedding_model=embedding,
        table_name=PG_TABLE,
        vector_size=VECTOR_SIZE,
    )
    return mgr.create_store()

def doc_to_text_id(doc: Document) -> str:
    """将文档内容的前80字符作为唯一标识（近似）"""
    return doc.page_content[:80]

def jaccard_similarity(set1: set, set2: set) -> float:
    """计算两个集合的 Jaccard 相似度"""
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def compare_query(query: str, chroma_store, pg_store) -> dict:
    """对比单个查询在两个后端的 top-k 结果"""
    chroma_docs = chroma_store.similarity_search(query, k=K)
    pg_docs = pg_store.similarity_search(query, k=K)

    chroma_ids = {doc_to_text_id(d) for d in chroma_docs}
    pg_ids = {doc_to_text_id(d) for d in pg_docs}
    jaccard = jaccard_similarity(chroma_ids, pg_ids)

    return {
        "query": query,
        "jaccard": round(jaccard, 3),
        "chroma_count": len(chroma_docs),
        "pgvector_count": len(pg_docs),
    }

def main():
    # print(f"📌 PGVector 连接字符串: {PG_CONN}")   # 添加这一行
    ...
    print("🔍 对比 Chroma vs PGVector 检索一致性")

    # 检查 PGVector 容器是否可用（简单连接测试）
    try:
        import socket
        host, port = "localhost", 5433
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        print("❌ 无法连接 PostgreSQL (localhost:5433)，请先启动 pgvector 容器。")
        sys.exit(1)

    chroma_store = get_chroma_store()
    pg_store = get_pgvector_store()

    results = []
    for q in TEST_QUERIES:
        res = compare_query(q, chroma_store, pg_store)
        results.append(res)
        print(f"  {q[:30]:30s}  Jaccard: {res['jaccard']}")

    avg_jaccard = sum(r["jaccard"] for r in results) / len(results)
    print(f"\n📊 平均 Jaccard 相似度: {avg_jaccard:.3f}")

    # 保存报告
    with open("backend_consistency.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "average_jaccard": avg_jaccard},
                  f, indent=2, ensure_ascii=False)
    print("✅ 一致性报告已保存至 backend_consistency.json")

    # 断言：平均相似度应 >= 0.7
    if avg_jaccard < 0.7:
        print(f"⚠️ 警告：一致性偏低 ({avg_jaccard:.3f} < 0.7)，请检查数据导入是否一致。")
    else:
        print("✅ 一致性检查通过")

if __name__ == "__main__":
    
    main()