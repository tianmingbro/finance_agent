"""
compare_retrievers.py
对比原向量检索器和混合检索器对相同问题的 top-3 结果
"""
import os
import yaml
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 导入混合检索相关
from hybrid_retriever import HybridRetriever, create_bm25_retriever
from financial_rag_skill import ResourceManager  # 若 ResourceManager 已更新则直接使用
from config import get_embedding_model_path, get_vector_size

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = get_embedding_model_path()
VECTOR_SIZE = get_vector_size()

def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_bm25_docs():
    """加载 BM25 索引文档（与 ResourceManager 中的逻辑保持一致）"""
    docs = []
    qa_path = Path("data/final_qa_dataset.yaml")
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        qa_list = data if isinstance(data, list) else data.get("final_qa_dataset", [])
        for item in qa_list:
            q = item.get("query") or item.get("question", "")
            a = item.get("answer", "")
            if q and a:
                content = f"问题：{q}\n答案：{a}"
                docs.append(Document(page_content=content, metadata={"source": "qa_dataset"}))
    return docs

def compare_queries():
    # 1. 向量检索器
    embedding = get_embedding()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
        collection_name="finance_qa",
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. BM25 检索器
    bm25_docs = load_bm25_docs()
    bm25_retriever = create_bm25_retriever(bm25_docs, k=3)

    # 3. 混合检索器
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fusion_strategy="rrf",
        k=3,
        fetch_k=20,
    )

    test_queries = [
        "存款保险最高偿付限额是多少？",
        "个人外汇便利化额度是多少？",
        "商业银行的核心一级资本充足率要求？",
        "反洗钱法规定的客户尽职调查是什么？",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"❓ 查询: {query}")

        # 向量检索 top-3
        vec_docs = vector_retriever.invoke(query)
        print("\n--- 向量检索 Top-3 ---")
        for i, doc in enumerate(vec_docs[:3], 1):
            print(f"  [{i}] {doc.page_content[:100]}...")

        # 混合检索 top-3
        hyb_docs = hybrid_retriever.get_relevant_documents(query)
        print("\n--- 混合检索 Top-3 ---")
        for i, doc in enumerate(hyb_docs[:3], 1):
            print(f"  [{i}] {doc.page_content[:100]}...")

if __name__ == "__main__":
    compare_queries()