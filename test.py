# check_counts.py
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from vector_store_manager import VectorStoreManager
import os
from config import get_embedding_model_path, get_vector_size

emb = HuggingFaceEmbeddings(model_name=get_embedding_model_path(),
                            model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

# Chroma
chroma_store = Chroma(persist_directory="./chroma_db", embedding_function=emb, collection_name="finance_qa")
chroma_count = len(chroma_store.get()["ids"])
print(f"Chroma 文档数: {chroma_count}")

# PGVector
conn = os.getenv("PGVECTOR_CONNECTION_STRING", "postgresql+psycopg://pgvector:pgvector@localhost:5433/ai_rag")
mgr = VectorStoreManager(conn, emb, table_name="finance_knowledge", vector_size=768)
pg_store = mgr.create_store()
pg_count = len(pg_store.get()["ids"]) if hasattr(pg_store, "get") else "无法直接获取（可尝试搜索验证）"
print(f"PGVector 文档数: {pg_count}")