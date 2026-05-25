"""
import_docs.py
命令行文档导入工具：将指定目录下的支持文档切片后写入向量库
用法示例：
  python import_docs.py --input data/source_docs --backend chroma
  python import_docs.py --input data/source_docs --backend pgvector --table finance_knowledge
"""
import argparse
import sys
from pathlib import Path
from loader_facade import LoaderFacade
from data_ingestion import split_documents, update_vectorstore  # 复用切片与入库逻辑
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="批量导入金融法规文档至向量库")
    parser.add_argument("--input", type=str, default="data/source_docs",
                        help="包含文档的目录路径 (默认: data/source_docs)")
    parser.add_argument("--backend", type=str, default="chroma",
                        choices=["chroma", "pgvector"],
                        help="向量库后端 (默认: chroma)")
    parser.add_argument("--persist", type=str, default="./chroma_db",
                        help="Chroma 持久化目录 (仅后端为 chroma 时有效)")
    parser.add_argument("--table", type=str, default="finance_knowledge",
                        help="PGVector 表名 (仅后端为 pgvector 时有效)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要加载的文件，不执行入库")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_path}")
        sys.exit(1)

    # 加载文档
    facade = LoaderFacade()
    docs = []
    for file in input_path.iterdir():
        if file.is_file():
            try:
                docs.extend(facade.load(file))
            except ValueError:
                pass  # 跳过不支持格式

    if not docs:
        print("⚠️ 没有找到可加载的文档")
        sys.exit(0)

    print(f"📄 加载 {len(docs)} 个文档片段")
    if args.dry_run:
        print("--dry-run 模式，仅列示文件：")
        for f in input_path.iterdir():
            if f.suffix.lower() in facade._loaders:
                print(f"  - {f.name}")
        sys.exit(0)

    # 切片
    chunks = split_documents(docs)

    # 入库
    if args.backend == "chroma":
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        EMBEDDING_MODEL =  os.path.join(BASE_DIR, "models1","all-MiniLM-L6-v2")

        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            persist_directory=args.persist,
            embedding_function=embedding,
            collection_name="finance_qa",
        )
        vectorstore.add_documents(chunks)
        print(f"✅ 已写入 Chroma ({args.persist})")
    elif args.backend == "pgvector":
        # 需提前设置 PGVECTOR_CONNECTION_STRING 环境变量
        from langchain_postgres import PGEngine, PGVectorStore
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDING_MODEL =  os.path.join(BASE_DIR, "models","text2vec-base-chinese","Jerry0", "text2vec-base-chinese")

        conn_str = os.getenv("PGVECTOR_CONNECTION_STRING",
                             "postgresql+psycopg://pgvector:pgvector@localhost:5432/ai_rag")
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        engine = PGEngine.from_connection_string(url=conn_str)
        try:
            engine.init_vectorstore_table(table_name=args.table, vector_size=384)
        except Exception:
            pass
        store = PGVectorStore.create_sync(
            engine=engine,
            table_name=args.table,
            embedding_service=embedding,
        )
        store.add_documents(chunks)
        print(f"✅ 已写入 PGVectorStore (表: {args.table})")

if __name__ == "__main__":
    main()