"""
data_ingestion.py
Day36 交付物：使用 LoaderFacade 批量导入原始文档至 Chroma
"""
"""
data_ingestion_v2.py
使用 SplitterFactory 的多策略文档导入管道
"""
import os
import sys
from pathlib import Path
from typing import List

# LangChain 组件
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 自定义模块
from loader_facade import LoaderFacade
from splitter_factory import SplitterFactory

# -------------------- 配置 --------------------
SOURCE_DIR = "data/source_docs"
CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_STRATEGY = "recursive"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL =  os.path.join(BASE_DIR, "models","text2vec-base-chinese","Jerry0", "text2vec-base-chinese")

# 文件扩展名 → 分割策略映射
EXTENSION_STRATEGY_MAP = {
    ".md": "markdown",
    ".txt": "recursive",
    ".pdf": "recursive",
    ".docx": "recursive",
}

def load_all_documents(source_dir: str) -> List[Document]:
    """批量加载目录下所有支持的文档"""
    facade = LoaderFacade()
    all_docs = []
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ 目录不存在: {source_dir}")
        sys.exit(1)

    for file_path in source_path.iterdir():
        if file_path.is_file():
            try:
                docs = facade.load(file_path)
                print(f"✅ 加载成功: {file_path.name} ({len(docs)} 个片段)")
                all_docs.extend(docs)
            except ValueError:
                print(f"⏭️ 跳过不支持格式: {file_path.name}")
            except Exception as e:
                print(f"❌ 加载失败: {file_path.name} ({e})")
    print(f"\n📄 总计加载 {len(all_docs)} 个文档对象")
    return all_docs

def split_documents_by_type(documents: List[Document]) -> List[Document]:
    """
    根据文档原始文件扩展名选择分割策略。
    未识别扩展名或未知类型默认使用递归策略。
    """
    factory = SplitterFactory()
    chunks = []

    # 按策略分组
    strategy_docs: dict = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        ext = Path(source).suffix.lower()
        strategy = EXTENSION_STRATEGY_MAP.get(ext, DEFAULT_STRATEGY)
        strategy_docs.setdefault(strategy, []).append(doc)

    for strategy, docs in strategy_docs.items():
        if not docs:
            continue
        print(f"🔪 使用 '{strategy}' 策略分割 {len(docs)} 个文档...")
        # 透传常用参数（markdown 策略不需要 chunk_size）
        kwargs = {}
        if strategy != "markdown":
            kwargs["chunk_size"] = CHUNK_SIZE
            kwargs["chunk_overlap"] = CHUNK_OVERLAP
        # 对 semantic 策略需要 embeddings，此处暂不启用
        split_chunks = factory.split(docs, strategy=strategy, **kwargs)
        chunks.extend(split_chunks)
        print(f"   -> 得到 {len(split_chunks)} 个片段")

    print(f"\n🔪 切片完成: 总计 {len(chunks)} 个片段")
    return chunks

def update_vectorstore(chunks: List[Document]):
    """将切片写入 Chroma（追加模式）"""
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding,
        collection_name="finance_qa",
    )
    vectorstore.add_documents(chunks)
    print(f"✅ 已写入 Chroma ({CHROMA_PERSIST_DIR})")

def verify_retrieval():
    """检索测试：验证新增内容可被搜索到"""
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding,
        collection_name="finance_qa",
    )

    test_queries = [
        "存款保险最高偿付限额是多少？",
        "个人外汇年度便利化额度是多少？",
        "资本充足率指标有哪些？",
        "Markdown文档中的存款保险条例第一条是什么？",  # 示例，如果导入了 md 文件
    ]
    print("\n" + "=" * 60)
    print("  检索验证")
    print("=" * 60)
    for q in test_queries:
        results = vectorstore.similarity_search(q, k=2)
        print(f"\n❓ {q}")
        for i, doc in enumerate(results, 1):
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] {snippet}...")

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  SplitterFactory 多策略文档导入管道")
    print("=" * 60)

    # 1. 加载
    docs = load_all_documents(SOURCE_DIR)
    if not docs:
        print("⚠️ 没有加载到任何文档。")
        sys.exit(0)

    # 2. 按策略切片
    chunks = split_documents_by_type(docs)

    # 3. 写入向量库
    update_vectorstore(chunks)

    # 4. 验证检索
    verify_retrieval()