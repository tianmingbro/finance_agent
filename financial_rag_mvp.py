# ═══════════════════════════════════════════════════════════════
# 金融 RAG 最小可行管道 - LangChain v1.2 版本
# ═══════════════════════════════════════════════════════════════
import os
from pathlib import Path
from typing import List
import shutil

# ===== Part 1: 文档加载 =====
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# ===== Part 2: 文本切片 =====
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== Part 3: 向量库 (使用独立包，替代已弃用的 langchain_community.vectorstores) =====
from langchain_chroma import Chroma

# ===== Part 4: Embedding (Sentence-Transformer 本地模型) =====
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings

# ===== Part 5: LLM (OpenAI API) =====
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
# ===== Part 6: RAG 链 (替代已弃用的 RetrievalQA) =====
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

SCRIPT_DIR = Path(__file__).resolve().parent
# ---------------------- 可调配置 ----------------------
# 替换为你的 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
os.environ["DASHSCOPE_API_KEY"] = "sk-991aa8d5210f42fab50ce7f59dfca11a"
DATA_FILE = SCRIPT_DIR / "data" / "finance_qa.txt"
PERSIST_DIR = SCRIPT_DIR / "chroma_db"
# ------------------------------------------------------
def build_finance_rag_pipeline():
    print("=" * 60)
    print("金融 RAG 管道开始构建...")
    print("=" * 60)

    print("\n[Step 1] 加载金融文档...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()
    print(f"  ✅ 已加载 {len(documents)} 个文档")

    print("\n[Step 2] 切片中...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  ✅ 产生了 {len(chunks)} 个切片")

    print("\n[Step 3] 加载 Embedding 模型...")
    # 使用千问云端 Embedding（无需下载，无需 GPU）
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v1",  # 千问通用 Embedding 模型
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],  # 或直接写你的 Key
    )
    print("  ✅ 千问云端 Embedding 就绪 (text-embedding-v1)")

    print("\n[Step 4] 构建向量库...")
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"  ⚠️  已清空旧向量库 {PERSIST_DIR}/")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name="finance_qa",
    )
    print(f"  ✅ 向量库已持久化至 {PERSIST_DIR}/")

    print("\n[Step 5] 构建 RAG 链...")
    llm = ChatTongyi(model="qwen-plus", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个专业的金融法规咨询助手。\n"
            "使用以下检索到的上下文来回答问题。\n"
            "如果上下文中找不到答案，请诚实地说'该信息未在已知法规中收录'。\n"
            "回答请保持简洁、专业，不超过三句话。\n"
            "\n上下文:\n{context}"
        )),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("  ✅ RAG 链构建完成!")
    print("=" * 60)
    return rag_chain


def ask_question(chain, query):
    print(f"\n{'='*60}")
    print(f"📝 查询: {query}")
    print(f"{'='*60}")
    result = chain.invoke({"input": query})
    print(f"\n📚 检索上下文:")
    for i, doc in enumerate(result.get("context", []), 1):
        print(f"  [{i}] {doc.page_content[:120]}...")
    print(f"\n🤖 AI 回答: {result['answer']}")
    print(f"{'='*60}")
    return result


if __name__ == "__main__":
    rag_chain = build_finance_rag_pipeline()
    test_queries = [
        "商业银行的资本充足率要求是多少？",
        "LPR是什么？它和房贷有什么关系？",
        "个人购汇的年度额度限制是什么？",
        "股票交易有哪些限制？",
    ]
    for q in test_queries:
        ask_question(rag_chain, q)