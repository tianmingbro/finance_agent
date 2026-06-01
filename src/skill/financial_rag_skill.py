"""
金融 RAG Skill 完整封装 (三层渐进加载)
Day30 核心交付物
"""
# financial_rag_skill.py 最顶部
import logging
import sys
import asyncio
import threading

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import os
import re
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field

# LangChain v1.2 核心组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# 文档处理 (已验证可用)
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# RAG 链 (替代弃用的 RetrievalQA)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# LLM 使用 Qwen-plus (通过 OpenAI 兼容接口)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from src.vectordb.vector_store_interface import create_vector_store
from src.retriever.hybrid_retriever import HybridRetriever, create_bm25_retriever
from langchain_core.documents import Document
from src.cache.caching_manager import CachingManager
from config import get_embedding_model_path, get_vector_size
logger = logging.getLogger(__name__)

# -------------------- 配置 --------------------
# 请确保已设置环境变量 DASHSCOPE_API_KEY
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-dashscope-api-key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "finance_qa.txt")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
MODEL_PATH = get_embedding_model_path()
VECTOR_SIZE = get_vector_size() 

# ==================== 1. 元数据定义 ====================
@dataclass
class SkillMetadata:
    """金融问答技能元数据"""
    name: str = "financial_rag_qa"
    version: str = "0.1.0"
    description: str = "面向中国金融法规的知识问答技能，支持资本充足率、LPR、外汇管理等咨询"
    author: str = "AI转型训练营"
    dependencies: List[str] = field(default_factory=lambda: [
        "langchain>=1.2",
        "langchain-openai",
        "langchain-chroma",
        "sentence-transformers"
    ])
    trigger_keywords: List[str] = field(default_factory=lambda: [
    # 原有
    "资本充足率", "LPR", "贷款市场报价利率", "购汇额度",
    "外汇管理", "金融法规", "监管要求", "商业银行",
    "房贷利率", "外汇额度", "资本管理办法", "核心一级资本",
    "便利化额度",
    # 新增：覆盖存款保险、反洗钱、LPR改革等领域
    "存款保险", "偿付限额", "最高偿付",
    "反洗钱", "洗钱", "反洗钱法",
    "LPR改革", "LPR调整", "LPR报价", "LPR机制",
    "外汇便利化", "结汇", "购汇", "外汇",
    "系统重要性银行", "资本监管", "资本达标",
    "金融机构监管", "金融监管", "银行监管",
    "金融政策", "金融法规咨询", "金融法规问答"])
    performance_baseline: Dict[str, float] = field(default_factory=lambda: {
        "p95_latency_ms": 2000,
        "expected_accuracy": 0.9
    })
    eval_dataset_version: str = "0.1"


# ==================== 2. 指令加载器 ====================
class InstructionLoader:
    """负责基于触发词匹配加载系统指令"""

    def __init__(self, metadata: SkillMetadata):
        self.metadata = metadata
        # 编译正则，忽略大小写
        escaped = [re.escape(kw) for kw in metadata.trigger_keywords]
        self._pattern = re.compile("|".join(escaped), re.IGNORECASE)

        # 可配置指令模板
        self.instructions = {
            "default": (
                "你是一个专业的中国金融法规咨询助手。"
                "请使用提供的上下文信息回答问题。"
                "保持回答简洁、专业，不超过三句话。"
            ),
            "safety": (
                "【重要规则】首先检查上下文是否包含与问题相关的信息。"
                "如果包含，直接根据上下文回答，不要添加任何拒绝语句。"
                "仅当上下文中完全没有相关信息时，才回复'该信息未在已知法规中收录'。"
            ),
            "clarification": (
                "如果用户问题模糊或有歧义，请主动请求澄清具体细节。"
            )
        }

    def should_trigger(self, user_input: str) -> bool:
        return bool(self._pattern.search(user_input))

    def get_instruction(self, include_safety: bool = True,
                       include_clarification: bool = True) -> str:
        parts = [self.instructions["default"]]
        if include_safety:
            parts.append(self.instructions["safety"])
        if include_clarification:
            parts.append(self.instructions["clarification"])
        return "\n".join(parts)

    def load_instruction(self, user_input: str,
                        include_safety: bool = True,
                        include_clarification: bool = True) -> str:
        """条件加载：匹配触发词则返回指令，否则空字符串"""
        if not self.should_trigger(user_input):
            return ""
        return self.get_instruction(include_safety, include_clarification)


# ==================== 3. 资源管理器 (延迟加载) ====================
class ResourceManager:
    """管理重资源（向量库、LLM）的延迟加载与缓存"""

    def __init__(self):
        self._vectorstore = None
        self._retriever = None
        self._llm = None
        self._loaded = False
        self._lock = threading.Lock()
        # 默认使用纯向量检索器（安全基线）
        self.retrieval_mode = os.getenv("RETRIEVAL_MODE", "vector").lower()
        # 资源配置
        self.chunk_size = 300
        self.chunk_overlap = 50
        self.search_k = 5
        self.embedding_model_name = MODEL_PATH
        # 读取环境变量决定后端
        self._backend = os.getenv("VECTOR_STORE_BACKEND", "chroma")
        self._hybrid_retriever = None

    def load_resources(self):
        
        """首次调用时加载所有重资源，后续直接返回缓存"""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            print(f"💾 [延迟加载] 初始化向量库（后端: {self._backend}）...")
            # 第二次检查（加锁后），防止多个线程同时通过第一次检查
            # Embedding
            self._embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            # 在原有缓存初始化代码前增加开关
            if os.getenv("DISABLE_CACHE") != "1":
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                cache_mode = os.getenv("CACHE_MODE", "semantic")          # ← 定义变量
                self._cache_mgr = CachingManager(
                                redis_url=redis_url,
                                embedding_model=self._embedding,
                                mode=cache_mode,
                                ttl=3600,
                        )
                self._cache_mgr.enable_llm_cache()
                self._embedding = self._cache_mgr.enable_embedding_cache()
                logger.info("缓存层已启用 (mode=%s, redis=%s)", cache_mode, redis_url)
           

            if self._backend == "chroma":
                
                self._vectorstore = Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=self._embedding,
                    collection_name="finance_qa"
                )

            elif self._backend == "pgvector":
            

                # 使用抽象层工厂创建向量库
                self._vectorstore = create_vector_store(
                    embedding_model=self._embedding,
                    config={
                        "backend": self._backend,
                        "collection_name": "finance_qa",
                        "persist_directory": PERSIST_DIR,
                        # pgvector 相关配置（仅在 backend=pgvector 时生效）
                        "table_name": "finance_knowledge",
                        "vector_size": VECTOR_SIZE,
                    },
                )
            # 2. 生成基础向量检索器
            vector_retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": self.search_k}
            )
            if self.retrieval_mode == "hybrid":
            
                # ── 新增：构建 BM25 检索器 ──────────────────────
                bm25_docs = self._load_bm25_documents()
                bm25_retriever = create_bm25_retriever(bm25_docs, k=2)

                # ── 创建混合检索器 ─────────────────────────────
                self._hybrid_retriever = HybridRetriever(
                    vector_retriever=self._retriever ,
                    bm25_retriever=bm25_retriever,
                    fusion_strategy="rrf",   # 默认 RRF，可配置
                    k=self.search_k,
                    fetch_k=10,              # 扩大候选池
                )

                # 将混合检索器赋值给 self._retriever，供 RAG 链使用
                self._retriever = self._hybrid_retriever
            else:
                # 检索器
                self._retriever = vector_retriever
            # LLM (Qwen-plus, 使用 OpenAI 兼容模式)
            print("🧠 [延迟加载] 正在连接 Qwen-plus 模型...")
            self._llm = ChatOpenAI(
                model="qwen-plus",
                temperature=0,
                openai_api_key=QWEN_API_KEY,
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

             # ── 缓存层初始化（新增） ─────────────────────
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            cache_mode = os.getenv("CACHE_MODE", "semantic")  # 或 "exact"

            self._cache_mgr = CachingManager(
                redis_url=redis_url,
                embedding_model=self._embedding,      # 原始的 HuggingFaceEmbeddings
                mode=cache_mode,
                ttl=3600,                             # 1 小时
            )
            # 启用 LLM 缓存 —— 之后所有 ChatOpenAI 调用都会走缓存
            self._cache_mgr.enable_llm_cache()

            # 将 Embedding 替换为带缓存的版本 —— 之后 embed_query / embed_documents
            # 会先查 Redis，命中则直接返回向量，不再重复计算
            self._embedding = self._cache_mgr.enable_embedding_cache()

            self._loaded = True
            logger.info("✅ 缓存层已启用 (mode=%s, redis=%s)", cache_mode, redis_url)
            print("✅ 所有重资源加载完成,混合检索器已就绪 (向量 + BM25)")

    def _ingest_documents(self):
        """从原始文档构建向量索引"""
        loader = TextLoader(DATA_FILE, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        print(f"📝 正在将 {len(chunks)} 个切片写入 {self._backend}...")
        self._vectorstore.add_documents(chunks)

    def _load_bm25_documents(self) -> List[Document]:
        """从知识库文档中加载用于 BM25 索引的文档"""
        import yaml
        from pathlib import Path
        from langchain_core.documents import Document

        qa_path = Path("data/final_qa_dataset.yaml")
        docs = []
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            qa_list = data if isinstance(data, list) else data.get("final_qa_dataset", [])
            for item in qa_list:
                query = item.get("query") or item.get("question", "")
                answer = item.get("answer", "")
                if len(answer) < 10:   # 过滤过短、可能无意义的答案
                    continue
                if query and answer:
                    # 将问答拼接为文档，让 BM25 能匹配问题和答案中的关键词
                    content = f"问题：{query}\n答案：{answer}"
                    docs.append(Document(page_content=content, metadata={"source": "qa_dataset"}))
        # 如果 QA 数据集不存在，回退到原始文档
        if not docs:
            from langchain_community.document_loaders import TextLoader
            for file in Path("data/source_docs").glob("*.txt"):
                loader = TextLoader(str(file), encoding="utf-8")
                docs.extend(loader.load())
        return docs  
      
    @property
    def retriever(self):
        if not self._loaded:
            raise RuntimeError("资源尚未加载，请先调用 load_resources()")
        return self._retriever

    @property
    def llm(self):
        if not self._loaded:
            raise RuntimeError("资源尚未加载，请先调用 load_resources()")
        return self._llm


# ==================== 4. Skill 主类 ====================
class FinancialRAGSkill:
    """金融法规问答技能 - 三层渐进加载"""

    def __init__(self):
        # 第一层：元数据（常驻内存，极轻量）
        self.metadata = SkillMetadata()

        # 第二层：指令加载器（仅含正则，无重资源）
        self.instruction_loader = InstructionLoader(self.metadata)

        # 第三层：资源管理器（默认不加载向量库和LLM）
        self.resource_mgr = ResourceManager()

        # 一些元问题关键词，不需要检索
        self._meta_questions = ["你能做什么", "帮助", "功能", "你是谁", "你好"]

    def should_trigger(self, user_input: str) -> bool:
        """基于触发词判断是否激活本技能"""
        return self.instruction_loader.should_trigger(user_input)

    def _needs_retrieval(self, user_input: str) -> bool:
        """简单判断是否需要检索（事实查询需要，元问题不需要）"""
        lower = user_input.lower()
        if any(m in lower for m in self._meta_questions):
            return False
        return True

    def run(self, user_input: str) -> str:
        """技能执行入口"""
        # 第一层：未触发则返回通用提示
        if not self.should_trigger(user_input):
            return "我是金融法规助手，请输入您想咨询的金融法规问题（如资本充足率、LPR、购汇额度等）。"

        # 第二层：加载指令
        instruction = self.instruction_loader.load_instruction(user_input)

        # 判断是否需要检索
        if self._needs_retrieval(user_input):
            # 第三层：延迟加载完整资源（仅在需要检索时）
            self.resource_mgr.load_resources()

            # 构建 RAG 链
            prompt = ChatPromptTemplate.from_messages([
                ("system", instruction),
                ("system", "检索到的上下文如下（仅基于此回答）：\n{context}"),
                ("human", "{input}"),
            ])
            combine_chain = create_stuff_documents_chain(
                self.resource_mgr.llm, prompt
            )
            rag_chain = create_retrieval_chain(
                self.resource_mgr.retriever, combine_chain
            )
            result = rag_chain.invoke({"input": user_input})
            return result["answer"]
        else:
            # 简单元问题，直接调用 LLM（此时也需加载 LLM）
            self.resource_mgr.load_resources()  # 仍需要 LLM
            messages = [
                SystemMessage(content=instruction),
                HumanMessage(content=user_input)
            ]
            response = self.resource_mgr.llm.invoke(messages)
            return response.content

    def run_with_context(self, user_input: str) -> dict:
        """扩展方法：执行 RAG 并同时返回检索上下文，供评测使用"""

        # print(f"DEBUG: needs_retrieval = {self._needs_retrieval(user_input)}")
        if not self.should_trigger(user_input):
            return {"input": user_input, "answer": "我是金融法规助手，请输入金融问题。", "context": []}

        instruction = self.instruction_loader.load_instruction(user_input)

        if self._needs_retrieval(user_input):
            self.resource_mgr.load_resources()
            retriever = self.resource_mgr.retriever

            # 1. 独立检索
            docs = retriever.invoke(user_input)
            context_texts = [doc.page_content for doc in docs]  # 关键：提取文本

            # print(f"DEBUG context: {len(context_texts)}")  # 临时调试
            prompt = ChatPromptTemplate.from_messages([
                ("system", instruction + "\n\n上下文:\n{context}"),
                ("human", "{input}"),
            ])
            combine_chain = create_stuff_documents_chain(self.resource_mgr.llm, prompt)
            rag_chain = create_retrieval_chain(self.resource_mgr.retriever, combine_chain)
            result = rag_chain.invoke({"input": user_input})

            # 兼容多种可能的键名
            # raw_context = (
            #     result.get("context") or
            #     result.get("retrieved_documents") or
            #     []
            # )
            # # 统一转为字符串列表
            # context_texts = []
            # for doc in raw_context:
            #     if hasattr(doc, 'page_content'):
            #         context_texts.append(doc.page_content)
            #     elif isinstance(doc, str):
            #         context_texts.append(doc)
            #     else:
            #         context_texts.append(str(doc))

            return {
                "input": user_input,
                "answer": result["answer"],
                "context": context_texts
            }
        else:
            self.resource_mgr.load_resources()
            messages = [
                SystemMessage(content=instruction),
                HumanMessage(content=user_input)
            ]
            response = self.resource_mgr.llm.invoke(messages)
            return {"input": user_input, "answer": response.content, "context": []}


if __name__ == "__main__":
    skill = FinancialRAGSkill()
    query = "金融机构开展客户尽职调查的法定情形"
    skill.resource_mgr.load_resources()           # 强制加载
    retriever = skill.resource_mgr.retriever

    # 1. 测试 retriever 能否直接返回文档
    raw_docs = retriever.invoke(query)
    print(f"检索到的原始文档数: {len(raw_docs)}")
    for i, doc in enumerate(raw_docs):
        print(f"[{i}] {doc.page_content[:100]}...")

    # 2. 测试完整的 RAG 链
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", skill.instruction_loader.load_instruction(query) + "\n\n上下文:\n{context}"),
        ("human", "{input}"),
    ])
    combine_chain = create_stuff_documents_chain(skill.resource_mgr.llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)
    result = rag_chain.invoke({"input": query})
    print("result keys:", result.keys())
    print("context:", result.get("context"))