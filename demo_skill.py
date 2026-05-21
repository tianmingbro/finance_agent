"""
金融 RAG Skill 渐进式加载演示
支持三层加载：元数据常驻，指令按需注入，资源懒加载
"""

import os
import time
from pathlib import Path
import yaml

# ---------- 模拟 LangChain 相关导入（实际环境请确保已安装） ----------
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    print(f"请安装所需依赖: {e}")
    exit(1)


class FinancialRAGSkill:
    """金融法规问答 RAG Skill，支持渐进式三层加载"""

    def __init__(self, config_path="skill_finance_qa.yaml"):
        # 第一层：元数据（始终加载）
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["skill"]
        self.metadata = self.config["metadata"]
        self.instruction_cfg = self.config["instruction"]
        self.resource_cfg = self.config["resources"]

        # 延迟加载的资源初始状态为 None
        self._vectorstore = None
        self._llm = None
        self._rag_chain = None
        print(f"✅ Skill '{self.metadata['name']}' 元数据已加载（触发词: {', '.join(self.metadata['trigger_keywords'])}）")

    def should_trigger(self, user_input: str) -> bool:
        """检查用户输入是否匹配任一触发词（简单包含匹配）"""
        user_lower = user_input.lower()
        return any(kw in user_lower for kw in self.metadata["trigger_keywords"])

    def load_instruction(self) -> str:
        """第二层：加载系统指令（无资源消耗）"""
        return self.instruction_cfg["system_prompt"]

    def _load_resources(self):
        """第三层：按需加载完整 RAG 资源（向量库、LLM等）"""
        if self._rag_chain is not None:
            print("  ℹ️ 资源已加载，直接使用...")
            return

        print("  🔄 开始加载完整资源...")
        start = time.time()

        # 1. 加载文档
        loader = TextLoader(self.resource_cfg["data_source"]["path"], encoding="utf-8")
        docs = loader.load()

        # 2. 切片
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.resource_cfg["chunking"]["chunk_size"],
            chunk_overlap=self.resource_cfg["chunking"]["chunk_overlap"]
        )
        chunks = splitter.split_documents(docs)

        # 3. Embedding + 向量库
        embeddings = HuggingFaceEmbeddings(
            model_name=self.resource_cfg["embedding"]["model_name"],
            model_kwargs={"device": self.resource_cfg["embedding"]["device"]},
            encode_kwargs={"normalize_embeddings": self.resource_cfg["embedding"]["normalize"]}
        )
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.resource_cfg["vectorstore"]["persist_directory"],
            collection_name=self.resource_cfg["vectorstore"]["collection_name"]
        )

        # 4. LLM + 链
        self._llm = ChatOpenAI(
            model=self.resource_cfg["llm"]["model"],
            temperature=self.resource_cfg["llm"]["temperature"]
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instruction_cfg["system_prompt"]),
            ("human", "{input}")
        ])
        combine_chain = create_stuff_documents_chain(self._llm, prompt)
        retriever = self._vectorstore.as_retriever(
            search_kwargs=self.resource_cfg["vectorstore"]["search_kwargs"]
        )
        self._rag_chain = create_retrieval_chain(retriever, combine_chain)

        elapsed = time.time() - start
        print(f"  ✅ 完整资源加载完成，耗时 {elapsed:.2f} 秒")

    def run(self, user_input: str, verbose=True) -> str | None:
        """执行 Skill 主流程：触发->指令->按需资源加载"""
        if not self.should_trigger(user_input):
            if verbose:
                print(f"⏭️  输入 '{user_input}' 未匹配触发词，跳过该 Skill。")
            return None

        if verbose:
            print(f"🎯 触发词匹配，加载指令层...")
        # 指令可在此处注入到对话上下文（本演示中指令在资源层 prompt 中使用）

        # 按需加载完整资源并运行
        self._load_resources()

        if verbose:
            print(f"📝 正在执行 RAG 查询...")
        result = self._rag_chain.invoke({"input": user_input})

        if verbose:
            print(f"🤖 回答: {result['answer']}")
        return result["answer"]


# -------------------- 演示入口 --------------------
if __name__ == "__main__":
    # 设置你的 API Key（可通过环境变量或直接修改）
    os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

    # 初始化 Skill（仅加载元数据）
    skill = FinancialRAGSkill("skill_finance_qa.yaml")

    print("\n" + "="*60)
    print("场景1: 用户提问 '资本充足率是多少？'")
    print("="*60)
    skill.run("资本充足率是多少？")

    print("\n" + "="*60)
    print("场景2: 用户提问 '今天天气如何？'（不触发）")
    print("="*60)
    skill.run("今天天气如何？")

    print("\n" + "="*60)
    print("场景3: 再次提问 'LPR是什么？'（验证资源复用）")
    print("="*60)
    skill.run("LPR是什么？")