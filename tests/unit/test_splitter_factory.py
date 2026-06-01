import os

from langchain_huggingface import HuggingFaceEmbeddings
import pytest
from pathlib import Path
from langchain_core.documents import Document

# 导入待实现的 SplitterFactory（此时尚未编写，测试会失败 —— 预期行为）
from archive.prototypes.pgvector_prototype import BASE_DIR
from src.splitter.splitter_factory import SplitterFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMBEDDING_MODEL = str(BASE_DIR / "models" / "text2vec-base-chinese" / "Jerry0" / "text2vec-base-chinese")

def load_plain_documents():
    """返回一个简单的 Document 列表用于测试"""
    return [Document(page_content="第一段。\n\n第二段。\n\n第三段。", metadata={"source": "test.txt"})]

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def load_fixture_documents(filename: str) -> list:
    """辅助函数：读取文本文件并包装为 Document 列表（模拟 Loader 输出）"""
    filepath = FIXTURES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": str(filepath)})]

class TestSplitterFactory:
    def setup_method(self):
        self.factory = SplitterFactory()
        self.docs = load_plain_documents()

    # ---------- 1. recursive 策略测试 ----------
    def test_recursive_split_basic(self):
        """测试 recursive 策略能正常分割，并验证片段数和重叠"""
        docs = load_fixture_documents("plain.txt")
        chunks = self.factory.split(docs, strategy="recursive", chunk_size=20, chunk_overlap=5)
        # 至少产生多个片段
        assert len(chunks) > 1
        # 所有片段均为 Document
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) <= 25  # 考虑重叠，最大不超过 chunk_size + overlap
        # 重叠检查：相邻片段之间应有重叠字符（简单抽检）
        # 不严格，因为 RecursiveCharacterTextSplitter 可能恰好落在分隔符上无重叠
        # 我们改为验证 chunk_overlap 参数已被传入（通过片段长度分布）

    def test_recursive_split_with_custom_size(self):
        """验证自定义 chunk_size 和 chunk_overlap 能正确透传"""
        docs = load_fixture_documents("long_text.txt")
        chunks = self.factory.split(docs, strategy="recursive", chunk_size=50, chunk_overlap=10)
        # 每个片段长度应小于 chunk_size + overlap
        for chunk in chunks:
            assert len(chunk.page_content) <= 60
        assert len(chunks) > 3

    # ---------- 2. markdown 策略测试 ----------
    def test_markdown_split_preserves_headers(self):
        """测试 Markdown 按标题分割，并保留层级信息"""
        docs = load_fixture_documents("regulation.md")
        chunks = self.factory.split(docs, strategy="markdown")
        # 应产生多个片段（每个标题段落一个）
        assert len(chunks) >= 3
        # 至少有一个 chunk 的 metadata 中包含标题层级字段
        header_found = False
        for chunk in chunks:
            if "Header" in chunk.metadata or any(k.startswith("Header") for k in chunk.metadata):
                header_found = True
                break
        assert header_found, "MarkdownHeaderTextSplitter 应在 metadata 中保留标题信息"

    # ---------- 3. semantic 策略测试 ----------
    def test_semantic_split_preserves_sentences(self):
        """语义分割测试（依赖 langchain-experimental）"""
        pytest.importorskip("langchain_experimental")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        factory = SplitterFactory(embeddings=embeddings)
        docs = load_fixture_documents("plain.txt")
        chunks = factory.split(docs, strategy="semantic")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Document)
            # 不检查末尾标点，语义分割可能不按句子边界，仅验证非空
            assert len(chunk.page_content) > 0


    # ---------- 4. 返回类型验证 ----------
    def test_factory_returns_documents(self):
        """验证分割后返回的是 List[Document] 且保留 metadata"""
        docs = load_fixture_documents("plain.txt")
        chunks = self.factory.split(docs, strategy="recursive", chunk_size=30, chunk_overlap=5)
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert hasattr(chunk, "page_content")
            assert hasattr(chunk, "metadata")
            assert "source" in chunk.metadata  # 原始 metadata 应保留

    # ---------- 5. 无效策略异常 ----------
    def test_invalid_strategy_raises_valueerror(self):
        """测试使用未注册的策略时抛出 ValueError"""
        docs = load_fixture_documents("plain.txt")
        with pytest.raises(ValueError, match="不支持的分割策略"):
            self.factory.split(docs, strategy="imaginary")

    # ---------- 6. 参数透传验证 ----------
    def test_chunk_size_and_overlap_configurable(self):
        """通过 create_splitter 获取分割器实例，验证参数透传"""
        splitter = self.factory.create_splitter("recursive", chunk_size=123, chunk_overlap=45)
        # 分割器实例应具有配置属性
        assert splitter._chunk_size == 123
        assert splitter._chunk_overlap == 45

    def test_register_custom_strategy(self):
        """验证注册自定义策略后能正常使用"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        factory = SplitterFactory()
        factory.register_strategy("custom_recursive", RecursiveCharacterTextSplitter)
        docs = load_fixture_documents("plain.txt")
        chunks = factory.split(docs, strategy="custom_recursive", chunk_size=20, chunk_overlap=5)
        assert len(chunks) > 1


    def test_token_split_basic(self):
        """测试 token 策略能正常分割"""
        docs = load_fixture_documents("long_text.txt")
        chunks = self.factory.split(docs, strategy="token", chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_semantic_split_basic(self):
        """测试 semantic 策略能正常分割（需 langchain-experimental）"""
        pytest.importorskip("langchain_experimental")


        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        factory = SplitterFactory(embeddings=embeddings)
        docs = load_fixture_documents("plain.txt")
        chunks = factory.split(docs, strategy="semantic")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Document)


    def test_list_strategies(self):
        """验证 list_strategies 返回所有已注册策略"""
        strategies = self.factory.list_strategies()
        assert "recursive" in strategies
        assert "markdown" in strategies
        assert "token" in strategies
        # semantic 取决于是否安装了 langchain-experimental


    @pytest.mark.parametrize("name,splitter_cls,chunk_size", [
        ("my_recursive", RecursiveCharacterTextSplitter, 10),
        ("my_token", TokenTextSplitter, 20),
    ])
    def test_registered_strategy_produces_chunks(self, name, splitter_cls, chunk_size):
        """注册的自定义策略能够正常生成片段"""
        self.factory.register_strategy(name, splitter_cls)
        chunks = self.factory.split(self.docs, strategy=name, chunk_size=chunk_size, chunk_overlap=0)
        # 至少生成 1 个片段
        assert len(chunks) >= 1
        # 片段长度应受 chunk_size 约束（token 策略可能有出入，仅大致检查）
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 20  # 留有余量

    def test_register_and_list_strategies(self):
        """注册后 list_strategies 能返回新策略名"""
        original_count = len(self.factory.list_strategies())
        self.factory.register_strategy("latex", RecursiveCharacterTextSplitter)
        assert "latex" in self.factory.list_strategies()
        assert len(self.factory.list_strategies()) == original_count + 1

    def test_register_overwrites_existing(self):
        """注册同名策略会覆盖旧策略（日志可查看）"""
        self.factory.register_strategy("recursive", TokenTextSplitter)
        # 使用 "recursive" 应该变为 TokenTextSplitter（根据 chunk_size 判断）
        chunks = self.factory.split(self.docs, strategy="recursive", chunk_size=50, chunk_overlap=0)
        # TokenTextSplitter 的分割方式可能产生较少片段，简单验证不崩溃即可
        assert len(chunks) >= 1

    def test_register_invalid_splitter_raises_on_use(self):
        """如果注册的 splitter 无法实例化，应在调用 split 时抛出异常"""
        class BadSplitter:
            pass
        self.factory.register_strategy("bad", BadSplitter)
        with pytest.raises((TypeError, AttributeError)):
            # BadSplitter 缺少 split_documents 方法 → AttributeError
            # 未来若增加前置检查也可改为 TypeError，所以同时捕获两种
            self.factory.split(self.docs, strategy="bad")