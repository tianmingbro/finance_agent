import pytest
from pathlib import Path
from langchain_core.documents import Document

from loader_facade import LoaderFacade
from langchain_community.document_loaders import TextLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoaderFacade:
    def setup_method(self):
        self.facade = LoaderFacade()

    def test_load_txt_returns_nonempty_documents(self):
        """验证 TXT 加载内容非空"""
        file_path = FIXTURES_DIR / "sample.txt"
        docs = self.facade.load(file_path)
        assert len(docs) > 0
        assert len(docs[0].page_content) > 0

    def test_load_pdf_returns_pages(self):
        """验证 PDF 加载页数 ≥ 1"""
        file_path = FIXTURES_DIR / "sample.pdf"
        docs = self.facade.load(file_path)
        # PyPDFLoader 默认每页生成一个 Document
        assert len(docs) >= 1
        # 至少有一页包含预期关键词
        assert any("存款保险" in doc.page_content for doc in docs)

    def test_load_docx_returns_paragraphs(self):
        """验证 DOCX 加载段落数 ≥ 1"""
        file_path = FIXTURES_DIR / "sample.docx"
        docs = self.facade.load(file_path)
        # Docx2txtLoader 返回整个文档内容（通常 1 个 Document）
        assert len(docs) >= 1
        full_text = " ".join(doc.page_content for doc in docs)
        assert "外汇" in full_text

    def test_unsupported_format_raises_valueerror(self):
        """验证不支持格式抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的文件格式"):
            self.facade.load("test.xyz")

    def test_facade_returns_langchain_documents(self):
        """验证返回对象类型为 LangChain Document 列表"""
        file_path = FIXTURES_DIR / "sample.txt"
        docs = self.facade.load(file_path)
        assert isinstance(docs, list)
        for doc in docs:
            assert isinstance(doc, Document)
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert "source" in doc.metadata

    def test_register_custom_loader_and_use(self):
        """验证自定义加载器注册后能正常调用"""
        facade = LoaderFacade()
        facade.register_loader("custom", TextLoader)  # 假设实现支持
        # 将 sample.txt 重命名为 sample.custom 测试
        custom_path = FIXTURES_DIR / "sample.custom"
        custom_path.write_text("自定义格式测试", encoding="utf-8")
        docs = facade.load(custom_path)
        assert len(docs) == 1
        assert "自定义格式测试" in docs[0].page_content

    # 新增：参数化测试自定义加载器注册
    @pytest.mark.parametrize("ext,loader_cls,content", [
        (".md", TextLoader, "# 存款保险条例\n最高偿付限额50万元"),
        (".rst", TextLoader, "个人外汇管理办法\n===============\n每人每年等值5万美元"),
        (".log", TextLoader, "[INFO] 反洗钱新规已生效"),
    ])
    def test_register_custom_loader_parametrized(self, ext, loader_cls, content):
        """验证不同扩展名注册后能正确加载内容"""
        facade = LoaderFacade()
        facade.register_loader(ext, loader_cls)

        # 创建临时文件
        temp_file = FIXTURES_DIR / f"sample{ext}"
        temp_file.write_text(content, encoding="utf-8")

        try:
            docs = facade.load(temp_file)
            assert len(docs) == 1
            assert content[:20] in docs[0].page_content
        finally:
            temp_file.unlink(missing_ok=True)  # 清理临时文件