"""
test_document_parser.py
Day36 测试驱动开发：文档解析器单元测试
"""
import os
import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from document_parser import load_document
from document_parser import DocumentLoadError

# -------------------- 测试辅助：创建临时文件 --------------------

@pytest.fixture
def sample_txt():
    """创建临时 TXT 文件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("存款保险最高偿付限额为50万元人民币。\n")
        f.write("商业银行资本充足率不得低于8%。")
        path = f.name
    yield path
    os.unlink(path)

@pytest.fixture
def sample_md():
    """创建临时 Markdown 文件"""
    content = "# 存款保险条例\n\n## 最高偿付限额\n\n存款保险实行限额偿付，最高偿付限额为人民币50万元。\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)

# --- 测试辅助：创建临时 HTML 文件 ---
@pytest.fixture
def sample_html():
    """创建包含金融法规内容的临时 HTML 文件"""
    content = """<!DOCTYPE html>
    <html>
    <head>
        <title>存款保险条例 - 官方解读</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>存款保险条例</h1>
        <p>存款保险实行限额偿付，最高偿付限额为人民币50万元。</p>
        <p>同一存款人在同一家投保机构所有被保险存款账户的存款本金和利息合并计算。</p>
    </body>
    </html>"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)

# -------------------- 测试用例 --------------------

class TestTextLoader:
    """TextLoader 测试：TXT 格式加载"""

    def test_load_txt_returns_documents(self, sample_txt):
        """正常 TXT 文件应返回至少一个 Document"""
        docs = load_document(sample_txt)
        assert isinstance(docs, list), "返回值应为 list"
        assert len(docs) > 0, "应返回至少一个 Document"
        assert all(isinstance(d, Document) for d in docs), "每个元素都应为 Document"

    def test_txt_page_content_not_empty(self, sample_txt):
        """Document 的 page_content 不应为空"""
        docs = load_document(sample_txt)
        assert len(docs[0].page_content) > 0, "page_content 不应为空"

    def test_txt_metadata_contains_source(self, sample_txt):
        """metadata 应包含 source 字段，且值为文件路径"""
        docs = load_document(sample_txt)
        assert "source" in docs[0].metadata, "metadata 应包含 source"
        assert docs[0].metadata["source"] == sample_txt, (
            f"source 应为原始文件路径，实际为 {docs[0].metadata['source']}"
        )


class TestMarkdownLoader:
    """UnstructuredMarkdownLoader 测试：Markdown 格式加载"""

    def test_load_md_returns_documents(self, sample_md):
        """正常 MD 文件应返回至少一个 Document"""
        docs = load_document(sample_md)
        assert len(docs) > 0

    def test_md_page_content_not_empty(self, sample_md):
        """Document 的 page_content 不应为空"""
        docs = load_document(sample_md)
        assert len(docs[0].page_content) > 0

    def test_md_metadata_contains_source(self, sample_md):
        """metadata 应包含 source 字段"""
        docs = load_document(sample_md)
        assert "source" in docs[0].metadata


class TestPDFLoader:
    """PyPDFLoader 测试：PDF 格式加载"""

    def test_load_pdf_returns_documents(self):
        """正常 PDF 文件应返回至少一个 Document"""
        # 依赖项目中已有的 PDF 测试文件
        pdf_path = "data/source_docs/deposit_insurance_regulation.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"测试 PDF 文件不存在: {pdf_path}")
        docs = load_document(pdf_path)
        assert len(docs) > 0, f"应加载至少一页，实际 {len(docs)} 页"

    def test_pdf_page_content_not_empty(self):
        """PDF 每页的 page_content 不应为空"""
        pdf_path = "data/source_docs/deposit_insurance_regulation.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"测试 PDF 文件不存在: {pdf_path}")
        docs = load_document(pdf_path)
        for i, doc in enumerate(docs):
            assert len(doc.page_content) > 0, f"第 {i+1} 页内容为空"

    def test_pdf_metadata_contains_source_and_page(self):
        """PDF 的 metadata 应包含 source 和 page 字段"""
        pdf_path = "data/source_docs/deposit_insurance_regulation.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"测试 PDF 文件不存在: {pdf_path}")
        docs = load_document(pdf_path)
        assert "source" in docs[0].metadata
        assert "page" in docs[0].metadata, "PDF 加载器应提供页码信息"


class TestDOCXLoader:
    """Docx2txtLoader 测试：DOCX 格式加载"""

    def test_load_docx_returns_documents(self):
        """正常 DOCX 文件应返回至少一个 Document"""
        docx_path = "data/source_docs/deposit_insurance_regulation.docx"
        if not Path(docx_path).exists():
            pytest.skip(f"测试 DOCX 文件不存在: {docx_path}")
        docs = load_document(docx_path)
        assert len(docs) > 0

    def test_docx_page_content_not_empty(self):
        """Document 的 page_content 不应为空"""
        docx_path = "data/source_docs/deposit_insurance_regulation.docx"
        if not Path(docx_path).exists():
            pytest.skip(f"测试 DOCX 文件不存在: {docx_path}")
        docs = load_document(docx_path)
        assert len(docs[0].page_content) > 0

    def test_docx_metadata_contains_source(self):
        """metadata 应包含 source 字段"""
        docx_path = "data/source_docs/deposit_insurance_regulation.docx"
        if not Path(docx_path).exists():
            pytest.skip(f"测试 DOCX 文件不存在: {docx_path}")
        docs = load_document(docx_path)
        assert "source" in docs[0].metadata

# --- 新增 HTML 测试类 ---
class TestHTMLLoader:
    """BSHTMLLoader 测试：HTML 格式加载"""

    def test_load_html_returns_documents(self, sample_html):
        """正常 HTML 文件应返回至少一个 Document"""
        docs = load_document(sample_html)
        assert isinstance(docs, list), "返回值应为 list"
        assert len(docs) > 0, f"应返回至少一个 Document，实际 {len(docs)}"
        assert all(isinstance(d, Document) for d in docs), "每个元素都应为 Document"

    def test_html_page_content_not_empty(self, sample_html):
        """Document 的 page_content 不应为空，且包含 HTML 文本"""
        docs = load_document(sample_html)
        content = docs[0].page_content
        assert len(content) > 0, "page_content 不应为空"
        # 应解析出页面内的中文内容
        assert "存款保险" in content, f"HTML 文本解析异常: {content[:80]}"

    def test_html_metadata_contains_source(self, sample_html):
        """metadata 应包含 source 字段，且值为文件路径"""
        docs = load_document(sample_html)
        assert "source" in docs[0].metadata, "metadata 应包含 source"
        assert docs[0].metadata["source"] == sample_html, (
            f"source 应为原始文件路径，实际为 {docs[0].metadata['source']}"
        )

    def test_html_metadata_contains_title(self, sample_html):
        """BSHTMLLoader 应从 <title> 标签提取标题到 metadata"""
        docs = load_document(sample_html)
        assert "title" in docs[0].metadata, (
            "BSHTMLLoader 应提取页面标题到 metadata['title']"
        )
        assert "存款保险条例" in docs[0].metadata["title"], (
            f"标题内容异常: {docs[0].metadata.get('title')}"
        )

class TestErrorHandling:
    """错误处理与边界测试"""

    def test_file_not_found_raises(self):
        """不存在的文件应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            load_document("/nonexistent/path/file.txt")

    def test_unsupported_format_raises(self):
        """不支持的后缀应抛出 ValueError"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False, encoding="utf-8") as f:
            f.write("some content")
            path = f.name
        try:
            with pytest.raises(ValueError, match="不支持的文档格式"):
                load_document(path)
        finally:
            os.unlink(path)

    def test_empty_txt_file(self):
        """空 TXT 文件应能正常加载，但 page_content 可为空"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            path = f.name  # 空文件
        try:
            docs = load_document(path)
            # 空文件加载不应崩溃，返回 Document 即可
            assert isinstance(docs, list)
            assert len(docs) > 0
        finally:
            os.unlink(path)

    def test_load_txt_with_chinese_encoding(self, sample_txt):
        """中文 TXT 文件应正确加载，不乱码"""
        docs = load_document(sample_txt)
        assert "存款保险" in docs[0].page_content, (
            f"中文内容加载异常: {docs[0].page_content[:50]}"
        )
        
    """扩展边界测试：损坏文件、加密文件等"""

    def test_load_corrupted_pdf_raises(self):
        """损坏的 PDF 文件应抛出 DocumentLoadError"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as f:
            f.write(b"this is not a valid pdf")  # 无效内容
            path = f.name
        try:
            with pytest.raises(DocumentLoadError):
                load_document(path)
        finally:
            os.unlink(path)

    def test_load_encrypted_pdf_raises(self):
        """加密 PDF（需要密码）应抛出 DocumentLoadError"""
        # 使用 pypdf 创建一个带密码的 PDF，或使用预存的测试文件
        encrypted_pdf = "data/test_docs/encrypted_sample.pdf"
        if not Path(encrypted_pdf).exists():
            pytest.skip(f"加密 PDF 测试文件不存在: {encrypted_pdf}")
        with pytest.raises(DocumentLoadError):
            load_document(encrypted_pdf)

    def test_load_empty_markdown_does_not_crash(self):
        """空 Markdown 文件应能加载（可能返回空内容文档）"""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8") as f:
            path = f.name
        try:
            docs = load_document(path)
            assert isinstance(docs, list)
            # 允许返回一个 Document，但内容可能为空
            assert len(docs) > 0
        finally:
            os.unlink(path)

    def test_load_large_txt_file(self):
        """大型 TXT 文件应能正常加载，不超时"""
        # 创建一个约 1MB 的文本文件
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("金融法规测试。\n" * 100000)  # 约 1.2 MB
            path = f.name
        try:
            docs = load_document(path)
            assert len(docs) == 1
            assert len(docs[0].page_content) > 100000
        finally:
            os.unlink(path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])