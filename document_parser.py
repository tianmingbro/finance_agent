"""
document_parser.py
Day36 核心模块：统一文档解析器，支持 PDF、DOCX、TXT、Markdown、HTML。
使用示例:
    >>> from document_parser import load_document
    >>> docs = load_document("data/test_docs/deposit_insurance_regulation.pdf")
    >>> print(docs[0].page_content[:50])
    存款保险条例
    >>> len(docs)  # PDF 按页加载，一页一个 Document
    23
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
)
from langchain_core.documents import Document

class DocumentLoadError(Exception):
    """文档加载过程中出现的错误（损坏文件、加密文件等）"""
    pass

# 文件后缀 → 加载器映射
LOADER_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": BSHTMLLoader,  # HTML 文件也用 HTML 加载器处理
    ".htm": BSHTMLLoader,   # 兼容短后缀
}

def load_document(file_path: str) -> List[Document]:
    """
    根据文件后缀自动选择加载器，返回 Document 列表。

    Args:
        file_path: 文档路径，支持 .txt / .md / .pdf / .docx / .html

    Returns:
        List[Document]: 每个 Document 包含 page_content 和统一的 metadata。
            metadata 保证包含：
                - "source" (str)：原始文件路径
                - "format" (str)：文件格式后缀（不含点）
            对于 PDF，额外包含 "page" (int) 和 "total_pages" (int)。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
        DocumentLoadError: 文件加载或解析失败（如损坏的 PDF、加密文档等）

    示例:
        >>> docs = load_document("report.pdf")
        >>> print(docs[0].metadata["source"])
        report.pdf
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in LOADER_MAP:
        raise ValueError(f"不支持的文档格式: {suffix}，目前支持: {list(LOADER_MAP.keys())}")

    loader_cls = LOADER_MAP[suffix]
    # PyPDFLoader 和 TextLoader 的初始化略有不同：TextLoader 需要指定 encoding
    try:
        if suffix == ".txt":
            loader = loader_cls(str(path), encoding="utf-8")
        elif suffix == ".html" or suffix == ".htm":
            loader = loader_cls(str(path), open_encoding='utf-8')
        else:
            loader = loader_cls(str(path))
    
        documents = loader.load()
    except Exception as e:
        # 将底层加载器的异常统一包装
        raise DocumentLoadError(
            f"加载文档失败 ({path.name}): {e}"
        ) from e

    # 统一 metadata 格式：确保所有 Document 都至少有 "source"
    for doc in documents:
        doc.metadata["source"] = str(path)
        # 可选：统一添加文件格式信息
        doc.metadata["format"] = suffix.lstrip(".")

    return documents