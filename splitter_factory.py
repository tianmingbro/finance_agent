"""
splitter_factory.py
Day37 核心交付物：文本分割工厂，支持 RecursiveCharacterTextSplitter、
MarkdownHeaderTextSplitter、TokenTextSplitter、SemanticChunker 四种策略，
并预留 HTML/Code 策略接口。增加二次安全切割，防止超长片段导致评测异常。
兼容：langchain v1.2 + langchain-text-splitters + langchain-experimental
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)

logger = logging.getLogger(__name__)

# SemanticChunker 来自 experimental，导入失败时优雅降级
try:
    from langchain_experimental.text_splitter import SemanticChunker
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False
    logger.warning(
        "langchain-experimental 未安装，semantic 策略不可用。"
        "安装: pip install langchain-experimental"
    )

# HTMLHeaderTextSplitter 来自 langchain_text_splitters，可能不存在
try:
    from langchain_text_splitters import HTMLHeaderTextSplitter
    _HAS_HTML = True
except ImportError:
    _HAS_HTML = False


class SplitterFactory:
    """
    文本分割工厂：根据策略名创建分割器，或直接对文档列表执行分割。

    内置策略：
      - "recursive": RecursiveCharacterTextSplitter（通用递归分割）
      - "markdown":   MarkdownHeaderTextSplitter（按 Markdown 标题层级分割）
      - "token":      TokenTextSplitter（按 token 数量分割，基于 tiktoken）
      - "semantic":   SemanticChunker（按语义相似度分割，实验性）
      - "html":       HTMLHeaderTextSplitter（按 HTML 标题分割，需 langchain-text-splitters）
      - "code":       预留策略，需用户注册自定义分割器
    """

    def __init__(self, embeddings: Optional[Any] = None):
        """
        Args:
            embeddings: 语义分割器所需的 Embedding 模型（仅在 strategy="semantic" 时使用）。
        """
        self._embeddings = embeddings
        self._strategies: Dict[str, Optional[type]] = {}
        self._register_builtin_strategies()

    def _register_builtin_strategies(self) -> None:
        """注册内置策略，不可用的策略设为 None"""
        self._strategies["recursive"] = RecursiveCharacterTextSplitter
        self._strategies["markdown"] = MarkdownHeaderTextSplitter
        self._strategies["token"] = TokenTextSplitter
        self._strategies["semantic"] = SemanticChunker if _HAS_SEMANTIC else None
        self._strategies["html"] = HTMLHeaderTextSplitter if _HAS_HTML else None
        self._strategies["code"] = None  # 预留，需用户注册

    # ── 公开 API ───────────────────────────────────────
    def register_strategy(self, name: str, splitter_cls: type) -> None:
        """注册自定义分割策略（开闭原则）"""
        self._strategies[name] = splitter_cls
        logger.info("注册自定义分割策略: %s -> %s", name, splitter_cls.__name__)

    def list_strategies(self) -> List[str]:
        """列出所有可用策略名称（仅返回非 None 的策略）"""
        return [name for name, cls in self._strategies.items() if cls is not None]

    # ── 核心方法 ───────────────────────────────────────
    def create_splitter(self, strategy: str, **kwargs) -> Any:
        """
        根据策略名创建并返回对应的分割器实例。

        Args:
            strategy: 策略名称
            **kwargs: 透传给分割器构造函数的参数

        Returns:
            分割器实例

        Raises:
            ValueError: 策略不存在或未实现
        """
        if strategy not in self._strategies:
            raise ValueError(
                f"不支持的分割策略 '{strategy}'。可用: {self.list_strategies()}"
            )

        splitter_cls = self._strategies[strategy]
        if splitter_cls is None:
            raise NotImplementedError(
                f"分割策略 '{strategy}' 尚未实现或依赖未安装。"
                "请使用 register_strategy 注册自定义分割器，或安装必要的依赖。"
            )

        # ── 策略 1: recursive ──
        if strategy == "recursive":
            chunk_size = kwargs.pop("chunk_size", 400)
            chunk_overlap = kwargs.pop("chunk_overlap", 100)
            separators = kwargs.pop(
                "separators",
                ["\n\n", "\n", "。", "！", "？", "；", " ", ""]
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                **kwargs,
            )

        # ── 策略 2: markdown ──
        if strategy == "markdown":
            headers_to_split_on = kwargs.pop(
                "headers_to_split_on",
                [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
            )
            strip_headers = kwargs.pop("strip_headers", False)
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=strip_headers,
                **kwargs,
            )

        # ── 策略 3: token ──
        if strategy == "token":
            chunk_size = kwargs.pop("chunk_size", 512)       # token 数
            chunk_overlap = kwargs.pop("chunk_overlap", 50)
            encoding_name = kwargs.pop("encoding_name", "cl100k_base")
            return TokenTextSplitter(
                encoding_name=encoding_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )

        # ── 策略 4: semantic ──
        if strategy == "semantic":
            embeddings = kwargs.pop("embeddings", self._embeddings)
            if embeddings is None:
                raise ValueError(
                    "semantic 策略需要提供 embeddings 参数，"
                    "或在构造 SplitterFactory 时传入。"
                )
            breakpoint_threshold_type = kwargs.pop(
                "breakpoint_threshold_type", "percentile"
            )
            breakpoint_threshold_amount = kwargs.pop(
                "breakpoint_threshold_amount", None
            )
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
                **kwargs,
            )

        # ── 策略 5: html ──
        if strategy == "html":
            headers_to_split_on = kwargs.pop(
                "headers_to_split_on",
                [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            )
            return HTMLHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                **kwargs,
            )

        # ── 策略 6: code ──
        if strategy == "code":
            # 目前未实现，代码分割需自定义注册
            raise NotImplementedError(
                "code 分割策略尚未内置实现。请使用 register_strategy 注册自定义分割器，"
                "例如 langchain_text_splitters 的 LatexTextSplitter 或 CodeSplitter。"
            )

        # ✨ 通用分支：处理用户注册的自定义策略（不含特殊默认参数）
        return splitter_cls(**kwargs)

    def split(
        self,
        documents: List[Document],
        strategy: str = "recursive",
        max_chunk_size: int = 0,  # 0 表示不启用二次切割
        **kwargs,
    ) -> List[Document]:
        """
        对文档列表执行分割，返回切分后的 Document 列表。

        Args:
            documents: 原始文档列表（通常来自 LoaderFacade）
            strategy: 分割策略名称，默认 "recursive"
            max_chunk_size: 强制限制片段最大字符数（>0 时启用二次安全切割）
            **kwargs: 透传给 create_splitter 的参数

        Returns:
            切分后的 Document 列表
        """
        splitter = self.create_splitter(strategy, **kwargs)

        # MarkdownHeaderTextSplitter 特殊处理（按文本分割）
        if isinstance(splitter, MarkdownHeaderTextSplitter):
            all_chunks: List[Document] = []
            for doc in documents:
                chunks = splitter.split_text(doc.page_content)
                for chunk in chunks:
                    chunk.metadata = {**doc.metadata, **chunk.metadata}
                all_chunks.extend(chunks)
            logger.info(
                "markdown 分割: %d 文档 → %d 片段", len(documents), len(all_chunks)
            )
        else:
            # 通用分割器：split_documents
            all_chunks = splitter.split_documents(documents)
            logger.info(
                "%s 分割: %d 文档 → %d 片段", strategy, len(documents), len(all_chunks)
            )

        # ── 二次安全切割：强制限制片段最大长度 ──
        if max_chunk_size > 0:
            safety_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=min(50, max_chunk_size // 5),
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            )
            safe_chunks = safety_splitter.split_documents(all_chunks)
            logger.info(
                "长度安全过滤: %d 片段 → %d 片段 (max_chunk_size=%d)",
                len(all_chunks), len(safe_chunks), max_chunk_size,
            )
            all_chunks = safe_chunks

        # 片段长度统计（DEBUG 级别）
        if all_chunks and logger.isEnabledFor(logging.DEBUG):
            lengths = [len(doc.page_content) for doc in all_chunks]
            logger.debug(
                "片段长度统计 - 最小:%d, 最大:%d, 平均:%d, 总数:%d",
                min(lengths), max(lengths),
                sum(lengths) // len(lengths), len(lengths),
            )

        return all_chunks