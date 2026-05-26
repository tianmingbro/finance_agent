"""
document_processor.py
组合加载与分割的上层门面
"""
from pathlib import Path
from typing import List, Union, Optional
from langchain_core.documents import Document
from loader_facade import LoaderFacade
from splitter_factory import SplitterFactory


class DocumentProcessor:
    """
    文档处理器：加载 + 自动选择分割策略 + 切片。
    """

    def __init__(
        self,
        loader_facade: Optional[LoaderFacade] = None,
        splitter_factory: Optional[SplitterFactory] = None,
    ):
        self.loader = loader_facade or LoaderFacade()
        self.splitter = splitter_factory or SplitterFactory()

        # 扩展名到策略的默认映射
        self.extension_strategy = {
            ".txt": "recursive",
            ".pdf": "recursive",
            ".docx": "recursive",
            ".md": "markdown",
            ".html": "html",
        }

    def process(
        self,
        file_path: Union[str, Path],
        strategy: Optional[str] = None,
        **splitter_kwargs,
    ) -> List[Document]:
        """
        加载文档，使用指定策略（或按扩展名自动选择）切片。

        Args:
            file_path: 文件路径
            strategy: 分割策略。若为 None，则根据文件扩展名自动选择。
            **splitter_kwargs: 透传给 SplitterFactory.split 的参数

        Returns:
            切分后的 Document 列表
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # 确定策略
        if strategy is None:
            strategy = self.extension_strategy.get(ext, "recursive")

        # 加载
        docs = self.loader.load(path)

        # 分割
        chunks = self.splitter.split(docs, strategy=strategy, **splitter_kwargs)
        return chunks

    def process_directory(
        self,
        directory: Union[str, Path],
        **splitter_kwargs,
    ) -> List[Document]:
        """
        批量处理目录下所有支持的文件。
        """
        directory = Path(directory)
        all_chunks = []
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    chunks = self.process(file_path, **splitter_kwargs)
                    all_chunks.extend(chunks)
                    print(f"✅ 处理完成: {file_path.name} → {len(chunks)} 个片段")
                except ValueError:
                    print(f"⏭️ 跳过不支持的格式: {file_path.name}")
                except Exception as e:
                    print(f"❌ 处理失败: {file_path.name} ({e})")
        return all_chunks