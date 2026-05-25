"""
loader_facade.py
Day36 增强版：统一文档加载门面，支持动态注册、编码处理、日志与耗时统计
兼容：langchain v1.2 + langchain-community
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Type, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class LoaderFacade:
    """统一文档加载门面，根据文件扩展名自动选择加载器"""

    def __init__(self):
        self._loaders: Dict[str, Type[BaseLoader]] = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
        }

    def register_loader(self, extension: str, loader_cls: Type[BaseLoader]) -> None:
        """
        注册自定义加载器。扩展名需以 '.' 开头（如 '.epub'）。
        支持动态扩展，符合开闭原则。
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        self._loaders[extension.lower()] = loader_cls
        logger.info("注册自定义加载器: %s -> %s", extension, loader_cls.__name__)

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()

        # 1. 格式校验（最早抛出，避免文件存在等无关检查）
        if ext not in self._loaders:
            raise ValueError(
                f"不支持的文件格式 '{ext}'。当前支持: {list(self._loaders.keys())}"
            )

        # 2. 文件存在校验
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        loader_cls = self._loaders[ext]

        # 3. 实例化并加载（TextLoader 强制 UTF-8，避免 chardet 兼容问题）
        start = time.time()
        if loader_cls is TextLoader:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            loader = loader_cls(str(path))

        documents = loader.load()
        elapsed = time.time() - start

        logger.info("加载完成 [%s] %s (耗时 %.3fs, %d 个文档)",
                     loader_cls.__name__, path.name, elapsed, len(documents))

        # 4. 统一补全元数据
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = str(path)

        return documents
    
    async def aload(self, file_path: Union[str, Path]) -> List[Document]:
        """
        异步加载文档，适合大文件或批量并发场景。
        内部使用 asyncio.to_thread 将同步加载操作放入线程池。
        """
        return await asyncio.to_thread(self.load, file_path)