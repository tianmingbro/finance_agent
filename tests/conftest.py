# tests/conftest.py
import sys
from pathlib import Path

# 将项目根目录（financial-rag-agent/）和 src 目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))        # 让 `from loader_facade import ...` 可直接工作
sys.path.insert(0, str(PROJECT_ROOT))    # 让 `from src.xxx import ...` 也可工作（备用）

# tests/conftest.py 或直接写在测试文件中
from langchain_core.documents import Document

import pytest

from config import get_embedding_model_path
MODEL_PATH=get_embedding_model_path()



@pytest.fixture(scope="session")
def embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

