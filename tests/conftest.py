# tests/conftest.py
import os
import sys
import asyncio
from pathlib import Path

# ============ Windows 兼容性修复 ============
# Psycopg 与 Windows 的 ProactorEventLoop 不兼容，必须在所有 psycopg 导入前设置
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============ 环境变量覆盖（必须在任何其他导入之前）============
# 强制使用 Chroma 后端，避免依赖外部 pgvector/Redis 服务
os.environ["VECTOR_STORE_BACKEND"] = "chroma"
os.environ["DISABLE_CACHE"] = "1"

# 将项目根目录和 src 目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))        # 让 `from loader_facade import ...` 可直接工作
sys.path.insert(0, str(PROJECT_ROOT))    # 让 `from src.xxx import ...` 也可工作（备用）

from langchain_core.documents import Document

import pytest

from config import get_embedding_model_path
MODEL_PATH = get_embedding_model_path()


@pytest.fixture(scope="session")
def embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
