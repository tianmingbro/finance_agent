# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 中的环境变量

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_embedding_model_path():
    """返回完整的 Embedding 模型路径"""
    rel_path = os.getenv("EMBEDDING_MODEL_PATH", "models/text2vec-base-chinese/Jerry0/text2vec-base-chinese")
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Embedding 模型未找到: {abs_path}")
    return abs_path

def get_vector_size():
    """返回向量维度，默认 768"""
    return int(os.getenv("VECTOR_SIZE", "768"))