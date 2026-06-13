from src.config import get_embedding_model_path
MODEL_PATH=get_embedding_model_path()
import time
import pytest
import socket
from unittest.mock import patch

from src.cache.caching_manager import CachingManager
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.llms import LLM
from langchain_redis import RedisCache, RedisSemanticCache
# embedding 缓存仍使用社区版（有 embed_query，弃用警告不影响功能）
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.storage import RedisStore
from langchain_classic.embeddings.cache import CacheBackedEmbeddings

def is_redis_available(host="localhost", port=6379):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False

requires_redis = pytest.mark.skipif(
    not is_redis_available(),
    reason="Redis 未启动，请执行 docker run -d -p 6379:6379 redis:latest"
)

class FakeLLM(LLM):
    call_count: int = 0
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        self.call_count += 1
        return f"回答：这是关于「{prompt[:20]}...」的答案。"
    @property
    def _llm_type(self) -> str:
        return "fake"

@requires_redis
class TestRedisCache:
    def test_redis_connection(self):
        import redis
        r = redis.Redis(host="localhost", port=6379)
        assert r.ping()

    def test_llm_exact_cache_hit(self):
        cache = RedisCache(redis_url="redis://localhost:6379", ttl=600)
        cache.clear()                              # ✅ 清理残留
        set_llm_cache(cache)
        llm = FakeLLM()
        prompt = "商业银行资本充足率是多少？"
        first = llm.invoke(prompt)
        assert llm.call_count == 1
        second = llm.invoke(prompt)
        assert second == first
        assert llm.call_count == 1                 # 命中缓存，不再增加

    def test_llm_semantic_cache_hit(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        cache = RedisSemanticCache(
            redis_url="redis://localhost:6379",
            embeddings=embeddings,
            distance_threshold=0.2,
            ttl=600,
        )
        cache.clear()                              # ✅ 清理残留
        set_llm_cache(cache)
        llm = FakeLLM()
        first = llm.invoke("存款保险最高赔付金额")
        assert llm.call_count == 1
        second = llm.invoke("存款保险最多赔多少钱")
        assert second == first
        assert llm.call_count == 1                 # 语义命中

    def test_embedding_cache_hit(self):
        underlying = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        store = RedisStore(redis_url="redis://localhost:6379", ttl=600)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying, store, namespace="test_emb", query_embedding_cache=True
        )
        text = "核心一级资本充足率不得低于5%"
        vec1 = cached_embeddings.embed_query(text)
        vec2 = cached_embeddings.embed_query(text)
        assert vec1 == vec2
        with patch.object(type(underlying), 'embed_query', wraps=underlying.embed_query) as mock_embed:
            cached_embeddings.embed_query("另一段文本")
            cached_embeddings.embed_query("另一段文本")
            assert mock_embed.call_count == 1

    def test_cache_ttl_expiry(self):
        cache = RedisCache(redis_url="redis://localhost:6379", ttl=1)
        cache.clear()
        set_llm_cache(cache)
        llm = FakeLLM()
        prompt = "LPR最新报价是多少？"
        llm.invoke(prompt)
        assert llm.call_count == 1
        time.sleep(1.5)
        llm.invoke(prompt)
        assert llm.call_count == 2

    def test_cache_invalidation(self):
        cache = RedisCache(redis_url="redis://localhost:6379", ttl=600)
        cache.clear()                              # ✅ 清理残留
        set_llm_cache(cache)
        llm = FakeLLM()
        prompt = "个人外汇便利化额度"
        llm.invoke(prompt)
        assert llm.call_count == 1
        cache.clear()
        llm.invoke(prompt)
        assert llm.call_count == 2


@requires_redis
class TestCachingManager:
    def test_enable_exact_llm_cache(self):
        manager = CachingManager(redis_url="redis://localhost:6379", mode="exact", ttl=600)
        manager.enable_llm_cache()
        manager._llm_cache.clear()
        llm = FakeLLM()
        prompt = "个人外汇便利化额度"
        first = llm.invoke(prompt)
        assert llm.call_count == 1
        second = llm.invoke(prompt)
        assert second == first
        assert llm.call_count == 1
        manager.clear_cache()

    def test_enable_semantic_llm_cache(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        manager = CachingManager(
            redis_url="redis://localhost:6379",
            mode="semantic",
            embedding_model=embeddings,
            distance_threshold=0.2,
            ttl=600,
        )
        manager.enable_llm_cache()
        manager._llm_cache.clear()
        llm = FakeLLM()
        first = llm.invoke("存款保险最高赔付金额")
        assert llm.call_count == 1
        second = llm.invoke("存款保险最多赔多少钱")
        assert second == first
        assert llm.call_count == 1
        manager.clear_cache()

    def test_enable_embedding_cache(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        manager = CachingManager(
            redis_url="redis://localhost:6379",
            embedding_model=embeddings,
            ttl=600,
            namespace="test_emb"
        )
        cached_embeddings = manager.enable_embedding_cache()
        text = "核心一级资本充足率不得低于5%"
        vec1 = cached_embeddings.embed_query(text)
        vec2 = cached_embeddings.embed_query(text)
        assert vec1 == vec2
        manager.clear_cache()
