"""
caching_manager.py
Day41 核心交付物：Redis 双模式缓存管理器（已适配 LangChain v1.2）
"""
import logging
import time
from typing import Optional, Dict, Any, Literal

from langchain_core.globals import set_llm_cache               # ✅ v1.2 正确路径
from langchain_classic.embeddings.cache import CacheBackedEmbeddings  # ✅ v1.2 正确路径
from langchain_community.storage import RedisStore              # ✅ v1.2 正确路径
from langchain_redis import RedisCache, RedisSemanticCache      # 保持不变

logger = logging.getLogger(__name__)
CacheMode = Literal["exact", "semantic"]


class CachingManager:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        embedding_model: Any = None,
        mode: CacheMode = "exact",
        ttl: Optional[int] = 3600,
        distance_threshold: float = 0.2,
        namespace: str = "embedding_cache",
    ):
        self.redis_url = redis_url
        self.embedding_model = embedding_model
        self.mode = mode
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.namespace = namespace

        self._llm_cache = None
        self._embedding_store = None
        self._cached_embeddings = None
        self._stats = {
            "enabled": False,
            "mode": mode,
            "ttl": ttl,
            "llm_cache_enabled": False,
            "embedding_cache_enabled": False,
            "created_at": time.time(),
        }

    def enable_llm_cache(self):
        if self.mode == "exact":
            self._llm_cache = RedisCache(redis_url=self.redis_url, ttl=self.ttl)
            logger.info("LLM 精确匹配缓存已启用")
        elif self.mode == "semantic":
            if self.embedding_model is None:
                raise ValueError("语义缓存需要 embedding_model")
            self._llm_cache = RedisSemanticCache(
                embeddings=self.embedding_model,
                redis_url=self.redis_url,
                distance_threshold=self.distance_threshold,
                ttl=self.ttl,
            )
            logger.info("LLM 语义缓存已启用")
        else:
            raise ValueError(f"不支持的缓存模式: {self.mode}")

        set_llm_cache(self._llm_cache)
        self._stats["llm_cache_enabled"] = True
        self._stats["enabled"] = True

    def enable_embedding_cache(self):
        if self.embedding_model is None:
            raise ValueError("Embedding 缓存需要 embedding_model")

        self._embedding_store = RedisStore(redis_url=self.redis_url, ttl=self.ttl)
        self._cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            self.embedding_model,
            self._embedding_store,
            namespace=self.namespace,
        )
        self._stats["embedding_cache_enabled"] = True
        self._stats["enabled"] = True
        logger.info("Embedding 缓存已启用")
        return self._cached_embeddings

    def clear_cache(self):
        if self._llm_cache is not None:
            self._llm_cache.clear()
            logger.info("LLM 缓存已清除")
            self._llm_cache = None
        if self._embedding_store is not None:
            try:
                import redis
                client = redis.Redis.from_url(self.redis_url)
                keys = client.keys(f"{self.namespace}*")
                if keys:
                    client.delete(*keys)
                logger.info("Embedding 缓存已清除 (%d 条)", len(keys))
            except Exception as e:
                logger.warning("清除 Embedding 缓存失败: %s", e)
            finally:
                self._embedding_store = None
                self._cached_embeddings = None
        self._stats["enabled"] = False
        self._stats["llm_cache_enabled"] = False
        self._stats["embedding_cache_enabled"] = False

    def get_cache_stats(self):
        stats = {**self._stats}
        stats["uptime_seconds"] = time.time() - self._stats["created_at"]
        stats["llm_cache_type"] = self.mode if self._stats["llm_cache_enabled"] else None
        return stats

    @property
    def cached_embeddings(self):
        return self._cached_embeddings