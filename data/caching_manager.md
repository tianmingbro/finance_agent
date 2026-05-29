caching_manager.md 使用说明

版本：V1.0
日期：2026-05-28
模块：caching_manager.py
概述

CachingManager 基于 Redis 为金融法规智能体提供双模式缓存：

    LLM 响应缓存：避免重复调用 Qwen‑plus API，降低延迟与成本。

    Embedding 计算缓存：避免重复计算 Sentence‑Transformer 向量，加速文档索引与检索。

支持精确匹配与语义匹配两种策略，通过一行代码集成，全局生效。
快速开始
python

from caching_manager import CachingManager
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

mgr = CachingManager(
    redis_url="redis://localhost:6379",
    embedding_model=embedding,
    mode="semantic",       # 或 "exact"
    ttl=3600,
)

mgr.enable_llm_cache()               # 全局 LLM 缓存
cached_emb = mgr.enable_embedding_cache()  # 返回带缓存的 Embedding 实例

缓存策略选择
策略	类	命中条件	优点	缺点
exact	RedisCache	完全相同的 Prompt（MD5 哈希）	零额外计算，命中速度极快	无法处理同义改写
semantic	RedisSemanticCache	向量相似度 ≤ distance_threshold（默认 0.2）	可捕获不同表述的相同意图	每次查询需计算 Embedding，增加少量开销

建议：

    金融法规问题同义变体多，优先使用 semantic。

    若 Embedding 服务负载高或对延迟极度敏感，使用 exact。

    可通过环境变量 CACHE_MODE 动态切换。

TTL 配置建议

    金融法规更新频率极低（数月～数年），可设置较长 TTL，如 3600 秒（1 小时） 至 86400 秒（24 小时）。

    开发/调试阶段建议缩短，例如 60 秒，以便及时看到修改效果。

    当知识库发生重大更新（如新法规发布）时，应调用 mgr.clear_cache() 主动刷新。

性能基准

基于 benchmark_cache.py 对 10 个高频金融问题各重复调用 3 次的测试结果（示例）：
模式	总耗时（秒）	LLM 调用次数	估算费用（USD）	节省时间（vs 无缓存）
无缓存	~45.0	30	~0.06	-
精确缓存	~15.0	10	~0.02	66.7%
语义缓存	~18.5	10	~0.02	58.9%

    注：语义缓存因需额外计算 Embedding，耗时略高于精确缓存，但仍可减少约 60% 的 API 调用。

集成方式

    金融 RAG Skill：在 ResourceManager.load_resources() 中初始化 CachingManager，全局启用 LLM 缓存，并用 enable_embedding_cache() 返回的实例替换原始 Embedding。

    Agent：通过 _get_financial_skill() 预热，确保缓存就绪。Agent 的工具调用自动受益。

    评测管道：运行 run_evaluation_pipeline.py 前确保 Redis 可用，即可自动利用缓存。

常见问题

Q：如何确认缓存是否生效？
A：查看日志中是否出现“LLM 语义/精确缓存已启用”，并使用 mgr.get_cache_stats() 查看状态。在评测脚本中统计 LLM 调用次数，第二次运行若大幅减少即表示生效。

Q：缓存命中后答案会变化吗？
A：不会。缓存的是相同的 LLM 响应或 Embedding 向量，因此核心质量指标（忠实度、答案相关性、上下文召回率）应保持不变。

Q：如何清除特定问题的缓存？
A：RedisCache 和 RedisSemanticCache 的 clear() 会清空所有缓存。若需精细控制，可考虑为每个用户或会话使用不同的 Redis 前缀。