# A2A 上下文透传设计

**版本**：V1.0  
**日期**：第8周 Day51  
**概述**：在多 Agent 协作流程中，用户往往需要进行连续交互（如“提问 → 评测”）。为避免用户重复输入已生成的内容，我们在 A2A 协议的 `Task` 数据结构中引入 `context` 字典，并由 Orchestrator 维护会话存储，实现跨 Agent 的上下文自动传递。

## 数据结构

在 `agents/a2a_types.py` 中，`Task` 数据类增加了 `context` 字段：

```python
@dataclass
class Task:
    id: str
    session_id: str = ""
    messages: List[Message] = field(default_factory=list)
    artifacts: List[Any] = field(default_factory=list)
    status: str = "created"
    context: Dict[str, Any] = field(default_factory=dict)   # 新增

    context 是一个自由字典，可携带任意的跨 Agent 共享信息。

    典型的上下文包含：query（原始问题）、answer（上一次生成的答案）、sources（检索来源列表）。

    该字段由上游 Agent 填充，下游 Agent 读取，可贯穿整个调用链。

会话管理策略
内存存储

Orchestrator 使用 Python 字典 session_store 存储每个会话的最新上下文，以 session_id 为键。每个键对应的值包含：

    data：上下文字典（query, answer, sources 等）

    timestamp：最后更新时间（用于 TTL 过期）

TTL 过期

为防止内存无限增长，SESSION_TTL 默认设为 300 秒。每次获取会话时，检查时间戳，若超过 TTL 则自动删除并返回空。同时，后台异步任务每分钟扫描并清理所有过期会话。
python

SESSION_TTL = 300  # 秒

def get_session(session_id: str) -> dict:
    if session_id in session_store:
        entry = session_store[session_id]
        if time.time() - entry["timestamp"] < SESSION_TTL:
            return entry["data"]
        else:
            del session_store[session_id]
    return {}

上下文生命周期

    生成：当 Orchestrator 调用 RAG Agent 并收到成功响应后，如果返回的 Task 中包含 context（通常携带 query、answer、sources），则调用 set_session() 将其存入 session_store。

    消费：当用户发出“评测一下”这类模糊指令时，Orchestrator 会调用 get_session() 检索该会话的上一次问答上下文，填入新 Task 的 context 字段，然后发送给 Eval Agent。

    销毁：超过 TTL 未被访问的会话自动被清理；用户新的问答会覆盖同一 session_id 下的旧上下文。

协作流程示例

    用户提问 → Orchestrator 调用 RAG Agent，RAG 返回 artifacts（答案）和 context（query, answer, sources）。

    Orchestrator 将 context 存入 session_store（键 = session_id）。

    用户点击“评测” → Orchestrator 从 session_store 取出上下文，构造一个携带该 context 的 Task，发送给 Eval Agent。

    Eval Agent 优先从 context 中读取 query 和 answer，直接进行评测，无需用户再次提供。

异常处理

    如果 get_session() 返回空（会话不存在或已过期），Orchestrator 会向用户返回明确的错误信息：“没有找到上一次对话，请先提问或手动输入”。

    Eval Agent 在 context 为空时会自动回退到消息文本解析，保持向后兼容。

日志与调试

所有 Agent 在处理任务时都会在日志中输出 session_id 和 context 摘要，便于追踪调用链：
text

发送任务到 http://localhost:8101, session=user123, context keys=['query', 'answer']
RAG Agent 收到任务 session=user123, context={'query': '...', 'answer': '...'}

未来优化

    将 session_store 替换为 Redis，支持多实例部署和持久化。

    定义标准化的上下文 Schema，避免字段滥用。

    引入 trace_id 全链路跟踪。