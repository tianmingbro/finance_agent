# RAG Agent 函数式工作流使用说明

**版本**：V1.0  
**日期**：2026-06-02  
**依赖**：`langgraph >= 1.0`、`langchain >= 1.2`

## 概述

`workflow.py` 使用 LangGraph 的函数式 API（`@entrypoint` + `@task`）定义了金融法规 RAG 的核心流程。  
工作流将**检索**、**生成**、**评测**拆分为独立任务，由框架自动处理状态传递、错误重试与耗时记录。

## 快速开始

```python
from workflow import rag_agent_workflow

# 仅检索 + 生成
result = rag_agent_workflow.invoke("资本充足率是多少？")
print(result["answer"])          # 生成的答案
print(result["retrieved_docs"])  # 检索到的文档

# 同时进行评测
result = rag_agent_workflow.invoke("资本充足率是多少？", need_eval=True)
print(result["evaluation"])      # 评测分数 JSON

工作流架构
text

用户查询
   │
   ▼
@entrypoint (rag_agent_workflow)
   │
   ├── @task retrieve_task       ← 调用 MCP 检索工具
   │       │
   │       └── 返回 JSON 文档列表
   │
   ├── @task generate_answer_task ← 基于文档生成答案
   │       │
   │       └── 返回自然语言答案
   │
   └── @task evaluate_task        ← 可选：评测答案质量
           │
           └── 返回忠实度/相关性分数

每个 @task 都是可独立执行、可重试的 Python 函数。任务之间的依赖关系通过参数传递自然表达：
generate_answer_task 接收 retrieve_task 的输出；evaluate_task 接收原始问题和生成答案。

关键特性
声明式重试
python

@task(retry=2)                # 失败后自动重试 2 次
def retrieve_task(query: str) -> str:
    ...

适用于网络波动、LLM 临时不可用等场景。重试次数可在每个任务上灵活配置。
自动耗时日志

每个任务内部通过 time.time() 记录耗时，并在 logger 输出。生产环境可接入 Prometheus 等监控系统。

状态自动传递

无需手动定义 TypedDict。
@entrypoint 中的变量在多次调用间由框架自动持久化（需要 checkpointer 时），任务间直接通过返回值传递数据。
如何扩展新任务

以新增“敏感词过滤”任务为例：

    定义新任务

python

@task(retry=0)
def guardrails_task(user_query: str) -> bool:
    """检查查询是否包含敏感词"""
    sensitive_words = ["洗钱", "非法集资"]
    return any(word in user_query for word in sensitive_words)

    在入口点中编排

python

@entrypoint()
def rag_agent_workflow(query: str, need_eval: bool = False) -> dict:
    # 先执行安全检查
    is_blocked = guardrails_task(query).result()
    if is_blocked:
        return {"answer": "包含敏感内容，无法回答。", "retrieved_docs": {}}

    # 原有流程
    retrieved = retrieve_task(query).result()
    answer = generate_answer_task(query, retrieved).result()
    # ...

新任务可以放置在流程的任何位置，也可以与其他任务并行执行（多个 @task 在无依赖时自动并行）。
测试策略

    单元测试：Mock 外部依赖（如 search_finance_docs、ChatOpenAI），验证任务间数据传递和行为分支。

    集成测试：实际调用工作流，验证端到端输出。

    重试测试：使用 side_effect 抛出异常，断言重试次数。

部署注意事项

    工作流实例是无状态的，可水平扩展。

    若需持久化对话状态，可传入 checkpointer（如 SqliteSaver）。

    所有敏感配置（API Key、数据库连接）应通过环境变量注入，禁止硬编码。

text


