# A2A 多 Agent 协作架构

**版本**：V1.0  
**日期**：第8周 Day50  
**概述**：基于 Google A2A 开放协议规范，将金融 RAG 应用拆分为三个独立、可复用的标准化 Agent 服务，通过轻量级的 HTTP/JSON 通信实现任务分发、检索增强生成和质量评测的协作流程。

## 架构图

```mermaid
graph TD
    User[用户 / Streamlit 界面] -->|HTTP| Orchestrator[Orchestrator Agent<br/>端口 8100]
    Orchestrator -->|A2A Task| RAGAgent[RAG Agent<br/>端口 8101]
    Orchestrator -->|A2A Task| EvalAgent[Eval Agent<br/>端口 8102]

    subgraph RAG Agent
        RAGServer[FastAPI 服务]
        MCPRetriever[search_finance_docs]
        LLM[Qwen-plus 生成]
    end

    subgraph Eval Agent
        EvalServer[FastAPI 服务]
        DeepEval[DeepEval 评测引擎]
    end

    RAGServer --> MCPRetriever --> VectorDB[(向量库)]
    RAGServer --> LLM
    EvalServer --> DeepEval

    Orchestrator Agent：主控代理，负责解析用户自然语言意图，将金融问答请求路由给 RAG Agent，将评测请求路由给 Eval Agent。自身也是一个 A2A 服务，暴露 /a2a/task 端点。

    RAG Agent：封装了检索增强生成能力，通过 search_finance_docs（混合检索）和 Qwen-plus 生成答案，并将检索上下文与最终答案一并返回。

    Eval Agent：使用 DeepEval 引擎对问答对进行忠实度和答案相关性评分，以结构化 JSON 返回结果。

Agent 配置
Agent 信息（Agent Card）

每个 Agent 通过 /.well-known/agent-card 端点公开其元数据，方便服务发现。
Agent	名称	端口	核心技能	描述
Orchestrator	orchestrator	8100	dispatch	意图解析与任务分发
RAG Agent	rag_agent	8101	rag_query	金融法规检索增强生成
Eval Agent	eval_agent	8102	evaluate	答案忠实度与相关性评测
环境变量
变量	说明	默认值
DASHSCOPE_API_KEY	阿里云 DashScope API 密钥（用于 Qwen-plus LLM）	无（必需）
RAG_AGENT_URL	Orchestrator 调用 RAG Agent 的地址	http://localhost:8101
EVAL_AGENT_URL	Orchestrator 调用 Eval Agent 的地址	http://localhost:8102
通信流程

各 Agent 通过 HTTP POST 请求到 /a2a/task 端点交换 Task 对象，遵循 A2A 任务生命周期（created → working → completed / failed）。   

典型交互示例

    用户提问
    bash

    curl -X POST http://localhost:8100/a2a/task \
      -H "Content-Type: application/json" \
      -d '{"task": {"id": "1", "session_id": "s1", "messages": [{"role":"user","parts":[{"text":"资本充足率是多少？"}]}]}}'

    Orchestrator 解析到不含“评测”关键字，构造新 Task 发送给 RAG Agent：
    bash

    POST http://localhost:8101/a2a/task

    RAG Agent 检索、生成后返回：
    json

    {
      "id": "1",
      "status": "completed",
      "artifacts": [
        {
          "answer": "根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%……",
          "sources": ["capital_management_measures_2024.txt"]
        }
      ]
    }

    Orchestrator 将此结果直接返回给用户。

    评测请求
    用户输入 评测 资本充足率|核心一级资本充足率不低于5%，Orchestrator 解析到“评测”，提取问题和答案，发送给 Eval Agent：
    bash

    POST http://localhost:8102/a2a/task

    Eval Agent 返回：
    json

    {
      "id": "2",
      "status": "completed",
      "artifacts": [
        {
          "faithfulness": 0.92,
          "answer_relevancy": 0.95
        }
      ]
    }

异常处理与回退

    Orchestrator 在调用下游 Agent 时设置 30 秒超时，并捕获 ConnectionError 和 TimeoutException，返回 failed 状态和友好错误信息。

    各 Agent 内部通过 @agent_task 装饰器统一记录耗时和异常，将错误信息放入 artifacts 并置状态为 failed。

部署与启动

    安装依赖
    bash

    pip install fastapi uvicorn httpx langchain-openai

    启动 Agent 服务
    bash

    python agents/rag_agent.py      # 端口 8101
    python agents/eval_agent.py     # 端口 8102
    python agents/orchestrator.py   # 端口 8100

    Streamlit 协作界面
    bash

    streamlit run streamlit_app/app.py

    在侧边栏选择“多 Agent 协作”即可进行问答与评测。

    一键演示脚本
    bash

    python demo_a2a.py

    自动启动三个服务，模拟一次完整交互，演示结束后自动终止进程。