# 🏦 金融 RAG 多智能体系统

基于 **A2A (Agent-to-Agent)** 协议构建的企业级金融法规智能问答与评测平台，通过主控 Agent 协同 RAG Agent 和评测 Agent，支持复杂指令拆分、上下文透传、混合检索、多维质量评估和可视化监控。

---

## 🧭 系统架构

### 多 Agent 协作模型

```mermaid
graph TD
    User[用户 / Streamlit 界面] -->|HTTP| Orch[Orchestrator Agent<br/>端口 8100]
    Orch -->|A2A Task| RAG[RAG Agent<br/>端口 8101]
    Orch -->|A2A Task| Eval[Eval Agent<br/>端口 8102]

    subgraph RAG Agent 内部
        MCP[MCP 检索工具]
        LLM[Qwen-plus 生成]
    end

    subgraph Eval Agent 内部
        DeepEval[DeepEval 引擎]
    end

    RAG --> MCP --> VectorDB[(向量库<br/>Chroma/PGVector)]
    RAG --> LLM
    Eval --> DeepEval

    Prometheus -->|抓取 /metrics| Orch
    Prometheus -->|抓取 /metrics| RAG
    Prometheus -->|抓取 /metrics| Eval
    Grafana --> Prometheus

Orchestrator Agent：意图解析、任务分发、结果汇总、会话管理

RAG Agent：混合检索（向量 + BM25）+ LLM 生成，暴露 rag_query 技能

Eval Agent：忠实度/答案相关性评分，暴露 evaluate 技能

通信：标准化 JSON 消息（A2A Task/Message），支持上下文透传与断点续查

监控：Prometheus 采集指标，Grafana 仪表盘可视化

🧩 Skill 规范说明

1. RAG Skill (financial_qa)

    功能：接收金融法规相关自然语言问题，通过混合检索（语义向量 + BM25）获取上下文，调用 Qwen‑plus 生成精确答案，并返回引用来源。

    触发词：资本充足率、LPR、存款保险、外汇额度、反洗钱、商业银行 等金融法规关键词。

    输入：
    json

    {
      "query": "商业银行的核心一级资本充足率要求是多少？"
    }

    输出：
    json

    {
      "answer": "根据《商业银行资本管理办法》，核心一级资本充足率不得低于5%...",
      "sources": ["capital_management_measures_2024.txt"]
    }

    配置项：

        chunk_size：文档切片大小（默认 500）

        chunk_overlap：切片重叠（默认 100）

        search_k：返回文档数（默认 4）

        fusion_strategy：混合检索融合策略（rrf 或 weighted）

    依赖外部服务：

        阿里云 DashScope API（Qwen‑plus 模型）

        向量数据库（Chroma 或 PGVector）

        Redis（可选，用于缓存）

        本地 search_finance_docs 工具（已封装为 MCP 服务）

2. 评测 Skill (evaluate_answer)

    功能：对给定的问答对进行忠实度与答案相关性评估，返回量化分数和信任等级。

    触发词：评测、评估、检查质量、忠实度、相关性。

    输入：
    json

    {
      "query": "存款保险最高偿付限额是多少？",
      "answer": "最高偿付限额为人民币50万元。"
    }

    输出：
    json

    {
      "faithfulness": 0.92,
      "answer_relevancy": 0.95,
      "trust": "高"
    }

    配置项：

        model：评测底层 LLM（默认 qwen-plus）

        threshold：通过阈值（默认 0.7）

    依赖外部服务：

        DeepEval 评测框架（FaithfulnessMetric、AnswerRelevancyMetric）

        阿里云 DashScope API（用于 LLM‑as‑Judge）

任务对象 (Task) 包含 session_id、messages、context 等字段，支持跨 Agent 上下文透传。

📦 安装步骤
环境要求

    Python 3.10+

    Docker & Docker Compose

    Redis (Docker 内提供)

    DashScope API Key (用于 Qwen-plus LLM)

1. 获取代码
bash

git clone git@github.com:tianmingbro/finance_agent.git
cd finance-rag-agents

2. 配置环境变量

复制模板并填写关键密钥：
bash

cp .env.example .env
# 编辑 .env 文件：
# DASHSCOPE_API_KEY=sk-your-key-here
# OPENAI_API_KEY=sk-... (可选，用于 Promptfoo)

3. 本地开发启动
bash

# 安装 Python 依赖
pip install -r requirements.txt

# 启动向量库（Chroma 本地无需额外操作；若用 PGVector 需提供连接串）
初始化向量库
# 若使用 Chroma
python scripts/ingest_data.py --backend chroma
# 启动 Redis
docker run -d -p 6379:6379 redis:latest

# 分别启动三个 Agent
python agents/rag_agent.py &
python agents/eval_agent.py &
python agents/orchestrator.py &

# 启动 Streamlit 控制台（可选）
streamlit run streamlit_app/app.py

4. Docker Compose 一键部署（推荐）
bash

docker-compose up -d

自动启动所有服务（Agent、Redis、Prometheus、Grafana）。
服务清单：

    orchestrator → http://localhost:8100

    rag-agent → http://localhost:8101

    eval-agent → http://localhost:8102

    redis → localhost:6379

    prometheus → http://localhost:9090

    grafana → http://localhost:3000（admin/admin）

📡 API 文档
服务发现

GET /.well-known/agent-card
返回 Agent 的元数据（名称、技能、端点）。

示例：
bash

curl http://localhost:8100/.well-known/agent-card

任务发送（核心接口）

POST /a2a/task
向 Agent 发送任务。所有 Agent 共用此接口格式。

请求体：
json

{
  "task": {
    "id": "唯一ID",
    "session_id": "会话ID（可选）",
    "messages": [
      {
        "role": "user",
        "parts": [{"text": "资本充足率是多少？"}]
      }
    ],
    "context": {}
  }
}

响应体（完成时）：
json

{
  "id": "...",
  "session_id": "...",
  "status": "completed",
  "artifacts": [
    {
      "answer": "根据《商业银行资本管理办法》...",
      "sources": ["doc1.txt"]
    }
  ],
  "context": { ... }
}

复合指令示例
bash

curl -X POST http://localhost:8100/a2a/task \
  -H "Content-Type: application/json" \
  -d '{"task": {"id":"c1","session_id":"s1","messages":[{"role":"user","parts":[{"text":"查一下LPR并评估答案是否准确"}]}]}}'

Orchestrator 自动解析为「问答+评测」链，返回答案与分数。
流式响应（实验性）

POST /a2a/task/stream
以 Server-Sent Events 流式返回生成结果。

📊 评测模块使用指南
1. DeepEval 组件级评估
DeepEval 用于对 RAG 系统的三大组件（规划、检索、生成）进行独立打分，快速定位质量短板。
准备评测数据集

    数据集格式：YAML，结构与训练数据一致，需包含 categories 或 test_cases 字段，每条用例至少提供 query 和 expected_answer。

    示例：
    yaml

    categories:
      - category: factual_query
        entries:
          - query: "商业银行的核心一级资本充足率要求是多少？"
            expected_answer: "不低于5%。"

    推荐使用已有的 data/eval_dataset_v2.yaml，或通过 Streamlit 上传自定义数据集。

通过 Streamlit 管理界面或命令行执行：
bash

streamlit run streamlit_app/app.py
# 侧边栏选择“评测中心”，上传 eval_dataset_v2.yaml，点击开始评测。

解读报告

    报告（JSON / 界面卡片）展示每个组件的指标均值（0~1）：

        Planning：工具选择准确率、参数合理性

        Retrieval：上下文召回率、上下文精确度

        Generation：忠实度、答案相关性

    卡片颜色：绿色（≥ 阈值，默认0.7）、红色（未通过）。

    详细结果可下载 CSV 逐条分析。
    
2. Promptfoo 多模型对比与安全红队
Promptfoo 用于横向对比不同 LLM 在金融法规问答上的表现，并执行安全漏洞扫描。
配置文件说明

    主配置：promptfooconfig.yaml，定义 providers（模型）、prompts（模板）、tests（断言）。

    测试用例：tests/promptfoo/promptfoo_tests.yaml，包含三类场景（直接回答、RAG增强、安全边界）。

    模板：configs/promptfoo/prompts/rag_qa.json，使用 Nunjucks 语法注入变量。

# 多模型对比
npx promptfoo@latest eval -c promptfooconfig.yaml

# 红队扫描
npx promptfoo@latest redteam run -c promptfooconfig.yaml

报告查看

    评测完成后自动生成 HTML 报告（promptfoo_output.html），用浏览器打开。

    报告包含：

        每个模型的通过率、延迟。

        失败用例的详情（输入、输出、未满足的断言）。

        安全扫描结果（漏洞类型、攻击样例、风险等级）。

    也可在 Streamlit 的“评测中心”选择 Promptfoo 框架，上传数据集后一键运行并下载报告。

3. 自动化评测流水线
bash

# 完整组件评估 + 回归检测
python scripts/run_evaluation_pipeline.py --component

# 多模型 A/B 对比
python scripts/compare_prompts.py

4. 评测数据集

    测试集 data/eval_dataset_v2.yaml（60条金融法规问答，含事实/推理/对抗）

    构建流程：原始文档 → 自动生成 QA → 人工校验 → 分类标注
    详见 docs/benchmark_construction.md

📈 监控与告警

    Prometheus 自动抓取各 Agent 的 /metrics 端点。

    Grafana 预置仪表盘 configs/grafana/dashboards/a2a-overview.json，展示请求速率、延迟分布、子任务成功率等。

    访问 http://localhost:3000 查看（默认凭证 admin/admin）。

📂 项目结构（核心部分）
text
fin-rag-agent/
├
├── api/                        # FastAPI 独立端点（备用）
│   └── main.py
├── configs/                    # 配置文件
│   ├── promptfoo/              # Promptfoo 模板与测试
│   ├── grafana/                # Grafana 仪表盘与数据源配置
│   └── prometheus/             # Prometheus 抓取配置
├── data/                       # 数据集与向量库持久化
│   ├── eval_dataset_v2.yaml    # 评测数据集
│   ├── final_qa_dataset.yaml   # 最终 QA 数据集
│   └── source_docs/            # 原始法规文档
├── docs/                       # 文档
│   ├── a2a_architecture.md
│   ├── complex_task_handling.md
│   └── ...
├── scripts/                    # 运维与评测脚本
│   ├── run_evaluation_pipeline.py
│   ├── compare_prompts.py
│   ├── benchmark_complex.py
│   └── ingest_data.py
├── src/                        # 核心 Skill 与检索器
│   ├── retriever/              # 混合检索器
│   ├── skill/                  # RAG Skill、AI Test Skill
│   │   agent/                  # A2A Agent 服务
│   │   ├── a2a_types.py        # A2A 核心数据类（Task, Message, AgentCard）
│   │   ├── rag_agent.py        # RAG Agent (端口 8101)
│   │   ├── eval_agent.py       # Eval Agent (端口 8102)
│   │   ├── orchestrator.py     # 主控 Orchestrator (端口 8100)
│   │   └── decorators.py       # 日志、重试等装饰器
│   └── ...
├── streamlit_app/              # Streamlit 评测管理界面
│   └── app.py
├── tests/                      # 测试用例
│   ├── unit/                   # 单元测试
│   ├── integration/            # 集成测试
│   └── promptfoo/              # Promptfoo 测试用例
├── docker-compose.yml
├── Dockerfile (多阶段，见 agents/)
├── requirements.txt
├── README.md
└── LICENSE

如何添加新 Agent（遵循 A2A 规范）

    创建 Agent 文件：在 src/agent/ 下新建 Python 文件（如 compliance_agent.py）。

    实现 FastAPI 应用：
    python

    from fastapi import FastAPI
    from agents.a2a_types import Task, AgentCard, AgentSkill
    app = FastAPI()

    @app.post("/a2a/task")
    async def handle_task(task: Task):
        # 实现业务逻辑
        return task

    @app.get("/.well-known/agent-card")
    def agent_card():
        return AgentCard(
            name="compliance_agent",
            description="合规审查代理",
            url="http://localhost:8103",
            skills=[AgentSkill(id="compliance_check", name="合规检查")]
        )

    定义 Agent 技能：在 AgentCard 中声明技能 ID 和描述，供 Orchestrator 发现。

    注册到 Orchestrator：在 orchestrator.py 中更新 parse_complex_intent 或在配置中添加新 Agent 的 URL，使其能被分发调用。

    容器化：为新 Agent 编写 Dockerfile，并在 docker-compose.yml 中添加服务定义。

如何注册新工具

工具指 MCP 服务器或 A2A 代理内部可调用的函数。在现有架构中，工具通过 @agent_task 装饰器或 @tool 装饰器注册。

方式一：A2A Agent 内部工具（如新增 OCR 工具）

    在目标 Agent 中定义异步函数。

    使用 @agent_task("tool_name") 装饰器，自动记录耗时与异常。

    在 /a2a/task 端点中根据消息内容调用该函数。

方式二：LangChain 工具（用于 Agent 自动决策）

    定义函数并用 @tool 装饰。

    将其加入 create_agent 的 tools 列表。

    在 Orchestrator 中调用 get_tools() 获取工具集。

测试
运行测试

    单元测试：pytest tests/unit/ -v

    集成测试（含 A2A 通信）：pytest tests/integration/ -v

    性能基准测试：python scripts/benchmark_complex.py

    全量评测回归：python scripts/run_evaluation_pipeline.py --component

所有测试均可在本地或 CI 环境中执行，依赖 Redis 和向量库的测试会自动跳过（如果服务未启动）。
CI/CD 简介

项目使用 GitHub Actions 自动执行以下流程：

    代码推送/PR：触发单元测试、集成测试、代码风格检查。

    主分支合并：额外运行 Promptfoo 安全扫描和性能基准测试，生成报告归档。

    发布：通过 Tag 触发 Docker 镜像构建并推送至容器仓库。

配置文件位于 .github/workflows/，可根据团队需求调整触发条件和测试集。
许可证

本项目采用 Apache License 2.0 开源。
致谢

    LangChain —— LLM 应用框架

    LangGraph —— Agent 工作流编排

    DeepEval —— 单元测试与评估

    Promptfoo —— 多模型对比与红队测试

    FastAPI —— 高性能 Web 框架

    Streamlit —— 可视化界面

    Redis —— 缓存与会话存储

    Prometheus & Grafana —— 监控与可视化

