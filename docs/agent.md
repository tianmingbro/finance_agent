# 金融法规智能体使用说明

**版本**：V1.0  
**日期**：2026-05-28  
**模块**：`agent.py`, `tools.py`

## 架构概览

金融法规智能体基于 LangGraph 的 `create_agent()` 预构建组件，采用 **LLM 自主决策 + 工具调用** 的架构模式，替代了早期手工编写的条件路由（`integrated_graph.py`）。

用户输入 → [Agent 决策] ──→ 需要工具? ──是──→ 调用 financial_qa 或 evaluate_answer
│ │
否 返回结果
│ │
└── 直接回复 ←────────── 工具结果 ────┘


- **决策核心**：qwen-plus 模型根据对话上下文和工具描述，自主判断是否调用工具、调用哪个工具。
- **状态管理**：所有消息（用户提问、工具调用、工具结果、AI 回复）自动追加到 `messages` 历史中，通过 `checkpointer` 持久化，支持多轮对话和会话恢复。
- **工具层**：底层复用已封装的 `FinancialRAGSkill`（混合检索）和 `EvaluationRunner`（DeepEval 评测），通过 `@tool` 装饰器暴露为 Agent 可调用的函数。

## 工具清单

### 1. `financial_qa`
- **功能**：回答金融法规相关问题
- **输入**：`query: str` —— 用户问题
- **输出**：包含答案和检索上下文的文本
- **底层**：`FinancialRAGSkill.run_with_context()` → 混合检索器（向量 + BM25）
- **适用场景**：事实查询、法规解释、条款咨询

### 2. `evaluate_answer`
- **功能**：对给定回答进行忠实度与相关性评测
- **输入**：`query: str` —— 原始问题；`answer: str` —— 待评测的回答
- **输出**：忠实度分数、答案相关性分数、综合信任等级
- **底层**：`EvaluationRunner` → DeepEval 指标（Faithfulness, AnswerRelevancy）
- **适用场景**：用户要求评估某个答案的准确性、开发调试

## 快速开始

```python
from agent import build_agent

# 创建 Agent（自动加载工具和检查点）
agent = build_agent()

# 单轮对话
config = {"configurable": {"thread_id": "user-123"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "资本充足率要求是多少？"}]},
    config
)
print(response["messages"][-1].content)

如何扩展新工具

使用 @tool 装饰器定义新工具函数，然后在 build_agent() 的 tools 列表中添加即可。

示例：添加一个法规更新通知工具
# tools.py
@tool
def check_regulation_update(regulation_name: str) -> str:
    """检查指定法规是否有最新修订"""
    # 实现逻辑，例如查询数据库或爬取官网
    return f"{regulation_name} 暂无更新。"

在 agent.py 中引入并加入工具列表：

from tools import financial_qa, evaluate_answer, check_regulation_update

def build_agent():
    return create_agent(
        model="qwen-plus",
        tools=[financial_qa, evaluate_answer, check_regulation_update],
        ...
    )

Agent 将自动识别新工具，无需修改图结构或路由逻辑。

配置参数
参数	默认值	说明
model	"qwen-plus"	决策模型，支持任何 LangChain 兼容的 chat model
tools	[financial_qa, evaluate_answer]	可用工具列表
system_prompt	金融法规助手角色描述	定义 Agent 的行为边界
checkpointer	MemorySaver	会话检查点，可替换为 SqliteSaver 或 PostgresSaver

已知限制

    工具调用结果超出模型上下文窗口时，Agent 可能截断历史消息；生产环境建议监控 token 使用量。

    evaluate_answer 评测工具每次调用会消耗较多 API token（DeepEval 内部多次调用 LLM），高频使用需注意成本。

    当前系统 prompt 未强制要求 Agent 在每次回答后主动提供评测，若需此功能，可修改 prompt 或增加事后触发规则