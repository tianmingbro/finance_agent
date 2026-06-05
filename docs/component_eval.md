# 组件级评估方案 V1.0

**版本**：V1.0  
**日期**：2026-06-04  
**适用项目**：金融 RAG Agent 评测体系

## 1. 设计思路

传统整体评估将 RAG 系统视为黑盒，只能得到“好”或“坏”的结论，难以定位问题根源。**组件级评估（Component‑Level Evaluation）** 将智能体工作流拆分为**规划、检索、生成**三个独立组件，为每个组件定义专属的测试用例和指标，实现：

- **快速定位瓶颈**：当整体回答质量下降时，可立刻判断是检索遗漏了关键文档、生成模型产生幻觉，还是规划器选错了工具。
- **独立回归测试**：修改检索策略后，只需重新运行检索组件的评估，无需重新评测生成端。
- **可组合的覆盖率**：每个组件可单独维护其测试集，未来新增组件（如“合规检查”）时可无缝集成。

与 Day34 的整体评测相比，组件级评估提供**更细粒度的诊断信息**，两者互补：整体评测用于宏观质量跟踪，组件评测用于微观问题定位。

## 2. 组件与指标定义

### 2.1 规划组件（Planner）

**目标**：评估 Agent 在接收到用户请求后，能否正确选择工具并构造参数。  
**测试输入**：用户查询 `query`，对话历史（可选），期望的工具名 `expected_tool` 和参数 `expected_args`。  
**评估指标**：

- **工具选择准确率**：实际调用的工具是否与期望一致（1.0 表示完全匹配）。
- **参数合理度**：工具参数的核心部分（如 `query`）是否与期望一致。

**适用场景**：多工具 Agent、动态路由的工作流。当前系统采用固定编排（检索→生成），本指标可用于验证未来扩展的工具调用逻辑。

### 2.2 检索组件（Retriever）

**目标**：验证检索器返回的文档是否相关且能覆盖答案所需信息。  
**测试输入**：查询 `query`，关键信息列表 `key_info`（标准答案中必须出现的事实点）。  
**评估指标**：

- **上下文召回率 (Contextual Recall)**：关键信息被检索文档覆盖的比例。≥ 0.7 视为通过。
- **上下文精确度 (Contextual Precision)**：检索结果中真正相关的文档占比。≥ 0.7 视为通过。

**实现**：内部调用 `tools_mcp.search_finance_docs` 获取实际检索结果，通过 DeepEval 的 `ContextualRecallMetric` 和 `ContextualPrecisionMetric` 自动计算。

### 2.3 生成组件（Generator）

**目标**：确保 LLM 基于检索上下文生成的答案**忠实、相关、完整**。  
**测试输入**：查询 `query`，可选的期望答案 `expected_answer`（仅用于参考，不参与指标计算）。  
**评估指标**：

- **忠实度 (Faithfulness)**：答案中的每条信息是否能从检索上下文推导。≥ 0.8 视为通过。
- **答案相关性 (Answer Relevancy)**：答案与问题的语义相关程度。≥ 0.8 视为通过。

**实现**：内部会先调用检索获取上下文，再调用 LLM 生成答案，然后基于同一上下文使用 DeepEval 的 `FaithfulnessMetric` 和 `AnswerRelevancyMetric` 评分。

## 3. 测试数据格式

组件级测试数据采用 YAML 格式，每条用例需包含各组件所需的标注字段：

```yaml
test_cases:
  - id: "fact_001"
    query: "商业银行的核心一级资本充足率要求是多少？"
    expected_answer: "核心一级资本充足率不得低于5%。"
    category: "factual_query"
    planning:
      expected_tool: "financial_qa"
      expected_args:
        query: "商业银行的核心一级资本充足率要求是多少？"
    retrieval:
      key_info: ["不低于5%"]
    generation:
      key_entities: ["5%"]

4. 评估脚本与集成
4.1 独立运行
bash

python run_evaluation_pipeline.py --component

该命令会加载 data/component_eval_data.yaml，执行三个组件的评估，并输出 component_eval_report.json。
4.2 API 端点

POST /eval/component 接受单个查询及可选标注，返回规划、检索、生成三个组件的评分数：
json

{
  "query": "资本充足率是多少？",
  "expected_tool": "financial_qa",
  "expected_args": {"query": "资本充足率是多少？"},
  "key_info": ["不低于5%", "不低于8%"],
  "expected_answer": "核心一级资本充足率不低于5%，资本充足率不低于8%。"
}

响应：
json

{
  "planning": {
    "tool_accuracy": 1.0,
    "arg_reasonableness": 1.0
  },
  "retrieval": {
    "contextual_recall": 0.85,
    "contextual_precision": 0.80
  },
  "generation": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.95
  }
}

5. 扩展新组件

若要为工作流添加新组件（如“合规检查”），只需：

    在 eval_components.py 中新增类似 evaluate_compliance(query, answer) 的函数，返回指标字典。

    在测试数据 YAML 中增加 compliance 字段。

    在 generate_component_report() 中调用新函数并汇总。

    （可选）在 API 中增加对应字段。

这种设计保持了高度模块化，新组件不会影响已有评估逻辑。
6. 与整体评估的互补关系
维度	整体评估 (Day34)	组件级评估 (Day46)
粒度	系统级（端到端）	组件级（规划/检索/生成）
指标	忠实度、答案相关性、上下文召回率	同上，但按组件独立计算，增加工具准确率、精确度
定位速度	只知道“有问题”，需人工排查	可直接指出是哪个组件未达标
依赖	完整 RAG 流程必须跑通	每个组件可独立测试，检索评估不依赖生成
使用场景	日常回归、版本发布前的整体质量门禁	开发调试、组件优化、技术选型对比

推荐工作流：开发阶段频繁运行组件级评估，确保每次改动只影响目标组件；CI 中同时运行整体评估与组件评估，前者保证系统端到端可用，后者防止组件级退化。