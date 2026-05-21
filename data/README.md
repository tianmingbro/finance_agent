markdown

# 金融法规问答数据集

## 概述
本目录包含用于金融 RAG（检索增强生成）应用的所有数据集与资源，涵盖原始法规文档、生成的问答对、评测数据集以及向量库文件。数据遵循“采集→生成→校验→入库”的自动化流水线，确保知识覆盖与质量可控。

## 目录结构

data/
├── README.md # 本说明文件
├── index.yaml # 源文档索引
├── source_docs/ # 原始法规文本
│ ├── capital_management_measures_2024.txt
│ ├── personal_forex_measures_2007.txt
│ ├── personal_forex_implementation_rules_2007.txt
│ ├── deposit_insurance_regulation_2015.txt
│ ├── lpr_formation_mechanism_2026.md
│ └── anti_money_laundering_law_2025.txt
├── candidate_qa.yaml # 大模型生成的候选 QA（未校验）
├── flagged_qa.yaml # 自动标记潜在问题的 QA
├── reviewed_qa.yaml # 人工校验通过的 QA
├── rewrite_list.yaml # 需要重写的知识点清单
├── rewrite_candidates.yaml # 二次生成的结果
├── final_qa_dataset.yaml # 最终 QA 数据集（≥50条）
├── eval_dataset.yaml # 原始评测数据集（12条骨架）
├── eval_dataset_v2.yaml # 扩充后的评测数据集（≥60条）
├── finance_qa.txt # 早期手工问答对（已弃用，仅保留兼容）
└── chroma_db/ # Chroma 向量库持久化目录
text


## 数据来源
- **法规原文**：来自中国政府官方网站（银保监会、央行、国家外汇管理局、国务院行政法规库等），均为现行有效版本，具体文号与生效日期见 `index.yaml`。
- **问答对**：基于上述法规原文，由 `qwen-plus` 模型辅助生成，并经人工逐条核验、修正，保证答案严格符合法规原文。每个 QA 标记来源文档，可追溯至具体条款。
- **评测数据集**：根据 RAG 应用特点设计了事实查询、推理查询、模糊查询、对抗查询四种类型，用于全面评估系统能力。

## 数据使用
- **向量库构建**：运行 `python update_vectordb.py`，将 `final_qa_dataset.yaml` 中的 QA 对切片后写入 Chroma（`./chroma_db`），供金融 RAG Skill 检索使用。
- **评测执行**：评测脚本可直接读取 `eval_dataset_v2.yaml` 进行全量指标计算。
- **扩展维护**：
  - 新增法规：将文本放入 `source_docs/`，更新 `index.yaml`，重新运行 `generate_qa.py` 生成候选 QA，然后走校验流程。
  - 纠正错误：直接修改 `final_qa_dataset.yaml` 或 `reviewed_qa.yaml`，重新运行 `update_vectordb.py` 更新向量库。

## 版本历史
- V0.1 (Day29) : 手工编写的金融问答对，12条评测种子数据。
- V0.5 (Day33) : 引入自动化流水线，最终 QA 数据集扩至 50+ 条，评测数据集扩至 60+ 条。