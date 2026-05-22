markdown

# 金融法规问答数据集

## 概述
本目录包含用于金融 RAG（检索增强生成）应用的所有数据集与资源，涵盖原始法规文档、生成的问答对、评测数据集以及向量库文件。数据遵循“采集→生成→校验→入库”的自动化流水线，确保知识覆盖与质量可控。

## 目录结构

data/
├── source_docs/ # 原始法规文档（多格式）
│ ├── index.yaml # 文档索引
│ ├── *.txt # TXT 文档
│ ├── *.md # Markdown 文档
│ ├── *.pdf # PDF 文档
│ └── *.docx # Word 文档
├── test_docs/ # 用于测试的多格式样本
├── candidate_qa.yaml # 自动生成的候选 QA（待校验）
├── reviewed_qa.yaml # 人工校验通过的 QA
├── final_qa_dataset.yaml # 最终标准化数据集（≥50条）
├── eval_dataset.yaml # 评测数据集骨架
├── eval_dataset_v2.yaml # 扩充后的评测数据集
└── holdout_dataset.yaml # 独立 holdout 集


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


---

## 如何添加新文档

### 1. 收集法规原文

从官方渠道（国家金融监督管理总局、中国人民银行、国家外汇管理局等）获取法规文本，支持以下格式：

| 格式 | 后缀 | 说明 |
|------|------|------|
| 纯文本 | `.txt` | 通用格式，推荐使用 UTF-8 编码 |
| Markdown | `.md` | 带有标题层级的文本，便于结构化解析 |
| PDF | `.pdf` | 官方发布的 PDF 文件，注意文本层必须可复制 |
| Word 文档 | `.docx` | Microsoft Word 格式，支持段落样式 |

## 如何增加新文档格式

本文档解析器使用 `LOADER_MAP` 字典将文件后缀映射到 LangChain 加载器。若要增加对新格式（如 `.epub`、`.rtf`）的支持，请按以下步骤操作：

### 1. 安装依赖
LangChain 社区为许多格式提供了加载器，需安装相应的第三方库。
例如，增加 `.epub` 支持需安装 `pypandoc` 和 `unstructured[epub]`：
```bash
pip install pypandoc "unstructured[epub]"

- **编码要求**：文本类文件（TXT、MD）必须使用 **UTF-8** 编码；PDF 必须有可复制的文本层，不支持扫描件；DOCX 仅提取段落文本

### 2. 放置文档

将文件放入 `data/source_docs/` 目录下。命名规范：`{法规简称}_{年份}.{后缀}`，例如：
capital_management_measures_2024.txt
deposit_insurance_regulation_2015.pdf


如果一份法规有多个格式版本，建议保留最权威的一个格式，测试用多格式样本放入 `data/test_docs/`。

### 3. 更新索引

在 `data/source_docs/index.yaml` 中添加新文档条目，至少包含：
- `filename`：实际文件名
- `title`：法规标题
- `topics`：覆盖的知识点标签

### 4. 运行数据流水线

```bash
# 生成候选 QA（基于新文档）
python generate_qa.py

# 人工校验后更新向量库
python update_vectordb.py

# 扩充评测数据集
python build_final_dataset.py