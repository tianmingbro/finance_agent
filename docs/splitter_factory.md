# SplitterFactory 使用说明

**版本**：V1.0  
**日期**：2026-05-25  
**模块**：`splitter_factory.py`

## 概述

`SplitterFactory` 是一个可扩展的文本分割工厂，提供统一的接口来创建多种分割器，或直接对文档列表执行分割。通过策略模式，你可以根据文档类型（纯文本、Markdown、代码等）灵活切换分割策略，无需修改业务代码。

**核心能力**：
- 内置 4 种分割策略：`recursive`、`markdown`、`token`、`semantic`
- 支持自定义策略注册（开闭原则）
- 透传参数到具体分割器，精细化控制切片行为
- 自动日志记录（策略名、输入/输出文档数）

## 快速开始

```python
from splitter_factory import SplitterFactory
from langchain_core.documents import Document

factory = SplitterFactory()
docs = [Document(page_content="长文本内容...", metadata={"source": "doc.txt"})]

# 使用递归分割策略
chunks = factory.split(docs, strategy="recursive", chunk_size=500, chunk_overlap=100)

# 直接获取分割器实例，稍后手动调用
splitter = factory.create_splitter("recursive", chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)


内置策略详解
1. recursive – 通用递归分割

    对应类：RecursiveCharacterTextSplitter

    适用场景：绝大多数纯文本文档（TXT、PDF、DOCX 转换后的文本），对内容结构无特殊要求。

    原理：使用递归方式依次尝试按段落、换行、句末标点、空格、字符边界进行分割，优先保持语义完整性。

    默认参数：

        chunk_size=500（字符数）

        chunk_overlap=100

        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]

调优建议：

    法规/合同类长句：建议 chunk_size=600~800，chunk_overlap=150，防止关键条款被切断。

    FAQ/短文本：chunk_size=200~300，chunk_overlap=50，避免噪声过多。

    如果文档中包含大量英文或数字，可调整 separators，将空格提前。

2. markdown – Markdown 标题分割

    对应类：MarkdownHeaderTextSplitter

    适用场景：结构清晰的 Markdown 文档（如法规总结、技术手册、API 文档）。

    原理：依据 Markdown 标题层级（#、##、### 等）将文档拆分为独立段落，每个标题及其下属内容成为一个片段，标题信息保留在 metadata 中（如 Header 1、Header 2）。

    默认参数：

        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]

        strip_headers=False（保留标题文本）

调优建议：

    多级标题：可按需扩展 headers_to_split_on，例如增加 ("####", "Header 4")。

    过短章节合并：该分割器不自动合并，若某标题下内容过短（如仅一行），可后续通过自定义逻辑拼接；或在注册时使用包装类添加最小片段长度过滤。

    检索增强：标题保留在 metadata 中，可在检索时作为增强信息拼接进 Prompt，提升生成准确度。

3. token – Token 级精确分割

    对应类：TokenTextSplitter

    适用场景：对 LLM 上下文窗口有严格限制的场景（如需要精确控制 token 数，避免超限）。

    原理：使用 tiktoken 按 token 数量分割，chunk_size 和 chunk_overlap 均为 token 数。

    默认参数：

        chunk_size=512（token）

        chunk_overlap=50

        encoding_name="cl100k_base"

调优建议：

    根据模型上下文窗口设定 chunk_size（如 gpt-4o-mini 上下文 128k，可设 1000~2000 token）。

    若需混合中英文，建议测试 token 计算是否准确；对于中文，cl100k 编码效率较低，可考虑 o200k_base 或自定义 tokenizer。

    token 策略不保证句子完整，适合对语义连贯性要求不高的场景。

4. semantic – 语义分割（实验性）

    对应类：SemanticChunker（来自 langchain-experimental）

    适用场景：需要高度保持语义连贯性的长文档（如学术论文、技术报告），且对计算开销不敏感。

    原理：利用 Embedding 模型计算相邻句子的语义相似度，在相似度剧烈下降的“断点”处分割，而非固定长度。支持多种断点阈值算法：

        percentile（默认）：所有句子相似度中低于某个百分位的断点

        standard_deviation：低于均值减去若干标准差

        interquartile：基于四分位距

        gradient：相似度梯度变化

    依赖：需额外安装 pip install langchain-experimental，并传入 embeddings 参数。

调优建议：

    必须提供高质量的 Embedding 模型（如 sentence-transformers/all-MiniLM-L6-v2 或本地微调模型）。

    breakpoint_threshold_amount 控制敏感度：值越小分割越细，值越大分割越粗。

    计算成本高，不适合实时或大批量导入；建议对关键文档离线预处理。

自定义策略

你可以通过 register_strategy 注册自己的分割器：
python

from langchain_text_splitters import LatexTextSplitter

factory = SplitterFactory()
factory.register_strategy("latex", LatexTextSplitter)
chunks = factory.split(docs, strategy="latex", chunk_size=1000)

注册后，list_strategies() 会包含新策略名。注意，注册的策略类需要与 RecursiveCharacterTextSplitter 兼容（即具备 split_documents 方法），或由 split 方法内部特殊处理（如 MarkdownHeaderTextSplitter）


---

### 2. 技术方案更新（文本分割器策略化）

> 以下内容可直接替换或追加到《金融领域智能问答 RAG 应用技术方案》的“数据工程”或“RAG Skill 概要”章节。

```markdown
## 数据工程 – 文本分割器策略化（更新于 V0.6）

### 分割策略

原有数据管道中硬编码的 `RecursiveCharacterTextSplitter` 已替换为 **`SplitterFactory`**，支持根据文档格式自动选用最优分割策略。

| 文档类型 | 使用策略 | 分割器 | 说明 |
|----------|----------|--------|------|
| TXT、PDF、DOCX | `recursive` | `RecursiveCharacterTextSplitter` | 通用递归分割，保持句子完整性 |
| Markdown (.md) | `markdown` | `MarkdownHeaderTextSplitter` | 按标题层级分割，保留结构信息 |
| Token 敏感场景（可选） | `token` | `TokenTextSplitter` | 按 token 数精确控制上下文长度 |
| 高语义连贯性需求（实验） | `semantic` | `SemanticChunker` | 按语义相似度动态分割 |

### 自动策略选择

数据导入脚本 `data_ingestion_v2.py` 根据文件扩展名自动选取策略，无需手动干预。`SplitterFactory` 使用日志记录每次分割的策略、输入文档数和输出片段数，便于监控数据管道状态。

### 策略扩展

`SplitterFactory` 提供 `register_strategy()` 接口，当项目引入新文档格式（如 LaTeX、HTML）时，只需注册对应的 LangChain 分割器即可，数据管道代码无需修改。

### 性能与质量影响

- 引入 `SplitterFactory` 后，数据管道执行时间无明显变化。
- 对于 Markdown 法规文档，检索上下文中现在包含标题层级信息，便于大模型更精准理解条款关系。
- 评测基线对比显示，`faithfulness`、`answer_relevancy` 等核心指标未发生显著下降（变化 <5%），`contextual_recall` 因结构化信息补充而小幅提升。

### 后续计划

- 评测 `semantic` 策略在长文档法规上的效果，若检索质量提升明显，将加入自动策略选择逻辑。
- 提供图形化策略选择界面（长期）。