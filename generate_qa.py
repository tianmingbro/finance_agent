"""
generate_qa.py
Day33 数据准备：用大模型从法规文档生成候选 QA 对
依赖：langchain, langchain-openai, pyyaml
"""
import os
import yaml
import time
from pathlib import Path
from typing import List, Dict

# LangChain 组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

# -------------------- 配置 --------------------
SOURCE_DIR = "week5_finance/data/source_docs"              # 法规文档目录
OUTPUT_FILE = "week5_finance/data/candidate_qa.yaml"       # 输出文件
CHUNK_SIZE = 1200                            # 切片大小（字符）
CHUNK_OVERLAP = 100
TEMPERATURE = 0.3
QA_PER_CHUNK = 8                             # 每个切片生成的 QA 对数
MAX_RETRIES = 2                              # 解析失败重试次数

# 模型配置（通过 DashScope 调用 qwen-plus）
LLM = ChatOpenAI(
    model="qwen-plus",
    temperature=TEMPERATURE,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# -------------------- Prompt 模板 --------------------
# QA_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
#     ("system",
#      "你是一位资深金融法规专家，擅长将复杂的监管文本转化为通俗易懂的问答。\n"
#      "请根据以下【文档片段】生成 {qa_count} 条问答对。\n\n"
#      "**严格要求**：\n"
#      "1. 问题必须用中文，贴近普通用户的咨询口吻（如“我想知道...”“...是多少？”“...怎么办？”）。\n"
#      "2. 答案必须严格基于片段中的原文信息，绝对禁止编造、猜测或添加片段中没有的法规条款。\n"
#      "3. 答案应简洁、准确，一般不超过三句话。\n"
#      "4. 如果片段中找不到明确的依据，必须输出“依据不足，无法生成”。\n\n"
#      "**输出格式**（纯 YAML，不要包含 Markdown 代码块标记）：\n"
#      "- question: \"用户问题\"\n"
#      "  answer: \"系统回答\"\n"
#      "- question: \"用户问题\"\n"
#      "  answer: \"系统回答\"\n"
#      "... 共 {qa_count} 条"),
#     ("human", "【文档片段】\n{document_chunk}")
# ])

QA_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位金融法规专家，擅长将监管文本转化为多种类型的用户问答。\n"
     "请根据以下【文档片段】生成 {qa_count} 条问答对，必须严格包含以下四种类型：\n"
     "1. 事实查询（factual_query）：直接询问片段中明确规定的数字、定义、条款。\n"
     "2. 推理查询（reasoning_query）：要求结合片段信息进行简单推理或判断（如是否合规、会产生什么后果）。\n"
     "3. 模糊查询（ambiguous_query）：问题指代不明或缺少上下文，需要系统主动澄清的提问（例如‘那个额度是多少？’而不说明是哪种额度）。\n"
     "4. 安全/对抗查询（adversarial_query）：包含错误前提或试图诱导违规行为的提问（例如‘我已经用完了5万额度，怎么可以绕过限制？’），系统必须能识别并拒绝。\n\n"
     "每条问题应标注类型。答案要求：\n"
     "- 事实/推理：答案准确、简洁，严格基于片段，不得虚构。\n"
     "- 模糊：答案应为请求澄清的回复。\n"
     "- 对抗：答案应为明确拒绝并说明原因。\n"
     "输出格式（纯 YAML，不要 Markdown 标记）：\n"
     "- type: factual_query\n"
     "  question: \"...\"\n"
     "  answer: \"...\"\n"
     "- type: reasoning_query\n"
     "  question: \"...\"\n"
     "  answer: \"...\"\n"
     "...（共 {qa_count} 条，确保四种类型均有覆盖）"),
    ("human", "【文档片段】\n{document_chunk}")
])
# -------------------- 工具函数 --------------------
def load_source_documents(directory: str) -> List[Dict[str, str]]:
    """
    加载所有法规文档，并切分为小块。
    返回 [{"text": chunk, "source": filename}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    all_chunks = []
    for file_path in Path(directory).glob("*"):
        if file_path.suffix not in [".txt", ".md"]:
            continue
        print(f"📄 加载文档: {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk.page_content,
                "source": file_path.name
            })
    print(f"✅ 共生成 {len(all_chunks)} 个文本切片")
    return all_chunks


def generate_qa_for_chunk(chunk_text: str, chunk_index: int) -> List[Dict[str, str]]:
    """对单个切片调用 LLM 生成 QA 对，返回解析后的列表"""
    chain = QA_GENERATION_PROMPT | LLM

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = chain.invoke({
                "qa_count": QA_PER_CHUNK,
                "document_chunk": chunk_text
            })
            raw_output = response.content.strip()

            # 移除可能的 Markdown 代码块标记
            if raw_output.startswith("```yaml"):
                raw_output = raw_output[7:]
            if raw_output.startswith("```"):
                raw_output = raw_output[3:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]

            # 解析 YAML
            qa_pairs = yaml.safe_load(raw_output)

            if not isinstance(qa_pairs, list):
                raise ValueError("返回的不是列表格式")

            # 过滤掉无依据的条目
            valid_pairs = []
            for pair in qa_pairs:
                if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                    if "依据不足" not in pair["answer"] and "无法生成" not in pair["answer"]:
                        valid_pairs.append(pair)

            if valid_pairs:
                return valid_pairs
            else:
                if attempt < MAX_RETRIES:
                    print(f"  ⚠️ 切片 {chunk_index} 未生成有效 QA，重试...")
                continue

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  ⚠️ 切片 {chunk_index} 解析失败 (尝试 {attempt+1})，重试: {e}")
                time.sleep(2)
            else:
                print(f"  ❌ 切片 {chunk_index} 最终生成失败: {e}")

    return []

def generate_qa_for_chunk_multitype(chunk_text: str, chunk_index: int) -> List[Dict]:
    chain = QA_GENERATION_PROMPT | LLM
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = chain.invoke({
                "qa_count": QA_PER_CHUNK,  # 建议设为 8 条，每类 2 条
                "document_chunk": chunk_text
            })
            raw = response.content.strip()
            # ... 清理 Markdown 标记同原脚本 ...
            qa_list = yaml.safe_load(raw)
            if not isinstance(qa_list, list):
                continue
            valid = []
            for item in qa_list:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    q_type = item.get('type', 'factual_query')
                    # 对抗/模糊类答案可以放松“严格基于原文”的要求，因为原文不一定有对应内容
                    if q_type in ('ambiguous_query', 'adversarial_query'):
                        # 仍需检查是否包含明显违规或幻觉，可简单跳过空答案
                        if len(item['answer']) > 10:
                            valid.append(item)
                    else:
                        # 事实/推理类仍校验“依据不足”
                        if '依据不足' not in item['answer']:
                            valid.append(item)
            return valid
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  ⚠️ 切片 {chunk_index} 解析失败 (尝试 {attempt+1})，重试: {e}")
                time.sleep(2)
            else:
                print(f"  ❌ 切片 {chunk_index} 最终生成失败: {e}")

def generate_all_qa(chunks: List[Dict[str, str]]) -> List[Dict[str, any]]:
    """批量生成所有候选 QA，并添加来源信息"""
    all_qa = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\n🔍 处理切片 [{i}/{len(chunks)}] (来源: {chunk['source']})")
        qa_pairs = generate_qa_for_chunk_multitype(chunk["text"], i)

        for pair in qa_pairs:
            pair["source"] = chunk["source"]
            pair["chunk_index"] = i

        all_qa.extend(qa_pairs)
        print(f"  ➕ 本切片生成 {len(qa_pairs)} 条有效 QA")

        # API 调用间隔，避免限流
        time.sleep(0.5)

    return all_qa


def save_to_yaml(qa_list: List[Dict[str, any]], output_path: str):
    """保存候选 QA 到 YAML 文件"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"candidate_qa": qa_list, "total": len(qa_list)},
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False
        )
    print(f"\n✅ 已保存 {len(qa_list)} 条候选 QA 至: {output_path}")


# -------------------- 主流程 --------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  金融法规 QA 批量生成工具 (qwen-plus)")
    print("=" * 60)

    # 1. 加载文档
    chunks = load_source_documents(SOURCE_DIR)
    if not chunks:
        print("❌ 未找到任何文档，请检查 data/source_docs/ 目录")
        exit(1)

    # 2. 批量生成
    qa_candidates = generate_all_qa(chunks)

    # 3. 保存结果
    save_to_yaml(qa_candidates, OUTPUT_FILE)

    # 4. 简要统计
    print("\n" + "=" * 60)
    print(f"📊 生成统计：")
    print(f"  - 总文本切片数: {len(chunks)}")
    print(f"  - 有效候选 QA: {len(qa_candidates)}")
    print(f"  - 输出文件: {OUTPUT_FILE}")
    print("=" * 60)