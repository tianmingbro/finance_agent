"""
update_vectordb.py
Day33 数据准备：将 final_qa_dataset.yaml 写入 Chroma 向量库，
并扩充评测数据集，运行金融 RAG 质量抽查。
"""
import hashlib
import re
import yaml
import os
import sys
import random
from pathlib import Path
from typing import List, Dict

# LangChain 组件（与 FinancialRAGSkill 保持一致）
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 导入金融 Skill（用于质量验证）
from financial_rag_skill import FinancialRAGSkill

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models","text2vec-base-chinese","Jerry0", "text2vec-base-chinese")

# -------------------- 配置 --------------------
FINAL_QA_FILE = "week5_finance/data/final_qa_dataset.yaml"      # 输入：最终 QA 数据集
EVAL_DATASET_FILE = "week5_finance/data/eval_dataset.yaml"      # 评测数据集（Day29 骨架）
EXPANDED_EVAL_FILE = "week5_finance/data/eval_dataset_v2.yaml"  # 扩充后的评测数据集
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")  # Chroma 向量库目录（与 skill 同目录，确保路径一致）
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TEST_QUESTIONS = [
    "存款保险的最高偿付限额是多少？",
    "LPR改革后房贷利率如何变化？",
    "反洗钱法要求金融机构履行哪些义务？",
    "个人外汇便利化额度用完了还能继续购汇吗？",
    "资本充足率不达标的银行会有什么后果？",
]


def load_final_qa_dataset(file_path: str) -> List[Dict[str, str]]:
    """加载 final_qa_dataset.yaml"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 兼容不同的键名
    if isinstance(data, list):
        return data
    for key in ["final_qa_dataset", "reviewed_qa", "candidate_qa"]:
        if key in data:
            return data[key]
    return []


def build_documents_from_qa(qa_list: List[Dict[str, str]]) -> List[Document]:
    """
    将 QA 对转换为 LangChain Document 列表。
    每个 QA 对格式化为：问题：...\n答案：...
    """
    docs = []
    for item in qa_list:
        # 优先使用标准化字段 query/answer，兼容 question/answer
        query = item.get("query") or item.get("question", "")
        answer = item.get("answer", "")
        if not query or not answer:
            continue
        content = f"问题：{query}\n答案：{answer}"
        metadata = {
            "source": item.get("source", "final_qa_dataset"),
            "category": item.get("category", "unknown"),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def update_vectorstore(docs: List[Document]):
    """将文档切片后写入 Chroma（增量添加，不清空旧数据）"""
    print(f"📄 准备将 {len(docs)} 条 QA 对写入向量库...")

    # 切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"🔪 切片得到 {len(chunks)} 个片段")

    # 初始化 Embedding
    embedding = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        
    )

    # 连接 Chroma（若目录不存在会创建）
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding,
        collection_name="finance_qa",
    )

    # # 添加文档
    # vectorstore.add_documents(chunks)
    # print(f"✅ 已成功写入向量库 (Chroma @ {CHROMA_PERSIST_DIR})")
    # 为每个文档生成基于内容的唯一 ID
    new_docs = []
    new_ids = []
    for doc in docs:
        doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
        # 检查是否已存在（可选，upsert 本身就会覆盖，不会重复）
        existing = vectorstore.get(doc_id)
        if existing and existing["ids"]:
            continue  # 已存在，跳过（如不需覆盖则保留；若内容可能更新则改用 upsert）
        new_docs.append(doc)
        new_ids.append(doc_id)

    if new_docs:
        vectorstore.add_documents(documents=new_docs, ids=new_ids)
        print(f"✅ 新增 {len(new_docs)} 个切片，已跳过 {len(docs)-len(new_docs)} 个重复")
    else:
        print("✅ 所有切片均已存在，无重复添加")


import re

def classify_query(query: str) -> str:
    # ====== 对抗类（试探规则、诱导违规、挑战系统） ======
    adversarial_patterns = [
        r"(真的|确实|是不是)?.{0,5}(取消|废除|失效|不管用|改了|变了|不强制|没有执行)",  # 质疑法规存在/有效性
        r"(绕过|避开|不按|钻.*空子|变通|有其他办法|可以不.{0,3}吗|能偷偷|能私下)",    # 试探违规路径
        r"(多开.*账户|多个账户.*额度|分开.*买|分拆|化整为零|规避.*限额)",            # 试探额度规避
        r"(随便|可以随便|不查|不会被发现|怎么操作.*才不|有没有办法.*不)",            # 直接寻求违规操作
        r"(骗|假造|伪造|作假|顶替|冒充)",                                        # 明显违法字眼
        r"(忽略|别管|别听|不要管).{0,5}(之前|刚才|上面|规定|指令)",                # 指令注入
        r"(真的.*吗|是不是.*可以|难道.*不|不是说|可是.*规定)",                     # 否定/质疑语调（弱对抗）
        r"(洗钱|逃汇|骗汇|非法买卖|地下钱庄)",                                     # 违法活动
        r"(反洗钱.*太麻烦|不想.*客户调查|能不能不提供|信息可以.*不)",               # 规避合规
    ]

    for pat in adversarial_patterns:
        if re.search(pat, query):
            return "adversarial_query"

    # ====== 模糊类（指代不明、需要澄清） ======
    ambiguous_keywords = ["那个", "这个", "指什么", "什么意思", "即什么", "你说的是"]
    if any(kw in query for kw in ambiguous_keywords):
        return "ambiguous_query"

    # ====== 推理类（需要解释、判断、对比） ======
    reasoning_keywords = [
        "为什么", "怎么办", "是否合规", "如果", "如何判断",
        "会有什么", "怎么判断", "能全赔", "还安全", "如何变化",
        "有什么影响", "怎么算", "可以吗", "能不能", "会怎样",
        "有什么后果", "需要满足", "什么条件", "区别", "差异"
    ]
    if any(kw in query for kw in reasoning_keywords):
        return "reasoning_query"

    # ====== 事实类（默认） ======
    return "factual_query"

def expand_eval_dataset(final_qa_list: List[Dict], eval_path: str, output_path: str):
    # 加载原评测数据集
    if Path(eval_path).exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = yaml.safe_load(f)
    else:
        eval_data = {"categories": []}

    if "categories" not in eval_data:
        eval_data["categories"] = []
    categories = {cat["category"]: cat for cat in eval_data["categories"]}

    target = 15  # 每类最少 15 条
    required = {
        "factual_query": "事实查询",
        "reasoning_query": "推理查询",
        "ambiguous_query": "模糊查询",
        "adversarial_query": "对抗查询",
    }

    # 确保所有类别都存在
    for c_name, desc in required.items():
        if c_name not in categories:
            categories[c_name] = {"category": c_name, "description": desc, "entries": []}

    # 当前各类别数量
    current_counts = {c: len(categories[c]["entries"]) for c in required}

    # 从 final QA 中分配新条目，直至各类别均达标
    new_entries = []
    for item in final_qa_list:
        query = item.get("query") or item.get("question", "")
        answer = item.get("answer", "")
        if not query or not answer:
            continue

        # 自动分类
        cat = classify_query(query)
        if current_counts.get(cat, 0) < target:
            new_entries.append({
                "id": f"new_{len(new_entries):03d}",
                "query": query,
                "expected_answer": answer,
                "source": item.get("source", ""),
            })
            current_counts[cat] += 1

        # 如果所有类别都已达标，提前退出
        if all(v >= target for v in current_counts.values()):
            break

    # 将新条目追加到对应类别
    for entry in new_entries:
        # 需要根据内容重新确定类别，这里简单保留分类
        cat = classify_query(entry["query"])
        categories[cat]["entries"].append(entry)

    eval_data["categories"] = list(categories.values())

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(eval_data, f, allow_unicode=True, sort_keys=False)

    print(f"✅ 评测数据集已扩充，保存至 {output_path}")
    for cat in eval_data["categories"]:
        print(f"   - {cat['category']}: {len(cat['entries'])} 条")


def test_rag_quality(questions: List[str]):
    """运行金融 RAG 抽查，验证回答质量"""
    print("\n" + "=" * 60)
    print("  金融 RAG 质量抽查（更新后）")
    print("=" * 60)
    skill = FinancialRAGSkill()
    for q in questions:
        print(f"\n❓ 问题: {q}")
        try:
            answer = skill.run(q)
            print(f"🤖 回答: {answer[:200]}...")
            # 简单检查是否包含预期关键词（不严格）
            if any(kw in answer for kw in ["%", "万", "元", "不得", "应当", "规定"]):
                print("   ✅ 回答可能有效")
            else:
                print("   ⚠️ 回答疑似不完整")
        except Exception as e:
            print(f"   ❌ 检索或生成失败: {e}")
    print("=" * 60)


# -------------------- 主流程 --------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Day33 向量库更新 & 评测数据集扩充")
    print("=" * 60)

    # 1. 加载最终 QA 数据集
    if not Path(FINAL_QA_FILE).exists():
        print(f"❌ 未找到 {FINAL_QA_FILE}，请先完成人工校验并生成 final_qa_dataset.yaml")
        sys.exit(1)
    qa_list = load_final_qa_dataset(FINAL_QA_FILE)
    print(f"📋 加载最终 QA 数据集，共 {len(qa_list)} 条")

    # 2. 转换并写入向量库
    docs = build_documents_from_qa(qa_list)
    if docs:
        update_vectorstore(docs)
    else:
        print("⚠️ 没有有效文档，跳过向量库更新")

    # 3. 扩充评测数据集
    expand_eval_dataset(qa_list, EVAL_DATASET_FILE, EXPANDED_EVAL_FILE)

    from dotenv import load_dotenv
    load_dotenv()  # 从 .env 文件加载环境变量

    # 4. 质量验证
    if os.environ.get("DASHSCOPE_API_KEY"):
        test_rag_quality(TEST_QUESTIONS)
    else:
        print("⚠️ 未设置 DASHSCOPE_API_KEY，跳过质量抽查。请设置后手动运行验证。")