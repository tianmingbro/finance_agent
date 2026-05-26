# tests/conftest.py 或直接写在测试文件中
from langchain_core.documents import Document

def get_sample_documents():
    """返回 3 个简单文档，用于向量库增删查测试"""
    return [
        Document(page_content="核心一级资本充足率不得低于5%。", 
                 metadata={"source": "test_capital.txt"}),
        Document(page_content="存款保险最高偿付限额为50万元。", 
                 metadata={"source": "test_deposit.txt"}),
        Document(page_content="个人购汇年度便利化额度为5万美元。", 
                 metadata={"source": "test_forex.txt"}),
    ]
import pytest

@pytest.fixture
def sample_documents():
    """提供一组金融法规文档用于构建 BM25 索引和向量检索器"""
    return [
        Document(page_content="存款保险最高偿付限额为人民币50万元。", metadata={"source": "deposit"}),
        Document(page_content="商业银行核心一级资本充足率不得低于5%。", metadata={"source": "capital"}),
        Document(page_content="个人每年便利化购汇额度为等值5万美元。", metadata={"source": "forex"}),
        Document(page_content="LPR由各报价行按公开市场操作利率加点形成。", metadata={"source": "lpr"}),
        Document(page_content="反洗钱法要求金融机构建立客户尽职调查制度。", metadata={"source": "aml"}),
    ]