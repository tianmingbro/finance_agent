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