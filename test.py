from langchain_chroma import Chroma
from financial_rag_skill import ResourceManager

# 临时代码
mgr = ResourceManager()
mgr.load_resources()
store = mgr._vectorstore

# 获取随机几条
results = store.similarity_search("反洗钱", k=3)
for doc in results:
    print(doc.page_content[:200])