LoaderFacade 使用说明
1. 概述

LoaderFacade 是金融 RAG 项目中的统一文档加载门面，旨在屏蔽底层不同格式文档加载器的差异，为数据工程管道提供简单一致的接口。
当前支持格式：TXT、PDF、DOCX，并可动态扩展新格式。
2. 设计决策
2.1 为什么使用门面模式（Facade Pattern）？

    屏蔽复杂性：不同文档格式依赖不同的第三方库（PyPDFLoader、Docx2txtLoader、TextLoader），门面模式将这些加载器封装在一个类中，对外暴露单一的 load() 方法。

    开闭原则：通过 register_loader() 方法，无需修改原有代码即可添加新格式支持，符合“对扩展开放，对修改关闭”的设计原则。

    统一元数据：门面自动补全 metadata 中的 source 字段，确保下游向量库存储一致性。

    易于测试与替换：整个加载逻辑集中在一处，便于单元测试，将来切换底层库也只需改门面内部。

2.2 架构简图
text

用户/管道脚本
     │
     ▼
LoaderFacade.load(file_path)
     │
     ├─ 扩展名识别
     ├─ 加载器路由 (字典映射)
     │    ├─ .txt  → TextLoader
     │    ├─ .pdf  → PyPDFLoader
     │    ├─ .docx → Docx2txtLoader
     │    └─ .xyz  → 自定义加载器 (通过 register_loader)
     │
     ▼
统一返回 List[langchain_core.documents.Document]

3. 快速开始
3.1 安装依赖
bash

pip install langchain langchain-community pypdf docx2txt

3.2 基本用法
python

from loader_facade import LoaderFacade

facade = LoaderFacade()
documents = facade.load("data/source_docs/capital_rules.pdf")

for doc in documents:
    print(doc.page_content[:200])

3.3 批量加载目录

配合 data_ingestion.py 脚本，可以一键导入 data/source_docs/ 下所有支持格式的文档：
bash

python data_ingestion.py

该脚本自动完成：
加载 → 切片 → 写入 Chroma 向量库 → 检索验证
4. 扩展新格式

通过 register_loader() 方法注册新的加载器类。加载器必须继承自 langchain_community.document_loaders.base.BaseLoader。

示例：注册 EPUB 加载器
python

from langchain_community.document_loaders import UnstructuredEPubLoader
from loader_facade import LoaderFacade

facade = LoaderFacade()
facade.register_loader(".epub", UnstructuredEPubLoader)
documents = facade.load("book.epub")

    注意：扩展名必须以 . 开头，大小写不敏感。

5. 错误处理
场景	异常类型	说明
文件不存在	FileNotFoundError	检查路径是否正确
不支持的文件格式	ValueError	异常消息中会列出当前支持的扩展名列表
文件损坏或不兼容	底层加载器抛出	根据具体加载器文档处理
6. 测试

项目包含完整的 pytest 测试套件，位于 tests/test_loader_facade.py。执行：
bash

pytest tests/test_loader_facade.py -v

测试覆盖：TXT/PDF/DOCX 加载、不支持格式异常、返回类型验证、自定义加载器注册。
7. 设计日志

    2026-05-24：初始版本，支持 TXT、PDF、DOCX，基于 TDAD 开发，所有测试先于实现。

    待办：未来可根据需要加入 OCR 模式（扫描版 PDF）、云存储文件流式加载等高级特性。

