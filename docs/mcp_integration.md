# MCP 工具集成文档

## 概述
本项目的金融文档检索能力已通过 **MCP (Model Context Protocol)** 标准化。任何支持 MCP 的客户端（如 Claude Desktop、LangChain Agent）均可安全调用我们的检索和主题查询工具。

## MCP 服务器配置
服务器以 `stdio` 传输方式运行，启动命令：
```bash
python mcp_finance_server.py

1. 在 Claude Desktop 中连接

在 claude_desktop_config.json 中添加：
json

{
  "mcpServers": {
    "finance-search": {
      "command": "python",
      "args": ["mcp_finance_server.py"]
    }
  }
}

2. 在 LangChain Agent 中连接
python

from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "finance": {
        "command": "python",
        "args": ["mcp_finance_server.py"],
        "transport": "stdio",
    }
})
tools = client.get_tools()

可用工具列表
search_finance_docs

检索金融法规相关文档，支持混合检索（语义 + 关键词）。

    参数：

        query (string)：查询文本

        top_k (int)：返回文档数量，默认 4

    返回：JSON 字符串
    json

    {
      "documents": [
        {
          "index": 1,
          "content": "文档片段...",
          "source": "capital.md"
        }
      ],
      "sources": ["capital.md"]
    }

list_available_topics

列出当前知识库覆盖的主题。

    参数：无

    返回：JSON 字符串
    json

    {
      "topics": ["LPR", "外汇管理", "存款保险", "资本充足率"]
    }

测试与调试

使用 MCP Inspector 实时测试：
bash

mcp dev mcp_finance_server.py

或运行单元测试：
bash

pytest test_mcp_tools.py -v