"""
mcp_client.py
通过 langchain-mcp-adapters 获取 MCP 服务器中的工具。
"""
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_finance_mcp_tools():
    """连接到金融检索 MCP 服务器，返回 LangChain 工具列表"""
    # 配置服务器连接：通过 stdio 启动 Python 脚本
    servers_config = {
        "finance_search": {
            "command": "python",
            "args": ["mcp_finance_server.py"],
            "transport": "stdio",
        }
    }
    client = MultiServerMCPClient(servers_config)
    # 获取所有工具（自动转换为 LangChain BaseTool）
    tools = client.get_tools()
    return await tools