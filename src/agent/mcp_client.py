"""
mcp_client.py
通过 langchain-mcp-adapters 获取 MCP 服务器中的工具。
"""
import os
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_finance_mcp_tools():
    """连接到金融检索 MCP 服务器，返回 LangChain 工具列表"""
    # 配置服务器连接：通过 stdio 启动 Python 脚本
    current_dir = Path(__file__).resolve().parent          # src/agent
    project_root = current_dir.parent.parent               # finance_agent
    server_script = project_root / "servers" / "mcp_finance_server.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    servers_config = {
        "finance_search": {
            "command": "python",
            "args": [str(server_script)],
            "transport": "stdio",
            "cwd": str(project_root),  
            "env": env,             # 工作目录设为项目根，确保相对路径有效
        }
    }
    client = MultiServerMCPClient(servers_config)
    # 获取所有工具（自动转换为 LangChain BaseTool）
    tools = client.get_tools()
    return await tools