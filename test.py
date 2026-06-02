# mcp_server.py 临时简化版
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test")

@mcp.tool()
def hello(name: str = "World") -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="stdio")