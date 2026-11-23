"""Local stdio fixture: no network or real credentials."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("sector-test")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def forbidden() -> str:
    """This tool must never be exposed."""
    return "not allowed"


if __name__ == "__main__":
    server.run(transport="stdio")
