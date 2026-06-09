"""MCP Server for OpenCareer - exposes career tools via MCP protocol.

Uses FastMCP with Streamable HTTP transport so langchain-mcp-adapters
can connect via MultiServerMCPClient.
"""

import logging
import functools
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

from .prompts import create_skill_registry
from .tools.resume_tool import resume_skill as _resume_skill_impl

logger = logging.getLogger("careers.mcp.server")

# ------------------------------------------------------------------
# FastMCP application
# ------------------------------------------------------------------

if FASTMCP_AVAILABLE:
    try:
        mcp = FastMCP(
            name="OpenCareer MCP Server",
            description="AI Career Companion tools",
            host="127.0.0.1",
            port=8001,
        )
    except TypeError:
        mcp = FastMCP(
            name="OpenCareer MCP Server",
            host="127.0.0.1",
            port=8001,
        )

    mcp.tool()(_resume_skill_impl)


def main():
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not FASTMCP_AVAILABLE:
        logger.error("FastMCP not available. Run: pip install fastmcp")
        sys.exit(1)

    logger.info("Starting OpenCareer MCP Server...")

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()