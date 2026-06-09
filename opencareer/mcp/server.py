"""
MCP Server for OpenCareer — exposes career companion tools via MCP protocol.

Uses FastMCP with Streamable HTTP transport so langchain-mcp-adapters
can connect via MultiServerMCPClient.

Run:
    python -m opencareer.mcp.server
or:
    python opencareer/mcp/server.py
"""

import logging
import functools
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from opencareer.mcp.prompts import create_skill_registry
from opencareer.mcp.tools.resume_tool import resume_skill as _resume_skill_impl

logger = logging.getLogger("opencareer.mcp.server")

# ------------------------------------------------------------------
# FastMCP application
# ------------------------------------------------------------------

mcp = FastMCP(
    name="OpenCareer MCP Server",
    instructions="OpenCareer tools",
    host="127.0.0.1",
    port=8001,
)

# Register resume_skill as MCP tool (preserves type annotations)
mcp.tool()(_resume_skill_impl)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main():
    """Start the MCP server."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting OpenCareer MCP Server...")

    # Run with Streamable HTTP transport
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
