"""Tier-2: expose the SAME VisionService methods as an MCP server, so other MCP
clients (Claude Desktop, a teammate's agent) can use the perception too. One impl,
two surfaces (DRY) -- the board's own agent still calls vision in-process.

Run: python -m voice_agent.vision_mcp_server         (stdio)
Requires: pip install fastmcp"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_service import VisionService   # noqa: E402


def build_server(vs=None):
    from fastmcp import FastMCP             # lazy: only needed to run the server
    vs = vs or VisionService()
    mcp = FastMCP("ece180-vision")

    # register the same read-only methods; FastMCP infers schema from signature/docstring
    mcp.tool()(vs.who_is_in_view)
    mcp.tool()(vs.describe_scene)
    mcp.tool()(vs.get_person_emotion)
    mcp.tool()(vs.emotion_timeline)
    mcp.tool()(vs.presence_events)
    mcp.tool()(vs.list_enrolled)
    return mcp, vs


def main():
    mcp, vs = build_server()
    vs.start()
    try:
        mcp.run()          # stdio by default; mcp.run(transport="streamable-http", ...) to host
    finally:
        vs.stop()


if __name__ == "__main__":
    main()
