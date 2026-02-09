"""
Deep Search Agent - Remote MCP Server (SSE Transport)
=====================================================
HTTP-based MCP server for remote deployment on Coolify/Docker.
Exposes research tools via Server-Sent Events protocol.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ============================================================================
# SETUP
# ============================================================================

server = Server("deep-search-agent")
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def get_report_files() -> List[str]:
    """Get all saved report filenames sorted by name."""
    if not REPORTS_DIR.exists():
        return []
    return sorted([f.name for f in REPORTS_DIR.glob("*.md")])


# ============================================================================
# TOOL HANDLERS
# ============================================================================

async def handle_research(arguments: Any) -> list[TextContent]:
    """Run deep research via async subprocess."""
    query = arguments.get("query")
    max_iterations = arguments.get("max_iterations", 5)
    providers = arguments.get("providers")

    cmd = [sys.executable, str(BASE_DIR / "main.py")]
    if providers:
        cmd.extend(["--providers", providers])
    cmd.extend(["--iterations", str(max_iterations)])
    cmd.append(query)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR),
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=600
        )

        reports = get_report_files()
        if reports:
            latest = reports[-1]
            filepath = REPORTS_DIR / latest
            content = filepath.read_text(encoding="utf-8")
            return [TextContent(
                type="text",
                text=f"# Research Complete\n\n**Query:** {query}\n**Report:** {latest}\n\n---\n\n{content}",
            )]

        out = (stdout.decode() if stdout else "")[-2000:]
        err = (stderr.decode() if stderr else "")[-2000:]
        return [TextContent(
            type="text",
            text=f"Research completed but no report file found.\n\nStdout:\n{out}\n\nStderr:\n{err}",
        )]

    except asyncio.TimeoutError:
        return [TextContent(type="text", text="Research timed out after 10 minutes. Try fewer iterations or a simpler query.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error during research: {str(e)}")]


async def handle_list_reports(arguments: Any) -> list[TextContent]:
    """List saved research reports."""
    limit = arguments.get("limit", 10)
    reports = get_report_files()

    if not reports:
        return [TextContent(type="text", text="No reports found. Run a research query first.")]

    lines = [f"# Saved Reports ({len(reports)} total)\n"]
    for i, filename in enumerate(reversed(reports[-limit:]), 1):
        filepath = REPORTS_DIR / filename
        try:
            first_lines = filepath.read_text(encoding="utf-8").split("\n")[:10]
            query = "Unknown"
            for line in first_lines:
                if line.startswith("# Research Report:"):
                    query = line.replace("# Research Report:", "").strip()
                    break
            lines.append(f"{i}. **{filename}** - {query}")
        except Exception:
            lines.append(f"{i}. **{filename}**")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_report(arguments: Any) -> list[TextContent]:
    """Get full content of a specific report."""
    filename = arguments.get("filename")
    filepath = REPORTS_DIR / filename

    if not filepath.exists():
        reports = get_report_files()
        similar = [r for r in reports if filename.lower() in r.lower()]
        if similar:
            return [TextContent(type="text", text=f"File not found: {filename}\n\nDid you mean: {similar[0]}?")]
        return [TextContent(
            type="text",
            text=f"File not found: {filename}\n\nAvailable: {', '.join(reports[:5]) if reports else 'none'}",
        )]

    content = filepath.read_text(encoding="utf-8")
    return [TextContent(type="text", text=content)]


# ============================================================================
# MCP TOOL DEFINITIONS
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="research",
            description=(
                "Run deep research on any topic using multi-provider search agent. "
                "Uses DeepSeek-R1 for routing, Exa/Tavily/Serper for search, "
                "and synthesizes a comprehensive academic report. "
                "Supports auto-language detection and translation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question or topic (any language)",
                    },
                    "max_iterations": {
                        "type": "number",
                        "description": "Max research iterations (default: 5, max: 10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "providers": {
                        "type": "string",
                        "description": "Force specific providers (comma-separated): exa,tavily,serper. Bypasses auto-router.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_reports",
            description="List all saved research reports with filenames and queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Max reports to show (default: 10)",
                        "default": 10,
                    }
                },
            },
        ),
        Tool(
            name="get_report",
            description="Get full content of a saved research report by filename",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Report filename from list_reports",
                    }
                },
                "required": ["filename"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    handlers = {
        "research": handle_research,
        "list_reports": handle_list_reports,
        "get_report": handle_get_report,
    }
    handler = handlers.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments)


# ============================================================================
# SSE TRANSPORT & HTTP APP
# ============================================================================

sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


async def health(request):
    """Health check endpoint for Docker/Coolify."""
    reports = get_report_files()
    keys_configured = all([
        os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("TAVILY_API_KEY") or os.getenv("SERPER_API_KEY"),
    ])
    return JSONResponse({
        "status": "ok" if keys_configured else "degraded",
        "service": "deep-search-agent",
        "reports_count": len(reports),
        "keys_configured": keys_configured,
    })


app = Starlette(
    routes=[
        Route("/health", endpoint=health),
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Deep Search Agent MCP Server (SSE)")
    print(f"Listening on {host}:{port}")
    print(f"SSE endpoint: http://{host}:{port}/sse")
    print(f"Health check: http://{host}:{port}/health")
    uvicorn.run(app, host=host, port=port)
