"""
Deep Search Agent - MCP Server (Python)
==================================
Model Context Protocol server for universal LLM access.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# SETUP
# ============================================================================

server = Server("deep-search-agent")
REPORTS_DIR = Path("reports")

# Make sure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)


def get_report_files() -> List[str]:
    """Get all saved report filenames."""
    if not REPORTS_DIR.exists():
        return []
    return sorted([f.name for f in REPORTS_DIR.glob("*.md")])


# ============================================================================
# TOOL HANDLERS
# ============================================================================

async def handle_research(arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle research tool call."""
    query = arguments.get("query")
    max_iterations = arguments.get("max_iterations", 5)
    providers = arguments.get("providers")

    print(f"\n[MCP] Research: {query}")
    if providers:
        print(f"[MCP] Forced providers: {providers}")

    try:
        cmd = [sys.executable, "main.py"]
        if providers:
            cmd.extend(["--providers", providers])
        cmd.extend(["--iterations", str(max_iterations)])
        cmd.append(query)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        reports = get_report_files()
        if reports:
            latest = reports[-1]
            return [TextContent(
                type="text",
                text=f" Research complete!\n\n**Query:** {query}\n**Max Iterations:** {max_iterations}\n**Report Saved:** {latest}\n\nFile: {REPORTS_DIR}/{latest}\n\nUse 'get_report' tool to view full report."
            )]
        return [TextContent(
            type="text",
            text="Research completed but no report file found. Check logs for errors."
        )]
        
    except subprocess.TimeoutExpired:
        return [TextContent(
            type="text",
            text="L Research timed out after 10 minutes. Try with a simpler query."
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"L Error during research: {str(e)}"
        )]


async def handle_list_reports(arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle list_reports tool call."""
    limit = arguments.get("limit", 10)
    
    print(f"\n[MCP] Listing reports (limit: {limit})")
    
    reports = get_report_files()
    
    if not reports:
        return [TextContent(
            type="text",
            text="No reports found. Run a research query first using the 'research' tool."
        )]
    
    lines = [f"=� Saved Research Reports ({len(reports)} total)\n"]
    
    for i, filename in enumerate(reversed(reports[-limit:]), 1):
        filepath = REPORTS_DIR / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = f.readlines()[:10]
            
            query = "Unknown"
            generated = "Unknown"
            
            for line in first_lines:
                if line.startswith("# Research Report:"):
                    query = line.replace("# Research Report:", "").strip()
                elif "Generated:" in line:
                    generated = line.split("**Generated:**")[-1].strip().strip("**")
            
            lines.append(f"\n{i}. **{filename}**")
            lines.append(f"   Query: {query}")
            lines.append(f"   Generated: {generated}")
            lines.append("")
        except Exception as e:
            lines.append(f"\n{i}. **{filename}**")
            lines.append(f"   Query: Error reading file")
            lines.append("")
    
    lines.append("\nUse 'get_report' tool to view a specific report.")
    
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_report(arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle get_report tool call."""
    filename = arguments.get("filename")
    
    print(f"\n[MCP] Getting report: {filename}")
    
    filepath = REPORTS_DIR / filename
    
    if not filepath.exists():
        reports = get_report_files()
        if reports:
            similar = [r for r in reports if filename.lower() in r.lower()]
            if similar:
                return [TextContent(
                    type="text",
                    text=f"L File not found: {filename}\n\nDid you mean: {similar[0]}?"
                )]
            return [TextContent(
                type="text",
                text=f"L File not found: {filename}\n\nAvailable reports (showing 5):\n" + "\n".join([f"  - {r}" for r in reports[:5]]) + "\n\nUse 'list_reports' to see all reports."
            )]
        return [TextContent(
            type="text",
            text=f"L File not found: {filename}\n\nNo reports available. Use 'research' tool to create one."
        )]
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return [TextContent(
            type="text",
            text=f"=� Report: {filename}\n\n---\n{content}"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"L Error reading report: {str(e)}"
        )]


# ============================================================================
# SERVER SETUP
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="research",
            description="Run deep research on any topic using multi-platform agent (Exa, Tavily, Serper). Auto-detects language, selects best search engine, gathers 2000-3000+ sources, processes large contexts, and saves comprehensive academic report. Use 'providers' to force specific search engines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question or topic (any language - auto-translates if non-English)"
                    },
                    "max_iterations": {
                        "type": "number",
                        "description": "Maximum research iterations (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "providers": {
                        "type": "string",
                        "description": "Comma-separated list of search providers to force (bypasses auto-router). Options: exa, tavily, serper. Example: 'serper,tavily' will cycle through Serper and Tavily round-robin."
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_reports",
            description="List all saved research reports with filename, date, and query",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of reports to show (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_report",
            description="Get a specific research report by filename. Use list_reports first to see available reports.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Report filename (e.g., 'Quantum computing advances-20260131-133000.md')"
                    }
                },
                "required": ["filename"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    if name == "research":
        return await handle_research(arguments)
    elif name == "list_reports":
        return await handle_list_reports(arguments)
    elif name == "get_report":
        return await handle_get_report(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


# ============================================================================
# SERVER STARTUP
# ============================================================================

async def main():
    """Start MCP server."""
    print("=" * 60, file=sys.stderr)
    print("=� DEEP SEARCH AGENT - MCP SERVER", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("\nAvailable Tools:", file=sys.stderr)
    print("  1. research(query, max_iterations=5)", file=sys.stderr)
    print("  2. list_reports(limit=10)", file=sys.stderr)
    print("  3. get_report(filename)", file=sys.stderr)
    print("\nServer starting via stdio...", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())