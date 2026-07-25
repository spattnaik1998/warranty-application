"""Tool package. Importing it registers all built-in tools in REGISTRY."""

from __future__ import annotations

from warrant.tools.registry import REGISTRY, ToolRegistry, ToolResult, ToolRole, ToolSpec

# Import for side effects: each module registers its tools on import.
from warrant.tools import arxiv_tool  # noqa: E402,F401
from warrant.tools import pdf_read_tool  # noqa: E402,F401
from warrant.tools import youtube_tool  # noqa: E402,F401
from warrant.tools import factcheck_tool  # noqa: E402,F401

__all__ = ["REGISTRY", "ToolRegistry", "ToolResult", "ToolRole", "ToolSpec"]
