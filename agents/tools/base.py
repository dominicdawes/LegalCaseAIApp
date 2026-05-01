# agents/tools/base.py
"""
ToolContext binds project_id + source_ids into a closure so individual
@tool functions don't need those parameters in their LangChain signature
(the LLM would hallucinate them).  Call make_tools() once per agent run.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolContext:
    project_id: str
    source_ids: List[str] = field(default_factory=list)
    use_voyage: bool = False  # mirrors USE_VOYAGE_EMBEDDINGS at call time


def make_tools(
    project_id: str,
    source_ids: Optional[List[str]] = None,
    use_voyage: bool = False,
    tool_names: Optional[List[str]] = None,
) -> list:
    """
    Build a list of bound LangChain tools for a single agent run.

    Args:
        project_id  — Supabase project UUID
        source_ids  — restrict retrieval to these document IDs (empty = all)
        use_voyage  — True if the project was ingested with voyage-law-2
        tool_names  — optional allowlist; pass None for the full set

    Returns:
        List of LangChain BaseTool instances ready for bind_tools / ToolNode.
    """
    ctx = ToolContext(
        project_id=project_id,
        source_ids=source_ids or [],
        use_voyage=use_voyage,
    )

    from .discovery import build_discovery_tools
    from .section import build_section_tools
    from .chunk import build_chunk_tools
    from .tables import build_table_tools
    from .verification import build_verification_tools
    from .crossdoc import build_crossdoc_tools
    from .housekeeping import build_housekeeping_tools

    all_tools = (
        build_discovery_tools(ctx)
        + build_section_tools(ctx)
        + build_chunk_tools(ctx)
        + build_table_tools(ctx)
        + build_verification_tools(ctx)
        + build_crossdoc_tools(ctx)
        + build_housekeeping_tools(ctx)
    )

    if tool_names is not None:
        name_set = set(tool_names)
        all_tools = [t for t in all_tools if t.name in name_set]

    return all_tools
