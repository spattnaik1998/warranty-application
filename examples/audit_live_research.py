"""Run the live async research agent and produce a real delegation audit.

This is the "basic artefact": a self-contained HTML audit built from genuine
arXiv retrieval + real OpenAI calls, with measured token costs. Run it with::

    WARRANT_MOCK=0 python -m examples.audit_live_research

It fails fast if live credentials are missing.
"""

from __future__ import annotations

import asyncio

import warrant
from warrant.config import get_settings
from warrant.logging_setup import get_logger, log_event

from examples.live_research_agent import NODE_TOOLS, build_live_graph

log = get_logger("warrant.examples.live")

# Three distinct queries so a redundant reviewer is caught across varied inputs.
QUERIES = [
    "retrieval augmented generation for large language models",
    "mixture of experts routing efficiency",
    "speculative decoding inference acceleration",
]


async def _run_queries(app: object) -> None:
    """Populate the trace store with one recorded run per query."""
    for query in QUERIES:
        log_event(log, "running query", stage="example.live", query=query, status="start")
        await app.ainvoke({"query": query, "messages": []})  # type: ignore[attr-defined]


def main() -> None:
    settings = get_settings()
    settings.require_live()  # raise clearly if WARRANT_MOCK!=0 without keys

    warrant.reset()
    app = warrant.instrument(
        build_live_graph(),
        node_tools=NODE_TOOLS,
        build_graph=build_live_graph,
        output_key="answer",
        graph_name="live-research-agent",
    )

    # Record the runs inside an event loop, then audit *outside* it: ablation
    # replays the async graph via a plain asyncio.run on the main thread, which
    # is far more robust with real async LLM clients than nesting a loop inside
    # a still-running one.
    asyncio.run(_run_queries(app))
    report = warrant.audit()

    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "live_research_audit.html"
    json_path = out_dir / "live_research_audit.json"
    report.to_html(html_path)
    report.to_json_file(json_path)

    print(report.to_cli())
    print(f"\nHTML report: {html_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
