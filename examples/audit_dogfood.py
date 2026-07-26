"""Example/dogfood runner: audit Warrant's own briefing pipeline.

    WARRANT_MOCK=1 python examples/audit_dogfood.py

Prints the audit and writes out/dogfood_audit.html. The audit should classify
``compose`` as a REORGANIZER — the SDK rediscovering the governed design.
"""

from __future__ import annotations

import warrant
from examples.dogfood_brief_graph import NODE_TOOLS, build_brief_graph
from warrant.schemas.tasks import BriefRequest

REQUESTS = [
    BriefRequest(arxiv_id="2603.26993"),
    BriefRequest(youtube_channel="Last Week in AI"),
]


def main() -> int:
    warrant.reset()
    app = warrant.instrument(
        build_brief_graph(),
        node_tools=NODE_TOOLS,
        build_graph=build_brief_graph,
        graph_name="warrant-briefing",
        output_key="markdown",
    )
    with warrant.session():
        for req in REQUESTS:
            app.invoke({"request": req})
        report = warrant.audit()

    print(report.to_cli())
    report.to_html("out/dogfood_audit.html")
    compose = {f.node_id: f for f in report.findings}.get("compose")
    if compose:
        print(f"\ncompose classified as: {compose.admissibility.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
