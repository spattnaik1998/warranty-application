"""Example: audit a generic LangGraph app in ~3 lines.

    python examples/audit_research_graph.py

Wraps the research graph, runs it a few times, and prints + writes a delegation
audit that names the redundant ``reviewer`` node (collapsible, ~$0 value) while
keeping the load-bearing ``writer``. Renders a self-contained HTML report to
``out/audit.html``.
"""

from __future__ import annotations

import warrant
from warrant.examples.research_graph import NODE_TOOLS, build_research_graph

TOPICS = ["kv-cache compression", "mixture-of-experts routing", "speculative decoding"]


def main() -> int:
    warrant.reset()
    app = warrant.instrument(
        build_research_graph(),
        node_tools=NODE_TOOLS,
        tools={"arxiv": "INJECTOR"},
        build_graph=build_research_graph,   # unlocks ablation (capability 2)
        graph_name="research-assistant",
        output_key="draft",
    )

    with warrant.session():
        for topic in TOPICS:
            app.invoke({"topic": topic})
        report = warrant.audit()

    print(report.to_cli())
    report.to_html("out/audit.html")
    report.to_json_file("out/audit.json")
    print("\nHTML report -> out/audit.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
