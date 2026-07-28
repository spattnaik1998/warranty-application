"""Command-line interface.

    warrant brief --youtube "Last Week in AI"
    warrant brief --arxiv-id 2603.26993
    warrant brief --arxiv-id 2603.26993 --ungoverned   # naive baseline
    warrant probe                                       # run the Delegation Ledger
"""

from __future__ import annotations

import argparse
import sys

from warrant.config import get_settings
from warrant.logging_setup import configure_logging
from warrant.schemas.tasks import BriefRequest


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _request_from_args(args: argparse.Namespace) -> BriefRequest:
    return BriefRequest(
        arxiv_id=args.arxiv_id,
        arxiv_query=args.query,
        youtube_channel=args.youtube,
    )


def cmd_brief(args: argparse.Namespace) -> int:
    request = _request_from_args(args)
    if args.ungoverned:
        from warrant.pipeline import run_brief

        result = run_brief(request)
        gate_summary = "(ungoverned: gate disabled)"
    else:
        from warrant.orchestrator import GovernedOrchestrator

        gov = GovernedOrchestrator().run(request)
        result = gov.brief
        gate_summary = (f"gate: {gov.admitted} admitted, {gov.rejected} collapsed, "
                        f"{gov.redundant} pruned")

    print(result.to_markdown())
    print(f"\n---\n{gate_summary}")
    if result.flagged_for_review:
        print(f"{len(result.flagged_for_review)} claim(s) flagged for human review.")
    if result.warnings:
        print("warnings:", "; ".join(result.warnings))

    out = get_settings().output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "brief.md").write_text(result.to_markdown(), encoding="utf-8")
    print(f"\nwritten to {out / 'brief.md'}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from warrant.ledger import run_probe, write_report
    from warrant.ledger.report import acceptance, render_report

    probe = run_probe(_request_from_args(args) if (args.arxiv_id or args.youtube or args.query)
                      else None)
    print(render_report(probe))
    path = write_report(probe)
    ok = all(acceptance(probe).values())
    print(f"\nreport + plots written to {path.parent}")
    print("ACCEPTANCE:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def cmd_audit(args: argparse.Namespace) -> int:
    """Run a bundled example graph and emit a delegation audit."""
    import warrant

    if args.example == "dogfood":
        from examples.dogfood_brief_graph import NODE_TOOLS, build_brief_graph
        from warrant.schemas.tasks import BriefRequest

        warrant.reset()
        app = warrant.instrument(
            build_brief_graph(), node_tools=NODE_TOOLS, build_graph=build_brief_graph,
            graph_name="warrant-briefing", output_key="markdown",
        )
        inputs = [{"request": BriefRequest(arxiv_id="2603.26993")},
                  {"request": BriefRequest(youtube_channel="Last Week in AI")}]
    else:
        from examples.research_graph import NODE_TOOLS, build_research_graph

        warrant.reset()
        app = warrant.instrument(
            build_research_graph(), node_tools=NODE_TOOLS, tools={"arxiv": "INJECTOR"},
            build_graph=build_research_graph, graph_name="research-assistant", output_key="draft",
        )
        inputs = [{"topic": t} for t in
                  ("kv-cache compression", "mixture-of-experts routing", "speculative decoding")]

    with warrant.session():
        for inp in inputs:
            app.invoke(inp)
        report = warrant.audit()

    print(report.to_cli())
    out = get_settings().output_dir
    html_path = out / f"{args.example}_audit.html"
    report.to_html(str(html_path))
    report.to_json_file(str(out / f"{args.example}_audit.json"))
    print(f"\nHTML report written to {html_path}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Statically audit an agent codebase (local dir or GitHub repo)."""
    import warrant

    try:
        reports = warrant.scan(args.target, ref=args.ref)
    except (ValueError, RuntimeError) as exc:
        print(f"scan failed: {exc}", file=sys.stderr)
        return 2

    if not reports:
        print(
            f"No LangGraph graph found in '{args.target}'. Warrant's static scan looks "
            "for StateGraph(...) with add_node/add_edge; if the app builds its graph "
            "another way, run it and use warrant.instrument()/audit() instead."
        )
        return 1

    out = get_settings().output_dir
    out.mkdir(parents=True, exist_ok=True)
    for i, report in enumerate(reports):
        print(report.to_cli())
        print()
        safe = report.graph_name.replace("/", "_").replace(":", "_") or f"graph{i}"
        html_path = out / f"scan_{safe}.html"
        report.to_html(str(html_path))
        report.to_json_file(str(out / f"scan_{safe}.json"))
        print(f"HTML report written to {html_path}\n")
    print(
        f"Scanned {len(reports)} graph(s). This is a static structural audit — to prove "
        "a candidate and see its dollar cost, run the graph with warrant.instrument()."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warrant",
                                     description="Delegation-economics SDK for multi-agent systems.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="audit a bundled example graph and write a report")
    p_audit.add_argument("--example", choices=["research", "dogfood"], default="research",
                         help="which bundled graph to audit (default: research)")
    p_audit.set_defaults(func=cmd_audit)

    p_scan = sub.add_parser(
        "scan",
        help="statically audit an agent codebase (a GitHub repo or local dir) without running it",
    )
    p_scan.add_argument("target", help="owner/repo, a github.com URL, or a local directory path")
    p_scan.add_argument("--ref", default=None, help="git branch/tag to clone (GitHub targets)")
    p_scan.set_defaults(func=cmd_scan)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arxiv-id", dest="arxiv_id", default=None)
    common.add_argument("--query", dest="query", default=None)
    common.add_argument("--youtube", dest="youtube", default=None)

    p_brief = sub.add_parser("brief", parents=[common], help="produce a technical briefing")
    p_brief.add_argument("--ungoverned", action="store_true",
                         help="run the naive pipeline with the admissibility gate disabled")
    p_brief.set_defaults(func=cmd_brief)

    p_probe = sub.add_parser("probe", parents=[common], help="run the Delegation Ledger probe")
    p_probe.set_defaults(func=cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    configure_logging()
    args = build_parser().parse_args(argv)
    # Default the brief to the fixture channel if nothing was specified.
    if args.command == "brief" and not (args.arxiv_id or args.query or args.youtube):
        args.youtube = "Last Week in AI"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
