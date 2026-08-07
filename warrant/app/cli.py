"""Command-line interface.

    warrant audit --app myapp.graph:build --cases cases.jsonl  # audit your graph
    warrant audit --example research                            # or a bundled demo
    warrant scan owner/repo                                     # static, no execution
    warrant brief --youtube "Last Week in AI"
    warrant brief --arxiv-id 2603.26993 --ungoverned            # naive baseline
    warrant probe                                               # the Delegation Ledger
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

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


def _load_attr(spec: str) -> Any:
    """Import ``module:attr`` and return the attribute.

    The one place the CLI touches user code. Failures are the user's most likely
    mistake, so they carry the spec and the underlying error rather than a
    bare traceback.
    """
    if ":" not in spec:
        raise ValueError(f"'{spec}' must be in module:attribute form, e.g. myapp.graph:build")
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"could not import '{module_name}' ({exc}). Is it on your PYTHONPATH? "
            "Running from your project root usually fixes this."
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"'{module_name}' has no attribute '{attr}'.") from exc


def _load_cases(path: str) -> list[Any]:
    """Read one JSON input per line (JSONL) — the graph inputs to replay."""
    cases: list[Any] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON — {exc}") from exc
    if not cases:
        raise ValueError(f"{path} contained no cases. Put one JSON graph input per line.")
    return cases


def _materialize(obj: Any) -> Any:
    """Accept a compiled graph, a zero-arg factory, or a ``build_graph(disabled)``.

    Users reach for whichever they already have; requiring one shape would just
    make them write a wrapper.
    """
    if hasattr(obj, "invoke") or not callable(obj):
        return obj
    takes_arg = bool(inspect.signature(obj).parameters)
    return obj(frozenset()) if takes_arg else obj()


def _audit_user_graph(args: argparse.Namespace) -> int:
    """Audit the caller's own graph: `--app module:attr --cases cases.jsonl`."""
    import warrant

    if not args.cases:
        print("audit failed: --app requires --cases PATH (one JSON graph input per line).",
              file=sys.stderr)
        return 2
    try:
        app_obj = _materialize(_load_attr(args.app))
        build_graph = _load_attr(args.build_graph) if args.build_graph else None
        cases = _load_cases(args.cases)
        node_tools = (
            json.loads(Path(args.node_tools).read_text(encoding="utf-8"))
            if args.node_tools
            else None
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2

    warrant.reset()
    instrumented = warrant.instrument(
        app_obj,
        node_tools=node_tools,
        build_graph=build_graph,
        graph_name=args.name or args.app,
        output_key=args.output_key,
    )
    with warrant.session():
        for case in cases:
            instrumented.invoke(case)
        report = warrant.audit(runs_per_month=args.runs_per_month)

    print(report.to_cli())
    out = get_settings().output_dir
    stem = (args.name or "audit").replace("/", "_").replace(":", "_")
    html_path = out / f"{stem}.html"
    report.to_html(str(html_path))
    report.to_json_file(str(out / f"{stem}.json"))
    print(f"\nHTML report written to {html_path}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit a graph — the caller's own with ``--app``, else a bundled example."""
    import warrant

    if args.app:
        return _audit_user_graph(args)

    if args.example == "dogfood":
        from warrant.examples.dogfood_brief_graph import NODE_TOOLS, build_brief_graph
        from warrant.schemas.tasks import BriefRequest

        warrant.reset()
        app = warrant.instrument(
            build_brief_graph(), node_tools=NODE_TOOLS, build_graph=build_brief_graph,
            graph_name="warrant-briefing", output_key="markdown",
        )
        inputs = [{"request": BriefRequest(arxiv_id="2603.26993")},
                  {"request": BriefRequest(youtube_channel="Last Week in AI")}]
    else:
        from warrant.examples.research_graph import NODE_TOOLS, build_research_graph

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
        report = warrant.audit(runs_per_month=args.runs_per_month)

    print(report.to_cli())
    out = get_settings().output_dir
    html_path = out / f"{args.example}_audit.html"
    report.to_html(str(html_path))
    report.to_json_file(str(out / f"{args.example}_audit.json"))
    print(f"\nHTML report written to {html_path}")
    return 0


_MAX_STEM = 60


def _unique_stem(name: str, used: set[str]) -> str:
    """A filesystem-safe, collision-free, *short* stem for one report.

    Three constraints at once. A real repo yields dozens of graphs, so names that
    sanitize alike must not overwrite each other. Graph names carry their source
    path, so they get long — and a long name plus a deep output directory exceeds
    Windows' 260-character path limit, which crashed a real scan partway through.
    So: keep the distinctive tail, and buy uniqueness with a short digest of the
    full name rather than with length.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "graph"
    if len(safe) > _MAX_STEM:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        safe = safe[-(_MAX_STEM - 9):].lstrip("_") + "_" + digest
    stem, n = safe, 2
    while stem in used:
        stem, n = f"{safe}_{n}", n + 1
    used.add(stem)
    return stem


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
    used: set[str] = set()
    unwritten = 0
    for i, report in enumerate(reports):
        print(report.to_cli())
        print()
        stem = _unique_stem(report.graph_name or f"graph{i}", used)
        html_path = out / f"scan_{stem}.html"
        try:
            report.to_html(str(html_path))
            report.to_json_file(str(out / f"scan_{stem}.json"))
        except OSError as exc:
            # One unwritable report must not discard the other fifty-three. The
            # analysis already succeeded and was printed above; say what was lost.
            unwritten += 1
            print(f"could not write report for {report.graph_name}: {exc}", file=sys.stderr)
            continue
        print(f"HTML report written to {html_path}\n")
    candidates = sum(len(r.candidates()) for r in reports)
    print(
        f"Scanned {len(reports)} graph(s), {candidates} reorganizer candidate(s). This is a "
        "static structural audit — to prove a candidate and see its dollar cost, run the "
        "graph with warrant.instrument()."
    )
    if unwritten:
        print(f"{unwritten} report(s) could not be written to {out}; see the errors above.",
              file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warrant",
                                     description="Delegation-economics SDK for multi-agent systems.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser(
        "audit",
        help="audit a running graph — yours via --app, or a bundled example",
        description=(
            "Audit your own LangGraph app:\n"
            "  warrant audit --app myapp.graph:build_graph --build-graph myapp.graph:build_graph \\\n"
            "                --cases cases.jsonl --output-key answer [--runs-per-month 30000]\n"
            "Or try it on a bundled demo:  warrant audit --example research"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_audit.add_argument("--example", choices=["research", "dogfood"], default="research",
                         help="which bundled graph to audit (default: research)")
    p_audit.add_argument("--app", default=None, metavar="MODULE:ATTR",
                         help="your compiled graph, or a factory returning one "
                              "(overrides --example)")
    p_audit.add_argument("--build-graph", dest="build_graph", default=None, metavar="MODULE:ATTR",
                         help="factory build_graph(disabled: frozenset) -> graph; "
                              "unlocks the ablation proof")
    p_audit.add_argument("--cases", default=None, metavar="PATH",
                         help="JSONL file, one graph input per line (required with --app)")
    p_audit.add_argument("--output-key", dest="output_key", default=None, metavar="FIELD",
                         help="state field holding the final answer; without it ablation "
                              "diffs the whole state and biases toward KEEP")
    p_audit.add_argument("--node-tools", dest="node_tools", default=None, metavar="PATH",
                         help='JSON file mapping {"node": ["tool", ...]}; only needed for '
                              "tools your framework does not report")
    p_audit.add_argument("--name", default=None, help="graph name for the report and filenames")
    p_audit.add_argument("--runs-per-month", dest="runs_per_month", type=int, default=None,
                         metavar="N",
                         help="your production traffic, to project a monthly cost. Without "
                              "it the report stays in dollars per 1,000 runs — Warrant "
                              "never invents a volume.")
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
