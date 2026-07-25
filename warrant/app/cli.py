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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warrant",
                                     description="Delegation-economics orchestrator.")
    sub = parser.add_subparsers(dest="command", required=True)

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
