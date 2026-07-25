"""Command-line interface for assurance and experimental research demos.

    warrant init .
    warrant validate . --format markdown --output warrant-report.md

Experimental legacy commands:
    warrant brief --youtube "Last Week in AI"
    warrant brief --arxiv-id 2603.26993
    warrant brief --arxiv-id 2603.26993 --ungoverned   # naive baseline
    warrant probe                                       # run the Delegation Ledger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from warrant.assurance.engine import ValidationEngine
from warrant.assurance.models import Verdict
from warrant.assurance.policy import PolicyError, load_policy, write_starter_policy
from warrant.assurance.render import render_json, render_markdown, render_terminal
from warrant.assurance.source import materialize_source
from warrant.config import get_settings
from warrant.exceptions import WarrantError
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


def cmd_init(args: argparse.Namespace) -> int:
    try:
        target = write_starter_policy(args.path)
    except PolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Created {target}")
    return 0


def _render_report(report, format_name: str) -> str:
    if format_name == "json":
        return render_json(report)
    if format_name == "markdown":
        return render_markdown(report)
    return render_terminal(report)


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        with materialize_source(args.source) as (root, label):
            policy_path = Path(args.policy)
            if not policy_path.is_absolute():
                policy_path = root / policy_path
            policy, digest = load_policy(policy_path)
            if args.allow_exec:
                print(
                    "WARNING: executing commands declared by a trusted repository. "
                    "Warrant does not provide an operating-system security sandbox.",
                    file=sys.stderr,
                )
            report = ValidationEngine().validate(
                root,
                policy,
                digest,
                allow_exec=args.allow_exec,
                repository_label=label,
            )
    except (WarrantError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rendered = _render_report(report, args.format)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(rendered)
    return {
        Verdict.PASS: 0,
        Verdict.FAIL: 1,
        Verdict.INDETERMINATE: 2,
    }[report.verdict]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warrant",
                                     description="Evidence-based software assurance.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a starter warrant.yml policy")
    p_init.add_argument("path", nargs="?", default=".")
    p_init.set_defaults(func=cmd_init)

    p_validate = sub.add_parser("validate", help="validate a local or public GitHub repository")
    p_validate.add_argument("source", nargs="?", default=".")
    p_validate.add_argument("--policy", default="warrant.yml")
    p_validate.add_argument("--format", choices=("terminal", "json", "markdown"),
                            default="terminal")
    p_validate.add_argument("--output", default=None)
    p_validate.add_argument("--allow-exec", action="store_true",
                            help="run policy-declared commands in this trusted repository")
    p_validate.set_defaults(func=cmd_validate)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arxiv-id", dest="arxiv_id", default=None)
    common.add_argument("--query", dest="query", default=None)
    common.add_argument("--youtube", dest="youtube", default=None)

    p_brief = sub.add_parser("brief", parents=[common], help="experimental legacy: produce a technical briefing")
    p_brief.add_argument("--ungoverned", action="store_true",
                         help="run the naive pipeline with the admissibility gate disabled")
    p_brief.set_defaults(func=cmd_brief)

    p_probe = sub.add_parser("probe", parents=[common], help="experimental legacy: run the Delegation Ledger probe")
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
