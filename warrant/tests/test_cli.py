"""The CLI is the surface a pip-installed user meets first, so it gets tests.

Covers the two things most likely to break for someone who is not us: auditing
their *own* graph via ``--app``, and the bundled demo working from an install
rather than a checkout (``warrant.examples`` is inside the package for exactly
this reason). Error paths return exit codes rather than tracebacks.
"""

from __future__ import annotations

import json

import pytest

from warrant.app.cli import build_parser, main
from warrant.config import get_settings

pytest.importorskip("langgraph.graph")


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """Point report output at a temp dir instead of the repo's ./out."""
    monkeypatch.setenv("WARRANT_OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps({"topic": t}) for t in ("kv-cache", "moe routing")),
        encoding="utf-8",
    )
    return path


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_audit_bundled_example_runs_from_the_package(out_dir, capsys) -> None:
    """`warrant audit --example` must not depend on the repo's examples/ dir."""
    assert main(["audit", "--example", "research"]) == 0
    printed = capsys.readouterr().out
    assert "COLLAPSE reviewer" in printed
    assert "per 1,000 runs" in printed          # no volume declared, no /mo
    assert (out_dir / "research_audit.html").exists()
    assert (out_dir / "research_audit.json").exists()


def test_audit_declared_volume_switches_the_unit(out_dir, capsys) -> None:
    assert main(["audit", "--example", "research", "--runs-per-month", "30000"]) == 0
    printed = capsys.readouterr().out
    assert "30,000 runs/month (declared)" in printed
    assert "per 1,000 runs" not in printed


def test_audit_user_graph_end_to_end(out_dir, cases, capsys) -> None:
    code = main([
        "audit",
        "--app", "warrant.examples.research_graph:build_research_graph",
        "--build-graph", "warrant.examples.research_graph:build_research_graph",
        "--cases", str(cases),
        "--output-key", "draft",
        "--name", "mine",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "COLLAPSE reviewer" in printed
    assert "(n=2)" in printed                    # n travels with the verdict
    assert (out_dir / "mine.html").exists()

    report = json.loads((out_dir / "mine.json").read_text(encoding="utf-8"))
    assert report["runs_per_month"] is None
    assert report["projected_savings_per_month"] is None
    assert report["projected_savings_per_1k_runs"] > 0


def test_audit_app_without_cases_is_a_usage_error(out_dir, capsys) -> None:
    code = main(["audit", "--app", "warrant.examples.research_graph:build_research_graph"])
    assert code == 2
    assert "--cases" in capsys.readouterr().err


def test_audit_reports_a_bad_module_spec_clearly(out_dir, cases, capsys) -> None:
    assert main(["audit", "--app", "not_a_module", "--cases", str(cases)]) == 2
    assert "module:attribute" in capsys.readouterr().err

    assert main(["audit", "--app", "warrant.nope:thing", "--cases", str(cases)]) == 2
    assert "could not import" in capsys.readouterr().err

    assert main([
        "audit", "--app", "warrant.examples.research_graph:missing", "--cases", str(cases)
    ]) == 2
    assert "has no attribute" in capsys.readouterr().err


def test_audit_rejects_malformed_cases(out_dir, tmp_path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"topic": "ok"}\nnot json\n', encoding="utf-8")
    code = main([
        "audit", "--app", "warrant.examples.research_graph:build_research_graph",
        "--cases", str(bad),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "bad.jsonl:2" in err                 # names the offending line

    empty = tmp_path / "empty.jsonl"
    empty.write_text("# only a comment\n", encoding="utf-8")
    assert main([
        "audit", "--app", "warrant.examples.research_graph:build_research_graph",
        "--cases", str(empty),
    ]) == 2
    assert "no cases" in capsys.readouterr().err


def test_scan_local_dir_and_unusable_target(out_dir, tmp_path, capsys) -> None:
    src = tmp_path / "proj"
    src.mkdir()
    (src / "app.py").write_text(
        "from langgraph.graph import StateGraph, END\n"
        "def retriever(s):\n"
        "    return {'docs': web_search(s['q'])}\n"
        "def reviewer(s):\n"
        "    return s\n"
        "g = StateGraph(dict)\n"
        "g.add_node('retriever', retriever)\n"
        "g.add_node('reviewer', reviewer)\n"
        "g.set_entry_point('retriever')\n"
        "g.add_edge('retriever', 'reviewer')\n"
        "g.add_edge('reviewer', END)\n",
        encoding="utf-8",
    )
    assert main(["scan", str(src)]) == 0
    printed = capsys.readouterr().out
    assert "STATIC scan" in printed
    assert "$0.00" not in printed               # cost is absent, not zero

    # A target that is neither a directory nor a repo slug fails cleanly.
    assert main(["scan", "this is not a target"]) == 2
    assert "scan failed" in capsys.readouterr().err
