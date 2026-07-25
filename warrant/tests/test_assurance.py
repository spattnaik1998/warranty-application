"""Deterministic assurance policy, check, and report tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from warrant.assurance.engine import ValidationEngine, aggregate_verdict
from warrant.assurance.models import ClaimDefinition, ClaimResult, Verdict
from warrant.assurance.policy import PolicyError, load_policy_bytes
from warrant.assurance.render import render_json, render_markdown, render_terminal


def _policy(claims: str, name: str = "fixture"):
    data = f'version: "1"\nproject:\n  name: {name}\nclaims:\n{claims}'.encode()
    return load_policy_bytes(data)


def test_policy_rejects_unknown_fields_and_duplicate_ids():
    with pytest.raises(PolicyError, match="extra"):
        _policy("  - id: files\n    type: files\n    paths: [README.md]\n    surprise: true\n")
    with pytest.raises(PolicyError, match="duplicate claim ids"):
        _policy(
            "  - id: same\n    type: ci\n"
            "  - id: same\n    type: tests\n"
        )


def test_claim_type_specific_validation():
    with pytest.raises(PolicyError, match="minimum"):
        _policy("  - id: coverage\n    type: coverage\n")
    with pytest.raises(PolicyError, match="argument array"):
        _policy("  - id: command\n    type: command\n")
    with pytest.raises(PolicyError, match="unsafe"):
        _policy("  - id: files\n    type: files\n    paths: [../secret]\n")


def test_verdict_precedence_ignores_advisory_failures():
    def result(verdict, required=True):
        return ClaimResult(
            claim_id="x",
            claim_type="ci",
            required=required,
            verdict=verdict,
            summary="x",
        )

    assert aggregate_verdict([result(Verdict.PASS)]) is Verdict.PASS
    assert aggregate_verdict([result(Verdict.INDETERMINATE)]) is Verdict.INDETERMINATE
    assert aggregate_verdict([result(Verdict.FAIL)]) is Verdict.FAIL
    assert aggregate_verdict([result(Verdict.FAIL, required=False)]) is Verdict.PASS


def test_static_repository_passes_and_evidence_is_hashed(tmp_path: Path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "LICENSE").write_text(
        "MIT License\nPermission is hereby granted, free of charge, to anyone.",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "test.yml").write_text("name: test\n", encoding="utf-8")
    (tmp_path / "coverage.xml").write_text('<coverage line-rate="0.91" />', encoding="utf-8")

    policy, digest = _policy(
        "  - id: files\n    type: files\n    paths: [README.md]\n"
        "  - id: license\n    type: license\n    license: MIT\n"
        "  - id: tests\n    type: tests\n"
        "  - id: coverage\n    type: coverage\n    minimum: 90\n"
        "  - id: ci\n    type: ci\n"
    )
    report = ValidationEngine().validate(tmp_path, policy, digest)

    assert report.verdict is Verdict.PASS
    assert len(report.claims) == 5
    assert all(len(item.sha256) == 64 for claim in report.claims for item in claim.evidence)
    assert json.loads(render_json(report))["verdict"] == "pass"
    assert "Warrant verdict: PASS" in render_terminal(report)
    assert "## Evidence" in render_markdown(report)


def test_missing_and_unavailable_evidence_are_distinct(tmp_path: Path):
    policy, digest = _policy(
        "  - id: missing-file\n    type: files\n    paths: [README.md]\n"
        "  - id: missing-coverage\n    type: coverage\n    minimum: 80\n"
    )
    report = ValidationEngine().validate(tmp_path, policy, digest)
    assert report.verdict is Verdict.FAIL
    assert report.claims[0].verdict is Verdict.FAIL
    assert report.claims[1].verdict is Verdict.INDETERMINATE


def test_advisory_failure_becomes_warning(tmp_path: Path):
    policy, digest = _policy(
        "  - id: files\n    type: files\n    paths: [README.md]\n    required: false\n"
    )
    report = ValidationEngine().validate(tmp_path, policy, digest)
    assert report.verdict is Verdict.PASS
    assert report.warnings


def test_command_refused_without_permission(tmp_path: Path):
    policy, digest = _policy(
        f"  - id: command\n    type: command\n"
        f"    command: [{json.dumps(sys.executable)}, -c, \"print('ok')\"]\n"
    )
    report = ValidationEngine().validate(tmp_path, policy, digest)
    assert report.verdict is Verdict.INDETERMINATE
    assert "permission" in report.claims[0].summary


def test_command_execution_success_failure_and_sanitized_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("WARRANT_TEST_SECRET", "must-not-leak")
    success = ClaimDefinition(
        id="success",
        type="command",
        command=[sys.executable, "-c", "import os; assert 'WARRANT_TEST_SECRET' not in os.environ"],
    )
    failure = ClaimDefinition(
        id="failure",
        type="command",
        command=[sys.executable, "-c", "raise SystemExit(7)"],
    )
    from warrant.assurance.checks import check_command

    assert check_command(tmp_path, success, True).verdict is Verdict.PASS
    failed = check_command(tmp_path, failure, True)
    assert failed.verdict is Verdict.FAIL
    assert "exit code 7" in failed.summary


def test_command_timeout_is_indeterminate(tmp_path: Path):
    claim = ClaimDefinition(
        id="timeout",
        type="command",
        command=[sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=1,
    )
    from warrant.assurance.checks import check_command

    result = check_command(tmp_path, claim, True)
    assert result.verdict is Verdict.INDETERMINATE
    assert "timed out" in result.summary
