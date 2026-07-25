"""Built-in deterministic claim evaluators."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from warrant.assurance.models import ClaimDefinition, ClaimResult, Evidence, Verdict

MAX_COMMAND_OUTPUT = 64 * 1024


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {relative!r}") from exc
    return candidate


def evidence_for_file(
    root: Path,
    path: Path,
    observation: str,
    *,
    failure_reason: str | None = None,
) -> Evidence:
    data = path.read_bytes() if path.is_file() else observation.encode("utf-8")
    try:
        locator = path.relative_to(root).as_posix()
    except ValueError:
        locator = str(path)
    return Evidence(
        source="repository",
        locator=locator,
        sha256=_digest(data),
        captured_at=datetime.now(timezone.utc),
        observation=observation,
        failure_reason=failure_reason,
    )


def _result(
    claim: ClaimDefinition,
    verdict: Verdict,
    summary: str,
    evidence: list[Evidence],
    started: float,
) -> ClaimResult:
    return ClaimResult(
        claim_id=claim.id,
        claim_type=claim.type,
        required=claim.required,
        verdict=verdict,
        summary=summary,
        evidence=tuple(evidence),
        duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
    )


def check_files(root: Path, claim: ClaimDefinition, allow_exec: bool) -> ClaimResult:
    del allow_exec
    started = time.perf_counter()
    evidence: list[Evidence] = []
    missing: list[str] = []
    for relative in claim.paths:
        path = _safe_path(root, relative)
        if path.exists():
            evidence.append(evidence_for_file(root, path, f"Required path exists: {relative}"))
        else:
            missing.append(relative)
            evidence.append(
                evidence_for_file(
                    root,
                    path,
                    f"Required path is missing: {relative}",
                    failure_reason="path not found",
                )
            )
    if missing:
        return _result(
            claim,
            Verdict.FAIL,
            f"Missing required paths: {', '.join(missing)}",
            evidence,
            started,
        )
    return _result(claim, Verdict.PASS, "All required paths are present.", evidence, started)


_LICENSE_MARKERS = {
    "MIT": ("permission is hereby granted, free of charge", "mit license"),
    "Apache-2.0": ("apache license", "version 2.0"),
    "BSD-3-Clause": ("redistribution and use in source and binary forms", "neither the name"),
}


def check_license(root: Path, claim: ClaimDefinition, allow_exec: bool) -> ClaimResult:
    del allow_exec
    started = time.perf_counter()
    candidates = [
        path
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")
        if (path := root / name).is_file()
    ]
    if not candidates:
        missing = root / "LICENSE"
        ev = evidence_for_file(
            root,
            missing,
            "No recognized license file was found.",
            failure_reason="license file missing",
        )
        return _result(claim, Verdict.FAIL, "License file is missing.", [ev], started)

    expected = claim.license or ""
    markers = _LICENSE_MARKERS.get(expected)
    if markers is None:
        ev = evidence_for_file(
            root,
            candidates[0],
            f"Warrant does not yet recognize SPDX identifier {expected}.",
            failure_reason="unsupported SPDX identifier",
        )
        return _result(
            claim,
            Verdict.INDETERMINATE,
            f"Unsupported SPDX license identifier: {expected}",
            [ev],
            started,
        )

    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if all(marker in text for marker in markers):
            ev = evidence_for_file(root, path, f"License text matches SPDX {expected}.")
            return _result(claim, Verdict.PASS, f"Detected {expected} license.", [ev], started)

    ev = evidence_for_file(
        root,
        candidates[0],
        f"License text does not match SPDX {expected}.",
        failure_reason="license mismatch",
    )
    return _result(
        claim,
        Verdict.FAIL,
        f"License does not match required SPDX identifier {expected}.",
        [ev],
        started,
    )


def _test_files(root: Path) -> list[Path]:
    return sorted(
        {
            *root.glob("test_*.py"),
            *root.glob("tests/**/test_*.py"),
            *root.glob("**/tests/**/test_*.py"),
        }
    )


def _sanitized_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_command(root: Path, claim: ClaimDefinition, command: list[str]) -> ClaimResult:
    started = time.perf_counter()
    display = " ".join(command)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=_sanitized_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=claim.timeout_seconds,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[:MAX_COMMAND_OUTPUT]
        ev = evidence_for_file(
            root,
            root,
            f"$ {display}\n{output}",
            failure_reason=f"timed out after {claim.timeout_seconds}s",
        )
        return _result(
            claim,
            Verdict.INDETERMINATE,
            f"Command timed out after {claim.timeout_seconds}s.",
            [ev],
            started,
        )
    except (OSError, ValueError) as exc:
        ev = evidence_for_file(
            root,
            root,
            f"$ {display}",
            failure_reason=str(exc),
        )
        return _result(
            claim,
            Verdict.INDETERMINATE,
            f"Command could not start: {exc}",
            [ev],
            started,
        )

    combined = f"$ {display}\n{completed.stdout}\n{completed.stderr}".strip()
    if len(combined) > MAX_COMMAND_OUTPUT:
        combined = combined[:MAX_COMMAND_OUTPUT] + "\n[output truncated]"
    verdict = Verdict.PASS if completed.returncode == 0 else Verdict.FAIL
    reason = None if completed.returncode == 0 else f"exit code {completed.returncode}"
    ev = evidence_for_file(root, root, combined, failure_reason=reason)
    summary = (
        "Command completed successfully."
        if completed.returncode == 0
        else f"Command failed with exit code {completed.returncode}."
    )
    return _result(claim, verdict, summary, [ev], started)


def _execution_refused(root: Path, claim: ClaimDefinition, action: str) -> ClaimResult:
    started = time.perf_counter()
    ev = evidence_for_file(
        root,
        root,
        f"{action} was not executed.",
        failure_reason="execution requires explicit --allow-exec",
    )
    return _result(
        claim,
        Verdict.INDETERMINATE,
        f"{action} requires explicit execution permission.",
        [ev],
        started,
    )


def check_tests(root: Path, claim: ClaimDefinition, allow_exec: bool) -> ClaimResult:
    started = time.perf_counter()
    tests = _test_files(root)
    if not tests:
        ev = evidence_for_file(
            root,
            root / "tests",
            "No Python test files matching test_*.py were found.",
            failure_reason="tests not found",
        )
        return _result(claim, Verdict.FAIL, "No Python tests were discovered.", [ev], started)
    if not claim.execute:
        evidence = [
            evidence_for_file(root, path, f"Discovered Python test: {path.relative_to(root)}")
            for path in tests[:100]
        ]
        return _result(
            claim,
            Verdict.PASS,
            f"Discovered {len(tests)} Python test file(s).",
            evidence,
            started,
        )
    if not allow_exec:
        return _execution_refused(root, claim, "Test execution")
    command = list(claim.command) or [sys.executable, "-m", "pytest", "-q"]
    return _run_command(root, claim, command)


def _read_coverage(path: Path) -> float:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["totals"]["percent_covered"])
    root = ET.parse(path).getroot()
    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise ValueError("coverage XML has no line-rate")
    return float(line_rate) * 100.0


def check_coverage(root: Path, claim: ClaimDefinition, allow_exec: bool) -> ClaimResult:
    del allow_exec
    started = time.perf_counter()
    relative = claim.report or "coverage.xml"
    path = _safe_path(root, relative)
    if not path.is_file():
        ev = evidence_for_file(
            root,
            path,
            f"Coverage report is missing: {relative}",
            failure_reason="coverage evidence unavailable",
        )
        return _result(
            claim,
            Verdict.INDETERMINATE,
            "Coverage report is unavailable.",
            [ev],
            started,
        )
    try:
        percentage = _read_coverage(path)
    except (ET.ParseError, ValueError, KeyError, json.JSONDecodeError) as exc:
        ev = evidence_for_file(
            root,
            path,
            "Coverage report could not be parsed.",
            failure_reason=str(exc),
        )
        return _result(
            claim,
            Verdict.INDETERMINATE,
            f"Coverage report is invalid: {exc}",
            [ev],
            started,
        )
    minimum = float(claim.minimum or 0)
    verdict = Verdict.PASS if percentage >= minimum else Verdict.FAIL
    ev = evidence_for_file(
        root,
        path,
        f"Measured line coverage: {percentage:.2f}%; required: {minimum:.2f}%.",
        failure_reason=None if verdict is Verdict.PASS else "coverage below threshold",
    )
    return _result(
        claim,
        verdict,
        f"Coverage is {percentage:.2f}% (minimum {minimum:.2f}%).",
        [ev],
        started,
    )


def check_ci(root: Path, claim: ClaimDefinition, allow_exec: bool) -> ClaimResult:
    del allow_exec
    started = time.perf_counter()
    workflows = sorted((root / ".github" / "workflows").glob("*.y*ml"))
    if not workflows:
        ev = evidence_for_file(
            root,
            root / ".github" / "workflows",
            "No GitHub Actions workflow was found.",
            failure_reason="CI workflow missing",
        )
        return _result(claim, Verdict.FAIL, "GitHub Actions is not configured.", [ev], started)
    evidence = [
        evidence_for_file(root, path, f"Discovered CI workflow: {path.name}")
        for path in workflows
    ]
    return _result(
        claim,
        Verdict.PASS,
        f"Discovered {len(workflows)} GitHub Actions workflow(s).",
        evidence,
        started,
    )


def check_dependency_vulnerabilities(
    root: Path, claim: ClaimDefinition, allow_exec: bool
) -> ClaimResult:
    if not allow_exec:
        return _execution_refused(root, claim, "Dependency vulnerability scanning")
    requirements = _safe_path(root, claim.requirements)
    if not requirements.is_file():
        started = time.perf_counter()
        ev = evidence_for_file(
            root,
            requirements,
            f"Dependency manifest is missing: {claim.requirements}",
            failure_reason="requirements file missing",
        )
        return _result(
            claim,
            Verdict.INDETERMINATE,
            "Dependency manifest is unavailable.",
            [ev],
            started,
        )
    return _run_command(
        root,
        claim,
        [sys.executable, "-m", "pip_audit", "-r", str(requirements), "--progress-spinner", "off"],
    )


def check_command(root: Path, claim: ClaimDefinition, allow_exec: bool) -> ClaimResult:
    if not allow_exec:
        return _execution_refused(root, claim, "Configured command")
    return _run_command(root, claim, list(claim.command))


Check = Callable[[Path, ClaimDefinition, bool], ClaimResult]

CHECKS: dict[str, Check] = {
    "files": check_files,
    "license": check_license,
    "tests": check_tests,
    "coverage": check_coverage,
    "ci": check_ci,
    "dependency_vulnerabilities": check_dependency_vulnerabilities,
    "command": check_command,
}
