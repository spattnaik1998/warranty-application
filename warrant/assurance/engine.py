"""Validation orchestration and authoritative verdict aggregation."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from warrant import __version__
from warrant.assurance.checks import CHECKS
from warrant.assurance.models import (
    ClaimResult,
    ValidationPolicy,
    ValidationReport,
    Verdict,
)
from warrant.assurance.policy import load_policy


def aggregate_verdict(results: list[ClaimResult]) -> Verdict:
    """Apply required-claim precedence; advisory failures become warnings."""
    required = [result for result in results if result.required]
    if any(result.verdict is Verdict.FAIL for result in required):
        return Verdict.FAIL
    if any(result.verdict is Verdict.INDETERMINATE for result in required):
        return Verdict.INDETERMINATE
    return Verdict.PASS


def repository_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


class ValidationEngine:
    """Evaluate a policy against one repository checkout."""

    def validate(
        self,
        root: str | Path,
        policy: ValidationPolicy,
        policy_sha256: str,
        *,
        allow_exec: bool = False,
        repository_label: str | None = None,
    ) -> ValidationReport:
        started = time.perf_counter()
        repository_root = Path(root).resolve()
        if not repository_root.is_dir():
            raise ValueError(f"repository directory not found: {repository_root}")

        results: list[ClaimResult] = []
        warnings: list[str] = []
        for claim in policy.claims:
            result = CHECKS[claim.type](repository_root, claim, allow_exec)
            results.append(result)
            if not claim.required and result.verdict is not Verdict.PASS:
                warnings.append(f"{claim.id}: {result.summary}")

        return ValidationReport(
            warrant_version=__version__,
            project=policy.project,
            repository=repository_label or str(repository_root),
            repository_commit=repository_commit(repository_root),
            policy_sha256=policy_sha256,
            generated_at=datetime.now(timezone.utc),
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            verdict=aggregate_verdict(results),
            claims=tuple(results),
            warnings=tuple(warnings),
        )


def validate_repository(
    root: str | Path,
    *,
    policy_path: str | Path | None = None,
    allow_exec: bool = False,
) -> ValidationReport:
    repository_root = Path(root).resolve()
    resolved_policy = Path(policy_path) if policy_path else repository_root / "warrant.yml"
    policy, digest = load_policy(resolved_policy)
    return ValidationEngine().validate(
        repository_root,
        policy,
        digest,
        allow_exec=allow_exec,
    )
