"""Terminal, Markdown, and stable JSON report rendering."""

from __future__ import annotations

from warrant.assurance.models import ValidationReport, Verdict

_MARKS = {
    Verdict.PASS: "PASS",
    Verdict.FAIL: "FAIL",
    Verdict.INDETERMINATE: "INDETERMINATE",
}


def render_terminal(report: ValidationReport) -> str:
    lines = [
        f"Warrant verdict: {_MARKS[report.verdict]}",
        f"Project: {report.project.name}",
        f"Repository: {report.repository}",
        f"Policy: sha256:{report.policy_sha256}",
        "",
    ]
    for result in report.claims:
        requirement = "required" if result.required else "advisory"
        lines.append(
            f"[{_MARKS[result.verdict]}] {result.claim_id} "
            f"({result.claim_type}, {requirement}) - {result.summary}"
        )
        for evidence in result.evidence:
            lines.append(f"  evidence: {evidence.locator} sha256:{evidence.sha256[:12]}")
    if report.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines)


def render_markdown(report: ValidationReport) -> str:
    lines = [
        "# Warrant software-assurance report",
        "",
        f"**Verdict:** `{report.verdict.value}`  ",
        f"**Project:** {report.project.name}  ",
        f"**Repository:** `{report.repository}`  ",
        f"**Policy digest:** `sha256:{report.policy_sha256}`  ",
        f"**Commit:** `{report.repository_commit or 'unavailable'}`",
        "",
        "## Claims",
        "",
        "| Claim | Type | Requirement | Verdict | Summary |",
        "|---|---|---|---|---|",
    ]
    for result in report.claims:
        requirement = "required" if result.required else "advisory"
        summary = result.summary.replace("|", "\\|")
        lines.append(
            f"| `{result.claim_id}` | {result.claim_type} | {requirement} | "
            f"**{result.verdict.value}** | {summary} |"
        )
    lines.extend(["", "## Evidence", ""])
    for result in report.claims:
        lines.append(f"### {result.claim_id}")
        lines.append("")
        for item in result.evidence:
            reason = f"; failure: {item.failure_reason}" if item.failure_reason else ""
            lines.append(
                f"- `{item.locator}` — `sha256:{item.sha256}`{reason}. "
                f"{item.observation.splitlines()[0]}"
            )
        if not result.evidence:
            lines.append("- No evidence recorded.")
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    return "\n".join(lines)


def render_json(report: ValidationReport) -> str:
    return report.model_dump_json(indent=2)
