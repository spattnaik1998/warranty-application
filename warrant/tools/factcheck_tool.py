"""Fact-check tool (VALIDATOR) and a pure TRANSFORM tool.

`factcheck_metric` re-reads the source to verify a quoted number — an external,
non-redundant check that is Blackwell-admissible (it can move the Bayes
envelope). `draft_text` only reorganizes context it is given and injects no new
signal, so a worker equipped solely with it is a Reorganizer.
"""

from __future__ import annotations

from warrant.config import get_settings
from warrant.exceptions import ToolError
from warrant.schemas.belief import EvidenceRef
from warrant.tools.fixtures import PAPERS
from warrant.tools.registry import ToolResult, ToolRole, register


@register("factcheck_metric", ToolRole.VALIDATOR, "paper-text",
          "Re-read the source paper to verify a quoted metric actually appears.")
def factcheck_metric(arxiv_id: str, metric: str) -> ToolResult:
    settings = get_settings()
    if settings.mock:
        paper = PAPERS.get(arxiv_id)
        if paper is None:
            raise ToolError("factcheck_metric", f"no fixture for {arxiv_id!r}")
        source_text = paper["pdf_text"]
    else:  # pragma: no cover - live network
        from warrant.tools.pdf_read_tool import pdf_read

        source_text = pdf_read(arxiv_id=arxiv_id).observation
    present = metric.strip().lower() in source_text.lower()
    verdict = "supported" if present else "unsupported"
    obs = f"Metric {metric!r} is {verdict} by the source {arxiv_id}."
    return ToolResult(
        observation=obs,
        evidence_refs=[EvidenceRef(source_type="pdf", source_id=arxiv_id,
                                   locator="metric-check", snippet=metric)],
        artifacts={"verdict": verdict, "metric": metric},
    )


@register("draft_text", ToolRole.TRANSFORM, "none",
          "Reorganize / rephrase provided context into prose. Injects no new signal.")
def draft_text(context: str, instruction: str = "") -> ToolResult:
    # Deliberately trivial: this tool cannot add information, only reshape it.
    head = context.strip().split("\n")[0][:200] if context.strip() else ""
    obs = f"Draft based on given context: {head}"
    return ToolResult(observation=obs, artifacts={"instruction": instruction})
