"""Domain task schemas for the AI research -> technical briefing pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from warrant.schemas.belief import EvidenceRef


class BriefRequest(BaseModel):
    """A request to produce a technical briefing about a source."""

    # Exactly one of these is the entry point; the pipeline discovers the rest.
    arxiv_id: str | None = None
    arxiv_query: str | None = None
    youtube_channel: str | None = None
    topic: str | None = None
    audience: str = "ML engineer"


class Claim(BaseModel):
    """An atomic factual claim in the briefing, with its grounding.

    A claim without an evidence_ref is a grounding violation (the AgentLTL /
    CLAUDE.md 'no assertion without grounding' rule).
    """

    text: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    # Optional posterior over {supported, unsupported} for the grounding gate.
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def grounded(self) -> bool:
        return len(self.evidence_refs) > 0


class Section(BaseModel):
    """A section of the briefing (Title, Executive summary, Deep dive, ...)."""

    heading: str
    body: str
    claims: list[Claim] = Field(default_factory=list)


class BriefResult(BaseModel):
    """The finished briefing plus reliability metadata."""

    title: str
    paper_title: str | None = None
    arxiv_id: str | None = None
    sections: list[Section] = Field(default_factory=list)
    # Claims the risk-triggered escalator flagged for human review.
    flagged_for_review: list[Claim] = Field(default_factory=list)
    # Grounding / gray-error guard results.
    ungrounded_claims: list[Claim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.paper_title:
            ref = f" ({self.arxiv_id})" if self.arxiv_id else ""
            lines.append(f"*Paper: {self.paper_title}{ref}*")
            lines.append("")
        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            lines.append(section.body.strip())
            lines.append("")
        if self.flagged_for_review:
            lines.append("## ⚠️ Flagged for human review (high terminal posterior risk)")
            lines.append("")
            for claim in self.flagged_for_review:
                refs = ", ".join(r.short() for r in claim.evidence_refs) or "no evidence"
                lines.append(f"- {claim.text}  _(conf={claim.confidence:.2f}; {refs})_")
            lines.append("")
        return "\n".join(lines)
