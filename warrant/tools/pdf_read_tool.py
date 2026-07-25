"""PDF reading tool (exogenous injector: the paper's own text)."""

from __future__ import annotations

from warrant.config import get_settings
from warrant.exceptions import ToolError
from warrant.schemas.belief import EvidenceRef
from warrant.tools.fixtures import PAPERS
from warrant.tools.registry import ToolResult, ToolRole, register


@register("pdf_read", ToolRole.INJECTOR, "paper-text",
          "Read the full text of a paper (by arXiv id or local path).")
def pdf_read(arxiv_id: str | None = None, path: str | None = None) -> ToolResult:
    settings = get_settings()
    if settings.mock:
        if not arxiv_id or arxiv_id not in PAPERS:
            raise ToolError("pdf_read", f"no fixture text for {arxiv_id!r}")
        text = PAPERS[arxiv_id]["pdf_text"]
        source_id = arxiv_id
    else:  # pragma: no cover - live network / disk
        try:
            from pypdf import PdfReader

            if path:
                reader = PdfReader(path)
                source_id = path
            else:
                import io

                import requests

                url = f"https://arxiv.org/pdf/{arxiv_id}"
                data = requests.get(url, timeout=30).content
                reader = PdfReader(io.BytesIO(data))
                source_id = arxiv_id or url
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ToolError("pdf_read", "failed to read PDF", cause=exc)
    return ToolResult(
        observation=text,
        evidence_refs=[EvidenceRef(source_type="pdf", source_id=str(source_id),
                                   locator="full-text", snippet=text[:160])],
        artifacts={"text": text, "chars": str(len(text))},
    )
