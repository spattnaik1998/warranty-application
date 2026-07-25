"""arXiv discovery and fetch tools (exogenous injectors: external DB)."""

from __future__ import annotations

from warrant.config import get_settings
from warrant.exceptions import ToolError
from warrant.schemas.belief import EvidenceRef
from warrant.tools.fixtures import PAPERS
from warrant.tools.registry import ToolResult, ToolRole, register


def _fixture_search(query: str) -> dict:
    q = query.lower()
    for paper in PAPERS.values():
        if any(term in paper["title"].lower() for term in q.split()):
            return paper
    return next(iter(PAPERS.values()))


@register("arxiv_search", ToolRole.INJECTOR, "arxiv-db",
          "Search arXiv and return the best-matching paper's metadata.")
def arxiv_search(query: str) -> ToolResult:
    settings = get_settings()
    if settings.mock:
        paper = _fixture_search(query)
    else:  # pragma: no cover - live network
        try:
            import arxiv

            search = arxiv.Search(query=query, max_results=1)
            result = next(arxiv.Client().results(search))
            paper = {
                "arxiv_id": result.get_short_id(),
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": str(result.published.date()),
            }
        except Exception as exc:
            raise ToolError("arxiv_search", f"search failed for {query!r}", cause=exc)
    obs = f"Top arXiv match: {paper['title']} ({paper['arxiv_id']})"
    return ToolResult(
        observation=obs,
        evidence_refs=[EvidenceRef(source_type="arxiv", source_id=paper["arxiv_id"],
                                   snippet=paper["title"])],
        artifacts={"arxiv_id": paper["arxiv_id"], "title": paper["title"],
                   "authors": ", ".join(paper.get("authors", []))},
    )


@register("arxiv_fetch", ToolRole.INJECTOR, "arxiv-db",
          "Fetch a specific arXiv paper's metadata by id.")
def arxiv_fetch(arxiv_id: str) -> ToolResult:
    settings = get_settings()
    if settings.mock:
        paper = PAPERS.get(arxiv_id)
        if paper is None:
            raise ToolError("arxiv_fetch", f"unknown fixture paper {arxiv_id!r}")
    else:  # pragma: no cover - live network
        try:
            import arxiv

            result = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
            paper = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": str(result.published.date()),
            }
        except Exception as exc:
            raise ToolError("arxiv_fetch", f"fetch failed for {arxiv_id!r}", cause=exc)
    obs = (f"{paper['title']} — {', '.join(paper.get('authors', []))} "
           f"({paper.get('published', 'n/a')}).\nAbstract: {paper['abstract']}")
    return ToolResult(
        observation=obs,
        evidence_refs=[EvidenceRef(source_type="arxiv", source_id=arxiv_id,
                                   locator="abstract", snippet=paper["abstract"][:160])],
        artifacts={"title": paper["title"], "abstract": paper["abstract"],
                   "authors": ", ".join(paper.get("authors", []))},
    )
