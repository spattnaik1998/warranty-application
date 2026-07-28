"""Static (pre-execution) scan of a LangGraph codebase — no network, no exec."""

from __future__ import annotations

from pathlib import Path

from warrant.analysis.report import Recommendation
from warrant.analysis.static import analyze_source, scan_sources
from warrant.ingest.github import _parse_github, collect_python_sources, resolve_target
from warrant.schemas.belief import AdmissibilityClass

_EXPLICIT = '''
from langgraph.graph import StateGraph, END
import arxiv

def retriever(s):
    return {"papers": list(arxiv.Client().results(arxiv.Search(query=s["q"])))}

def analyze(s):
    llm = ChatOpenAI()
    return {"analysis": llm.invoke(s["papers"])}

def reviewer(s):
    return {"notes": "lgtm"}

def build():
    g = StateGraph(dict)
    g.add_node("retriever", retriever)
    g.add_node("analyze", analyze)
    g.add_node("reviewer", reviewer)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "analyze")
    g.add_edge("analyze", "reviewer")
    g.add_edge("reviewer", END)
    return g.compile()
'''

_DYNAMIC = '''
from langgraph.graph import StateGraph, END

def a(s): return s
def b(s): return s

def build(order):
    g = StateGraph(dict)
    fns = {"a": a, "b": b}
    for name in order:
        g.add_node(name, fns[name])
    return g.compile()
'''


def test_explicit_graph_classifies_injector_and_reorganizers() -> None:
    graphs = analyze_source({"app.py": _EXPLICIT})
    assert len(graphs) == 1
    g = graphs[0]
    assert set(g.nodes) == {"retriever", "analyze", "reviewer"}

    assert g.nodes["retriever"].admissibility is AdmissibilityClass.INJECTOR
    assert "arxiv" in g.nodes["retriever"].tools
    assert g.nodes["analyze"].admissibility is AdmissibilityClass.REORGANIZER
    assert g.nodes["analyze"].calls_llm is True
    assert g.nodes["reviewer"].admissibility is AdmissibilityClass.REORGANIZER
    assert g.nodes["reviewer"].calls_llm is False


def test_explicit_graph_report_keeps_injector_flags_candidates() -> None:
    report = scan_sources({"app.py": _EXPLICIT}, source_label="demo/repo")[0]
    verdicts = {f.node_id: f.recommendation for f in report.findings}
    assert verdicts["retriever"] is Recommendation.KEEP
    assert verdicts["analyze"] is Recommendation.REVIEW
    assert verdicts["reviewer"] is Recommendation.REVIEW
    # A static scan never invents ablation values or dollar figures.
    assert all(f.ablation_value is None and f.dollars_per_month == 0.0 for f in report.findings)
    assert report.projected_savings_per_month == 0.0
    assert any("STATIC scan" in n for n in report.notes)


def test_dynamic_graph_is_flagged_not_fabricated() -> None:
    reports = scan_sources({"app.py": _DYNAMIC})
    assert len(reports) == 1
    assert reports[0].findings == []
    assert any("assembled dynamically" in n for n in reports[0].notes)


def test_node_defined_elsewhere_is_unresolved() -> None:
    src = '''
from langgraph.graph import StateGraph, END
from mypkg.nodes import external_node

def build():
    g = StateGraph(dict)
    g.add_node("external", external_node)
    g.set_entry_point("external")
    g.add_edge("external", END)
    return g.compile()
'''
    g = analyze_source({"app.py": src})[0]
    assert g.nodes["external"].resolved is False
    report = scan_sources({"app.py": src})[0]
    assert report.findings[0].recommendation is Recommendation.REVIEW


def test_parse_github_forms() -> None:
    assert _parse_github("https://github.com/owner/repo") == ("owner", "repo")
    assert _parse_github("github.com/owner/repo.git") == ("owner", "repo")
    assert _parse_github("owner/repo") == ("owner", "repo")
    assert _parse_github("not a repo") is None


def test_resolve_local_dir_and_collect_skips_vendored(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    vendored = tmp_path / ".venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "junk.py").write_text("y = 2\n", encoding="utf-8")

    ref = resolve_target(str(tmp_path))
    assert ref.is_remote is False and ref.cleanup is None
    sources = collect_python_sources(ref.local_path)
    assert "app.py" in sources
    assert not any(".venv" in p for p in sources)
