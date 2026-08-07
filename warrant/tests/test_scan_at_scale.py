"""Scanning a real repo means dozens of graphs, not one.

A codebase of any size has many graphs whose state type is called ``State``. If
their reports collide, most of the scan is silently lost — which is exactly what
happened the first time this was pointed at a real repository (54 graphs found,
19 report files written).
"""

from __future__ import annotations

import json

import pytest

from warrant.analysis.static import scan_sources
from warrant.app.cli import _unique_stem, main
from warrant.config import get_settings

_GRAPH = """
from langgraph.graph import StateGraph, END

class State(dict):
    pass

def {node}(s):
    return s

g = StateGraph(State)
g.add_node("{node}", {node})
g.set_entry_point("{node}")
g.add_edge("{node}", END)
"""


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WARRANT_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    yield tmp_path / "out"
    get_settings.cache_clear()


def test_same_state_name_in_many_files_stays_distinguishable() -> None:
    sources = {
        f"pkg/mod{i}.py": _GRAPH.format(node=f"step{i}") for i in range(5)
    }
    reports = scan_sources(sources, source_label="acme/repo")
    assert len(reports) == 5
    names = [r.graph_name for r in reports]
    assert len(set(names)) == 5                    # not five graphs all called "State"
    assert all(name.startswith("acme/repo:pkg/mod") for name in names)
    assert all(name.endswith(":State") for name in names)


def test_unique_stem_never_reuses_a_filename() -> None:
    used: set[str] = set()
    stems = [_unique_stem("acme/repo:pkg/mod.py:State", used) for _ in range(3)]
    assert len(set(stems)) == 3
    assert all("/" not in s and ":" not in s for s in stems)

    # A name that sanitizes to nothing still gets a usable stem.
    assert _unique_stem("///", set())


def test_unique_stem_stays_short_enough_for_windows() -> None:
    """Long graph names plus a deep output dir used to exceed MAX_PATH and crash."""
    deep = "acme/repo:" + "/".join(f"very_long_package_segment_{i}" for i in range(12)) + ".py:State"
    stem = _unique_stem(deep, set())
    assert len(stem) <= 60

    # Two long names sharing a tail must still get different files: uniqueness is
    # bought with a digest, not with length.
    used: set[str] = set()
    a = _unique_stem("acme/repo:" + "a" * 200 + "/mod.py:State", used)
    b = _unique_stem("acme/repo:" + "b" * 200 + "/mod.py:State", used)
    assert a != b and len(a) <= 60 and len(b) <= 60


def test_scan_survives_a_report_it_cannot_write(out_dir, tmp_path, capsys, monkeypatch) -> None:
    """One unwritable report must not discard the analysis of all the others."""
    src = tmp_path / "proj"
    (src / "pkg").mkdir(parents=True)
    for i in range(3):
        (src / "pkg" / f"mod{i}.py").write_text(_GRAPH.format(node=f"step{i}"), encoding="utf-8")

    from warrant.analysis.report import AuditReport

    real_to_html = AuditReport.to_html
    calls = {"n": 0}

    def flaky(self, path=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(2, "No such file or directory", str(path))
        return real_to_html(self, path)

    monkeypatch.setattr(AuditReport, "to_html", flaky)

    assert main(["scan", str(src)]) == 0            # still a successful scan
    captured = capsys.readouterr()
    assert "Scanned 3 graph(s)" in captured.out     # all three were analyzed
    assert "could not write report" in captured.err
    assert len(list(out_dir.glob("scan_*.html"))) == 2


def test_scan_writes_one_report_per_graph(out_dir, tmp_path) -> None:
    src = tmp_path / "proj"
    (src / "pkg").mkdir(parents=True)
    for i in range(5):
        (src / "pkg" / f"mod{i}.py").write_text(_GRAPH.format(node=f"step{i}"), encoding="utf-8")

    assert main(["scan", str(src)]) == 0
    assert len(list(out_dir.glob("scan_*.html"))) == 5   # nothing overwritten
    assert len(list(out_dir.glob("scan_*.json"))) == 5


def test_scan_headline_counts_the_candidates_it_found(out_dir, tmp_path, capsys) -> None:
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
    # The old headline counted COLLAPSE verdicts, which a static scan never emits,
    # so it always read "0 candidates" underneath a table full of them.
    assert "1 reorganizer candidate(s)" in printed
    assert "0 reorganizer candidate(s)" not in printed

    report = json.loads(next(out_dir.glob("scan_*.json")).read_text(encoding="utf-8"))
    assert report["economics_available"] is False
