"""Static (pre-execution) structural audit of a LangGraph codebase.

Warrant's headline verdicts (ablation delegation-value, dollar savings) require
*running* a graph — you cannot measure what an agent contributes to an answer
without producing the answer. But capability level 1, the **structural** signal,
needs only source: a node that calls no *exogenous* tool (retrieval, web, DB, an
API) cannot inject a new signal and is a reorganizer *candidate* to collapse.

This module reads Python source with :mod:`ast` (never importing or executing
it), reconstructs each ``StateGraph`` — its nodes, edges, and entry point — and
classifies every node by the exogeneity signals detectable in its function body.
The result is deliberately labelled a *candidate* audit: it flags what to look
at and what to prove by running, and never fabricates an ablation value or a
dollar figure it has not measured.

Detection is heuristic by nature (it matches call/attribute names against curated
keyword tables), so it errs toward *under*-claiming exogeneity: a missed tool
shows up as a reorganizer candidate to confirm, not as a false savings number.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from warrant.analysis.report import AuditReport, NodeFinding, Recommendation
from warrant.schemas.belief import AdmissibilityClass

# Distinctive identifiers that imply a node reaches outside the system's own
# state. Matched as substrings against lowercased call/attribute/name tokens used
# *inside* a node's body (never against the node's own name, to avoid e.g.
# "re-SEARCH" tripping the "search" signal).
_INJECTOR_KEYWORDS = (
    "arxiv", "requests", "urlopen", "urllib", "httpx", "aiohttp", "wikipedia",
    "tavily", "serper", "duckduckgo", "googlesearch", "similarity_search",
    "as_retriever", "retriever", "vectorstore", "faiss", "chroma", "pinecone",
    "weaviate", "embed", "webbaseloader", "webloader", "firecrawl", "apify",
    "wikipedia_search", "google_search", "web_search", "bing",
)
_VALIDATOR_KEYWORDS = (
    "subprocess", "pytest", "run_tests", "factcheck", "verify_", "validate_",
    "lint", "sandbox_exec",
)
_LLM_KEYWORDS = (
    "chatopenai", "chatanthropic", "chatgoogle", "chatgroq", "chatollama",
    "completions", "langchain_openai", "langchain_anthropic", "llm", "chatvertexai",
    "generativemodel", "invoke_llm",
)

_GRAPH_CTORS = ("StateGraph", "MessageGraph", "Graph")
_EDGE_METHODS = {"add_edge", "add_conditional_edges", "set_entry_point", "set_finish_point"}


@dataclass
class StaticNode:
    """One graph node reconstructed from source, with detected signals."""

    node_id: str
    fn_name: str = ""
    tools: set[str] = field(default_factory=set)          # matched exogenous signals
    calls_llm: bool = False
    resolved: bool = True                                  # False if fn body not found
    file: str = ""
    lineno: int = 0

    @property
    def admissibility(self) -> AdmissibilityClass:
        if self.tools:
            return AdmissibilityClass.INJECTOR
        return AdmissibilityClass.REORGANIZER

    @property
    def is_reorganizer(self) -> bool:
        return self.admissibility is AdmissibilityClass.REORGANIZER


@dataclass
class StaticGraph:
    """A LangGraph graph reconstructed from source."""

    name: str
    nodes: dict[str, StaticNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    entry_point: str = ""
    file: str = ""
    dynamic: bool = False  # nodes added via a loop/dispatch — not statically reconstructable


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _const_or_name(node: ast.AST | None) -> str | None:
    """Best-effort literal endpoint: a string constant, or START/END/name id."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("START", "END"):
            return node.id
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_functions(tree: ast.AST) -> dict[str, list[ast.AST]]:
    """Map every def name (module-level or nested) to its function node(s)."""
    funcs: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, []).append(node)
    return funcs


def _body_tokens(fn: ast.AST) -> set[str]:
    """Lowercased identifiers *used* inside a function body (calls/attrs/names)."""
    tokens: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute):
            tokens.add(n.attr.lower())
        elif isinstance(n, ast.Name):
            tokens.add(n.id.lower())
    return tokens


def _classify_body(tokens: set[str]) -> tuple[set[str], bool]:
    """Return (matched exogenous signals, calls_llm) for a set of body tokens."""
    matched: set[str] = set()
    for tok in tokens:
        for kw in _INJECTOR_KEYWORDS + _VALIDATOR_KEYWORDS:
            if kw in tok:
                matched.add(kw)
    calls_llm = any(kw in tok for tok in tokens for kw in _LLM_KEYWORDS)
    return matched, calls_llm


def _graph_builders(tree: ast.AST) -> set[str]:
    """Names bound to a StateGraph(...)-like constructor anywhere in the module."""
    builders: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            ctor = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if ctor in _GRAPH_CTORS:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        builders.add(tgt.id)
    return builders


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze_source(sources: dict[str, str]) -> list[StaticGraph]:
    """Reconstruct every LangGraph graph found across ``{path: source}``.

    Node functions are resolved by name across *all* files, so a graph assembled
    from functions defined elsewhere still classifies. Files that fail to parse
    are skipped (a partial repo still yields a partial map).
    """
    trees: dict[str, ast.AST] = {}
    for path, src in sources.items():
        try:
            trees[path] = ast.parse(src.lstrip("﻿"))  # tolerate a stray BOM
        except SyntaxError:
            continue

    # Global function table so nodes resolve across module boundaries.
    functions: dict[str, list[ast.AST]] = {}
    file_of: dict[int, str] = {}
    for path, tree in trees.items():
        for name, fns in _collect_functions(tree).items():
            functions.setdefault(name, []).extend(fns)
            for fn in fns:
                file_of[id(fn)] = path

    graphs: list[StaticGraph] = []
    for path, tree in trees.items():
        builders = _graph_builders(tree)
        if not builders:
            continue
        graph = _build_graph_from_tree(tree, builders, functions, file_of, path)
        if graph.nodes or graph.dynamic:
            graphs.append(graph)
    return graphs


def _resolve_node_fn(
    fn_ref: ast.AST | None,
    functions: dict[str, list[ast.AST]],
    file_of: dict[int, str],
) -> tuple[str, set[str], bool, bool, str, int]:
    """Resolve a node's callable to (fn_name, tools, calls_llm, resolved, file, line)."""
    # Inline lambda: analyze it directly.
    if isinstance(fn_ref, ast.Lambda):
        tools, llm = _classify_body(_body_tokens(fn_ref))
        return "<lambda>", tools, llm, True, "", getattr(fn_ref, "lineno", 0)

    fn_name = ""
    if isinstance(fn_ref, ast.Name):
        fn_name = fn_ref.id
    elif isinstance(fn_ref, ast.Attribute):
        fn_name = fn_ref.attr

    defs = functions.get(fn_name, [])
    if not defs:
        return fn_name, set(), False, False, "", 0

    tools: set[str] = set()
    llm = False
    file = ""
    line = 0
    for d in defs:                       # union signals across same-named defs
        t, cl = _classify_body(_body_tokens(d))
        tools |= t
        llm = llm or cl
        file = file_of.get(id(d), "")
        line = getattr(d, "lineno", 0)
    return fn_name, tools, llm, True, file, line


def _build_graph_from_tree(
    tree: ast.AST,
    builders: set[str],
    functions: dict[str, list[ast.AST]],
    file_of: dict[int, str],
    path: str,
) -> StaticGraph:
    graph = StaticGraph(name="", file=path)
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        recv = call.func.value
        if not (isinstance(recv, ast.Name) and recv.id in builders):
            continue
        method = call.func.attr

        if method == "add_node":
            _handle_add_node(call, graph, functions, file_of)
        elif method == "add_edge":
            a = _const_or_name(call.args[0]) if call.args else None
            b = _const_or_name(call.args[1]) if len(call.args) > 1 else None
            if a and b:
                graph.edges.append((a, b))
        elif method == "set_entry_point" and call.args:
            ep = _const_or_name(call.args[0])
            if ep:
                graph.entry_point = ep
                graph.edges.append(("START", ep))
        elif method == "add_conditional_edges" and call.args:
            src = _const_or_name(call.args[0])
            for arg in call.args[1:]:
                if isinstance(arg, ast.Dict):
                    for v in arg.values:
                        dst = _const_or_name(v)
                        if src and dst:
                            graph.edges.append((src, dst))
    # Name the graph after its state type if we can find one, else the file stem.
    graph.name = graph.name or _infer_graph_name(tree, path)
    return graph


def _handle_add_node(
    call: ast.Call,
    graph: StaticGraph,
    functions: dict[str, list[ast.AST]],
    file_of: dict[int, str],
) -> None:
    args = call.args
    if not args:
        return
    # Forms: add_node("name", fn) | add_node(fn) | add_node("name", fn=...)
    fn_ref: ast.AST = args[1] if len(args) >= 2 else args[0]

    # A string-literal node name, or a bare function reference, is statically
    # reconstructable. A non-literal name (e.g. a loop variable) with a
    # non-referable callable (a subscript/call) means the graph is assembled
    # dynamically — we cannot know its real nodes without running it.
    literal_name = args[0].value if isinstance(args[0], ast.Constant) and isinstance(args[0].value, str) else None
    fn_name, tools, llm, resolved, file, line = _resolve_node_fn(fn_ref, functions, file_of)
    if literal_name is None and not fn_name:
        graph.dynamic = True
        return

    node_id = literal_name or fn_name or "<node>"
    graph.nodes[node_id] = StaticNode(
        node_id=node_id, fn_name=fn_name, tools=tools, calls_llm=llm,
        resolved=resolved, file=file or graph.file, lineno=line,
    )


def _infer_graph_name(tree: ast.AST, path: str) -> str:
    from pathlib import PurePosixPath

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            ctor = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if ctor in _GRAPH_CTORS and node.value.args:
                state = _const_or_name(node.value.args[0])
                if state and state not in ("dict", "list", "object", "Any", "None"):
                    return state
    return PurePosixPath(path).stem


# --------------------------------------------------------------------------- #
# Report assembly (reuses the runtime AuditReport renderer)
# --------------------------------------------------------------------------- #
def _static_finding(node: StaticNode) -> NodeFinding:
    if not node.resolved:
        rec, conf, reason = (
            Recommendation.REVIEW, 0.2,
            f"node function '{node.fn_name}' not found in the scanned source — "
            "could not classify. Include the file that defines it, or run the graph.",
        )
    elif node.tools:
        rec, conf, reason = (
            Recommendation.KEEP, 0.6,
            f"injector (static): calls exogenous signal(s) {sorted(node.tools)} — "
            "brings in information from outside the system.",
        )
    elif node.calls_llm:
        rec, conf, reason = (
            Recommendation.REVIEW, 0.45,
            "reorganizer candidate (static): calls a model but no exogenous tool, so it "
            "only re-processes existing context. Provide build_graph + run warrant.audit() "
            "to confirm zero delegation value by ablation.",
        )
    else:
        rec, conf, reason = (
            Recommendation.REVIEW, 0.4,
            "reorganizer candidate (static): no exogenous tool and no model call detected — "
            "a pure transform. Likely trivially collapsible; confirm by ablation.",
        )
    return NodeFinding(
        node_id=node.node_id,
        admissibility=node.admissibility,
        recommendation=rec,
        confidence=conf,
        ablation_tested=False,
        tools=sorted(node.tools),
        reason=reason,
    )


def build_static_report(graph: StaticGraph, source_label: str = "") -> AuditReport:
    """Turn a reconstructed :class:`StaticGraph` into an :class:`AuditReport`.

    All economic fields are zero and every ablation value is absent by design —
    a static scan measures structure, not cost or delegation value.
    """
    findings = [_static_finding(n) for n in graph.nodes.values()]
    name = f"{source_label}:{graph.name}" if source_label else graph.name

    if graph.dynamic and not graph.nodes:
        return AuditReport(
            graph_name=name,
            n_runs=0,
            findings=[],
            notes=[
                "This graph is assembled dynamically (nodes added via a loop or dispatch "
                "table), so its real nodes cannot be reconstructed from source alone. "
                "Run it and audit the live graph: warrant.instrument(graph, build_graph=...) "
                "then warrant.audit(). Static scan works best on explicit "
                'add_node("name", fn) construction.',
            ],
        )
    notes = [
        "STATIC scan: source parsed, never executed. This is capability level 1 "
        "(structural) only — it flags reorganizer *candidates*, not proven collapses.",
        "No ablation value or dollar figure is shown because neither can be measured "
        "without running the graph. To prove a candidate and price it, wrap the graph "
        "with warrant.instrument(build_graph=...) and run warrant.audit().",
        "Exogenous-tool detection is heuristic (keyword-based) and errs toward under- "
        "detection; a node flagged as a candidate may call a tool the scanner didn't "
        "recognize — verify before collapsing.",
    ]
    if any(not n.resolved for n in graph.nodes.values()):
        notes.append(
            "Some node functions were defined outside the scanned files and could not "
            "be classified — include their source for a complete map."
        )
    return AuditReport(
        graph_name=name,
        n_runs=0,
        findings=findings,
        total_tokens=0,
        total_dollars_per_month=0.0,
        projected_savings_per_month=0.0,
        notes=notes,
    )


def scan_sources(sources: dict[str, str], source_label: str = "") -> list[AuditReport]:
    """Analyze source files and return one static AuditReport per graph found."""
    return [build_static_report(g, source_label) for g in analyze_source(sources)]


__all__ = [
    "StaticNode",
    "StaticGraph",
    "analyze_source",
    "build_static_report",
    "scan_sources",
]
