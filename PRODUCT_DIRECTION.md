# PRODUCT_DIRECTION.md — read this before writing any code

> **This file is the single source of truth for what Warrant *is*.**
> If code you are about to write contradicts this document, stop and raise it
> with the maintainer. Do **not** "productize" Warrant in a new direction just
> because a task description is vague. A prior automated session already did
> that once and produced the wrong product (see *Rejected fork* below).

---

## 1. What Warrant is (the one-sentence thesis)

**Warrant is a delegation-economics SDK for multi-agent LLM systems.** You wrap
your existing agent graph, run it, and Warrant tells you **which of your agents
should not exist** — and proves the token/latency/dollar savings from collapsing
or pruning them.

The positioning wedge, verbatim, so no one dilutes it:

> *"LangSmith shows you your traces. Warrant tells you which of your agents
> shouldn't exist — and proves the savings."*

We do **not** compete on general LLM observability (LangSmith, Langfuse, Arize,
Braintrust own that). Our defensible niche is **multi-agent architecture
economics**: every redundant delegation is a quantifiable, CFO-legible cost.

## 2. Why this thesis is defensible (the moat)

Warrant operationalizes a specific result — the **Reliability Limits theorem**
(Ao, Gao & Simchi-Levi, arXiv:2603.26993): *without a new exogenous signal, any
delegated multi-agent network is decision-theoretically dominated by a single
centralized Bayes decision-maker.* Extra relay hops only add **posterior
distortion** (the "telephone game": accuracy collapses ~90.7% → 22.5% over 5
empty relay hops; KL divergence predicts the accuracy drop at r ≈ 0.72–0.97).

The product is the **enforcement and measurement** of that theorem on somebody
else's graph. That research grounding **is** the moat. A generic repo linter or
trace viewer has no moat. **If a change removes the connection to the theorem,
it is off-strategy.**

## 3. Strategy: open-core

- **`warrant-core`** (this repo, OSS, Apache-2.0, `pip install warrant`):
  the framework-agnostic measurement SDK. Drives adoption + academic
  credibility + enterprise top-of-funnel.
- **`warrant-cloud`** (commercial, later): hosted continuous monitoring,
  dashboards, CI gate, cross-customer baselines. Seams are stubbed now
  (`export()`, the SQLite store as the ingestion boundary), **built later**.

## 4. The capability ladder (works out of the box, rewards deeper integration)

Warrant must deliver value at annotation level 0 and get sharper as the user
opts in. **Never make level 1 require what only level 4 provides.**

| Lvl | Capability | User must provide | Reuses |
|-----|-----------|-------------------|--------|
| 1 | **Structural gate audit** — flag every agent node that calls no *exogenous* tool as a reorganizer candidate | nothing (tool tags auto-inferred) | `tools/registry.py` `ToolRole`, `gate/admissibility.py` |
| 2 | **Ablation delegation-value** — re-run graph with a node disabled, diff outputs; "removing X changed 3/50 answers, costs $Y/mo → collapse" | a graph factory (`build_graph(disabled_nodes)`) | new `analysis/ablation.py` + `analysis/cost.py` |
| 3 | **Embedding-novelty economics** — novelty of a delegation's output vs context it already had; low novelty ⇒ redundant | an embeddings key | `gate/novelty_audit.py` generalized to pluggable distance |
| 4 | **Posterior-distortion + matched-condition probe** — the rigorous theorem-grade proof | `warrant.decision(key, options, posterior)` annotations | `belief/distortion.py`, `ledger/probe.py` |

**Level 2 (ablation + dollars) is the headline, demo-able, CFO-legible feature.**

## 5. Two modes

- **Observe (MVP, non-invasive):** instrument, capture the inter-agent trace,
  compute the audit. No behavior change. This is the adoption wedge.
- **Govern (opt-in, post-MVP):** apply the admissibility gate at runtime to
  block/collapse reorganizer delegations. Reuses `gate/`. **Not in the MVP.**

## 6. Architecture invariants (do not violate)

1. **The Trace Contract is the only seam.** Everything — the internal
   orchestrator, the LangGraph adapter, any future OTel/cloud ingestion, and
   every analyzer — reads/writes `warrant/trace/contract.py`. No analyzer may
   import LangGraph or any framework SDK directly. Adapters translate
   framework events → Trace Contract; analyzers consume only the contract.
2. **Reuse the research engine; do not reinvent it.** The following are load-
   bearing and already correct — build *on* them, don't duplicate:
   - `warrant/schemas/belief.py` — `Posterior`, `AdmissibilityClass`
     (INJECTOR/VALIDATOR/REORGANIZER), `Delegation`, `SignalClaim`.
   - `warrant/tools/registry.py` — `ToolRole` (INJECTOR/VALIDATOR/TRANSFORM),
     `ToolSpec.exogenous`.
   - `warrant/gate/admissibility.py` — `classify_delegation`, `gate_decision`.
   - `warrant/gate/novelty_audit.py` — `DelegationEconomics`.
   - `warrant/belief/distortion.py`, `warrant/belief/state.py`.
   - `warrant/ledger/probe.py`, `metrics.py`, `report.py` (matplotlib, Agg).
   - `warrant/providers/base.py` — `get_provider`, `MockProvider`.
3. **Offline determinism is mandatory.** `WARRANT_MOCK=1` (default) must make
   the whole pipeline and every test run without network or API keys. Every new
   feature ships with a mock path.
4. **No hallucinated certainty.** Per the theorem's own guardrails: if evidence
   is weak (sparse trace, no decision points, no scorer), the audit must
   **degrade gracefully** and say so, not fabricate a redundancy verdict.
5. **Do not break the 24 existing research tests.** The delegation engine is the
   asset; the SDK is a layer on top of it.

## 7. Rejected fork — DO NOT REBUILD THIS

On branch `agent/software-assurance-v1` a prior session repurposed the name
"Warrant" into a **`warrant.yml`-driven repository-claims validator**
(pass/fail/indeterminate over `files`/`license`/`tests`/`coverage`/`ci`/
`dependency_vulnerabilities`/`command`). It is competent code but it is the
**wrong product**: generic CI hygiene with **no** connection to multi-agent
systems or the reliability theorem — i.e. exactly the "humdrum" project this
effort exists to avoid. It reuses only the *word* "warrant."

That branch is **abandoned, not merged.** The canonical line is
`agent/delegation-sdk-v1`, branched from the research prototype commit
(`2059234 Import Warrant research prototype`). If you find yourself adding repo-
assurance checks, coverage parsing, or SPDX detection: **you are on the wrong
product.** Stop.

## 8. MVP definition of done

`pip install -e .` → wrap any LangGraph app in ~2 lines → run it → get an
HTML/CLI audit that (a) lists reorganizer/collapsible agents, (b) quantifies
each one's ablation delegation-value and dollar cost, and (c) works with **zero
annotation** — with an optional `warrant.decision()` path unlocking the rigorous
posterior-distortion proof and the matched-condition ledger.

**Acceptance demo:** a generic research/RAG LangGraph graph with a deliberately
redundant "reviewer" node → `warrant.instrument(app); app.invoke(...); r =
warrant.audit()` must name the reviewer as collapsible with ~0 ablation value
and a non-zero $/latency saving, while *not* flagging a load-bearing node whose
removal breaks the output. **Dogfood:** instrumenting the built-in briefing
pipeline must independently rediscover that its `compose` step is a reorganizer.

## 9. Public API (frozen surface)

```python
import warrant

app = warrant.instrument(app, tools=None, mode="observe")  # wrap a compiled graph
with warrant.session():                                     # scope a run (or set of runs)
    app.invoke(...)
warrant.tool_tag("web_search", role="INJECTOR")             # optional manual tag
warrant.decision("headline_metric_matches", options=[...], posterior=...)  # level-4 opt-in
report = warrant.audit()          # -> AuditReport
report.to_html("out/audit.html"); report.to_json(); print(report.to_cli())
warrant.export()                  # cloud ingestion seam (stub)
```

Changing these signatures is a breaking change to the OSS contract — treat it as
one.
