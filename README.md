# Warrant — a delegation-economics SDK for multi-agent LLM systems

> **LangSmith shows you your traces. Warrant tells you which of your agents
> shouldn't exist — and proves the savings.**
>
> You wrap your existing agent graph, run it, and Warrant reports which agent
> nodes are *reorganizers* — they inject no new exogenous signal and only
> re-process context the system already has — quantifies each one's delegation
> value by **ablation**, and prints the token/latency/dollar cost of keeping it.

Every sub-agent must earn a **warrant** to exist: it justifies itself only if it
injects a **new exogenous signal** (a tool output, a retrieval, an environment
observation) or performs a **non-redundant external check**. Nodes that merely
reorganize evidence already in the shared state are candidates to *collapse*.

> **Product direction is pinned in [`PRODUCT_DIRECTION.md`](PRODUCT_DIRECTION.md)
> — read it before extending the codebase.** (A prior effort mistakenly rebuilt
> Warrant as a generic repo-claims validator; that fork is abandoned.)

Most multi-agent demos add a planner, a critic, and a reviewer and hope the
system gets better. **On the Reliability Limits of LLM-Based Multi-Agent
Planning** (Ao, Gao & Simchi-Levi, [arXiv:2603.26993](https://arxiv.org/abs/2603.26993))
proves the opposite: *without new exogenous signals, any delegated network is
decision-theoretically dominated by a single centralized Bayes decision-maker
with the same information* — extra hops only add **posterior distortion** (the
telephone game). Empirically the paper watches accuracy fall from 90.7% to 22.5%
over five empty relay stages, with the KL divergence between agent posteriors
predicting the accuracy drop at r≈0.72.

Warrant takes that theorem seriously and turns it into a measurement tool you
point at *your own* graph.

---

## Quickstart — audit your LangGraph app

```python
import warrant

# 1. Wrap your compiled graph. build_graph unlocks the ablation proof;
#    node_tools is only needed for tools your framework doesn't report itself.
app = warrant.instrument(
    compiled_graph,
    build_graph=build_graph,          # build_graph(disabled_nodes) -> compiled graph
    output_key="answer",
    node_tools={"retriever": ["web_search"]},   # optional
)

# 2. Run it as usual.
with warrant.session():
    for case in cases:
        app.invoke(case)
    report = warrant.audit()          # add runs_per_month=N for a monthly figure

# 3. Read the verdict.
print(report.to_cli())
report.to_html("out/audit.html")      # self-contained, shareable
```

...or from the shell, with no harness to write:

```bash
warrant audit --app myapp.graph:build_graph \
              --build-graph myapp.graph:build_graph \
              --cases cases.jsonl --output-key answer
```

```
node            verdict   class              ablation   nov      $/1k runs
--------------------------------------------------------------------------
retriever       KEEP      INJECTOR                  -  1.00           0.00
writer          KEEP      REORGANIZER     1.00 (n=50)  0.34           3.90
reviewer        COLLAPSE  REORGANIZER     0.00 (n=50)  0.04           1.42
--------------------------------------------------------------------------
Collapsing 1 redundant agent(s) saves $1.42 per 1,000 runs (27% of measured spend).
```

**The capability ladder** — value at zero annotation, sharper as you opt in:

| Level | Signal | You provide |
|------|--------|-------------|
| 1 | Structural: flag nodes that call no exogenous tool | nothing — tool calls are observed |
| 2 | **Ablation delegation-value + $ savings** (headline) | a `build_graph(disabled)` factory |
| 3 | Output-novelty economics | an embeddings key (optional) |
| 4 | Posterior-distortion + matched-condition ledger | `warrant.decision(...)` annotations |

### What Warrant needs from your graph

1. **`build_graph(disabled: set[str]) -> compiled_graph`** — a factory that
   recompiles your graph with named nodes omitted and edges relinked. This
   unlocks the headline ablation proof (re-run without a node; did the answer
   change?). Optional, but it's what turns a guess into a verdict.
2. **`output_key="answer"`** — the single state field holding the final
   deliverable. Ablation diffs it to decide whether a node mattered. Without it
   Warrant diffs the whole state, which biases every verdict toward KEEP — so it
   says so in the report rather than letting you read a skewed table.
3. **`node_tools={node_id: [tool_name, ...]}`** — *optional*. Warrant already
   reads the tool activity LangChain reports (`ToolMessage`, `AIMessage.tool_calls`),
   so a standard tool-calling graph needs no annotation. Declare only what the
   framework can't see — a node that calls an API directly. Names Warrant doesn't
   recognize can be tagged with `warrant.tool_tag("my_tool", "INJECTOR")`.

### How the numbers are computed

Warrant's whole value is that you can quote its figures without being embarrassed,
so each one names its own evidence:

- **Tokens.** Real per-response usage (`usage_metadata` on LangChain messages, or
  any object exposing it) is labelled `measured`. A node that makes no model call
  (pure retrieval/tool) is costed at **$0** — never a phantom length estimate.
  Only a node that ran a model *without* reporting usage is `estimated`
  (~4 chars/token), and the report names it so you know what to instrument.
- **Price.** Each node is priced at **the model it actually called**, read from the
  trace, with input and output tokens billed at their own rates. A node whose model
  the framework never reported falls back to `WARRANT_WORKER_MODEL` — and the
  report names those nodes too, because that figure could be off by 30×.
- **Volume.** The headline is **dollars per 1,000 runs**, which follows from
  measured tokens and assumes nothing. Pass `--runs-per-month N` (or
  `warrant.audit(runs_per_month=N)`) and you get a monthly projection, always
  labelled *declared* — Warrant does not know how often your graph runs and will
  not pretend to.
- **Confidence.** Scored from the evidence, not asserted. A COLLAPSE verdict after
  *n* clean replays is bounded by the rule of three (≈ 3/n), so one run reads as
  ~20% and fifty reads as ~94%. `n` appears next to every ablation value.
- **Determinism.** Before ablating anything, Warrant replays your inputs with
  *nothing* disabled. If the graph doesn't reproduce its own output, every diff is
  noise — so it says so and caps confidence instead of reporting a verdict.

### Scan a GitHub repo without running it (static audit)

You don't always have the app running — sometimes you just have a repo. Point
Warrant at a codebase and it reconstructs each LangGraph graph from source
(**never importing or executing it**) and flags the nodes that inject no
exogenous signal — reorganizer *candidates* to collapse:

```bash
warrant scan owner/repo                     # a GitHub repo (shallow-cloned)
warrant scan https://github.com/owner/repo  # or the full URL
warrant scan ./path/to/local/checkout       # or a local directory
```

```python
import warrant
for report in warrant.scan("owner/repo"):   # one report per graph found
    print(report.to_cli())
```

This is the **level-1 (structural) audit, statically**: it names what to look at
and writes a self-contained HTML report, but it deliberately shows **no ablation
value and no dollar figure at all** — not even `$0.00`, which would read as a
measured zero. Neither can be measured without running the graph, and Warrant
never fabricates a number it hasn't measured. A node flagged as a
*candidate* is proven (and priced) by wrapping the live graph with
`warrant.instrument(build_graph=...)` and running `warrant.audit()`. Detection of
exogenous tools is keyword-based and errs toward under-claiming, and graphs
assembled dynamically (a `for … add_node(...)` loop) are reported as such rather
than guessed at.

Try it without your own graph:

```bash
pip install warrant                  # or: pip install -e ".[dev]" from a checkout
warrant audit --example research     # a graph with a deliberately redundant reviewer
warrant audit --example dogfood      # audits Warrant's own briefing pipeline
warrant audit --example research --runs-per-month 30000   # priced at your volume
```

The `research` audit names the redundant `reviewer` node COLLAPSE (ablation
value 0) while keeping the load-bearing `writer` (value 1); the `dogfood` audit
independently rediscovers that the briefing pipeline's `compose` step is a
REORGANIZER — the SDK reproducing the hand-built governed design.

### Live example — a real async agent (real arXiv + real OpenAI)

```bash
pip install -e ".[dev,live]"
WARRANT_MOCK=0 OPENAI_API_KEY=sk-... python -m examples.audit_live_research
```

`examples/live_research_agent.py` is a genuine **async** LangGraph agent
(`search → analyze → synthesize → review`) that retrieves real papers from
arXiv and generates with OpenAI. The audit records **measured** token usage from
the model responses, ablates each node, and correctly flags `review` COLLAPSE
(it never changes the final answer) with a real dollar figure — while keeping
`analyze` and `synthesize` (removing either changes the answer). Output:
`out/live_research_audit.html`. Retrieval is cached to `out/arxiv_cache/` so
reruns don't re-hit arXiv, which also keeps ablation deterministic.

---

## The reference implementation (dogfood)

Warrant also ships a fully governed orchestrator for one real task — reading an
AI paper and writing a technical briefing — which serves as the dogfood target
above and as a worked example of runtime governance.

### What it does

1. **Two-move governed orchestrator (LangGraph).** An Anthropic orchestrator may
   only `Delegate(Φ)` or `Finish(y)`; it never touches a tool directly
   (`Φ = (Instruction, Context, Tools, Model)`, after AOrchestra).
2. **The Admissibility Gate.** Every proposed delegation is classified —
   `INJECTOR` (new external signal → admit), `VALIDATOR` (non-redundant external
   check → admit), or `REORGANIZER` (only re-reads context → **collapse into the
   orchestrator**). The decision is *structural* (from tool tags — no prompt can
   fake a signal) and *economic* (a delegation type whose observations never move
   the posterior is pruned).
3. **Posterior-preserving interface.** Hops carry typed posteriors, not prose —
   the paper's structured-message format that degrades ~2.8 pts/stage instead of
   ~8.5.
4. **Risk-triggered escalation (Theorem 10).** Claims are flagged for human
   review when terminal posterior risk `R_a(H)` exceeds review cost `R_h(H)` —
   never by step count.
5. **Grounding + gray-error guards.** No claim without an evidence reference; every
   quoted figure is checked against the source (catching silent semantic errors).
6. **The Delegation Ledger.** A probe that runs the task under matched conditions
   and reproduces the paper's plots **on your own pipeline**.

## Architecture

```
warrant/
  trace/          contract.py store.py    the framework-agnostic Trace Contract (the only seam)
  adapters/       langgraph.py            translate LangGraph events -> Trace Contract
  analysis/       structural/ablation/    the audit engine (the four capability signals)
                  novelty/distortion/cost + run_audit
  report/         __init__.py             self-contained HTML / CLI / JSON audit report
  config/         settings.py             env-driven, fail-fast config
  schemas/        belief.py tasks.py ledger.py   Pydantic contracts (Posterior, Delegation Φ, ...)
  providers/      anthropic_ / openai_ / base    adapters + posterior elicitation + deterministic mock
  tools/          registry.py + tools            exogenous/validator/transform tagging
  belief/         state.py distortion.py         belief state + KL/Brier posterior distortion
  gate/           admissibility.py novelty_audit.py   the two-layer gate + delegation economics
  orchestrator/   graph.py executor.py escalation.py  LangGraph two-move loop (reference impl)
  pipeline/       steps.py brief_pipeline.py     the AI-paper -> briefing domain (dogfood target)
  ledger/         probe.py metrics.py report.py  matched conditions + reproduced plots
  app/            cli.py api.py                   CLI (audit / scan / brief / probe) + FastAPI
  examples/       research_graph dogfood_brief_graph   packaged demo graphs
  tests/                                          contract, adapter, audit, dogfood, gate, ...
examples/         runnable scripts driving the packaged graphs (repo only)
```

The **Trace Contract** (`warrant/trace/contract.py`) is the single seam:
adapters write it, analyzers read it, and nothing downstream imports a framework
SDK. See [`PRODUCT_DIRECTION.md`](PRODUCT_DIRECTION.md) §6.

## Quickstart

```bash
pip install -r requirements.txt          # deps (pydantic, langgraph, anthropic, openai, ...)
cp .env.example .env                     # runs in mock mode by default (WARRANT_MOCK=1)

# Produce a governed briefing (offline, deterministic):
python -m warrant.app.cli brief --arxiv-id 2603.26993
python -m warrant.app.cli brief --youtube "Last Week in AI"

# Run the naive (ungoverned) baseline for contrast:
python -m warrant.app.cli brief --arxiv-id 2603.26993 --ungoverned

# Run the Delegation Ledger probe (writes out/ledger_report.md + PNG plots):
python -m warrant.app.cli probe

# Or the API:
uvicorn warrant.app.api:app --reload     # POST /brief, POST /probe, GET /ledger
```

**Live mode:** set `WARRANT_MOCK=0` and provide `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.
The mock provider makes the whole system — and CI — run offline: posterior
confidence tracks the presence of exogenous evidence in context, which is exactly
what reproduces the telephone game without a network.

## Results — the Delegation Ledger reproduces the paper

Running `warrant probe` on our own pipeline (mock mode):

| Condition | Accuracy | Comm. loss | Delegations |
|---|---:|---:|---|
| **A** centralized (all evidence, one call) | 1.000 | 0.000 | — |
| **B** governed (admissibility-gated) | 1.000 | 0.000 | 8 admitted, 1 collapsed |
| **B-** naive prose relay (gate off) | 0.667 | 2.456 | telephone game |
| **C** signal-starved (retrieval removed) | 0.833 | 1.475 | signal never enters |

- **KL ↔ accuracy-drop correlation: r ≈ 0.97** (paper: r≈0.72).
- **Prose relay degrades ~6.7 pts/stage; the posterior interface ~0** (paper: 8.5 vs 2.8).
- The governed network matches the centralized upper bound while the naive relay
  collapses — extra agents without new signal *only* added distortion.

Figures written to `out/`: `accuracy_vs_depth.png`, `kl_vs_accuracy_drop.png`,
`accuracy_by_condition.png`.

## Design lineage

Warrant is built from the reliability corpus it briefs:
the two-move orchestrator (AOrchestra), the decision-theoretic gate and
posterior distortion (Reliability Limits, arXiv:2603.26993), the
"no assertion without grounding" rule (AgentLTL), and the silent-failure /
gray-error framing (MAESTRO). The demo workload is recursively fitting: the
system reads multi-agent-reliability papers and is itself governed by the
reliability engineering those papers prescribe.

## Testing

```bash
python -m pytest -q      # 87 tests
ruff check .
```

Covering posterior math, gate classification, distortion, the adapter (sync and
async), the audit's economics and confidence, report rendering, the CLI, config
validation, and the static scanner.

Two families are load-bearing. The ledger smoke tests assert its acceptance
criteria — governed ≈ centralized, naive < governed, signal-starved < governed,
r > 0.5. The guard tests (`test_audit_guards.py`) assert the audit *refuses* to
sound confident without evidence: no declared or observed tools, no `output_key`,
or a graph that can't reproduce itself each produce a note and a capped
confidence rather than a clean-looking verdict.

CI runs both on 3.11 and 3.12, plus an install job that builds the wheel and runs
`warrant audit` from outside the source tree — the check that keeps the headline
command working for someone who only ran `pip install`.
