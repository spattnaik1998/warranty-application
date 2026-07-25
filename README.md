# Warrant — a delegation-economics orchestrator

> Every sub-agent must earn a **warrant** to exist: it is admitted into the
> multi-agent network only if it injects a **new exogenous signal** (a tool
> output, a retrieval, an environment observation) or performs a
> **non-redundant external check**. Delegations that merely reorganize evidence
> already in the shared belief state are rejected or *collapsed* into a single
> call.

Most multi-agent demos add a planner, a critic, and a reviewer and hope the
system gets better. **On the Reliability Limits of LLM-Based Multi-Agent
Planning** (Ao, Gao & Simchi-Levi, [arXiv:2603.26993](https://arxiv.org/abs/2603.26993))
proves the opposite: *without new exogenous signals, any delegated network is
decision-theoretically dominated by a single centralized Bayes decision-maker
with the same information* — extra hops only add **posterior distortion** (the
telephone game). Empirically the paper watches accuracy fall from 90.7% to 22.5%
over five empty relay stages, with the KL divergence between agent posteriors
predicting the accuracy drop at r≈0.72.

Warrant is the runtime that takes that theorem seriously. It orchestrates a real
task — reading an AI paper and writing a technical briefing — and refuses to
spin up an agent that can't pay for itself in information, while instrumenting
every hop so you can *prove* the multi-agent structure is earning its keep.

---

## What it does

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
  config/         settings.py          env-driven, fail-fast config
  schemas/        belief.py tasks.py ledger.py   Pydantic contracts (Posterior, Delegation Φ, ...)
  providers/      anthropic_ / openai_ / base    adapters + posterior elicitation + deterministic mock
  tools/          registry.py + tools            exogenous/validator/transform tagging
  belief/         state.py distortion.py         belief state + KL/Brier posterior distortion
  gate/           admissibility.py novelty_audit.py   the two-layer gate + delegation economics
  orchestrator/   graph.py executor.py escalation.py  LangGraph two-move loop
  pipeline/       steps.py brief_pipeline.py     the AI-paper -> briefing domain
  ledger/         probe.py metrics.py report.py  matched conditions + reproduced plots
  app/            cli.py api.py                   CLI + FastAPI
  tests/                                          schema, gate, distortion, smoke
```

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
python -m pytest -q      # 24 tests: posterior math, gate classification, distortion, smoke + ledger acceptance
```

The smoke tests assert the ledger's acceptance criteria: governed ≈ centralized,
naive < governed, signal-starved < governed, and r > 0.5.
