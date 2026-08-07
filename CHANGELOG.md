# Changelog

All notable changes to Warrant are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions track `pyproject.toml`.

## [Unreleased]

### Changed — every number now names its own evidence

This release is about credibility rather than capability. The audit already found
the right nodes; several of the figures it printed alongside them were assumptions
wearing the clothes of measurements.

- **BREAKING (report schema): dollars are per 1,000 runs; monthly is opt-in.**
  Every `$/mo` figure used to be `tokens × price × 30_000`, where 30,000 was a
  private constant that no flag, argument, or env var could change and that the
  report disclosed nowhere. It is gone. `AuditReport` now carries
  `dollars_per_1k_runs` (measured, assumption-free) plus `runs_per_month` and
  `dollars_per_month`, which are `None` unless the caller declares their traffic
  via `warrant.audit(runs_per_month=N)`, `--runs-per-month N`, or
  `WARRANT_RUNS_PER_MONTH`. When declared, every surface prints the volume and the
  word *declared*.
- **BREAKING (report schema): `mean_chain_loss_bits` is `None` when distortion was
  not measured.** It used to serialize `0.0`, which a JSON consumer reads as "no
  distortion" rather than "not measured".
- **Per-node model attribution.** `Outcome` gained `model`, `provider`-adjacent
  `mixed_models`, and real `prompt_tokens` / `completion_tokens` (declared since
  the beginning, never written by any producer). The LangGraph adapter reads the
  model name off the same message it reads usage from, so each node is priced at
  the model it *actually called*, with input and output billed at their own rates.
  Previously every node was priced at one global `settings.worker_model` — auditing
  an Opus graph reported gpt-4o-mini rates, silently. Nodes whose model the
  framework never reported are priced at the configured default **and named in a
  note**.
- **Confidence tracks evidence.** The five hardcoded literals (0.92 / 0.9 / 0.8 /
  0.5 / 0.4) are replaced by a function of `n`. A COLLAPSE after zero changes in
  *n* replays is scored by the rule of three (95% upper bound on the true rate
  ≈ 3/n), so one run reads ~20% instead of 92%, and fifty reads ~94%. `n` now
  appears beside every ablation value in the CLI and HTML.
- **Prices refreshed and split.** The blended single-rate table is now an
  input/output `PriceCard` per model at current published rates, overridable with
  `WARRANT_PRICE_<MODEL>_IN` / `_OUT`. A model with no known rate is flagged in a
  note instead of silently billed a made-up number.
- Novelty's `jaccard_distance` is renamed `unseen_token_fraction` — it never was
  Jaccard, and the behaviour (deliberate asymmetry, so a long context can't dilute
  a novel output) was right while the name wasn't. Its threshold now has its own
  `WARRANT_NOVELTY_EPSILON`, separate from the gate's `WARRANT_NOVELTY_KL_EPSILON`;
  they measure different quantities and were being conflated.

### Added

- **The nondeterminism floor.** Before ablating anything, the audit replays every
  recorded input with *nothing* disabled. If the graph doesn't reproduce its own
  output, ablation diffs are indistinguishable from sampling noise — so the report
  says so and caps confidence rather than reporting a verdict it can't support.
  Ablation also now counts `changed` and `errored` separately: a rate limit is no
  longer silently reported as delegation value.
- **Observed tool calls.** The adapter reads the tool activity LangChain already
  reports (`ToolMessage`, `AIMessage.tool_calls`) and unions it with anything
  declared. A standard tool-calling graph now gets a real level-1 audit at zero
  annotation, which is what the capability ladder always promised.
- **Guards for the two verdict-inverting failure modes.** Omitting `node_tools`
  used to classify *every* node as a REORGANIZER, and omitting `output_key` used to
  diff the whole state and make *every* node look load-bearing — both silently.
  Each now emits a first-position note and caps confidence.
- **`warrant audit --app module:attr --cases cases.jsonl`** — audit your own graph
  from the shell, with `--build-graph`, `--output-key`, `--node-tools`, `--name`,
  and `--runs-per-month`. Previously the CLI could only run the two bundled demos;
  pointing Warrant at your own app meant writing a Python harness.
- `AdmissibilityClass.VALIDATOR` is now reachable from the audit. `structural.py`
  could only ever emit INJECTOR or REORGANIZER, so a validator node was reported
  as an injector.
- GitHub Actions CI (pytest + ruff on 3.11/3.12), plus an install job that builds
  the wheel and runs `warrant audit` from outside the source tree.
- Tests for the CLI, config validation, report rendering, cost/pricing, the
  confidence curve, and the guards — areas that previously had none. 87 total.

### Fixed — found by pointing `warrant scan` at a real repository

Scanning `langchain-ai/langgraph` (54 graphs, 19 MB) surfaced four defects that a
single-graph fixture could never show:

- **35 of 54 reports were silently overwritten.** A real codebase has many graphs
  whose state type is called `State`, and every one of them produced
  `scan_State.html`. Graph names now include their defining file, and report
  filenames are deduplicated.
- **Report filenames then blew Windows' 260-character path limit**, crashing the
  scan partway through with an unhandled `FileNotFoundError` — after 43 of 54
  graphs. Stems are capped at 60 characters, buying uniqueness with a short digest
  of the full name rather than with length. A report that still cannot be written
  is now reported and skipped instead of discarding the other 53.
- **The static headline always read "0 candidates"** above a table full of them:
  it counted COLLAPSE verdicts, which a static scan can never emit (that needs
  ablation). It now counts reorganizer candidates, and the CLI footer totals them
  across all graphs.
- **Every remote scan leaked a full repository clone into the temp directory.**
  Git marks objects read-only, so `rmtree(ignore_errors=True)` gave up silently —
  19 MB per scan. The read-only bit is now cleared and retried, and a genuine
  failure is logged rather than swallowed.

Also: a static report drops its Model and cost columns entirely rather than
filling them with dashes, which still read as "we tried to price this".

### Fixed

- **`warrant audit` crashed on a `pip install`.** `cli.py` imported `examples.*`,
  which the wheel never shipped. The demo graphs moved to `warrant/examples/`;
  `examples/` keeps the runnable scripts and stays unpackaged.
- **A static scan no longer renders `$0.00`.** `economics_available=False` now
  suppresses the money card, table column, and bar chart entirely — printing zero
  implied a measured zero rather than an unmeasurable one.
- `SQLiteTraceStore.clear()` deleted only the in-memory cache, so `session()` and
  `reset()` left the rows on disk to resurrect on the next `load()`.
- `InstrumentedApp.replays` was never cleared by `session()`: it grew without
  bound and let a previous session's inputs feed this session's ablation.
- `warrant scan` now sets `GIT_TERMINAL_PROMPT=0`, so a private or missing repo
  fails fast with git's own message instead of hanging invisibly for 180s behind a
  captured credential prompt.
- Dead code removed: `DelegationEconomics` was constructed and fed inside
  `novelty_audit` and never read; `estimate_cost`'s `n_runs_observed` parameter was
  never used.

## [0.1.0] — earlier

### Added
- **`warrant scan <target>` — static audit of a GitHub repo or local directory.**
  Reconstructs each LangGraph graph from source with `ast` (never importing or
  executing it) and flags nodes that inject no exogenous signal as reorganizer
  *candidates*. GitHub targets (`owner/repo` or a URL) are shallow-cloned; a local
  path is read in place. Also exposed as `warrant.scan(target, ref=None)`, which
  returns one `AuditReport` per graph. The scan shows **no ablation value or
  dollar figure** by design — structural signal only, with prominent notes on how
  to prove and price a candidate by running the graph.
  - Honest degradation: dynamically-assembled graphs (`for … add_node(...)`) are
    reported as non-reconstructable rather than guessed; nodes whose functions
    live outside the scanned files are marked unresolved; a UTF-8 BOM no longer
    breaks parsing.
  - New modules: `warrant/analysis/static.py`, `warrant/ingest/` (github).

### Fixed
- **Honest cost accounting for tool-only nodes.** A node that makes no model call
  (e.g. a retrieval / web / DB node) is now costed at **$0** instead of a phantom
  token estimate derived from the length of the text it fetched. Previously a
  Wikipedia/arXiv node could show several dollars/month of spend it never
  incurred, inflating the audit total.
- **Truthful token-provenance note.** The report now flags a node's cost as
  `estimated` **only** when that node actually ran a model without reporting
  usage, and names the specific nodes to instrument. Pure tool nodes no longer
  trip a "wire up `usage_metadata`" nag they cannot satisfy; a run whose every
  model node reports usage is reported as fully `measured`.

### Added
- Per-node token provenance on the trace contract (`Outcome.token_source`:
  `measured` / `estimated` / `none`) so downstream consumers can distinguish an
  exact figure from a proxy from a genuine zero.
- README "What Warrant needs from your graph" — the three-declaration contract
  (`node_tools`, `build_graph`, `output_key`) for auditing an external repo, plus
  a "how cost is computed" note.
- Adapter test covering the $0 tool-node path and the `measured` run label when a
  tool node coexists with a measured model node.

### Internal
- Cleared pre-existing lint (unused imports, ambiguous variable names); the tree
  is `ruff`-clean. Test suite at 47 passing.
