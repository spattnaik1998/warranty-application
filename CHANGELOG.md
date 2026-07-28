# Changelog

All notable changes to Warrant are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions track `pyproject.toml`.

## [Unreleased]

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
