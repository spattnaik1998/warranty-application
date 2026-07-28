# Changelog

All notable changes to Warrant are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions track `pyproject.toml`.

## [Unreleased]

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
