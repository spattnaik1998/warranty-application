# Architecture

Warrant separates deterministic assurance from its experimental multi-agent
research components.

## Validation flow

1. A local checkout, public GitHub clone, or uploaded ZIP becomes a bounded
   repository root.
2. A strict, versioned `warrant.yml` is parsed with unknown fields rejected.
3. The validation engine dispatches each claim to a deterministic evaluator.
4. Evaluators produce immutable, SHA-256-addressed evidence and a claim verdict.
5. Required claim results aggregate to `pass`, `fail`, or `indeterminate`.
6. The same typed report is rendered as JSON, Markdown, terminal text, or an
   OpenAPI response.

`fail` means the available evidence disproves a required claim.
`indeterminate` means the evidence needed to decide is missing, inaccessible,
invalid, timed out, or not permitted to run.

## Trust boundaries

- Static checks never import or execute target-repository code.
- CLI execution is disabled unless `--allow-exec` is supplied.
- Commands are argument arrays, use a fixed repository working directory, run
  without a shell, receive a filtered environment, have a timeout, and produce
  bounded output.
- These controls are not a kernel or container sandbox; execution is only for
  trusted repositories.
- The API accepts bounded ZIP files, rejects traversal and symbolic links, and
  refuses policies containing executable claims.

## Extension contract

Built-in evaluators are registered by claim type in `warrant.assurance.checks`.
A future ecosystem adapter implements the same callable contract:

```python
(repository_root, claim_definition, allow_exec) -> ClaimResult
```

An adapter must return evidence for every outcome and must use `indeterminate`
instead of guessing when required evidence is unavailable.

## Experimental subsystem

The delegation gate, posterior metrics, AI-paper briefing, and Delegation Ledger
remain under experimental API routes and legacy CLI commands. Their output can
augment explanations but cannot change an authoritative assurance verdict.
