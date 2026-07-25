# `warrant.yml` policy reference

Policies use schema version `"1"` and reject unknown fields.

```yaml
version: "1"
project:
  name: example
  version: 1.2.0
  repository: https://github.com/example/project
claims: []
```

Every claim has an identifier, a `type`, an optional description, and
`required: true` by default. Advisory failures are reported as warnings and do
not change the overall verdict.

## Claim types

- `files`: `paths` lists repository-relative files or directories that must exist.
- `license`: `license` is an SPDX identifier. V1 recognizes `MIT`,
  `Apache-2.0`, and `BSD-3-Clause`.
- `tests`: discovers Python `test_*.py` files. Set `execute: true` and optionally
  provide a command array to run them with CLI execution permission.
- `coverage`: `minimum` is a percentage and `report` points to Coverage.py XML
  or JSON evidence; the default is `coverage.xml`.
- `ci`: requires at least one `.github/workflows/*.yml` or `.yaml` file.
- `dependency_vulnerabilities`: invokes `pip-audit` against `requirements`
  (default `requirements.txt`) only with CLI execution permission.
- `command`: runs the required `command` argument array only with CLI execution
  permission. `timeout_seconds` defaults to 300 and is limited to 3600.

Paths must be relative and may not traverse outside the repository.
