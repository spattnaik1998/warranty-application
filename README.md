# Warrant

Warrant is an open-source, evidence-based software-assurance validator for
Python repositories. A project declares verifiable claims in `warrant.yml`;
Warrant inspects the source, records content-addressed evidence, and returns an
authoritative `pass`, `fail`, or `indeterminate` verdict.

Deterministic checks control the verdict. Warrant's original multi-agent
reliability research remains available as an experimental demo, but model output
cannot turn missing or failed evidence into a pass.

## Quickstart

```bash
git clone https://github.com/spattnaik1998/warranty-application.git
cd warranty-application
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

warrant init ./my-project
warrant validate ./my-project
warrant validate ./my-project --format json
warrant validate ./my-project --format markdown --output warrant-report.md
```

Validate a public GitHub repository by HTTPS URL:

```bash
warrant validate https://github.com/owner/repository
```

Commands declared by a repository are never run by default. For a repository
you trust:

```bash
warrant validate ./trusted-project --allow-exec
```

This permission guard is not an operating-system sandbox.

## Policy

A minimal `warrant.yml`:

```yaml
version: "1"
project:
  name: example-service
claims:
  - id: required-files
    type: files
    paths: [README.md, pyproject.toml]
  - id: license
    type: license
    license: MIT
  - id: tests
    type: tests
  - id: coverage
    type: coverage
    minimum: 80
    report: coverage.xml
    required: false
```

V1 supports required files, SPDX license detection, Python test discovery and
opt-in execution, Coverage.py XML/JSON reports, GitHub Actions discovery,
`pip-audit` dependency checks, and guarded command arrays. See
[the policy reference](docs/policy-reference.md) and the
[complete example](examples/python/warrant.yml).

## Verdicts

- `pass`: every required claim is supported.
- `fail`: available evidence disproves at least one required claim.
- `indeterminate`: no required claim failed, but required evidence was missing,
  inaccessible, invalid, timed out, or not permitted to run.
- Advisory claims produce warnings without changing the overall verdict.

Each claim result links to immutable evidence with a locator, SHA-256 digest,
timestamp, observation, and failure reason. JSON is schema-versioned for CI use;
terminal and Markdown formats make the same evidence readable by people.

CLI exit codes are `0` for pass, `1` for fail, and `2` for indeterminate or
invalid input.

## FastAPI

```bash
uvicorn warrant.app.api:app --reload
```

Open `http://127.0.0.1:8000/docs` and submit a ZIP source archive to
`POST /v1/validations`. Uploads are bounded and checked for traversal,
symbolic links, expansion size, and file count. API policies containing
executable claims are rejected; uploaded code is never executed.

## Development

```bash
python -m pytest --cov=warrant --cov-fail-under=80
python -m ruff check warrant/assurance warrant/app warrant/tests
```

CI tests Python 3.11–3.13 on Linux and Windows and runs linting, dependency
auditing, and CodeQL. See [architecture](docs/architecture.md),
[contributing](CONTRIBUTING.md), and [security](SECURITY.md).

## Experimental reliability demo

The original delegation-economics orchestrator and Delegation Ledger remain
available through the legacy `warrant brief` and `warrant probe` commands and
under `/experimental/*` API routes. These demonstrate posterior-preserving
multi-agent delegation and do not participate in authoritative software
assurance verdicts.

## License

[MIT](LICENSE)
