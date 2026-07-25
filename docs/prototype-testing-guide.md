# Prototype testing guide

This guide is for engineers trailblazing Warrant v1 before its first release.
It provides a repeatable setup, a core acceptance pass, exploratory scenarios,
and a standard way to report findings.

## What you are testing

Warrant validates assurance claims declared in a repository's `warrant.yml`.
Every claim produces evidence and one of three verdicts:

- `pass`: the claim is supported by available evidence.
- `fail`: available evidence disproves the claim.
- `indeterminate`: Warrant cannot decide because evidence is missing, invalid,
  inaccessible, timed out, or not permitted to run.

The authoritative result must come from deterministic checks. Experimental
multi-agent features must not change a software-assurance verdict.

## 1. Prepare a test environment

Prerequisites:

- Git
- Python 3.11, 3.12, or 3.13
- A repository you are allowed to inspect

Clone the prototype branch:

```bash
git clone https://github.com/spattnaik1998/warranty-application.git
cd warranty-application
git switch agent/software-assurance-v1
```

Create an isolated environment.

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the installation:

```bash
warrant --help
warrant validate --help
python -c "import warrant; print(warrant.__version__)"
```

Expected version: `1.0.0`.

## 2. Run the maintainer acceptance pass

Run these commands from the Warrant repository root:

```bash
python -m ruff check warrant/assurance warrant/app warrant/tests
python -m pytest --cov=warrant --cov-report=xml --cov-fail-under=80
warrant validate .
```

Expected results:

- Ruff reports `All checks passed!`.
- All tests pass and total coverage is at least 80%.
- Warrant reports an overall `PASS`.
- Required files, MIT license, tests, and continuous integration claims pass.
- The coverage claim passes when `coverage.xml` was produced by the test command.

Also exercise every output format:

```bash
warrant validate . --format terminal
warrant validate . --format json --output warrant-report.json
warrant validate . --format markdown --output warrant-report.md
```

Confirm that JSON contains `schema_version: "1"` and that every evidence item
has a 64-character SHA-256 digest, locator, timestamp, and observation.

## 3. Create a disposable fixture project

Do not modify a production repository for the following scenarios.

```bash
mkdir warrant-fixture
warrant init warrant-fixture
```

Add the files required by the starter policy:

```text
warrant-fixture/
├── README.md
├── pyproject.toml
├── warrant.yml
└── tests/
    └── test_example.py
```

Minimal test:

```python
def test_example():
    assert True
```

The starter policy also requires an MIT license. Copy the Warrant `LICENSE`
file into the fixture or create an equivalent complete MIT license file.

### Scenario A: supported claims

Run:

```bash
warrant validate warrant-fixture
```

Expected result: `PASS`. The advisory CI claim may warn if the fixture has no
GitHub Actions workflow, but it must not change the overall verdict.

### Scenario B: disproven required claim

Temporarily rename `README.md`, then run validation again.

Expected result:

- Overall verdict: `FAIL`
- `required-files`: `FAIL`
- CLI exit code: `1`
- Evidence identifies the missing `README.md`.

Restore the file after the test.

### Scenario C: unavailable required evidence

Replace the policy claims with:

```yaml
claims:
  - id: coverage
    type: coverage
    minimum: 80
    report: coverage.xml
```

Ensure `coverage.xml` does not exist and run validation.

Expected result:

- Overall verdict: `INDETERMINATE`
- Coverage claim: `INDETERMINATE`
- CLI exit code: `2`
- The result says coverage evidence is unavailable, not that coverage failed.

### Scenario D: advisory failure

Set `required: false` on a claim that will fail.

Expected result: the claim records its failure and produces a warning, while the
overall result remains `PASS` if every required claim passes.

## 4. Test guarded execution

Repository-defined code is untrusted. Use these scenarios only with the
disposable fixture.

Add a command claim:

```yaml
  - id: python-version
    type: command
    command: [python, --version]
    timeout_seconds: 30
```

Without permission:

```bash
warrant validate warrant-fixture
```

Expected result: `INDETERMINATE`; the command is not executed.

With explicit permission:

```bash
warrant validate warrant-fixture --allow-exec
```

Expected result:

- A trust warning is printed to standard error.
- The command runs without a shell and passes.
- Evidence records the command and bounded output.

Then test a nonzero command and a timeout:

```yaml
  - id: expected-failure
    type: command
    command: [python, -c, "raise SystemExit(7)"]
  - id: expected-timeout
    type: command
    command: [python, -c, "import time; time.sleep(10)"]
    timeout_seconds: 1
```

Expected results:

- Exit code 7 produces a claim `FAIL`.
- The timeout produces `INDETERMINATE`.
- Neither condition crashes Warrant.

`--allow-exec` is a permission boundary, not an operating-system sandbox. Never
use it for an unknown or hostile repository.

## 5. Test a public GitHub repository

Choose a small public Python repository that you trust:

```bash
warrant validate https://github.com/OWNER/REPOSITORY
```

The repository must contain `warrant.yml`. Confirm that:

- only public HTTPS `github.com` URLs are accepted;
- the clone is shallow and temporary;
- SSH, arbitrary hosts, and private repository authentication are rejected;
- the report identifies the source URL and checked-out commit.

Do not use `--allow-exec` during public-repository exploration.

## 6. Test the FastAPI boundary

Start the API:

```bash
uvicorn warrant.app.api:app --reload
```

Open `http://127.0.0.1:8000/docs` and confirm that
`POST /v1/validations` appears under the assurance tag.

Create a ZIP containing the fixture project.

Windows PowerShell:

```powershell
Compress-Archive -Path .\warrant-fixture\* -DestinationPath .\warrant-fixture.zip
```

Linux or macOS:

```bash
cd warrant-fixture
zip -r ../warrant-fixture.zip .
cd ..
```

Submit it:

```bash
curl -X POST http://127.0.0.1:8000/v1/validations \
  -F "archive=@warrant-fixture.zip"
```

Expected result: HTTP 200 with a typed JSON validation report.

API safety scenarios:

| Scenario | Expected result |
|---|---|
| Non-ZIP upload | HTTP 415 |
| Malformed or missing policy | HTTP 422 |
| Policy containing command, executable tests, or dependency audit | HTTP 422 |
| ZIP path traversal or symbolic link | HTTP 400 |
| Archive larger than 25 MiB | HTTP 413 |
| Policy larger than 1 MiB | HTTP 413 |

Uploaded repository code must never execute.

## 7. Explore policy edge cases

Useful exploratory cases:

- Unknown policy or claim fields are rejected.
- Duplicate claim IDs are rejected.
- Absolute paths and `..` traversal are rejected.
- Coverage thresholds outside 0–100 are rejected.
- Command strings are rejected; commands must be YAML argument arrays.
- Unsupported SPDX identifiers produce `INDETERMINATE`, not a guessed result.
- Missing license files and mismatched recognized licenses produce `FAIL`.
- Malformed coverage XML or JSON produces `INDETERMINATE`.
- Markdown evidence text cannot break the claims table.
- Very large command output is truncated without exhausting memory.

## 8. Regression-check experimental features

The original research prototype remains available:

```bash
warrant brief --arxiv-id 2603.26993
warrant probe
```

Expected result: both commands still complete in mock mode. Their output is
experimental and must not appear in or modify a `ValidationReport` verdict.

## 9. Report findings

Open a
[bug report](https://github.com/spattnaik1998/warranty-application/issues/new?template=bug.yml)
for incorrect behavior. Include:

```text
Warrant commit:
Python version:
Operating system:
Input type: local / public GitHub / ZIP API
Policy:
Command:
Expected verdict and exit/status code:
Actual verdict and exit/status code:
Relevant evidence or error:
Reproduces consistently: yes / no
Execution permission enabled: yes / no
```

Never attach proprietary source, credentials, private URLs, or complete command
output containing secrets. Reduce the finding to a disposable fixture whenever
possible.

## Completion checklist

- [ ] Installation succeeds in a fresh virtual environment.
- [ ] Maintainer acceptance commands pass.
- [ ] Pass, fail, and indeterminate scenarios match their expected semantics.
- [ ] Advisory failures do not block an otherwise passing validation.
- [ ] Commands never run without explicit permission.
- [ ] JSON and Markdown reports contain traceable evidence.
- [ ] Public GitHub validation cleans up its temporary clone.
- [ ] API upload protections behave as documented.
- [ ] Legacy experimental commands still run.
- [ ] Findings are reported with a minimal, non-sensitive reproduction.
