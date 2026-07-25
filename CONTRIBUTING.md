# Contributing to Warrant

Thank you for helping make software-assurance evidence easier to verify.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest --cov=warrant --cov-fail-under=80
python -m ruff check warrant/assurance warrant/app warrant/tests
```

## Pull requests

- Open an issue for substantial behavior or policy-schema changes.
- Add tests for every new claim evaluator and failure mode.
- Keep authoritative verdicts deterministic. LLM output may explain evidence but
  must not override deterministic checks.
- Never run repository-defined commands without explicit user permission.
- Update the policy reference and examples when changing a public contract.

Use focused commits and describe the user impact and validation performed in
the pull request.
