"""Strict YAML policy loading and starter-policy generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import ValidationError

from warrant.assurance.models import ValidationPolicy
from warrant.exceptions import WarrantError


class PolicyError(WarrantError):
    """Raised when a policy is missing, malformed, or semantically invalid."""


STARTER_POLICY = """\
version: "1"
project:
  name: my-python-project
claims:
  - id: required-files
    type: files
    description: Essential project files are present.
    paths:
      - README.md
      - pyproject.toml
  - id: mit-license
    type: license
    description: The project declares the MIT license.
    license: MIT
  - id: tests-present
    type: tests
    description: Automated tests are present.
  - id: github-actions
    type: ci
    description: A GitHub Actions workflow is configured.
    required: false
"""


def load_policy_bytes(data: bytes, *, source: str = "warrant.yml") -> tuple[ValidationPolicy, str]:
    """Parse policy bytes and return the model plus its SHA-256 digest."""
    digest = hashlib.sha256(data).hexdigest()
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PolicyError(f"{source}: invalid UTF-8 YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyError(f"{source}: policy root must be a mapping")
    try:
        return ValidationPolicy.model_validate(raw), digest
    except ValidationError as exc:
        raise PolicyError(f"{source}: invalid policy\n{exc}") from exc


def load_policy(path: str | Path) -> tuple[ValidationPolicy, str]:
    policy_path = Path(path)
    if not policy_path.is_file():
        raise PolicyError(f"policy file not found: {policy_path}")
    return load_policy_bytes(policy_path.read_bytes(), source=str(policy_path))


def write_starter_policy(directory: str | Path) -> Path:
    """Create ``warrant.yml`` without replacing an existing policy."""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "warrant.yml"
    if target.exists():
        raise PolicyError(f"refusing to overwrite existing policy: {target}")
    target.write_text(STARTER_POLICY, encoding="utf-8")
    return target
