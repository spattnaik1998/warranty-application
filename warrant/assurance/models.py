"""Public contracts for policies, evidence, claim results, and reports."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


ClaimType = Literal[
    "files",
    "license",
    "tests",
    "coverage",
    "ci",
    "dependency_vulnerabilities",
    "command",
]


class ProjectMetadata(BaseModel):
    """Human-readable project identity included in validation reports."""

    name: str = Field(min_length=1)
    version: str | None = None
    repository: str | None = None

    model_config = {"extra": "forbid"}


class ClaimDefinition(BaseModel):
    """A single deterministic assurance claim declared in ``warrant.yml``."""

    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    type: ClaimType
    description: str = ""
    required: bool = True
    paths: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    minimum: float | None = Field(default=None, ge=0, le=100)
    report: str | None = None
    license: str | None = None
    execute: bool = False
    requirements: str = "requirements.txt"

    model_config = {"extra": "forbid", "frozen": True}

    @field_validator("paths", "command")
    @classmethod
    def _nonempty_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("list values must not be empty")
        return values

    @field_validator("paths")
    @classmethod
    def _relative_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            normalized = value.replace("\\", "/")
            if normalized.startswith("/") or ".." in normalized.split("/"):
                raise ValueError(f"unsafe repository-relative path: {value!r}")
        return values

    @model_validator(mode="after")
    def _type_contract(self) -> "ClaimDefinition":
        if self.type == "files" and not self.paths:
            raise ValueError("files claims require at least one path")
        if self.type == "command" and not self.command:
            raise ValueError("command claims require a non-empty command argument array")
        if self.type == "coverage" and self.minimum is None:
            raise ValueError("coverage claims require a minimum percentage")
        if self.type == "license" and not self.license:
            raise ValueError("license claims require an SPDX license identifier")
        if self.type != "tests" and self.execute:
            raise ValueError("execute is only valid for tests claims")
        return self

    @property
    def requires_execution(self) -> bool:
        return (
            self.type in {"command", "dependency_vulnerabilities"}
            or (self.type == "tests" and self.execute)
        )


class ValidationPolicy(BaseModel):
    """Versioned policy document loaded from ``warrant.yml``."""

    version: Literal["1"]
    project: ProjectMetadata
    claims: list[ClaimDefinition] = Field(min_length=1)

    model_config = {"extra": "forbid", "frozen": True}

    @model_validator(mode="after")
    def _unique_claim_ids(self) -> "ValidationPolicy":
        ids = [claim.id for claim in self.claims]
        duplicates = sorted({claim_id for claim_id in ids if ids.count(claim_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate claim ids: {', '.join(duplicates)}")
        return self


class Evidence(BaseModel):
    """Immutable, content-addressed observation supporting a claim result."""

    source: str
    locator: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    observation: str
    failure_reason: str | None = None

    model_config = {"extra": "forbid", "frozen": True}


class ClaimResult(BaseModel):
    """Result of evaluating one claim."""

    claim_id: str
    claim_type: ClaimType
    required: bool
    verdict: Verdict
    summary: str
    evidence: tuple[Evidence, ...] = ()
    duration_ms: int = Field(default=0, ge=0)

    model_config = {"extra": "forbid", "frozen": True}


class ValidationReport(BaseModel):
    """Authoritative output of a Warrant validation."""

    schema_version: Literal["1"] = "1"
    warrant_version: str
    project: ProjectMetadata
    repository: str
    repository_commit: str | None = None
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = Field(default=0, ge=0)
    verdict: Verdict
    claims: tuple[ClaimResult, ...]
    warnings: tuple[str, ...] = ()
    experimental: dict[str, object] | None = None

    model_config = {"extra": "forbid", "frozen": True}
