"""FastAPI service for deterministic software-assurance validation.

Run with:  uvicorn warrant.app.api:app --reload
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from warrant.assurance.engine import ValidationEngine
from warrant.assurance.models import ValidationReport
from warrant.assurance.policy import PolicyError, load_policy, load_policy_bytes
from warrant.assurance.source import MAX_ARCHIVE_BYTES, SourceError, extract_zip
from warrant.orchestrator import GovernedOrchestrator
from warrant.schemas.tasks import BriefRequest, BriefResult

app = FastAPI(
    title="Warrant",
    version="1.0.0",
    description="Evidence-based software assurance for Python repositories.",
)


class BriefResponse(BaseModel):
    brief: BriefResult
    markdown: str
    gate: dict
    economics: dict


@app.post("/experimental/brief", response_model=BriefResponse, tags=["experimental"])
@app.post("/brief", response_model=BriefResponse, deprecated=True, include_in_schema=False)
def brief(request: BriefRequest) -> BriefResponse:
    """Produce a governed technical briefing and return gate/economics telemetry."""
    gov = GovernedOrchestrator().run(request)
    return BriefResponse(
        brief=gov.brief,
        markdown=gov.brief.to_markdown(),
        gate={"admitted": gov.admitted, "collapsed": gov.rejected,
              "pruned": gov.redundant, "log": gov.gate_log},
        economics=gov.econ.report(),
    )


@app.post("/experimental/probe", tags=["experimental"])
@app.post("/probe", deprecated=True, include_in_schema=False)
def probe(request: BriefRequest | None = None) -> dict:
    """Run the Delegation Ledger probe; return the scorecard and acceptance."""
    from warrant.ledger import run_probe
    from warrant.ledger.report import acceptance, render_report

    result = run_probe(request)
    return {
        "conditions": {c.value: r.summary_row() for c, r in result.runs.items()},
        "kl_accuracy_r": round(result.kl_accuracy_r, 3),
        "prose_slope_pts_per_stage": round(result.prose_slope(), 2),
        "posterior_slope_pts_per_stage": round(result.posterior_slope(), 2),
        "acceptance": acceptance(result),
        "report_markdown": render_report(result),
    }


@app.get("/experimental/ledger", tags=["experimental"])
@app.get("/ledger", deprecated=True, include_in_schema=False)
def ledger() -> dict:
    """Convenience GET that runs the default probe."""
    return probe(None)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


async def _write_bounded_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="source archive exceeds the 25 MiB limit",
                )
            handle.write(chunk)


@app.post(
    "/v1/validations",
    response_model=ValidationReport,
    tags=["assurance"],
    summary="Validate a source repository archive",
)
async def validate_archive(
    archive: UploadFile = File(..., description="ZIP archive containing a source repository"),
    policy_file: UploadFile | None = File(
        None, description="Optional warrant.yml; otherwise read from the archive"
    ),
) -> ValidationReport:
    """Validate static evidence from a bounded ZIP archive.

    Repository-defined commands are intentionally rejected by the API.
    """
    filename = archive.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="only ZIP source archives are supported")

    with tempfile.TemporaryDirectory(prefix="warrant-api-") as temp:
        temp_root = Path(temp)
        archive_path = temp_root / "source.zip"
        await _write_bounded_upload(archive, archive_path)
        extract_root = temp_root / "extracted"
        extract_root.mkdir()
        try:
            repository_root = extract_zip(archive_path, extract_root)
            if policy_file is None:
                policy, digest = load_policy(repository_root / "warrant.yml")
            else:
                data = await policy_file.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise HTTPException(status_code=413, detail="policy exceeds the 1 MiB limit")
                policy, digest = load_policy_bytes(
                    data, source=policy_file.filename or "policy"
                )
        except PolicyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SourceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        executable = [claim.id for claim in policy.claims if claim.requires_execution]
        if executable:
            raise HTTPException(
                status_code=422,
                detail="API policies may not contain executable claims: " + ", ".join(executable),
            )
        return ValidationEngine().validate(
            repository_root,
            policy,
            digest,
            allow_exec=False,
            repository_label=filename,
        )
