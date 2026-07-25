"""FastAPI service exposing /brief, /probe, and /ledger.

Run with:  uvicorn warrant.app.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from warrant.orchestrator import GovernedOrchestrator
from warrant.schemas.tasks import BriefRequest, BriefResult

app = FastAPI(title="Warrant", version="0.1.0",
              description="A delegation-economics orchestrator.")


class BriefResponse(BaseModel):
    brief: BriefResult
    markdown: str
    gate: dict
    economics: dict


@app.post("/brief", response_model=BriefResponse)
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


@app.post("/probe")
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


@app.get("/ledger")
def ledger() -> dict:
    """Convenience GET that runs the default probe."""
    return probe(None)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
