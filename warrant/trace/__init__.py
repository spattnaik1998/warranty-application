"""The Trace Contract — Warrant's single framework-agnostic seam.

Every producer (the internal orchestrator, the LangGraph adapter, any future
OTel/cloud ingestion) writes these types; every analyzer in
``warrant.analysis`` reads them. No analyzer may import a framework SDK
directly — adapters translate framework events into this contract, and the
analysis layer consumes only the contract. See ``PRODUCT_DIRECTION.md`` §6.
"""

from __future__ import annotations

from warrant.trace.contract import (
    DecisionPoint,
    NodeRun,
    Outcome,
    RunStatus,
    RunTrace,
    ToolCallRecord,
)
from warrant.trace.store import SQLiteTraceStore, TraceStore

__all__ = [
    "RunTrace",
    "NodeRun",
    "ToolCallRecord",
    "DecisionPoint",
    "Outcome",
    "RunStatus",
    "TraceStore",
    "SQLiteTraceStore",
]
