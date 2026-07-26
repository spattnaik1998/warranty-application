"""Framework adapters: translate a framework's run events into the Trace Contract.

Adapters are the *only* place a framework SDK (LangGraph, and later others) may
be imported. Everything downstream reads ``warrant.trace.contract`` and never a
framework type. See ``PRODUCT_DIRECTION.md`` §6.
"""

from __future__ import annotations

from warrant.adapters.langgraph import InstrumentedApp, instrument_langgraph

__all__ = ["InstrumentedApp", "instrument_langgraph"]
