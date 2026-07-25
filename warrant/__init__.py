"""Warrant — a delegation-economics orchestrator.

Every sub-agent must earn a *warrant* to exist: it is admitted into the
multi-agent network only if it injects a new *exogenous* signal (a tool
output, a retrieval, an environment observation) or performs a
non-redundant external check. Delegations that merely reorganize evidence
already in the shared belief state are rejected or collapsed into a single
call, because — per Ao, Gao & Simchi-Levi (arXiv:2603.26993) — such hops are
decision-theoretically dominated by a centralized Bayes decision-maker and
only add posterior distortion (the "telephone game").
"""

from __future__ import annotations

import warnings

# Silence a noisy pending-deprecation warning emitted by langgraph's serializer
# on import; it does not affect our usage.
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
)

__version__ = "1.0.0"
