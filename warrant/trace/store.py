"""Trace persistence: an in-memory store and a SQLite-backed store.

The store is the boundary the future ``warrant-cloud`` ingestion endpoint will
sit behind (see ``PRODUCT_DIRECTION.md`` §3). For the OSS SDK it is deliberately
minimal: append a ``RunTrace``, list them, fetch one. SQLite serializes each
trace as the contract's own JSON so the schema and the storage never drift.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock

from warrant.trace.contract import RunTrace


class TraceStore:
    """In-process trace store. Thread-safe append/read for a single session."""

    def __init__(self) -> None:
        self._runs: list[RunTrace] = []
        self._lock = Lock()

    def add(self, trace: RunTrace) -> RunTrace:
        with self._lock:
            self._runs.append(trace)
        return trace

    def all(self) -> list[RunTrace]:
        with self._lock:
            return list(self._runs)

    def get(self, run_id: str) -> RunTrace | None:
        with self._lock:
            for r in self._runs:
                if r.run_id == run_id:
                    return r
        return None

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)


class SQLiteTraceStore(TraceStore):
    """Durable trace store. Serializes each RunTrace as contract JSON.

    Reuses the in-memory list as a read cache; every ``add`` also writes
    through to SQLite so a process restart (or a cloud ingest job) can rehydrate
    via :meth:`load`.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = str(path)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                run_id     TEXT PRIMARY KEY,
                graph_name TEXT,
                created_at TEXT,
                payload    TEXT NOT NULL
            )
            """
        )
        self._db.commit()
        self.load()

    def add(self, trace: RunTrace) -> RunTrace:
        super().add(trace)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO traces (run_id, graph_name, created_at, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    trace.run_id,
                    trace.graph_name,
                    trace.created_at.isoformat(),
                    trace.model_dump_json(),
                ),
            )
            self._db.commit()
        return trace

    def clear(self) -> None:
        """Drop every trace, on disk as well as in memory.

        Overriding matters: without it ``session()`` and ``reset()`` would empty
        the cache while the rows survived, and the next ``load()`` would
        resurrect traces the caller believed they had discarded.
        """
        with self._lock:
            self._db.execute("DELETE FROM traces")
            self._db.commit()
        super().clear()

    def load(self) -> list[RunTrace]:
        """Rehydrate the in-memory cache from disk. Returns all loaded traces."""
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM traces ORDER BY created_at"
            ).fetchall()
            self._runs = [RunTrace.model_validate(json.loads(payload)) for (payload,) in rows]
            return list(self._runs)

    def close(self) -> None:
        self._db.close()


__all__ = ["TraceStore", "SQLiteTraceStore"]
