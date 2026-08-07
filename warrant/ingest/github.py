"""Fetch a repository's Python source for a static, read-only scan.

Accepts a GitHub URL (``https://github.com/owner/repo``), an ``owner/repo``
slug, or a path to a local directory. GitHub targets are fetched with a shallow
``git clone --depth 1`` into a temp dir; nothing is imported or run. Only source
text is read — never secrets, never execution.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warrant.logging_setup import get_logger, log_event

log = get_logger("warrant.ingest.github")

# Directories that never hold the app's own agent code — skip for speed + noise.
_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", "site-packages",
    ".tox", ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs",
}
_MAX_FILE_BYTES = 1_000_000  # skip vendored megafiles

_GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com[:/]+(?P<owner>[^/]+)/(?P<repo>[^/#?]+)",
    re.IGNORECASE,
)
_SLUG = re.compile(r"^(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)$")


@dataclass(frozen=True)
class RepoRef:
    """A resolved scan target.

    ``local_path`` is where the source lives on disk; ``label`` is a short human
    name for the report; ``cleanup`` is the temp dir to remove when done (``None``
    for a target that was already local).
    """

    local_path: Path
    label: str
    is_remote: bool
    cleanup: Path | None = None


def _parse_github(target: str) -> tuple[str, str] | None:
    m = _GITHUB_URL.match(target.strip())
    if m:
        return m["owner"], re.sub(r"\.git$", "", m["repo"])
    m = _SLUG.match(target.strip())
    if m and "." not in m["owner"] + m["repo"].replace(".git", ""):
        return m["owner"], re.sub(r"\.git$", "", m["repo"])
    return None


def resolve_target(target: str, ref: str | None = None) -> RepoRef:
    """Resolve a GitHub URL / ``owner/repo`` / local path to a :class:`RepoRef`.

    Raises ``ValueError`` for an unusable target and ``RuntimeError`` if a clone
    fails (e.g. private repo without credentials, or ``git`` not installed).
    """
    candidate = Path(target).expanduser()
    if candidate.is_dir():
        return RepoRef(local_path=candidate, label=candidate.name, is_remote=False)

    parsed = _parse_github(target)
    if parsed is None:
        raise ValueError(
            f"'{target}' is not a local directory, a GitHub URL, or an owner/repo slug."
        )
    owner, repo = parsed
    url = f"https://github.com/{owner}/{repo}.git"
    tmp = Path(tempfile.mkdtemp(prefix="warrant_scan_"))
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(tmp)]
    log_event(log, "cloning repo", stage="ingest", url=url, status="start")
    # Never let git block on a credential prompt: stdout/stderr are captured, so a
    # prompt would be invisible and look like a 180-second hang. A private or
    # missing repo must fail fast with git's own message instead.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "GCM_INTERACTIVE": "never"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    except FileNotFoundError as exc:  # git not installed
        raise RuntimeError("git is required to scan a GitHub repo but was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Cloning {url} timed out after 180s.") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to clone {url} (is it public?). git said:\n{proc.stderr.strip()}"
        )
    log_event(log, "cloned repo", stage="ingest", url=url, status="ok")
    return RepoRef(local_path=tmp, label=f"{owner}/{repo}", is_remote=True, cleanup=tmp)


def collect_python_sources(root: Path) -> dict[str, str]:
    """Return ``{relative_posix_path: source}`` for every .py file under ``root``.

    Skips vendored/virtualenv directories and unreadable or oversized files.
    """
    sources: dict[str, str] = {}
    for py in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in py.parts):
            continue
        try:
            if py.stat().st_size > _MAX_FILE_BYTES:
                continue
            # utf-8-sig strips a leading BOM (common on Windows) that ast rejects.
            text = py.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        rel = py.relative_to(root).as_posix()
        sources[rel] = text
    return sources


def _force_remove(func: Any, path: str, _exc: Any) -> None:
    """Clear the read-only bit and retry.

    Git marks files under ``.git/objects`` read-only, which makes ``rmtree`` fail
    on Windows. Previously that failure was swallowed by ``ignore_errors``, so a
    full repository copy was left in the temp directory after every scan.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup_ref(ref: RepoRef) -> None:
    """Remove a temp clone if one was created (no-op for local targets)."""
    if ref.cleanup is None:
        return
    # `onexc` replaced `onerror` in 3.12; both take the same three arguments.
    handler = "onexc" if sys.version_info >= (3, 12) else "onerror"
    try:
        shutil.rmtree(ref.cleanup, **{handler: _force_remove})
    except OSError as exc:
        # The scan itself already succeeded; say what was left behind rather than
        # failing the command or silently leaking megabytes per run.
        log_event(
            log, "temp clone not removed", stage="ingest",
            path=str(ref.cleanup), status="warn", error=str(exc),
        )


__all__ = ["RepoRef", "resolve_target", "collect_python_sources", "cleanup_ref"]
