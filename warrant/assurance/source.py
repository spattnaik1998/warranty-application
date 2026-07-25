"""Safe materialization of local repositories, public Git URLs, and ZIP archives."""

from __future__ import annotations

import re
import stat
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from warrant.exceptions import WarrantError

MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_FILES = 5_000

_GITHUB_HTTPS = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$"
)


class SourceError(WarrantError):
    """Raised when repository input is unsafe or cannot be materialized."""


def is_git_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "git@", "ssh://"))


@contextmanager
def materialize_source(value: str) -> Iterator[tuple[Path, str]]:
    """Yield a local repository root and clean up temporary clones."""
    if not is_git_url(value):
        root = Path(value).resolve()
        if not root.is_dir():
            raise SourceError(f"repository directory not found: {root}")
        yield root, str(root)
        return

    if not _GITHUB_HTTPS.fullmatch(value):
        raise SourceError("v1 supports only public HTTPS github.com repository URLs")
    with tempfile.TemporaryDirectory(prefix="warrant-clone-") as temp:
        target = Path(temp) / "repository"
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--", value, str(target)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SourceError(f"repository clone failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise SourceError(f"repository clone failed: {detail}")
        yield target, value


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def extract_zip(archive: Path, destination: Path) -> Path:
    """Extract a bounded ZIP without traversal or symbolic links."""
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise SourceError(f"archive exceeds {MAX_ARCHIVE_BYTES // (1024 * 1024)} MiB limit")
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise SourceError("unsupported or invalid source archive; expected ZIP") from exc
    with zf:
        members = zf.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise SourceError(f"archive contains more than {MAX_ARCHIVE_FILES} entries")
        total = sum(info.file_size for info in members)
        if total > MAX_EXTRACTED_BYTES:
            raise SourceError("archive expands beyond the 100 MiB safety limit")
        root = destination.resolve()
        for info in members:
            normalized = info.filename.replace("\\", "/")
            if (
                not normalized
                or normalized.startswith("/")
                or ".." in normalized.split("/")
                or _member_is_symlink(info)
            ):
                raise SourceError(f"unsafe archive member: {info.filename!r}")
            target = (root / normalized).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise SourceError(f"archive member escapes extraction root: {info.filename!r}") from exc
        zf.extractall(root)

    children = [path for path in destination.iterdir() if path.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return destination
