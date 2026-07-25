"""Repository source and archive safety tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from warrant.assurance.source import SourceError, extract_zip, materialize_source


def test_local_source_materializes_without_copy(tmp_path: Path):
    with materialize_source(str(tmp_path)) as (root, label):
        assert root == tmp_path.resolve()
        assert label == str(tmp_path.resolve())


def test_non_github_remote_is_rejected():
    with pytest.raises(SourceError, match="github.com"):
        with materialize_source("https://example.com/repository.git"):
            pass


def test_zip_slip_is_rejected(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "bad")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(SourceError, match="unsafe"):
        extract_zip(archive, destination)


def test_single_archive_root_is_selected(tmp_path: Path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("project/README.md", "hello")
    destination = tmp_path / "out"
    destination.mkdir()
    root = extract_zip(archive, destination)
    assert root.name == "project"
    assert (root / "README.md").read_text() == "hello"
