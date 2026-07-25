"""CLI and FastAPI acceptance tests for software assurance."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from warrant.app.api import app
from warrant.app.cli import main

STATIC_POLICY = """\
version: "1"
project:
  name: interface-fixture
claims:
  - id: readme
    type: files
    paths: [README.md]
  - id: tests
    type: tests
"""


def _repository(tmp_path: Path, policy: str = STATIC_POLICY) -> Path:
    (tmp_path / "README.md").write_text("fixture", encoding="utf-8")
    (tmp_path / "warrant.yml").write_text(policy, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    return tmp_path


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            archive.writestr(f"project/{path}", content)
    return buffer.getvalue()


def test_cli_init_and_validate_json(tmp_path: Path, capsys):
    target = tmp_path / "project"
    assert main(["init", str(target)]) == 0
    assert (target / "warrant.yml").is_file()

    _repository(target)
    assert main(["validate", str(target), "--format", "json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{"):])
    assert payload["verdict"] == "pass"
    assert payload["schema_version"] == "1"


def test_cli_exit_codes_for_fail_and_indeterminate(tmp_path: Path):
    failing = """\
version: "1"
project:
  name: failing
claims:
  - id: missing
    type: files
    paths: [missing.txt]
"""
    _repository(tmp_path, failing)
    assert main(["validate", str(tmp_path)]) == 1

    indeterminate = """\
version: "1"
project:
  name: unknown
claims:
  - id: coverage
    type: coverage
    minimum: 80
"""
    (tmp_path / "warrant.yml").write_text(indeterminate, encoding="utf-8")
    assert main(["validate", str(tmp_path)]) == 2


def test_api_validates_static_archive():
    client = TestClient(app)
    archive = _zip_bytes(
        {
            "README.md": "fixture",
            "tests/test_ok.py": "def test_ok(): assert True\n",
            "warrant.yml": STATIC_POLICY,
        }
    )
    response = client.post(
        "/v1/validations",
        files={"archive": ("source.zip", archive, "application/zip")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["verdict"] == "pass"


def test_api_rejects_executable_policy():
    client = TestClient(app)
    policy = """\
version: "1"
project:
  name: executable
claims:
  - id: command
    type: command
    command: [python, -V]
"""
    archive = _zip_bytes({"README.md": "fixture", "warrant.yml": policy})
    response = client.post(
        "/v1/validations",
        files={"archive": ("source.zip", archive, "application/zip")},
    )
    assert response.status_code == 422
    assert "executable claims" in response.json()["detail"]


def test_api_rejects_unsupported_archive_and_zip_slip():
    client = TestClient(app)
    unsupported = client.post(
        "/v1/validations",
        files={"archive": ("source.tar", b"not a zip", "application/octet-stream")},
    )
    assert unsupported.status_code == 415

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    unsafe = client.post(
        "/v1/validations",
        files={"archive": ("source.zip", buffer.getvalue(), "application/zip")},
    )
    assert unsafe.status_code == 400
