"""Tests for the C13 snapshot-from-GitHub-release boot fetch (no network)."""

import io
import json
import zipfile
from unittest.mock import MagicMock, patch

from pipeline.vector_store import fetch_snapshot_from_github_release


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _fake_response(data: bytes):
    response = MagicMock()
    response.read.return_value = data
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_noop_when_env_vars_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("SNAPSHOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SNAPSHOT_GITHUB_REPO", raising=False)
    monkeypatch.delenv("SNAPSHOT_GITHUB_RELEASE_ID", raising=False)

    assert fetch_snapshot_from_github_release(tmp_path / "snapshot") is False


def test_downloads_and_extracts_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPSHOT_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("SNAPSHOT_GITHUB_REPO", "elliepsc/homebarista")
    monkeypatch.setenv("SNAPSHOT_GITHUB_RELEASE_ID", "42")

    release_json = json.dumps({
        "assets": [
            {"name": "chroma_snapshot.zip", "url": "https://api.github.com/asset/1"},
        ]
    }).encode()
    zip_payload = _zip_bytes({"chroma.sqlite3": b"fake-db-bytes"})

    responses = [_fake_response(release_json), _fake_response(zip_payload)]
    with patch("pipeline.vector_store.urlopen", side_effect=responses) as mock_urlopen:
        dest = tmp_path / "snapshot"
        result = fetch_snapshot_from_github_release(dest)

    assert result is True
    assert (dest / "chroma.sqlite3").read_bytes() == b"fake-db-bytes"
    assert mock_urlopen.call_count == 2
    first_request = mock_urlopen.call_args_list[0].args[0]
    assert first_request.headers["Authorization"] == "Bearer tok"


def test_missing_asset_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPSHOT_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("SNAPSHOT_GITHUB_REPO", "elliepsc/homebarista")
    monkeypatch.setenv("SNAPSHOT_GITHUB_RELEASE_ID", "42")

    release_json = json.dumps({"assets": []}).encode()
    with patch("pipeline.vector_store.urlopen", return_value=_fake_response(release_json)):
        result = fetch_snapshot_from_github_release(tmp_path / "snapshot")

    assert result is False
    assert not (tmp_path / "snapshot").exists()


def test_network_error_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPSHOT_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("SNAPSHOT_GITHUB_REPO", "elliepsc/homebarista")
    monkeypatch.setenv("SNAPSHOT_GITHUB_RELEASE_ID", "42")

    from urllib.error import URLError

    with patch("pipeline.vector_store.urlopen", side_effect=URLError("no network")):
        result = fetch_snapshot_from_github_release(tmp_path / "snapshot")

    assert result is False
