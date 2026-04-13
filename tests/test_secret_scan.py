from __future__ import annotations

from pathlib import Path

from ask_seattle.secret_scan import ALLOW_MARKER, scan_path, scan_repository


def test_scan_path_detects_real_secret_token(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    path.write_text('token = "' + "ghp_" + ("A" * 24) + '"\n', encoding="utf-8")

    findings = scan_path(path=path, relative_path="example.py")

    assert len(findings) == 1
    assert findings[0].rule == "github-token"


def test_scan_path_ignores_placeholder_secret_value(tmp_path: Path) -> None:
    path = tmp_path / "example.env"
    path.write_text('RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"\n', encoding="utf-8")

    findings = scan_path(path=path, relative_path="example.env")

    assert findings == []


def test_scan_path_respects_allow_marker(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    path.write_text(
        'openai_api_key = "' + "sk-" + ("A" * 24) + f'"  # {ALLOW_MARKER}\n',
        encoding="utf-8",
    )

    findings = scan_path(path=path, relative_path="example.py")

    assert findings == []


def test_scan_repository_skips_ignored_paths(monkeypatch, tmp_path: Path) -> None:
    ignored = tmp_path / "data" / "processed"
    ignored.mkdir(parents=True, exist_ok=True)
    tracked = ignored / "fixture.jsonl"
    tracked.write_text('token = "' + "ghp_" + ("A" * 24) + '"\n', encoding="utf-8")

    monkeypatch.setattr(
        "ask_seattle.secret_scan.git_file_list",
        lambda repo_root, staged: ["data/processed/fixture.jsonl"],
    )

    findings = scan_repository(repo_root=tmp_path, staged=False)

    assert findings == []
