from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ALLOW_MARKER = "secret-scan: allow"
SKIP_PREFIXES = (
    ".git/",
    ".venv/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
    "data/processed/",
    "models/",
    "src/ask_seattle.egg-info/",
)
SKIP_SUFFIXES = (".pyc",)


@dataclass(frozen=True)
class SecretPattern:
    name: str
    regex: re.Pattern[str]
    secret_group: int | None = None


@dataclass(frozen=True)
class SecretFinding:
    path: str
    line_number: int
    rule: str
    snippet: str


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern(
        name="private-key",
        regex=re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    SecretPattern(
        name="github-token",
        regex=re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    SecretPattern(
        name="openai-key",
        regex=re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    ),
    SecretPattern(
        name="aws-access-key",
        regex=re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    SecretPattern(
        name="slack-token",
        regex=re.compile(r"\bxox[aboprs]-[A-Za-z0-9-]{10,}\b"),
    ),
    SecretPattern(
        name="generic-secret-assignment",
        regex=re.compile(
            r"""(?ix)
            \b(?:api[_-]?key|token|secret|password|passwd|auth[_-]?token)\b
            \s*[:=]\s*
            ["']?(?P<secret>[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})["']?
            """
        ),
        secret_group=1,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan tracked or staged files for likely secrets.")
    parser.add_argument("--staged", action="store_true", help="Scan staged files only.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Defaults to current directory.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    findings = scan_repository(repo_root=repo_root, staged=bool(args.staged))
    if findings:
        for finding in findings:
            print(
                f"{finding.path}:{finding.line_number}: {finding.rule}: {finding.snippet}",
                file=sys.stderr,
            )
        print(
            "\nSecret scan failed. Add a placeholder, move the value to local ignored files, "
            f"or suppress a false positive with '{ALLOW_MARKER}'.",
            file=sys.stderr,
        )
        return 1
    print("secret scan passed")
    return 0


def scan_repository(*, repo_root: Path, staged: bool) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for relative_path in git_file_list(repo_root=repo_root, staged=staged):
        path = repo_root / relative_path
        if should_skip_path(relative_path) or not path.exists() or path.is_dir():
            continue
        findings.extend(scan_path(path=path, relative_path=relative_path))
    return findings


def git_file_list(*, repo_root: Path, staged: bool) -> list[str]:
    command = (
        ("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR")
        if staged
        else ("git", "ls-files")
    )
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def should_skip_path(relative_path: str) -> bool:
    normalized = relative_path.replace(os.sep, "/")
    return normalized.startswith(SKIP_PREFIXES) or normalized.endswith(SKIP_SUFFIXES)


def scan_path(*, path: Path, relative_path: str) -> list[SecretFinding]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if b"\x00" in raw:
        return []
    text = raw.decode("utf-8", errors="ignore")
    findings: list[SecretFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        line_finding: SecretFinding | None = None
        for pattern in SECRET_PATTERNS:
            for match in pattern.regex.finditer(line):
                secret = extract_secret(match=match, pattern=pattern)
                if is_placeholder_secret(secret):
                    continue
                line_finding = SecretFinding(
                    path=relative_path,
                    line_number=line_number,
                    rule=pattern.name,
                    snippet=line.strip(),
                )
                break
            if line_finding is not None:
                findings.append(line_finding)
                break
    return findings


def extract_secret(*, match: re.Match[str], pattern: SecretPattern) -> str:
    if pattern.secret_group is None:
        return match.group(0)
    return match.group(pattern.secret_group)


def is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().strip("\"'").upper()
    if not normalized:
        return True
    placeholder_markers = (
        "YOUR_",
        "EXAMPLE",
        "PLACEHOLDER",
        "CHANGEME",
        "REPLACE_ME",
        "DUMMY",
        "TEST_",
        "<",
    )
    return any(marker in normalized for marker in placeholder_markers)


if __name__ == "__main__":
    raise SystemExit(main())
