"""Check repository text for common secrets without printing matched values."""

import argparse
from pathlib import Path
import re
import subprocess


PATTERNS = {
    "private key block": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "Google API key": rb"\bAIza[A-Za-z0-9_-]{35}\b",
    "Hugging Face token": rb"\bhf_[A-Za-z0-9]{30,}\b",
    "secret-style API token": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b",
    "JWT-style token": rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    "quoted credential assignment": rb'''(?i)(?:api[_-]?key|password|access[_-]?token|client[_-]?secret)\s*[=:]\s*["'][^"'\r\n]{12,}["']''',
}
COMPILED_PATTERNS = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}
PRIVATE_NAMES = {
    ".env", "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json",
    ".netrc", ".npmrc", ".pypirc",
}
PRIVATE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}


def detect_secrets(path: str, data: bytes) -> list[str]:
    """Return issue labels only; callers must never print matched content."""
    file = Path(path)
    name = file.name.casefold()
    issues = []
    if (name in PRIVATE_NAMES or file.suffix.casefold() in PRIVATE_SUFFIXES
            or (name.startswith(".env.") and name != ".env.example")
            or name.startswith(("client_secret", "service-account"))
            or any(part.casefold() in {".ssh", ".aws", ".ollama", "secrets"}
                   for part in file.parts)):
        issues.append("private credential filename")
    if b"\x00" not in data[:8000]:
        issues.extend(name for name, pattern in COMPILED_PATTERNS.items() if pattern.search(data))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Check the exact staged blobs.")
    args = parser.parse_args(argv)
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    command = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
               if args.staged else ["git", "ls-files", "-z"])
    names = subprocess.check_output(command, cwd=root).decode("utf-8").split("\0")
    failed = False
    count = 0
    for name in filter(None, names):
        data = (subprocess.check_output(["git", "show", ":" + name], cwd=root)
                if args.staged else (root / name).read_bytes())
        count += 1
        for issue in detect_secrets(name, data):
            print(f"Blocked: {name}: {issue}")
            failed = True
    if not failed:
        print(f"Secret-pattern check passed for {count} files.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
