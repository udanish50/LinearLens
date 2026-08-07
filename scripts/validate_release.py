from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build"}
BLOCKED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".mat",
    ".sav",
    ".sqlite",
    ".db",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pdf",
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".cff", ".bib", ".sh"}
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked release artifact: {rel}")
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Makefile", ".env.example"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    failures.append(f"possible secret pattern in: {rel}")
    if failures:
        print("Release validation FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("Release validation passed: no blocked research artifacts or obvious secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
