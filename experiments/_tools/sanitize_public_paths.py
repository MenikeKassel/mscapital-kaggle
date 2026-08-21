#!/usr/bin/env python
"""Replace machine-specific roots in active, publishable text files."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SKIP = {ROOT / "docs" / "_archive", ROOT / "_archive", ROOT / "research" / "paper-reading-2026-08"}

def skipped(p): return any(str(p).startswith(str(x)) for x in SKIP)

def main():
    files = [ROOT / "README.md", ROOT / "CONTEXT.md", ROOT / "experiments" / "registry.csv"]
    files += [p for p in (ROOT / "docs").rglob("*.md") if not skipped(p)]
    files += [p for p in (ROOT / "experiments").rglob("*.md") if not skipped(p)]
    patterns = [
        (r"[A-Za-z]:[/\\][^`\s|,)]+", "<local-path>"),
        (r"[A-Za-z]:[/\\]", "<local-path>/"),
    ]
    changed = 0
    for p in sorted(set(files)):
        if not p.exists(): continue
        text = p.read_text(encoding="utf-8")
        new = text
        for pat, repl in patterns: new = re.sub(pat, repl, new)
        if new != text:
            p.write_text(new, encoding="utf-8"); changed += 1
    print(f"sanitized {changed} files")

if __name__ == "__main__": main()
