#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTIONS_DIR="$SCRIPT_DIR/sections"

python3 - <<'PY'
import re
from pathlib import Path

sections_dir = Path("/home/baris/repos/trm-original/docs/project-report/sections")

pattern = re.compile(r"(?<!!)\[([a-z][a-z0-9-]*(?:;\s*[a-z][a-z0-9-]*)*)\](?!\()")

def convert(match: re.Match) -> str:
    content = match.group(1)
    if ";" in content:
        parts = [f"@{p.strip()}" for p in content.split(";")]
        return "[" + "; ".join(parts) + "]"
    return f"[@{content}]"

for path in sorted(sections_dir.glob("*.md")):
    text = path.read_text()
    updated = pattern.sub(convert, text)
    if updated != text:
        path.write_text(updated)
        print(f"Updated: {path.name}")
PY
