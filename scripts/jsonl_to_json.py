"""Convert a JSONL file (one JSON object per line) to a single JSON array file.

Usage:
    uv run python scripts/jsonl_to_json.py <input.jsonl> [output.json]

If ``output.json`` is omitted, writes alongside the input with a ``.json`` suffix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> int:
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(records)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print(__doc__, file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2]) if len(argv) == 3 else input_path.with_suffix(".json")
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1
    n = convert(input_path, output_path)
    print(f"wrote {n} records → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
