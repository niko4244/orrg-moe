"""Merge, de-duplicate and shuffle every verified repair file into one training set.

Workers write separately so they can run in parallel; this collapses them. Duplicates
happen because different commits can produce the same (prompt, fix) pair, so the key is
the exact user message plus the target, not the file it came from.

    PYTHONPATH=. python train/merge_repair.py --out train/data/repair_all.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="train/data/hist_h*.jsonl,train/data/repair_*.jsonl,"
                                             "train/data/repair.jsonl")
    parser.add_argument("--out", default="train/data/repair_all.jsonl")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    seen: set[str] = set()
    rows: list[dict] = []
    for pattern in args.pattern.split(","):
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row["messages"]
                key = hashlib.sha256(
                    (messages[1]["content"] + "\x00" + messages[2]["content"]).encode("utf-8")
                ).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                row["source_file"] = Path(path).name
                rows.append(row)

    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # Size statistics matter for the report: sequence length drives the step budget.
    user_chars = [len(row["messages"][1]["content"]) for row in rows]
    assistant_chars = [len(row["messages"][2]["content"]) for row in rows]
    sources = Counter(row.get("source_file", "?") for row in rows)
    print(f"wrote {len(rows)} unique pairs to {out}")
    print("by file:", dict(sources))
    if rows:
        print(f"prompt chars mean {sum(user_chars)/len(user_chars):.0f} "
              f"max {max(user_chars)}; target chars mean "
              f"{sum(assistant_chars)/len(assistant_chars):.0f} max {max(assistant_chars)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
