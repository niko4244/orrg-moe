"""Build a chat-format SFT corpus from a public code-instruct dataset.

The target model is a 4B base served behind the Spark harness, whose tasks are
"rewrite this function so the tests pass". That is exactly the shape of
`bigcode/self-oss-instruct-sc2-exec-filter-50k`: it is generated from real code,
filtered by execution. None of the 50 Spark suite tasks is in it, so the suite stays
a held-out measurement rather than a training signal.

    PYTHONPATH=. python train/build_sft.py --limit 6000 --out train/data/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("cache/hf").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path("cache/hf/datasets").resolve()))

DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"

# The Spark harness asks for a complete corrected function and nothing else. The raw
# dataset answers open with prose ("To handle the index out of range errors, we need
# to..."), so training on it teaches the opposite of what the suite scores: v1 of the
# adapter dropped the one-shot score from 29/50 to 20/50 exactly that way. Extract the
# code block and state the contract as a system message.
SYSTEM = (
    "You are a senior Python engineer. Return only code: a complete, syntactically "
    "valid function. Do not include prose, explanation, or markdown fences."
)
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def code_only(response: str) -> str:
    blocks = CODE_RE.findall(response)
    if blocks:
        return max(blocks, key=len).strip()
    lines = [line for line in response.splitlines() if not line.lstrip().startswith(("To ", "We ", "This ", "Here"))]
    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--out", default="train/data/sft.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    stream = load_dataset(DATASET, split="train", streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in stream:
            # The 50k build ships `instruction` / `response` (plus `seed`, `concepts`).
            problem = (row.get("instruction") or "").strip()
            solution = (row.get("response") or "").strip()
            if len(problem) < 80 or len(solution) < 40:
                continue
            if len(problem) + len(solution) > args.max_chars:
                continue
            code = code_only(solution)
            if "def " not in code and "class " not in code:
                continue
            handle.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": code},
                ],
            }) + "\n")
            kept += 1
            if kept >= args.limit:
                break
    print(f"wrote {kept} examples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
