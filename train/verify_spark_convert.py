"""Check the rebuilt HF checkpoint against the GGUF that Ollama is serving.

A conversion that loads without error can still be silently wrong (transposed weights,
off-by-one rope, bad tokenizer). Two independent checks are used here: token-id counts
must match llama.cpp's own prompt_eval_count for the same raw string, and greedy
generation must agree with the GGUF for the first several tokens.

    PYTHONPATH=. python train/verify_spark_convert.py --model cache/models/spark-x2.6-4b-hf
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch

OLLAMA = "http://127.0.0.1:11434"
PROBES = [
    "def add(a, b):\n    return a + b",
    "Fix the fractional seconds in isoparse('2021-01-01T00:00:00.123456Z').",
    "  leading spaces and\ttabs\n\nand newlines  ",
    "Ünïcödé — 中文测试 — emoji 🙂",
    "x = {'a': [1, 2, 3]}\nprint(x['a'][-1])",
]


def ollama_prompt_tokens(text: str) -> int:
    body = json.dumps({"model": "spark-x2.6-4b:q6_k", "prompt": text, "raw": True,
                       "stream": False, "options": {"num_predict": 1}}).encode()
    request = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                     headers={"content-type": "application/json"})
    payload = json.load(urllib.request.urlopen(request, timeout=180))
    return int(payload.get("prompt_eval_count", -1))


def ollama_chat(prompt: str, tokens: int) -> str:
    body = json.dumps({"model": "spark-x2.6-4b:q6_k",
                       "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "think": False,
                       "options": {"temperature": 0, "top_k": 1, "num_predict": tokens}}).encode()
    request = urllib.request.Request(f"{OLLAMA}/api/chat", data=body,
                                     headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=300))["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/models/spark-x2.6-4b-hf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--new-tokens", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    failures = 0
    for probe in PROBES:
        mine = len(tokenizer(probe, add_special_tokens=False)["input_ids"])
        theirs = ollama_prompt_tokens(probe)
        ok = mine == theirs
        failures += 0 if ok else 1
        print(f"tokens mine={mine} llama.cpp={theirs} {'OK' if ok else 'MISMATCH'} "
              f":: {probe[:40]!a}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype, device_map={"": 0}
        if args.device == "cuda" else None,
    ).eval()

    prompt = "Write a Python function that returns the sum of a list."
    expected = ollama_chat(prompt, args.new_tokens)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    produced = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\nGGUF  :", ascii(expected[:160]))
    print("HF    :", ascii(produced[:160]))
    shared = 0
    for left, right in zip(expected, produced):
        if left != right:
            break
        shared += 1
    print(f"identical leading characters: {shared}/{min(len(expected), len(produced))}")
    print(f"PROBE_FAILURES={failures}")
    return 0 if failures == 0 and shared >= args.new_tokens // 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
