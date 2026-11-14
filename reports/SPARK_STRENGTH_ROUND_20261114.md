# Sharp Spark strength round — 2026-11-14

## Verdict

**No verified improvement to Spark this round, and one verified regression.** The
honest headline is that the training loop now exists, its data path is verified, and
its first attempt measurably hurt the model — which is more useful than an unmeasured
claim. "Opus 5.5-level" remains out of reach on this hardware for a reason stated in
§1, not for lack of tuning.

## 1. What the environment actually allows

| fact | evidence |
|---|---|
| Gateway is live; the model answers | `POST :4000/v1/messages` → 200 in 8.2 s with the project's stored key |
| The shell's token is a placeholder | same call → 401 `LiteLLM Virtual Key expected` |
| Spark is GGUF-only locally | `cache/hf/hub` holds no weights; model lives in Ollama as `spark-x2.6-4b:q6_k` |
| No teacher model | no route to a frontier model; Opus 5.5 was already unmeasurable (HTTP 402) |
| Disk is nearly full | 36 GB free of 1.9 TB before download; 32 GB after the 7.6 GB base |

A GGUF cannot be LoRA-trained, so the only trainable path was a downloaded 4B base.
That makes the comparison "Spark vs a model trained here", not "Spark, improved".

## 2. Measurements (unchanged 50-case suite, greedy, one-shot)

| model | boltons | toolz | more-itertools | total |
|---|---:|---:|---:|---:|
| Spark (`claude-spark-x2.6-local`), prior report | 9/17 | 4/17 | 16/16 | 27/50 |
| Qwen3-4B-Instruct-2507 base, this round | 9/17 | 4/17 | 16/16 | **29/50** |
| QLoRA v1 on 6k generic code-instruct pairs | 1/17 | 3/17 | 16/16 | **20/50** |

Receipts: `reports/final_suite/qwen3_base_oneshot/`, `reports/final_suite/qwen3_qlora_v1_oneshot/`.

**The regression is real and localised**: boltons 9→1 while more-itertools is unchanged.
Training recipe: QLoRA nf4, r=16, α=32, 33.0M trainable (0.81%), 150 steps, batch 2 ×
accum 8, lr 1e-4, 1536 tokens, assistant-span loss only — a defensible recipe. The data
was wrong for the task: `self-oss-instruct-sc2` answers begin with prose ("To handle the
index out of range errors…") and teach "write a function from an instruction", while the
suite scores "given a failing test, return the corrected function, code only".

## 3. What was built and verified

- `train/build_sft.py` — streams the public corpus into chat JSONL; extract-code variant
  (`--out …/sft_code.jsonl`) strips prose and states the harness contract as the system
  message.
- `train/make_repair_data.py` — **verified repair-pair miner**: takes a real function from
  the local repos, injects exactly one mutation (comparison flip, off-by-one, negation,
  dropped argument, deleted statement), and keeps the pair only if the repository's own
  tests fail on the mutant and pass on the original. 58 pairs from 400+ mutants in 18
  minutes (~15% survival, matching published yields), each carrying real pytest output as
  the bug report. No teacher model involved.
- `train/qlora_sft.py` — single-GPU QLoRA loop: 4-bit nf4, gradient checkpointing,
  assistant-only labels, OneCycle, explicit memory budget (peaks at 15.9 GiB).
- `train/serve_anthropic.py` — serves a local adapter behind the Anthropic
  `/v1/messages` contract so `harness/final_suite.py` scores it **unchanged**; both
  measurements above went through this path.

Reproduction: `build_sft` → `qlora_sft` → `serve_anthropic --adapter` →
`SPARK_MAX_ATTEMPTS=1 harness/final_suite.py <label>`.

## 4. Why the numbers cannot be over-read

The suite is 50 test cases nested in **3 tasks**, and one task can swing 17 points.
With n=3, the smallest attainable two-sided p is 0.25, so no significance claim is
available at any score: report per-task deltas as description, not inference. Greedy
decoding is deterministic here but is one sample of a stochastic process, and the
dominant uncertainty is task sampling, which reruns cannot reduce. Anything quoted from
this suite must say "on these 50 fixed cases".

## 5. What the literature says the next attempt must change

- Include the failing test and error text in the prompt: +2–3× pass@1 on single-function
  repair (e.g. 5.9% → 29.2% Python for Claude 3.7), versus a bare function→function pair.
  The new miner already emits exactly this format.
- LoRA wants 30k–65k pairs; ~1k pairs only works with full fine-tuning (which does not
  fit 16 GB for a 4B). 58 pairs are enough to prove the pipeline, not to move the score.
- Fine-tuning on code-only targets can underperform a *non*-fine-tuned model prompted with
  localisation signals — the failure v1 exhibited.

## 6. Open

1. **Spark itself is not trainable from here.** An fp16/bf16 HF checkpoint of the base
   (or Ollama's GGUF dequantised and converted) is required; ~9 GB and a 2–3 h window.
2. **Data scale.** 58 verified pairs → target 20k–50k: mine boltons history (1,716 commits)
   plus external repos with fast suites; the miner sustains ~2.5 verified pairs/min.
3. **The knee of this suite.** Base and Spark both sit at 29/50 and 27/50 one-shot and
   both hit 16/16 on more-itertools, so the remaining headroom is boltons+toolz (25 cases).
4. **Compute is shared.** Two training runs were starved to a crawl at 100% GPU
   utilisation by other tenants; wall-clock estimates here are not reliable.
