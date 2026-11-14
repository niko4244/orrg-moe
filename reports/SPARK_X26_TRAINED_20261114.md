# Sharp Spark X2.6-4B trained on this machine — 2026-11-14

## Verdict

**Spark itself is trainable here.** The blocker was never the hardware or a missing
download: the only Spark X2.6 copy on this machine is a Q6_K GGUF, which PEFT cannot
train, but the machine also holds a complete HF implementation of Spark's own
`spark2_5` architecture. Dequantising the GGUF into that implementation produced a
normal bf16 checkpoint, and QLoRA then ran on Spark's own weights.

Measured, same 50 cases, same prompts (all 50 prompt sha256 identical), same decoding
(temperature 0, max_tokens 3000, one attempt, thinking disabled), same shim:

| arm | boltons | toolz | more-itertools | total |
|---|---:|---:|---:|---:|
| Spark X2.6 base (rebuilt HF weights, no adapter) | 9/17 | 4/17 | 16/16 | **29/50** |
| Spark X2.6 + repair-format LoRA (`spark_repair_v1`) | 4/17 | 5/17 | 15/16 | **24/50** |

Paired 2x2, per case:

| | adapter passes | adapter fails |
|---|---:|---:|
| **base passes** | 23 | 6 |
| **base fails** | 1 | 20 |

- **1 fail→pass**, **6 pass→fail**, net −5.
- McNemar exact on the discordant pairs (1 vs 6): **p = 0.125**.
- Per task: boltons −5 (0 up, 5 down), toolz **+1** (1 up, 0 down), more-itertools −1.
  Task-level McNemar with 3 tasks: p = 1.00, and 0.25 is the smallest attainable
  two-sided p at that granularity, so no task-level claim is available either way.

**Plain verdict: regression, not an improvement.** The one genuine gain
(`test_is_valid_curry`) is outweighed by six losses concentrated in the boltons
fractional-second ISO-8601 cases. This is the fourth adapter on this suite (instruction
corpus 20/50, Qwen repair r16 22/50, Qwen repair r8 27/50, Spark repair 24/50) against
29/50 bases, and the first one measured on Spark's real weights rather than a stand-in.

## Artefact inventory: what was actually searched and found

Whole-machine sweep (name sweep over `C:\Users\nikma`, `I:`, `D:`, `F:`, `G:`; `*.gguf`
and `*.safetensors` over 500 MB everywhere; HF hub caches; LM Studio; every Ollama store
including `manifests-v2`).

| path | size | format | what it is |
|---|---:|---|---|
| `F:\Ollama\models\blobs\sha256-15bbbe…74581` | 3,379,174,144 B | GGUF Q6_K | **Spark X2.6 4B** weights (`Spark-X2.6-4B-abliterated-FIT-REFERENCE-3G-Q6_K.gguf`) |
| `F:\Ollama\models\manifests-v2\ollama.com\library\spark-x2.6-4b\q6_k` | 380 B | JSON manifest | points at the blob above |
| `C:\Users\nikma\.cache\huggingface\hub\models--XHToken--Spark-X2.6-4B\refs\main` | 40 B | ref only | **no blobs, no snapshots** — download never completed |
| `…\models--huihui-ai--Huihui-Spark-X2.5-4B-abliterated\refs\main` | 40 B | ref only | same, empty |
| `C:\Users\nikma\.cache\huggingface\modules\transformers_modules\base_v13_hf\` | 24 KB | Python | Spark's own `configuration_spark.py` + `modeling_spark.py` (remote code) |
| `I:\llm-models\sharp-spark-x2.5-4b\Sharp-Spark-X2.5-4B-Q6_K_XL.gguf` | 3,612,423,424 B | GGUF Q6_K | Spark **X2.5**, not X2.6 |
| `I:\llm-models\sharp-spark-x2.5-4b\llama.cpp\conversion\spark2_5.py` | 3 KB | Python | HF→GGUF converter for this architecture |
| `I:\llm-models\sharp-spark-x2.5-4b\chat_template.jinja` | 9.6 KB | Jinja | chat template (also embedded in the GGUF) |
| `F:\Ollama\models\manifests-v2\…\spark-r5-q6, spark-r6-q6, spark-t1-q6` | 3.38 GB each | GGUF Q6_K | earlier fine-tune attempts, GGUF only |

No fp16/bf16 safetensors or `.bin` shards of Spark existed anywhere on this machine
before this round. The GGUF was therefore the only weight source, and conversion was the
only route to training.

## What the model says it is (from the GGUF's own metadata, not file names)

```
general.architecture        spark2_5
general.name                Spark X2.6 4B Abliterated
spark2_5.block_count        36
spark2_5.embedding_length   2560
spark2_5.feed_forward_length 10240
spark2_5.attention.head_count / head_count_kv  16 / 4
spark2_5.attention.key_length / value_length   256 / 256
spark2_5.vocab_size         131072
spark2_5.context_length     1048576
spark2_5.attention.sliding_window 512, pattern TTTF…
spark2_5.rope.freq_base / _swa    5e6 / 1e4, dims 64 / 256
tokenizer.ggml.model        gpt2 (BPE, 131072 tokens, 130716 merges)
```

## Conversion: six defects that had to be fixed to get a working model

The conversion is `train/spark_gguf_to_hf.py`; the vendored, patched implementation is
`train/spark_hf/`. Each of these produced a model that *loaded* but was wrong, which is
why the verification below is behavioural rather than structural:

1. **`_tied_weights_keys` is a dict in transformers 5.16**, not the list the shipped
   code used — construction failed outright.
2. **Mask API rename**: `create_causal_mask(input_embeds=…)` → `inputs_embeds=…`, and
   `cache_position` is no longer accepted.
3. **`allow_is_causal_skip=False`**: the helper returns `None` when it can defer to
   SDPA's `is_causal` flag, but this model always uses eager attention, so attention
   silently became bidirectional and generation degenerated into repetition.
4. **GGUF is `ne0`-fastest**: a 2-D tensor needs `reshape(shape[::-1])`. Reshaping to
   `(ne0, ne1)` and calling `.T` permutes rows instead of transposing them — every shape
   assertion passes, the statistics look right, and the model is garbage. This was the
   defect that mattered most.
5. **bitsandbytes `Linear.weight.dtype` is `uint8`**, so the layer's
   `hidden_states.to(weight.dtype)` turned activations into Byte tensors and attention
   raised `baddbmm_cuda not implemented for 'Byte'`. Replaced with a compute-dtype
   helper.
6. **Special tokens must be registered from `tokenizer.ggml.token_type`** (113 of them).
   Without that, `<|User|>`/`<|Bot|>` were byte-split instead of matched atomically and
   the model never saw its control tokens.

Two further traps worth recording: transformers caches remote code by repo id, so the
first several runs used a stale copy of the implementation (cache had zero of the fixes);
and the model's template defaults to thinking *on*, which changes the prompt entirely —
the shim renders with `enable_thinking=False` for both arms.

## Fidelity of the rebuilt checkpoint

- **Tokenizer: 12/12, 37/37, 13/13, 14/14, 23/23 token ids** — exact match against
  llama.cpp's own `prompt_eval_count` for code, prose, whitespace-heavy, non-ASCII/emoji
  and dictionary probes.
- **Generation**: the rebuilt model answers coherently (`Write a one-liner that sums a
  list named xs` → `` `sum(xs)` ``, 1.0 s); before defect 4 was fixed the same prompt
  produced repeated single tokens.
- **Suite-level**: the rebuilt base scores **29/50**, in line with the GGUF-backed Spark
  reference (27/50 one-shot) and byte-identical prompts to the earlier runs.

Rebuilt checkpoint: `cache/models/spark-x2.6-4b-hf`, 7.7 GB, 3 safetensors shards,
4.112B params, 290/290 GGUF tensors mapped, 131,072-token vocabulary.

## Training on Spark's own weights

| | value |
|---|---|
| base | `cache/models/spark-x2.6-4b-hf`, 4-bit nf4 (double-quant), bf16 compute |
| attention | `eager` (the architecture provides no SDPA path) |
| adapter | LoRA r8, α16 on `q_k_v_proj, g_proj, out_proj, gate_proj, up_proj, down_proj` |
| trainable | 16,224,768 / 4,128,304,128 = **0.39%** |
| data | `train/data/repair_all.jsonl`, 234 verified pairs, sha256 `5eab4b30…a978425f` |
| schedule | OneCycle, lr 1e-4, batch 1 × grad-accum 4, 1,216-token cap |
| optimiser steps | **60** |
| micro-batches | **240** |
| tokens trained | **230,672** (padding excluded) |
| final loss | 0.0976 |
| wall clock | 8.0 s/step |
| adapter size | 72 MB |

Serving footprint: 10,052 MiB used of 16,311 MiB on the shared card with the merged
adapter loaded (the card was held at 98–100% by other tenants at several points this
round, which is why the footprint stayed at batch 1 and a 1,216-token cap).

## Receipts

- `reports/final_suite/spark_x26_base_oneshot/` — Spark base arm, `receipt.json` + per-case
  `cases/*/case.json` (function, test output, per-case PASSED/FAILED).
- `reports/final_suite/spark_x26_repair_oneshot/` — Spark + adapter arm, same layout.
- `reports/spark_eval_pairs/spark_base_vs_repair.md` / `.json` — the 2x2, per-case
  McNemar, per-task deltas and the enumerated flips, regenerated from those receipts by
  `train/eval_pairs.py`.

## Reproduction

```bash
# 1. rebuild trainable Spark weights from the GGUF (≈6 min, 8 GB disk)
PYTHONPATH=. python train/spark_gguf_to_hf.py \
  --gguf "F:/Ollama/models/blobs/sha256-15bbbe…74581" --out cache/models/spark-x2.6-4b-hf

# 2. prove the rebuild matches llama.cpp
PYTHONPATH=. python train/verify_spark_convert.py --model cache/models/spark-x2.6-4b-hf

# 3. train on the verified repair corpus
PYTHONPATH=. python train/qlora_sft.py --model cache/models/spark-x2.6-4b-hf \
  --trust-remote-code --attn-impl eager \
  --target-modules q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj \
  --steps 60 --batch 1 --accum 4 --max-len 1216 --lr 1e-4 --rank 8 \
  --data train/data/repair_all.jsonl --out train/adapters/spark_repair_v1

# 4. score both arms through the unchanged shim
PYTHONPATH=. python train/serve_anthropic.py --model cache/models/spark-x2.6-4b-hf \
  --port 4020 --trust-remote-code --no-thinking            # base
PYTHONPATH=. python train/serve_anthropic.py --model cache/models/spark-x2.6-4b-hf \
  --adapter train/adapters/spark_repair_v1 --port 4021 --trust-remote-code --no-thinking

ANTHROPIC_BASE_URL=http://127.0.0.1:4020 ANTHROPIC_AUTH_TOKEN=x SPARK_MODEL=spark-x2.6-4b-hf \
  SPARK_MAX_ATTEMPTS=1 PYTHONPATH=. python harness/final_suite.py spark_x26_base_oneshot

# 5. paired analysis
PYTHONPATH=. python train/eval_pairs.py --base reports/final_suite/spark_x26_base_oneshot \
  --adapter reports/final_suite/spark_x26_repair_oneshot \
  --out reports/spark_eval_pairs/spark_base_vs_repair.md
```

## What this settles, and what it does not

Settled: Spark X2.6-4B **can** be trained on this machine, its own weights and tokenizer
are usable outside Ollama, and its paired base score on the 50 cases is 29/50. Nothing
about Spark now depends on the Qwen stand-in.

Not settled: any improvement. 234 pairs is ~1% of the 30k–65k that LoRA recipes report
for behaviour change, and the adapter's damage is concentrated in exactly the capability
the suite needs most (fractional-second ISO-8601 parsing). The honest next step is data
scale, not more epochs of the same 234 pairs; "Opus 5.5-level" remains out of reach and
nothing here is progress toward it.

Also open: the rebuilt checkpoint lives in `cache/` and is gitignored (8 GB,
regenerated in ~6 minutes from the GGUF by the committed converter); the GGUF blob must
remain at its current path for the command above to work.
