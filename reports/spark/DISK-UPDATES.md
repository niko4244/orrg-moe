# Spark material from the drives — what is in git, what is not, and why

Captured 2026-11-13. The Spark work lives outside this repository on two drives;
this file and the manifests beside it are the in-repo record. Every artifact is
pinned by SHA-256 in `MANIFEST_X3.json`, so the weights can be verified after
being fetched from wherever they are actually stored.

## What is here

| file | what it is |
|---|---|
| `MANIFEST_X3.md` | human-readable manifest: 332 artifacts, 113.1 GiB, size + SHA-256 per file |
| `MANIFEST_X3.json` | the same manifest, machine-readable (164 KB) |
| `r5v2-repos.json` | repo list for the R5v2 qualification workspace |

In-repo code and results that predate this capture:

- `harness/spark_eval.py` — the evaluation harness (byte-recorded prompts, per-test scoring).
- `reports/FINAL_SPARK_VS_OPUS_20261108.md` — 49/50 two-run verdict.
- `reports/final_suite/` — receipts, attempt prompts/responses and hashes for both runs.

## What is on the drives and not here

**`I:\spark-enh`** (root of the X3 manifest, 113.1 GiB total)

- bf16 parent `out/rawfix_v5/base_v13_hf/`: three safetensors shards (3.70 + 3.71 GiB + 250 MiB), tokenizer 9.6 MiB, `modeling_spark.py`, `configuration_spark.py`, `merge_receipt.json`.
- Adapters `out/rawfix_v5/train_r5/adapter/`: `adapter_model.safetensors` 123.8 MiB plus `adapter_config.json`; the manifest also carries the R6 adapter group.
- Training/qualification workspaces, venvs, gocache/report directories, `HANDOFF*.md`, `MANIFEST_X3.*`.

**`I:\spark-r5v2-next-qualification-20261111`** (709 KB of files)

- ksuid and go-logr qualification witnesses: each repo has a parent run and a fixed run, captured as `.stdout`/`.stderr` plus Go build caches, with ksuid JSON traces (`ksuid-json-parent.jsonl` 16.8 KB, `ksuid-json-fixed.jsonl` 15.1 KB).
- Preflight directories `post-21.24-preflight-20261111`, `post-round2-report-preflight-20261111`, `round2-screen-01`, `new-tranche-02`, `witness`, and the extractor scripts.

**`G:\spark-source-gold`** — benchmark source material only: `attrs-zerolog`, `benchmark-pool`, `candidates`, `trust`, `trust-witness-draft`.

## Why the weights and adapters are not committed

GitHub rejects files over 100 MB and soft-caps repositories at 1–2 GB. The bf16
parent alone is 7.7 GiB; the manifest root is 113.1 GiB. A Git LFS data pack for
that volume is a paid, per-GB decision, not something to do implicitly.

The manifest is the safe substitute: it records path, size and SHA-256 for all 332
artifacts, so a checkout can verify anything fetched from elsewhere. If the weights
must be published, in order of practicality:

1. Hugging Face Hub repo (weights + adapters, with the same hashes in the card).
2. A GitHub Release asset — 2 GiB per-file cap, so the shards would need splitting.
3. Git LFS with a purchased data pack, for 113 GiB.

## Open question

`I:\spark-enh` also carries venvs, caches and multiple `_spark_full_pytest_*` report
directories. Mirroring the whole tree into this repo would add hundreds of MB of
non-source files, so only the manifests were copied. Say the word if specific
subdirectories (for example `out/rawfix_v5/train_r5/adapter/adapter_config.json`)
should be checked in as well.
