"""Fused vs split gate/up on the real Qwen2.5-1.5B widths.

The 190 MB split threshold was tuned on the synthetic harness, where a segment is
384 wide. The real seed's segments are 1120 wide and the gate/up output 2240, so
this repeats the comparison on the checkpoint's own weights and widths.

The full 9.06B conversion needs ~18 GB in bf16 and does not fit a 16 GB card, so
four of the 28 layers are kept: identical widths, weights, router and 60-segment
geometry, one seventh of the depth. Logits are vocab-sized (152k), so the
equivalence check runs at the smallest batch and frees them immediately.

    HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/real_threshold.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from experiments.harness import DEVICE, GROUPS, SHARED, VARIANTS, moe_layers
from experiments.prefill_structure import split_forward
from orrg import MoEConfig, convert_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS_USED, SEQ = 4, 128
BATCHES = (8, 16, 32, 48)
REPS = 3


def load(layers: int = LAYERS_USED):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True,
    )
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:layers])
    model.config.num_hidden_layers = layers
    model.config.use_cache = False
    return model.to(DEVICE).eval()


def timed_tps(fn, tokens: int) -> float:
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return tokens / statistics.median(samples)


def main() -> int:
    torch.manual_seed(0)
    probe = load()
    hidden, vocab = probe.config.hidden_size, probe.config.vocab_size
    inter = probe.config.intermediate_size
    del probe
    torch.cuda.empty_cache()

    cfg = MoEConfig(shared=SHARED, groups=GROUPS, variants=VARIANTS, top_residuals=1,
                    hidden_size=hidden, dispatch="fused")
    model, report = convert_model(load(), cfg, verify=False)
    ids_by_batch = {b: torch.randint(0, vocab, (b, SEQ), device=DEVICE) for b in BATCHES}
    with torch.no_grad():
        model(input_ids=ids_by_batch[BATCHES[0]], use_cache=False)

    layers = moe_layers(model)
    state = layers[0]._fused
    width = state._stacked()[0].shape[-1]
    rows_per_token = len(state.shards) + GROUPS
    originals = [(lay, lay._fused.forward) for lay in layers]
    print(f"{MODEL}: {LAYERS_USED} layers, hidden {hidden}, ffn {inter}, "
          f"segment {width // 2}, gate/up output {width}, "
          f"active {report.active_params()/1e6:.0f}M, "
          f"allocated {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    def use_split(flag: bool) -> None:
        for lay, fwd in originals:
            lay._fused.forward = (
                (lambda flat, s=lay._fused: split_forward(s, flat)) if flag else fwd
            )

    with torch.no_grad():  # equivalence first, at the smallest batch
        use_split(False)
        reference = model(input_ids=ids_by_batch[BATCHES[0]], use_cache=False).logits.float()
        use_split(True)
        split_logits = model(input_ids=ids_by_batch[BATCHES[0]], use_cache=False).logits.float()
        use_split(False)
        drift = (reference - split_logits).abs().max().item()
        scale = reference.abs().max().item()
        del reference, split_logits
        torch.cuda.empty_cache()
    assert drift <= 0.05 * scale, f"split diverges: {drift}"
    print(f"split vs fused max|diff| {drift:.4g}", flush=True)

    for batch in BATCHES:  # discard a sweep
        for flag in (False, True):
            use_split(flag)
            with torch.no_grad():
                model(input_ids=ids_by_batch[batch], use_cache=False)
        torch.cuda.empty_cache()

    rows = []
    print("| batch | packed rows | intermediate | fused | split | split/fused |")
    print("|---|---|---|---|---|---|")
    for batch in BATCHES:
        times = {False: [], True: []}
        for _ in range(2):  # alternate the arms within each pair
            for flag in (False, True):
                use_split(flag)
                ids = ids_by_batch[batch]
                with torch.no_grad():
                    times[flag].append(timed_tps(lambda: model(input_ids=ids, use_cache=False),
                                                 batch * SEQ))
        fused, split = statistics.median(times[False]), statistics.median(times[True])
        packed_rows = batch * SEQ * rows_per_token
        rows.append({"batch": batch, "rows": packed_rows,
                     "intermediate_mb": packed_rows * width * 2 / 1e6,
                     "fused_tps": fused, "split_tps": split, "split_over_fused": split / fused})
        print(f"| {batch} | {packed_rows} | {packed_rows * width * 2 / 1e6:.0f} MB | "
              f"{fused:.0f} | {split:.0f} | {split / fused:.2f} |", flush=True)

    out = Path("experiments/results/real_threshold.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "layers": LAYERS_USED, "seq": SEQ,
                               "width": width, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
