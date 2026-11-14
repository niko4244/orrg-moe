"""Rebuild a trainable HF checkpoint of Sharp Spark from its local Q6_K GGUF.

Spark ships here only as a quantised GGUF, and PEFT cannot train GGUF tensors. The
machine does hold a complete HF implementation of the same architecture (`spark2_5`,
cached from the model's own repo code), so the weights can be dequantised, transposed
into HF layout and saved as a normal bf16 checkpoint. Everything about the model comes
from the GGUF's own metadata - architecture, head counts, rope, sliding-window pattern,
tokenizer, chat template - never from a file name.

    PYTHONPATH=. python train/spark_gguf_to_hf.py \
        --gguf "F:/Ollama/models/blobs/sha256-15bbbe...74581" \
        --out cache/models/spark-x2.6-4b-hf
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

# Vendored copy of the model's own HF implementation, patched for transformers>=5.16.
MODULES = Path(__file__).resolve().parent / "spark_hf"

# llama.cpp's LLAMA_VOCAB_PRE_TYPE_SPARK2_5, in the order it applies them.
SPARK_PRE = (
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+",
    r"\p{N}",
)

# GGUF tensor name -> HF parameter name, block ids filled in per layer.
BLOCK_MAP = {
    "attn_qkv": "self_attn.q_k_v_proj",
    "attn_gate": "self_attn.g_proj",
    "attn_output": "self_attn.out_proj",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
GLOBAL_MAP = {
    "token_embd.weight": "model.embedding.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def import_modeling():
    root = str(MODULES.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from spark_hf.configuration_spark import Spark2_5Config  # noqa: PLC0415
    from spark_hf.modeling_spark import Spark2_5ForCausalLM  # noqa: PLC0415

    return Spark2_5Config, Spark2_5ForCausalLM


def config_from_metadata(fields: dict) -> dict:
    get = lambda key: fields[key].contents()  # noqa: E731
    arch = get("general.architecture")
    blocks = int(get(f"{arch}.block_count"))
    pattern = list(get(f"{arch}.attention.sliding_window_pattern"))
    if len(pattern) != blocks:
        raise ValueError(f"sliding_window_pattern {len(pattern)} != block_count {blocks}")
    head_dim = int(get(f"{arch}.attention.key_length"))
    full_rope = float(get(f"{arch}.rope.freq_base"))
    swa_rope = float(get(f"{arch}.rope.freq_base_swa"))
    dim_full = int(get(f"{arch}.rope.dimension_count"))
    dim_swa = int(get(f"{arch}.rope.dimension_count_swa"))
    vocab = int(get(f"{arch}.vocab_size"))
    tokens = get("tokenizer.ggml.tokens")
    if len(tokens) != vocab:
        raise ValueError(f"tokenizer has {len(tokens)} tokens but vocab_size is {vocab}")
    return {
        "arch": arch,
        "vocab_size": vocab,
        "hidden_size": int(get(f"{arch}.embedding_length")),
        "intermediate_size": int(get(f"{arch}.feed_forward_length")),
        "num_hidden_layers": blocks,
        "num_attention_heads": int(get(f"{arch}.attention.head_count")),
        "num_key_value_heads": int(get(f"{arch}.attention.head_count_kv")),
        "head_dim": head_dim,
        "max_position_embeddings": int(get(f"{arch}.context_length")),
        "rms_norm_eps": float(get(f"{arch}.attention.layer_norm_rms_epsilon")),
        "hidden_act": "gelu",
        "sliding_window": int(get(f"{arch}.attention.sliding_window")),
        "layer_types": ["sliding_attention" if flag else "full_attention" for flag in pattern],
        "rope_parameters": {
            "full_attention": {"rope_theta": full_rope,
                               "partial_rotary_factor": dim_full / head_dim},
            "sliding_attention": {"rope_theta": swa_rope,
                                  "partial_rotary_factor": dim_swa / head_dim},
        },
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "tie_word_embeddings": True,  # the GGUF carries no separate output tensor
        "bos_token_id": int(get("tokenizer.ggml.bos_token_id")),
        "eos_token_id": int(get("tokenizer.ggml.eos_token_id")),
        "pad_token_id": int(get("tokenizer.ggml.padding_token_id")),
    }


def build_tokenizer(fields: dict, out: Path) -> PreTrainedTokenizerFast:
    tokens = [str(token) for token in fields["tokenizer.ggml.tokens"].contents()]
    merges = [tuple(str(merge).split(" ")) for merge in fields["tokenizer.ggml.merges"].contents()]
    unk_id = int(fields["tokenizer.ggml.unknown_token_id"].contents())
    types = [int(value) for value in fields["tokenizer.ggml.token_type"].contents()] if \
        "tokenizer.ggml.token_type" in fields else [1] * len(tokens)

    backend = Tokenizer(models.BPE(vocab={token: index for index, token in enumerate(tokens)},
                                   merges=merges, unk_token=tokens[unk_id]))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(pattern), behavior="isolated") for pattern in SPARK_PRE]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    backend.decoder = decoders.ByteLevel()

    chat_template = str(fields["tokenizer.chat_template"].contents())
    specials = [token for token, kind in zip(tokens, types) if kind != 1]  # 1 == NORMAL
    expected = {token: index for index, (token, kind) in enumerate(zip(tokens, types))
                if kind != 1}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=tokens[int(fields["tokenizer.ggml.bos_token_id"].contents())],
        eos_token=tokens[int(fields["tokenizer.ggml.eos_token_id"].contents())],
        pad_token=tokens[int(fields["tokenizer.ggml.padding_token_id"].contents())],
        unk_token=tokens[unk_id],
        model_max_length=int(fields["spark2_5.context_length"].contents()),
        chat_template=chat_template,
    )
    # Without this the chat template's turn markers are byte-split instead of matched as
    # single special tokens, so the model never sees its own control tokens.
    tokenizer.add_tokens(specials, special_tokens=True)
    drifted = {token: (tokenizer.convert_tokens_to_ids(token), index)
               for token, index in expected.items()
               if tokenizer.convert_tokens_to_ids(token) != index}
    print(f"special tokens: {len(specials)}; id drift: {len(drifted)}")
    if drifted:
        raise SystemExit(f"special-token ids moved: {list(drifted.items())[:5]}")
    tokenizer.save_pretrained(str(out))
    return tokenizer


def tensor_to_torch(tensor) -> torch.Tensor:
    qtype = tensor.tensor_type
    data = tensor.data
    if qtype.name not in ("F32", "F16", "BF16"):
        # Quantised blocks dequantise to a flat buffer, so reshape by the tensor's own
        # declared shape before anything else.
        data = dequantize(data, qtype)
    shape = tuple(int(dimension) for dimension in tensor.shape)
    flat = np.asarray(data)
    if len(shape) == 2:
        # GGUF is ne0-fastest and stores [in, out]; a C-order reshape to (out, in) is
        # exactly the transpose HF wants. Reshaping to (in, out) and calling .T is NOT
        # the same thing - it permutes rows and leaves shapes identical, which is how
        # a scrambled checkpoint can pass every shape assertion.
        array = flat.reshape(shape[::-1])
    else:
        array = flat.reshape(shape)
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32).copy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", default="cache/models/spark-x2.6-4b-hf")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="rebuild tokenizer files in an existing output directory")
    args = parser.parse_args()

    reader = GGUFReader(args.gguf)
    fields = reader.fields
    spec = config_from_metadata(fields)
    arch = spec.pop("arch")
    print(f"architecture from GGUF metadata: {arch}")
    print(json.dumps({k: v for k, v in spec.items() if k != "layer_types"}, indent=2)[:900])
    print("layer_types[0:8]:", spec["layer_types"][:8])

    if args.tokenizer_only:
        out = Path(args.out)
        tokenizer = build_tokenizer(fields, out)
        print(f"rebuilt tokenizer in {out}: vocab {len(tokenizer)}")
        return 0

    Spark2_5Config, Spark2_5ForCausalLM = import_modeling()
    config = Spark2_5Config(**spec)
    dtype = getattr(torch, args.dtype)
    with torch.device("cpu"):
        model = Spark2_5ForCausalLM(config)
    model = model.to(dtype)

    params = dict(model.named_parameters())
    missing, loaded = [], 0
    for tensor in reader.tensors:
        name = tensor.name
        if name in GLOBAL_MAP:
            target = GLOBAL_MAP[name]
        elif name.startswith("blk."):
            _, block, leaf = name.split(".", 2)
            leaf = leaf.removesuffix(".weight")
            if leaf not in BLOCK_MAP:
                missing.append(name)
                continue
            target = f"model.layers.{block}.{BLOCK_MAP[leaf]}.weight"
        else:
            missing.append(name)
            continue
        if target not in params:
            missing.append(f"{name} -> {target}")
            continue
        value = tensor_to_torch(tensor).to(dtype)
        if tuple(params[target].shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch for {name} -> {target}: "
                             f"gguf {tuple(value.shape)} vs model {tuple(params[target].shape)}")
        with torch.no_grad():
            params[target].copy_(value)
        loaded += 1

    if missing:
        raise SystemExit(f"unmapped tensors ({len(missing)}): {missing[:10]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.tie_weights()
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
    shutil.copy(MODULES / "configuration_spark.py", out / "configuration_spark.py")
    shutil.copy(MODULES / "modeling_spark.py", out / "modeling_spark.py")
    config_dict = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_spark.Spark2_5Config",
        "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
    }
    (out / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

    tokenizer = build_tokenizer(fields, out)
    total = sum(p.numel() for p in model.parameters())
    print(f"loaded {loaded}/{len(reader.tensors)} tensors; {total/1e9:.3f}B params; "
          f"vocab {len(tokenizer)}; saved to {out}")
    probe = "def isoparse(s):\n    return s"
    print("tokenizer probe:", tokenizer(probe)["input_ids"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
