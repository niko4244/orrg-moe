"""QLoRA SFT for a 4B coding model on a single 16 GB card.

Deliberately a plain training loop rather than a trainer-framework call: the fit here
is memory-bound (4-bit base + adapter + gradient checkpointing at 1536 tokens), and a
hand-written loop makes every one of those choices explicit and checkable. Loss is
computed on the assistant span only, so the model is not trained to reproduce the
prompt.

    PYTHONPATH=. python train/qlora_sft.py --steps 400 --out train/adapters/v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ChatSet(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        messages = self.rows[index]["messages"]
        # transformers 5 returns a tokenizers.Encoding from apply_chat_template(tokenize=True),
        # so render to text and tokenise explicitly.
        prompt_text = self.tok.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True,
        )
        full_text = self.tok.apply_chat_template(messages, tokenize=False)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"][: self.max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for position in range(prompt_len):
            labels[position] = -100
        if all(label == -100 for label in labels):
            labels[-1] = full_ids[-1]
        return {"input_ids": full_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/models/qwen3-4b")
    parser.add_argument("--data", default="train/data/sft.jsonl")
    parser.add_argument("--out", default="train/adapters/v1")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--4bit", dest="four_bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="four_bit", action="store_false")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = ChatSet(args.data, tokenizer, args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    # `steps` counts optimiser steps, so the schedule's horizon is exactly that.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=max(args.steps, 1), pct_start=0.03,
    )

    model.train()
    step, micro, running = 0, 0, 0.0
    started = time.time()
    while step < args.steps:
        for batch in loader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            running += float(loss.detach())
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == args.steps:
                elapsed = time.time() - started
                print(f"step {step}/{args.steps} loss {running / args.accum:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {elapsed / step:.2f}s/step",
                      flush=True)
            running = 0.0
            if step >= args.steps:
                break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved adapter to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
