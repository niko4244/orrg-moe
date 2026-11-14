"""Serve a local HF model+adapter behind the Anthropic /v1/messages contract.

The Spark harness is HTTP-only (`harness/spark_eval.py`), so a locally trained adapter
can be scored by the unchanged 50-case suite if something answers that contract
locally. This is that something: greedy decoding, no sampling, and the same
`thinking: disabled` handling the harness sends.

    PYTHONPATH=. python train/serve_anthropic.py --port 4010 --adapter train/adapters/v1
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STATE: dict = {}


def render(system: str, messages: list[dict]) -> str:
    parts = []
    if system:
        parts.append({"role": "system", "content": system})
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content)
        parts.append({"role": message["role"], "content": content})
    return STATE["tokenizer"].apply_chat_template(
        parts, tokenize=False, add_generation_prompt=True,
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - http.server API
        if self.path.rstrip("/") not in ("/v1/messages", "/messages"):
            self._json(404, {"error": {"type": "not_found", "message": self.path}})
            return
        length = int(self.headers.get("content-length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        text = render(request.get("system") or "", request.get("messages") or [])
        inputs = STATE["tokenizer"](text, return_tensors="pt").to("cuda")
        prompt_tokens = int(inputs["input_ids"].shape[1])
        max_new = int(request.get("max_tokens") or 1024)
        started = time.time()
        with torch.no_grad():
            output = STATE["model"].generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=STATE["tokenizer"].pad_token_id,
            )
        completion = output[0][prompt_tokens:]
        body = STATE["tokenizer"].decode(completion, skip_special_tokens=True)
        self._json(200, {
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "role": "assistant",
            "model": request.get("model") or STATE["name"],
            "content": [{"type": "text", "text": body}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": int(completion.shape[0]),
            },
            "latency_s": round(time.time() - started, 3),
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/models/qwen3-4b")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--port", type=int, default=4010)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    STATE.update(model=model, tokenizer=tokenizer, name=args.adapter or args.model)
    print(f"serving {STATE['name']} on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
