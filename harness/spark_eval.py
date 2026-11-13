"""Reproducible Spark coding evaluation primitives.

One task per repo; each task is scored per test in a fixed test selection.
Prompts are byte-recorded so every case can be audited input-to-output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:4000").rstrip("/")
TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
MODEL = os.environ.get("SPARK_MODEL", "claude-spark-x2.6-local")

SYSTEM = (
    "You are a senior Python engineer fixing a bug in a real open-source repository. "
    "Return only a unified diff that applies with `git apply -p1` from the repository root. "
    "Use `--- a/<path>` and `+++ b/<path>` headers and correct @@ hunk ranges. "
    "Do not include prose, explanation, or markdown fences."
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


@dataclass
class ModelReply:
    text: str
    thinking: str
    stop_reason: str | None
    usage: dict
    latency_s: float
    request_sha256: str
    error: str | None = None


def call_model(system: str, user: str, max_tokens: int = 12000,
               temperature: float = 0.0, retries: int = 3,
               thinking: str = "disabled") -> ModelReply:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if thinking == "disabled":
        # Without this the model can consume the whole max_tokens budget in a
        # thinking block and return no answer at all (observed: 12k/12k).
        body["thinking"] = {"type": "disabled"}
    raw = json.dumps(body, sort_keys=True).encode("utf-8")
    req_hash = sha256_bytes(raw)
    last_err = None
    for attempt in range(retries):
        started = time.time()
        req = urllib.request.Request(
            BASE_URL + "/v1/messages",
            data=raw,
            headers={
                "content-type": "application/json",
                "x-api-key": TOKEN,
                "authorization": "Bearer " + TOKEN,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
            return ModelReply(text, thinking, data.get("stop_reason"),
                              data.get("usage") or {}, time.time() - started, req_hash)
        except Exception as exc:  # noqa: BLE001 - record transport failures verbatim
            last_err = f"{type(exc).__name__}: {exc}"
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read(2000).decode("utf-8", "replace")
                except Exception:
                    pass
            last_err += (" | " + detail) if detail else ""
            time.sleep(2 * (attempt + 1))
    return ModelReply("", "", None, {}, 0.0, req_hash, error=last_err)


FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.S)


def extract_diff(text: str) -> str:
    """Return the diff payload; tolerate fences or leading prose."""
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1)
    lines = text.replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            return "\n".join(lines[i:]).strip() + "\n"
    return text.strip() + "\n"


def git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "eval@local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "eval"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$")


def normalize_diff(diff_text: str) -> str:
    """Recount hunk headers and drop non-hunk trailing lines.

    Models frequently emit a correct hunk body with a stale line count
    (e.g. `@@ -146,6 +146,10 @@` followed by 11 context lines). That is a
    patch-format artifact, not a semantic error, so repair the counts before
    judging the edit.
    """
    lines = diff_text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = HUNK_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        old_start, new_start = int(m.group(1)), int(m.group(2))
        j = i + 1
        body: list[str] = []
        while j < len(lines) and not lines[j].startswith(("@@", "diff --git", "--- ", "+++ ")):
            body.append(lines[j])
            j += 1
        while body and not body[-1].startswith((" ", "+", "-", "\\")):
            body.pop()
        old_n = sum(1 for b in body if b[:1] in (" ", "-"))
        new_n = sum(1 for b in body if b[:1] in (" ", "+"))
        out.append(f"@@ -{old_start},{old_n} +{new_start},{new_n} @@{m.group(3)}")
        out.extend(body)
        i = j
    return "\n".join(out).strip() + "\n"


def apply_diff(repo: Path, diff_text: str) -> tuple[bool, str]:
    diff_text = normalize_diff(extract_diff(diff_text))
    if not diff_text.strip():
        return False, "empty diff"
    for cmd in (["git", "apply", "-p1", "--whitespace=nowarn", "--recount", "-"],
                ["git", "apply", "-p1", "--whitespace=nowarn", "-"]):
        p = subprocess.run(cmd, cwd=repo, input=diff_text, text=True,
                           capture_output=True)
        if p.returncode == 0:
            return True, ""
    # last resort: patch(1)
    p = subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", "-"],
                       cwd=repo, input=diff_text, text=True, capture_output=True)
    if p.returncode == 0:
        return True, ""
    return False, (p.stdout + p.stderr)[-2000:]


def parse_shortstat(repo: Path) -> str:
    p = subprocess.run(["git", "diff", "--shortstat"], cwd=repo,
                       text=True, capture_output=True)
    return p.stdout.strip()


def changed_files(repo: Path) -> list[str]:
    p = subprocess.run(["git", "diff", "--name-only"], cwd=repo,
                       text=True, capture_output=True)
    return [x for x in p.stdout.splitlines() if x.strip()]


def run_pytest(repo: Path, target: str, timeout: int = 300) -> dict:
    """Run one test target and return per-test outcomes parsed from -v output."""
    p = subprocess.run(
        ["python", "-m", "pytest", target, "-v", "--tb=short", "-p", "no:cacheprovider"],
        cwd=repo, text=True, capture_output=True, timeout=timeout,
    )
    out = p.stdout + "\n" + p.stderr
    results: dict[str, str] = {}
    for line in p.stdout.splitlines():
        m = re.match(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)", line)
        if m:
            results[m.group(1)] = m.group(2)
    return {"returncode": p.returncode, "results": results,
            "tail": "\n".join(out.splitlines()[-40:]), "output": out}


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def excerpt(path: Path, start: int, end: int, numbered: bool = True) -> str:
    """1-indexed inclusive line excerpt."""
    lines = read_lines(path)
    out = []
    for i in range(start, min(end, len(lines)) + 1):
        text = lines[i - 1]
        out.append(f"{i:4d}| {text}" if numbered else text)
    return "\n".join(out)


def find_function_span(path: Path, name: str) -> tuple[int, int]:
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return start, node.end_lineno
    raise KeyError(name)


def touched_functions(repo: Path, module_rel: str) -> list[str]:
    """Map added/removed line numbers in the working diff to function names."""
    import ast
    mod = repo / module_rel
    src = mod.read_text(encoding="utf-8")
    tree = ast.parse(src)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    p = subprocess.run(["git", "diff", "-U0", "--", module_rel], cwd=repo,
                       text=True, capture_output=True)
    touched = set()
    for line in p.stdout.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2) or 1)
        for ln in range(start, start + max(count, 1)):
            for a, b, name in spans:
                if a <= ln <= b:
                    touched.add(name)
                    break
    return sorted(touched)


PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = PY_FENCE_RE.search(text)
    body = m.group(1) if m else text
    return body.strip("\n") + "\n"


def replace_function(repo: Path, module_rel: str, fn_name: str,
                     code: str) -> tuple[bool, str]:
    """Splice a complete replacement function into the module."""
    import ast
    mod = repo / module_rel
    start, end = find_function_span(mod, fn_name)
    original = mod.read_text(encoding="utf-8")
    lines = original.splitlines()
    new_lines = code.rstrip("\n").split("\n")
    mod.write_text("\n".join(lines[:start - 1] + new_lines + lines[end:]) + "\n",
                   encoding="utf-8")
    p = subprocess.run(["python", "-m", "py_compile", str(mod)], text=True,
                       capture_output=True)
    if p.returncode != 0:
        mod.write_text(original, encoding="utf-8")
        return False, (p.stdout + p.stderr)[-1200:]
    return True, ""


def build_prompt(issue: str, repo_name: str, module_rel: str, code: str,
                 contract: str,
                 final_instruction: str | None = "Return only the unified diff that fixes the issue.") -> str:
    parts = [
        contract,
        f"## Issue\n\n{issue}",
        f"## Repository context\n\nRepository: {repo_name}\nFile: {module_rel}",
        f"```python\n{code}\n```",
    ]
    if final_instruction:
        parts.append(final_instruction)
    return "\n\n".join(parts) + "\n"


FUNCTION_SYSTEM = (
    "You are a senior Python engineer fixing a bug in a real open-source repository. "
    "Return the complete replacement function in one ```python code block. "
    "Do not include prose or explanation."
)

FUNCTION_CONTRACT = """You are fixing one bug in a real repository.
Rules:
1. Read the issue and the file excerpt; change only what the issue requires.
2. Preserve public behavior unrelated to the bug.
3. Return a complete, syntactically valid replacement for the target function.
4. Every branch of a conditional that computes a value must assign that value.
5. Keep the function's public signature and documented behavior."""

DEFAULT_CONTRACT = """You are fixing one bug in a real repository.
Rules:
1. Read the issue and the file excerpt; change only what the issue requires.
2. Preserve public behavior unrelated to the bug.
3. The diff must apply cleanly to the repository at the stated path.
4. Every branch of a conditional that computes a value must assign that value.
5. Return only the diff, no prose or fences."""


def snapshot_case(case_dir: Path, payload: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for key in ("prompt.txt", "response.txt", "thinking.txt", "diff.txt",
                "excerpt.txt", "test_output.txt", "function.txt"):
        val = payload.get(key)
        if isinstance(val, str):
            (case_dir / key).write_text(val, encoding="utf-8")
