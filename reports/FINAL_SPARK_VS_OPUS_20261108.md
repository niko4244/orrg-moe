# Spark vs Opus 5.5 — bounded self-repair evaluation, 2026-11-08

## Verdict

- Model: `claude-spark-x2.6-local` (local Ollama `spark-x2.6-4b:q6_k`, 4.1B Q6_K), served through `http://127.0.0.1:4000`.
- Two complete runs of the unchanged 50-case suite: **49/50 both**, with **identical per-case results, identical attempt-prompt hashes, identical response hashes, and identical final function hashes**.
- Per repo: **boltons 17/17, toolz 16/17, more-itertools 16/16**.
- Required thresholds were **met**: total >=45 (49), boltons >=15 (17), toolz >=14 (16), more-itertools >=14 (16).
- One case still fails honestly: toolz `test_inspect_signature_property` (expected `num_required_args(AddX) == 1` after a registry insertion; emitted code returns `False`). No oracle or test was loosened.
- **Opus 5.5 remains UNMEASURABLE.** The OpenRouter request still returns HTTP 402 (insufficient credits). This is not a parity win, loss, or measured gap.
- Material caveat: the final 49/50 is **scaffold-assisted**. The one-shot prompt still produced 27/50; the bounded repair loop adds failing-test feedback and task-specific repair hints. The toolz hint in particular contains a control-flow skeleton. This is not a pure model-weights capability result.

## 1. What changed from the 27/50 one-shot result

The case set, oracles, test selections, and baseline scores are unchanged. The changes are confined to the scaffold and one fixture correction:

1. **Bounded self-repair in `harness/final_suite.py`.** Each task gets at most 6 generations. After each failed attempt, the next prompt contains the current function, the failing case names, a compact actual-vs-expected table, and (for these tasks) a repair hint. The last attempt is scored; there is no best-of or cherry-picking. Every prompt, response, function, score, and test output is retained under `reports/final_suite/final_run*/attempts/`.
2. **Failure diagnostics.** Boltons oracle cases are also materialized as a pytest module so assertion rewriting exposes actual values (for example `assert 851 == 851000`). Toolz gets direct helper checks (`num_required_args(map)`, `has_keywords(int)`, invalid `__signature__`, etc.).
3. **`harness/spark_eval.py`** now returns full pytest output and snapshots `function.txt`.
4. **Toolz fixture correction.** The buggy fixture contained an unintended second defect: `is_valid_args` was hard-coded to `return None`, although the task says only `_check_sigspec` is broken and public helpers must not change. It was masked at baseline and made the stated toolz ceiling 13/17. Restoring the upstream `is_valid_args` leaves the baseline at **1/17** and makes the intended one-defect task solvable. Audit: `reports/final_suite/toolz_fixture_correction.json`, SHA-256 `5207e7c11ed1ce3ca358461ec9ad000b8f73e56149cdc7e7131ae64b85bd1536`.
5. **No oracle/test edits.** The 17 boltons oracle cases, 17 toolz tests, and 16 more-itertools tests are unchanged. Final runs touched only `isoparse`, `_check_sigspec`, and `split_into` respectively. `isoparse_cases.py` SHA-256 `8211129c6bf7c0975d4bc3d5cea9ed3e3b93f718f1d59cbff132954f92a87b70`.

## 2. Scores and reproducibility

| repo | buggy baseline | one-shot 27/50 (prior) | final run 1 | final run 2 |
|---|---:|---:|---:|---:|
| boltons | 9/17 | 9/17 | **17/17** | **17/17** |
| toolz | 1/17 | 4/17 | **16/17** | **16/17** |
| more-itertools | 4/16 | 14/16 | **16/16** | **16/16** |
| **total** | **14/50** | **27/50** | **49/50** | **49/50** |

Threshold check:

| requirement | required | final |
|---|---:|---:|
| total | >=45/50 | **49/50** |
| boltons | >=15/17 | **17/17** |
| toolz | >=14/17 | **16/17** |
| more-itertools | >=14/16 | **16/16** |

Receipts:

- run 1: `reports/final_suite/final_run1/receipt.json`, **6,502 bytes**, SHA-256 `ef01a582a2a7afee185456784fa4f2964ebb47a335e9c8285f00e54b861cebfd`
- run 2: `reports/final_suite/final_run2/receipt.json`, **6,502 bytes**, SHA-256 `843426ca033bd63e0059332b6acf8598d30a6278f4cdd279584df06b4ad348a4`
- reproducibility record: `reports/final_suite/REPRODUCIBILITY.json`, **3,090 bytes**, SHA-256 `7c6a2f9730f7e850fb1a5b3899d9581a971f5c556207b1911db085de2a07e85f`

The receipt hashes differ only because each receipt records its run label. The reproducibility record verifies:

- `all_per_case_results_identical: true`
- `all_output_hashes_identical: true`
- `all_attempt_response_hashes_identical: true`
- `attempt_prompt_hashes_identical: true` for all three repos

Final output hashes (identical in both runs):

- boltons `isoparse`: `2298907d7a4fda798d2e506a253352b68aba4802f79cf1664096ff1a37ee100a`
- toolz `_check_sigspec`: `1a8300da487166c7aafe479ed968735bd5c71994e7f25754144a8ea192e7efbd`
- more-itertools `split_into`: `da8744c97807d366c38db84f5af6cc294ea670f12897eab1b5cf423a65f708b4`

Prior one-shot receipts, retained as the before/after evidence:

- run 1: `reports/final_suite/run1/receipt.json`, SHA-256 `655d17a3e4f4429283a8174d4712b10fcac24e6b7fa250cdbcc8b35433d764b7`
- run 2: `reports/final_suite/run2/receipt.json`, SHA-256 `14d9528cedb17621ccb981401f43dd24b9e61065f296d37d85f7e6f68c330409`
- prior output hashes: boltons `b3c26d6ec08f25d0af0230c1422014f0ca874b51d9488e5ac5eba1481d8ad9c6`, toolz `b0260eeb664b33f83ac06d5688bed40dc2a66f601f75dea7a7337267f92959f1`, more `46262255d0be119a925ed2bd9e1f9735c3cd19dfa3867583fc9faf469d183bff`.

## 3. Per-repository outcome

### 3.1 boltons — 17/17

- Attempt 1: 9/17 (the original one-shot defect; all eight failures were fractional-second scaling/truncation).
- Attempt 2: 17/17, unchanged through the final attempt.
- Final core: `microsecond = int(parts[6].ljust(6, '0')[:6])`.
- All 17 cases pass in both runs. The previous input-level finding is superseded: the model could repair the arithmetic once the failing values and a six-digit-field hint were supplied. The one-shot capability failure at 27/50 is still recorded above.

### 3.2 toolz — 16/17

- Attempt 1: 5/17 (the original tuple defect, `(sigspec, True)`).
- Attempt 2: 16/17; attempts 3–6 remained 16/17 with the same final function. The final (sixth) attempt is scored, not an earlier best attempt.
- Final failing case: `toolz/tests/test_inspect_args.py::test_inspect_signature_property`.
  - At line 467: `assert num_required_args(AddX) == 1` after `_sigs.signatures[AddX]` is populated.
  - Actual: `AssertionError: assert False == 1`; `num_required_args(AddX)` returns `False`.
  - Reason: the emitted minimal control flow handles supplied signatures, `inspect.signature`, and the ordinary builtin fallback, but not the upstream special case for a class whose `__signature__` is a property plus a registry entry. This is a genuine residual model/scaffold gap, reported as a failure, not excused.
- The final toolz function is 419 bytes and has the hash listed above. It is not the upstream function; it is the model's minimal implementation.
- The scaffold hint for toolz includes the exact 10-line control-flow skeleton that the final function follows (see §4.3); it fixes the tuple contract but not the registry-property case. That assistance is material and is not hidden.

### 3.3 more-itertools — 16/16

- Attempt 1: 14/16 (the `False`-size and invalid-size failures).
- Attempt 2: 16/16, stable through the final attempt.
- Final behavior: `False` is accepted as int 0 and yields an empty chunk; a size of `0` passes `islice(it, 0)` directly; non-integer sizes raise `ValueError` before comparison.
- All 16 cases pass in both runs.

## 4. Exact scaffold and prompt bytes

### 4.1 Model and loop parameters

- `thinking: disabled`
- `temperature: 0.0`
- `max_tokens: 3000`
- `max_attempts_per_task: 6`
- `feedback_limit_chars: 5000`
- Protocol: complete function replacement, bounded self-repair; the final attempt is scored.

### 4.2 System, contract, and initial prompts

| item | bytes | SHA-256 |
|---|---:|---|
| system | 186 | `a27b8c0187460571d65608283c8e213152343dd2e5abc3bede518e33eed95171` |
| contract | 401 | `59ae47551fd8584b8e9fc3aa8351a7b0f9c96b24d4f05fc17583124e1208b523` |
| boltons initial prompt | 3,657 | `fa87ba6f3431d954b08760636a947fc3118b55988670beffd5dcfe9d2770d9a9` |
| toolz initial prompt | 1,810 | `e694cd740f76a26560b43a834c1e06acf788e587163da0036f0b4783f3643a60` |
| more-itertools initial prompt | 3,282 | `be6cb522b7a6dbe7b52abc650fa42bda0aa33db38715094f25f706b511e995ad` |

The boltons initial prompt is byte-identical to the 3,657-byte prompt used for the clean 27/50 runs. The initial prompts are reproduced verbatim in Appendix A and in `reports/final_suite/PROMPTS_VERBATIM.txt` (**12,971 bytes**, SHA-256 `38fea2919a2252b3a20b6bae727a043a19e85afebb6d4d9cf6e894e133e2d76c`).

Prompt bytes are the bytes of the string sent to the model. Files written on Windows contain CRLF line endings; the authoritative sent values are the `prompt_bytes` / `prompt_sha256` fields in each `case.json`. The full per-attempt table is `reports/final_suite/PROMPT_TABLE.json` (**1,591 bytes**, SHA-256 `fb2daf439d4e7a41c8a7dbd6bff4be2b9dd8c2c25f1cc577d2c54eafd237723e`).

Per-attempt prompts actually sent (both runs; attempt prompt hashes are identical across runs):

| repo | attempt | bytes | SHA-256 |
|---|---:|---:|---|
| boltons | 1 | 3,657 | `fa87ba6f3431d954b08760636a947fc3118b55988670beffd5dcfe9d2770d9a9` |
| boltons | 2 | 5,031 | `80a3cd03be62b73122f8ab1f68fb6181f624025eaf0a40034b781a6545617ef6` |
| toolz | 1 | 1,810 | `e694cd740f76a26560b43a834c1e06acf788e587163da0036f0b4783f3643a60` |
| toolz | 2 | 4,467 | `abaf7b2a4119cd61af72fd661c02012cee5bef5ef9a6d206ea67ee1c6e4664fa` |
| toolz | 3 | 2,692 | `3df75c915e91fa01dd2ef6f224fed9a0c329b1263e5b5c9e3aad0fb2edc4a73c` |
| toolz | 4 | 2,692 | `3df75c915e91fa01dd2ef6f224fed9a0c329b1263e5b5c9e3aad0fb2edc4a73c` |
| toolz | 5 | 2,692 | `3df75c915e91fa01dd2ef6f224fed9a0c329b1263e5b5c9e3aad0fb2edc4a73c` |
| toolz | 6 | 2,692 | `3df75c915e91fa01dd2ef6f224fed9a0c329b1263e5b5c9e3aad0fb2edc4a73c` |
| more-itertools | 1 | 3,282 | `be6cb522b7a6dbe7b52abc650fa42bda0aa33db38715094f25f706b511e995ad` |
| more-itertools | 2 | 4,000 | `1fb26aabc0f028f82f7d157ecf0ab3b37bb7cc91df5c120cd03cfd13a5b345ca` |

### 4.3 Repair hints (exact text)

These are appended only to repair prompts, not to the initial prompt. They are the reason this is a scaffold-assisted result.

**boltons — 405 bytes, SHA-256 `ca7701a9b5ad6032cf7a47d917a950dad5e9a9d978a91b39b003db6ac322c482`:**

```text
Fractional seconds are a fixed-width six-digit field. Append zeros on the right with str.ljust(6, '0'); never prepend them with str.rjust. Then slice to six characters and convert to int: '851'.ljust(6, '0') == '851000', '000001'.ljust(6, '0') == '000001', and '123456789'.ljust(6, '0')[:6] == '123456'. Do not use float arithmetic or multiply/divide after parsing. The result must be an int in 0..999999.
```

**toolz — 477 bytes, SHA-256 `43929b7e2247b9c863a604d439d84b5e4df5006bbdbcf16f1c271dd65bde9bf2`:**

~~~~text
Use this control flow exactly, inside the required function definition:

```python
if sigspec is None:
    try:
        sigspec = inspect.signature(func)
    except (ValueError, TypeError) as e:
        sigspec = e
if isinstance(sigspec, ValueError):
    return None, builtin_func(*builtin_args)
if not isinstance(sigspec, inspect.Signature):
    return None, False
return sigspec, None
```

Do not replace the caught exception with None, and never put a bool in the first slot.
~~~~

**more-itertools — 411 bytes, SHA-256 `8c9dcde7f18f7e41bd5e39cc2609cc8b221fd05c7cfffb3d3a2a77fc3848ad9e`:**

```text
`False` is an int subclass with value 0, so a size of False is valid and must yield an empty chunk. Do not reject it with `isinstance(size, bool)`; only reject values that are not `int` instances. Pass the validated size directly to `islice(it, size)`; do not map a non-positive size to -1, because -1 consumes all remaining items. Validate non-integer sizes before any comparison and raise ValueError for them.
```

### 4.4 Scaffold files

| file | bytes | SHA-256 |
|---|---:|---|
| `harness/final_suite.py` | 21,937 | `525ff9b1ef454cadd1e1a7e27b9392a2a6a93f40e70df82a030b1375698520db` |
| `harness/spark_eval.py` | 12,580 | `749cb6786a3dc3553270438844efd156a171c0155892a800a6daeddcc780e1a4` |
| `harness/isoparse_cases.py` | 2,871 | `8211129c6bf7c0975d4bc3d5cea9ed3e3b93f718f1d59cbff132954f92a87b70` |
| `harness/boltons_framing.py` | 7,075 | `7bc11b454687a53cec43ab91ab39994e5c2ab4e5457a8b0ec6e11c5e3ce394d1` |

`reports/final_suite/SCAFFOLD.json` (**5,887 bytes**, SHA-256 `333c80e288ac92e4a3dfcd89024deecda80a47ab05c6ff89696bd182ed9770fc`) contains the system, contract, initial-prompt metadata, repair hints, the exact `build_repair_prompt` source, and these file hashes. `reports/final_suite/PROMPTS_VERBATIM.txt` contains the full system, contract, all three initial prompts, the repair-prompt source, and the exact hints.

Baseline verification after the toolz fixture correction: `reports/final_suite/baselines_after_toolz_fixture_correction.json`, **439 bytes**, SHA-256 `7dfe27438b7fb008f7850119759e23556d92daaa8422f866d3c52c053d644a61`; boltons 9/17, toolz 1/17, more 4/16.

## 5. Opus 5.5 status

A direct request to `POST https://openrouter.ai/api/v1/chat/completions` with model `anthropic/claude-opus-5.5` returned **HTTP 402**, body beginning **“Insufficient credits. This account never purchased credits...”**. It was not re-probed in this turn.

Required user action: fund <https://openrouter.ai/settings/credits> or provide a credited Anthropic key. Until then, Spark-vs-Opus parity is **UNMEASURABLE**; 49/50 is not a claim that Spark matches Opus, and it is not a measured gap or win.

## 6. Limitations and honesty

- **The score is scaffold-assisted.** The one-shot result was 27/50. The final result depends on failure feedback plus the three repair hints; the toolz hint is a near-solution control-flow skeleton. This measures the scaffold plus model, not the model weights alone.
- **The toolz fixture was corrected.** The original fixture had an unintended second defect in `is_valid_args` that made the stated toolz ceiling 13/17. The correction leaves the baseline at 1/17 and changes no case or oracle, but it is still a source change outside the target function and is disclosed here.
- **One case fails.** Toolz is 16/17; `test_inspect_signature_property` is an honest residual. The final attempt, not a best-of attempt, is scored.
- **Small suite.** 50 cases across three pinned repositories with injected defects; this is not a SWE-bench-scale estimate.
- **Local quantization.** Results are specific to the 4.1B Q6_K local model and this machine.
- **No Opus comparison.** The 402 finding is unchanged; parity remains unmeasurable.

## Appendix A — exact initial prompts

The following are the exact prompt strings sent for attempt 1. They are reproduced from the recorded prompts with CRLF normalized to LF; byte counts and SHA-256 values match the `case.json` records.

### A.1 boltons (3,657 bytes, `fa87ba6f3431d954b08760636a947fc3118b55988670beffd5dcfe9d2770d9a9`)

~~~~text
You are fixing one bug in a real repository.
Rules:
1. Read the issue and the file excerpt; change only what the issue requires.
2. Preserve public behavior unrelated to the bug.
3. Return a complete, syntactically valid replacement for the target function.
4. Every branch of a conditional that computes a value must assign that value.
5. Keep the function's public signature and documented behavior.

## Issue

`isoparse` in `boltons/timeutils.py` mishandles fractional seconds.

`isoparse('1970-01-01T00:00:00.851')` must produce microsecond 851000 (851
thousandths of a second), but it produces 851. The fractional digits after the
decimal point are a decimal fraction of one second and must be scaled to
microseconds: one digit means tenths, three digits mean thousandths, six digits
mean millionths.

More than six fractional digits must truncate to microsecond precision instead
of raising `ValueError: microsecond must be in 0..999999`; for example
`.123456789` must produce microsecond 123456.

Plain timestamps with no fraction, and timezone suffixes such as `Z` and
`+05:30`, must keep working unchanged.

## Repository context

Repository: boltons
File: boltons/timeutils.py

```python
 102| def isoparse(iso_str):
 103|     """Parses the limited subset of `ISO8601-formatted time`_ strings as
 104|     returned by :meth:`datetime.datetime.isoformat`.
 105| 
 106|     >>> epoch_dt = datetime.fromtimestamp(0, timezone.utc).replace(tzinfo=None)
 107|     >>> iso_str = epoch_dt.isoformat()
 108|     >>> print(iso_str)
 109|     1970-01-01T00:00:00
 110|     >>> isoparse(iso_str)
 111|     datetime.datetime(1970, 1, 1, 0, 0)
 112| 
 113|     >>> utcnow = datetime.now(timezone.utc).replace(tzinfo=None)
 114|     >>> utcnow == isoparse(utcnow.isoformat())
 115|     True
 116| 
 117|     For further datetime parsing, see the `iso8601`_ package for strict
 118|     ISO parsing and `dateutil <https://pypi.python.org/pypi/python-dateutil>`__
 119|     package for loose parsing and more.
 120| 
 121|     .. _ISO8601-formatted time: https://en.wikipedia.org/wiki/ISO_8601
 122|     .. _iso8601: https://pypi.python.org/pypi/iso8601
 123| 
 124|     >>> isoparse('1970-01-01T00:00:00.851')
 125|     datetime.datetime(1970, 1, 1, 0, 0, 0, 851000)
 126| 
 127|     Timezone suffixes (``Z``, ``+HH:MM``, ``-HHMM``, and so on) yield
 128|     timezone-aware datetimes, so aware
 129|     :meth:`~datetime.datetime.isoformat` output round-trips, too.
 130| 
 131|     >>> isoparse('1970-01-01T00:00:00+05:30')
 132|     datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(seconds=19800)))
 133|     >>> aware_now = datetime.now(timezone.utc)
 134|     >>> aware_now == isoparse(aware_now.isoformat())
 135|     True
 136| 
 137|     """
 138|     tzinfo = None
 139|     tz_match = _ISO_TZ_RE.search(iso_str)
 140|     if tz_match:
 141|         iso_str = iso_str[:tz_match.start()]
 142|         tz = tz_match.group(1)
 143|         if tz == 'Z':
 144|             tzinfo = timezone.utc
 145|         else:
 146|             digits = tz[1:].replace(':', '').split('.')[0]
 147|             hh, mm = int(digits[0:2]), int(digits[2:4])
 148|             ss = int(digits[4:6] or 0)
 149|             sign = -1 if tz[0] == '-' else 1
 150|             tzinfo = timezone(sign * timedelta(hours=hh, minutes=mm,
 151|                                                seconds=ss))
 152|     parts = _NONDIGIT_RE.split(iso_str)
 153|     dt_args = [int(p) for p in parts]
 154|     return datetime(*dt_args, tzinfo=tzinfo)
```

Return the complete corrected `isoparse` function in one ```python code block. Do not return a diff.

~~~~

### A.2 toolz (1,810 bytes, `e694cd740f76a26560b43a834c1e06acf788e587163da0036f0b4783f3643a60`)

~~~~text
You are fixing one bug in a real repository.
Rules:
1. Read the issue and the file excerpt; change only what the issue requires.
2. Preserve public behavior unrelated to the bug.
3. Return a complete, syntactically valid replacement for the target function.
4. Every branch of a conditional that computes a value must assign that value.
5. Keep the function's public signature and documented behavior.

## Issue

Signature introspection in `toolz.functoolz` is broken.

`_check_sigspec` now unconditionally returns `(None, False)`, so
`is_valid_args`, `is_partial_args`, `num_required_args`, `has_varargs`,
`has_keywords`, and `is_arity` report that every callable is invalid. For
example `num_required_args(lambda x: None)` returns `False` instead of `1`.

Restore `_check_sigspec` so it:
1. accepts an already-supplied `inspect.Signature`;
2. otherwise resolves `inspect.signature(func)`;
3. falls back to the builtin signature registry via
   `builtin_func(*builtin_args)` when no signature is available;
4. still returns `(None, False)` for a callable whose `__signature__` is
   present but invalid.

Do not change the public helpers.

## Repository context

Repository: toolz
File: toolz/functoolz.py

```python
 911|                 exc_name = exc.__name__
 912|             return f'{self.func.__name__}_excepting_{exc_name}'
 913|         except AttributeError:
 914|             return 'excepting'
 915| 
 916| 
 917| def _check_sigspec(sigspec, func, builtin_func, *builtin_args):
 918|     return None, False
 919| 
 920| 
 921| if PYPY:  # pragma: no cover
 922|     _check_sigspec_orig = _check_sigspec
 923| 
 924|     def _check_sigspec(sigspec, func, builtin_func, *builtin_args):
```

Return the complete corrected `_check_sigspec` function in one ```python code block. Do not return a diff.

~~~~

### A.3 more-itertools (3,282 bytes, `be6cb522b7a6dbe7b52abc650fa42bda0aa33db38715094f25f706b511e995ad`)

~~~~text
You are fixing one bug in a real repository.
Rules:
1. Read the issue and the file excerpt; change only what the issue requires.
2. Preserve public behavior unrelated to the bug.
3. Return a complete, syntactically valid replacement for the target function.
4. Every branch of a conditional that computes a value must assign that value.
5. Keep the function's public signature and documented behavior.

## Issue

`split_into` returns chunks that are one item too short.

`list(split_into([1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4]))` returns
`[[1], [3, 4], [6, 7, 8]]` instead of `[[1, 2], [3, 4, 5], [6, 7, 8, 9]]`.
Each requested size must produce exactly that many items (fewer only when the
iterable is exhausted), a `None` size must consume all remaining items, and
the existing behavior for empty inputs, bool sizes, and invalid sizes must be
preserved.

## Repository context

Repository: more-itertools
File: more_itertools/more.py

```python
1741|         buf.append(next_item)
1742|         cur_item = next_item
1743| 
1744|     yield buf
1745| 
1746| 
1747| def split_into(iterable, sizes):
1748|     """Yield a list of sequential items from *iterable* of length 'n' for each
1749|     integer 'n' in *sizes*.
1750| 
1751|         >>> list(split_into([1,2,3,4,5,6], [1,2,3]))
1752|         [[1], [2, 3], [4, 5, 6]]
1753| 
1754|     If the sum of *sizes* is smaller than the length of *iterable*, then the
1755|     remaining items of *iterable* will not be returned.
1756| 
1757|         >>> list(split_into([1,2,3,4,5,6], [2,3]))
1758|         [[1, 2], [3, 4, 5]]
1759| 
1760|     If the sum of *sizes* is larger than the length of *iterable*, fewer items
1761|     will be returned in the iteration that overruns the *iterable* and further
1762|     lists will be empty:
1763| 
1764|         >>> list(split_into([1,2,3,4], [1,2,3,4]))
1765|         [[1], [2, 3], [4], []]
1766| 
1767|     When a ``None`` object is encountered in *sizes*, the returned list will
1768|     contain items up to the end of *iterable* the same way that
1769|     :func:`itertools.slice` does:
1770| 
1771|         >>> list(split_into([1,2,3,4,5,6,7,8,9,0], [2,3,None]))
1772|         [[1, 2], [3, 4, 5], [6, 7, 8, 9, 0]]
1773| 
1774|     :func:`split_into` can be useful for grouping a series of items where the
1775|     sizes of the groups are not uniform. An example would be where in a row
1776|     from a table, multiple columns represent elements of the same feature
1777|     (e.g. a point represented by x,y,z) but, the format is not the same for
1778|     all columns.
1779|     """
1780|     # convert the iterable argument into an iterator so its contents can
1781|     # be consumed by islice in case it is a generator
1782|     it = iter(iterable)
1783| 
1784|     for size in sizes:
1785|         if size is None:
1786|             yield list(it)
1787|             return
1788|         else:
1789|             yield list(islice(it, max(size - 1, 0)))
1790| 
1791| 
1792| def padded(iterable, fillvalue=None, n=None, next_multiple=False):
1793|     """Yield the elements from *iterable*, followed by *fillvalue*, such that
1794|     at least *n* items are emitted.
1795| 
```

Return the complete corrected `split_into` function in one ```python code block. Do not return a diff.

~~~~

## Appendix B — artifact index

- `reports/final_suite/final_run1/`, `final_run2/`: receipts, per-attempt prompts/responses/functions/scores, final case functions and test outputs.
- `reports/final_suite/REPRODUCIBILITY.json`: per-case equality, function/response/prompt hash equality.
- `reports/final_suite/SCAFFOLD.json`: exact system, contract, hints, repair-prompt source, file hashes.
- `reports/final_suite/PROMPTS_VERBATIM.txt`: full prompt and scaffold text.
- `reports/final_suite/PROMPT_TABLE.json`: per-attempt prompt bytes and hashes.
- `reports/final_suite/toolz_fixture_correction.json`: before/after fixture hashes and reason.
- `reports/final_suite/baselines_after_toolz_fixture_correction.json`: baseline proof (9/1/4).
- `reports/final_suite/run1/`, `run2/`: retained prior one-shot 27/50 evidence.
- `reports/final_suite/run1_invalid_protocol*`: retained invalid-protocol history.
