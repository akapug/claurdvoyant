# score_dataset.py — verified-outcome dataset scorer

A maturity-bit for the post-training pipeline. `cv dataset` exports chatml
JSONL (one agent-coding session per line). This script scores each session for
training value and emits it annotated with `score` + per-signal `signals`, so
you can keep only the HIGH-SIGNAL trajectories worth distilling into a smaller
model.

## Usage

```sh
# score every record, write annotated JSONL to stdout
python3 scripts/score_dataset.py sessions.chatml.jsonl > scored.jsonl

# keep only premium records, with a stderr histogram + signal averages
python3 scripts/score_dataset.py sessions.chatml.jsonl \
    --min-score 0.6 --out premium.jsonl --stats
```

Pure Python 3 stdlib (`json`, `argparse`, `re`, `sys`) — no pip deps.

## Output contract

Each emitted line is the original record (`messages` preserved verbatim) plus:

```json
{"messages": [...], "score": 0.85, "signals": {"verified_outcome": 1.0, ...}}
```

## Signals & composite weights

| signal | weight | what it keys on |
|---|---|---|
| `verified_outcome` | 0.40 | `test result: ok`, `passed`, `PASS`, `0 failed`, `build succeeded`, `Finished \`release\``, `exit 0`, `✓`, `200 OK`, `merged` (saturating count) |
| `self_correction` | 0.25 | tool_result failure (`FAILED`/`panicked`/non-zero exit/`success:false`) **then** a later verified_outcome; OR a user correction (`no,`/`actually`/`wrong`/`you broke…`) **then** an assistant pivot |
| `tool_density` | 0.15 | ` ```tool_call ` blocks per assistant turn (saturates at 1.0) |
| `reasoning_present` | 0.10 | fraction of assistant turns containing `<thinking>` |
| `substance` | 0.10 | message count + total chars, capped; sessions `<4` messages penalized hard |

`composite = Σ weight·signal`, clamped to `[0,1]`. Weights live at the top of
`score_dataset.py` (`WEIGHTS`).

## Heuristic limitations (honest)

- String matching cannot distinguish a tool-emitted "tests passed" from an
  assistant *claiming* it. We prefer markers that overwhelmingly originate in
  tool_result blocks.
- Raw `error` is deliberately NOT a failure signal (400+ false hits inside
  diffs/source in real records); self_correction keys only on tool_result
  failure markers + explicit non-zero exits.
- Ordering for self_correction is block-position, not a causal graph.
- `reasoning_present` depends on `<thinking>` surviving the export; many
  harnesses strip it, so 0 ≠ "no reasoning".
