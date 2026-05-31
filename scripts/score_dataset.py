#!/usr/bin/env python3
"""score_dataset.py — verified-outcome scorer for fine-tuning datasets.

Reads chatml JSONL exported by `cv dataset` (one session per line:
`{"messages":[{"role","content"}, ...]}`; roles user/assistant/tool; tool
calls/results live inside `content` as fenced ```tool_call``` / ```tool_result```
blocks). Computes per-signal sub-scores + a composite `score` in [0,1] and
re-emits each record annotated with `{"score": F, "signals": {...}}` while
preserving the original `messages`.

The composite is a maturity-bit: it selects HIGH-SIGNAL trajectories worth
distilling into a smaller model. Correctness evidence (verified_outcome) and
recovery behaviour (self_correction) dominate the blend because those are the
properties that make agent-coding data premium.

======================================================================
COMPOSITE WEIGHTS (sum to 1.0) — tune here, documented on purpose:
    verified_outcome  0.40   the correctness label — weighted highest
    self_correction   0.25   teaches error -> recovery; second highest
    tool_density      0.15   real agentic work uses tools
    reasoning_present 0.10   visible <thinking> reasoning to imitate
    substance         0.10   drops trivial / toy sessions
======================================================================

Pure stdlib (json, argparse, re, sys). Python 3.

HEURISTIC LIMITATIONS (be honest):
  - String/regex matching cannot tell a *claimed* "tests passed" from one a
    tool actually emitted; a deceptive assistant turn that types "200 OK" gets
    credit. We mitigate by preferring markers that overwhelmingly originate in
    tool_result blocks (e.g. "test result: ok", "0 failed", "exit 0").
  - "error"-family words appear constantly inside diffs, log dumps, and source
    code (492+ hits in a real record), so we DELIBERATELY do not treat raw
    "error" as a failure signal. self_correction keys only on tool_result
    failure markers and explicit non-zero exits, then requires a *later*
    verified_outcome marker — ordering is approximate (block-position, not a
    causal graph).
  - reasoning_present depends on <thinking> being preserved in the export;
    many harnesses strip it, so a 0 here is not proof of no reasoning.
  - Counts are saturating (diminishing returns) so a 600-tool-call megasession
    does not dwarf a tight, fully-verified 30-call session.
"""

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------
# Signal weights (composite blend). Keep summing to 1.0.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "verified_outcome": 0.40,
    "self_correction": 0.25,
    "tool_density": 0.15,
    "reasoning_present": 0.10,
    "substance": 0.10,
}

# ---------------------------------------------------------------------------
# Pattern banks. Comments cite exactly what each substring/regex keys on.
# ---------------------------------------------------------------------------

# verified_outcome markers. These are evidence the work was actually verified.
# Mix of case-sensitive (PASS) and case-insensitive (passed) handled below.
# Each entry: (pattern_or_substr, is_regex, case_insensitive)
_VERIFY_MARKERS = [
    ("test result: ok", False, True),   # cargo/rust test summary line
    ("0 failed", False, True),           # pytest / generic "N passed, 0 failed"
    ("build succeeded", False, True),    # xcodebuild / msbuild success line
    ("finished `release`", False, True), # `cargo build --release` -> "Finished `release`"
    ("finished `dev`", False, True),     # cargo dev profile success (sibling of above)
    ("200 OK", False, False),            # HTTP success status (case-sensitive on purpose)
    ("✓", False, False),                 # check-mark used by many test/lint runners
    ("merged", False, True),             # PR merged
]
# Word-boundary regexes (avoid substring false hits like "passed" in "bypassed",
# "PASS" inside "BYPASSED", "exit 0" inside "exit 02").
_VERIFY_REGEX = [
    re.compile(r"\bpassed\b", re.IGNORECASE),     # "14 tests passed"
    re.compile(r"\bPASS\b"),                       # case-sensitive PASS token
    re.compile(r"\bexit(?:\s+code)?\s+0\b"),       # "exit 0" / "exit code 0"
    re.compile(r"\bexit(?:ed)?\s+with\s+0\b"),     # "exited with 0"
]

# self_correction: a FAILURE marker that appears in a tool_result, OR an
# explicit non-zero exit. We key tightly here (not raw "error") to avoid the
# 400+ false positives "error" produces inside diffs/source.
_FAIL_REGEX = [
    re.compile(r"\bFAILED\b"),                     # test runner "FAILED"
    re.compile(r"\bpanicked\b", re.IGNORECASE),    # rust panic
    re.compile(r"\btest result:\s*FAILED\b", re.IGNORECASE),
    re.compile(r"\bexit(?:\s+code)?\s+[1-9]\d*\b"),# non-zero exit
    re.compile(r"\bexit(?:ed)?\s+with\s+[1-9]\d*\b"),
    re.compile(r'"success"\s*:\s*false'),          # tool envelope success=false
    re.compile(r"\b[1-9]\d*\s+failed\b", re.IGNORECASE),  # "3 failed"
    re.compile(r"\bcompilation\s+failed\b", re.IGNORECASE),
    re.compile(r"\berror\[E\d+\]", re.IGNORECASE), # rustc diagnostic code e.g. error[E0432]
]

# user-correction cues -> followed by assistant pivot (the second self_correction path)
_USER_CORRECTION_REGEX = [
    re.compile(r"^\s*no[,.\s]", re.IGNORECASE),    # "no, ..." at start of a user turn
    re.compile(r"\bactually\b", re.IGNORECASE),    # "actually that's not right"
    re.compile(r"\bthat'?s\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:quite|right|what)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(?:broke|missed|forgot|misunderstood)\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
]

_TOOL_CALL_FENCE = "```tool_call"
_TOOL_RESULT_FENCE = "```tool_result"
_THINKING = re.compile(r"<thinking>", re.IGNORECASE)

# Saturation anchors (counts above these earn ~full credit; tunable).
_VERIFY_SAT = 6.0        # ~6 distinct verify hits -> full verified_outcome
_SUBSTANCE_CHAR_SAT = 40000.0   # chars at which substance char-component saturates
_SUBSTANCE_MSG_SAT = 40.0       # messages at which substance msg-component saturates
_MIN_MESSAGES = 4               # below this -> trivial, substance penalized hard


def _count_markers(text):
    """Count verified_outcome marker hits in `text` (cheap of substr + regex)."""
    n = 0
    low = text.lower()
    for pat, _is_re, ci in _VERIFY_MARKERS:
        if ci:
            n += low.count(pat.lower())
        else:
            n += text.count(pat)
    for rx in _VERIFY_REGEX:
        n += len(rx.findall(text))
    return n


def _has_any(regexes, text):
    return any(rx.search(text) for rx in regexes)


def score_record(messages):
    """Return (composite_score, signals_dict) for one session's messages array."""
    n_msgs = len(messages)
    total_chars = 0
    assistant_turns = 0
    assistant_with_thinking = 0
    tool_call_blocks = 0

    # Per-message verify / fail flags, in order, for self_correction ordering.
    verify_at = []   # message indices that contain a verified_outcome marker
    fail_at = []     # message indices (tool role) that contain a failure marker
    user_corr_at = []  # user-correction message indices
    assistant_after = set()  # indices of assistant turns (for pivot check)

    for idx, m in enumerate(messages):
        role = m.get("role", "")
        content = m.get("content", "") or ""
        total_chars += len(content)

        if role == "assistant":
            assistant_turns += 1
            assistant_after.add(idx)
            if _THINKING.search(content):
                assistant_with_thinking += 1

        tool_call_blocks += content.count(_TOOL_CALL_FENCE)

        if _count_markers(content) > 0:
            verify_at.append(idx)

        # Failure markers only counted when they live in a tool_result block or a
        # tool-role message (avoids "error" noise in assistant prose / diffs).
        if role == "tool" or _TOOL_RESULT_FENCE in content:
            if _has_any(_FAIL_REGEX, content):
                fail_at.append(idx)

        if role == "user" and _has_any(_USER_CORRECTION_REGEX, content):
            user_corr_at.append(idx)

    # --- Signal 1: verified_outcome (saturating count of markers) ------------
    verify_hits = sum(_count_markers(m.get("content", "") or "") for m in messages)
    verified_outcome = min(1.0, verify_hits / _VERIFY_SAT)

    # --- Signal 2: self_correction -------------------------------------------
    # Path A: a tool failure followed *later* by a verified_outcome marker.
    recovered = False
    if fail_at and verify_at:
        first_fail = min(fail_at)
        if any(v > first_fail for v in verify_at):
            recovered = True
    # Path B: a user correction followed by a later assistant turn (pivot).
    user_pivot = False
    if user_corr_at and assistant_after:
        first_corr = min(user_corr_at)
        if any(a > first_corr for a in assistant_after):
            user_pivot = True
    # Blend the two paths: full credit if both, partial if one.
    if recovered and user_pivot:
        self_correction = 1.0
    elif recovered:
        self_correction = 0.8   # tool-failure recovery is the stronger signal
    elif user_pivot:
        self_correction = 0.5   # user-correction pivot is weaker (no proof of fix)
    else:
        self_correction = 0.0

    # --- Signal 3: tool_density (tool_call blocks per assistant turn) --------
    if assistant_turns > 0:
        density = tool_call_blocks / assistant_turns
    else:
        density = 0.0
    # ~1 tool call per assistant turn -> full credit; saturate.
    tool_density = min(1.0, density)

    # --- Signal 4: reasoning_present (fraction of assistant turns w/ <thinking>)
    if assistant_turns > 0:
        reasoning_present = assistant_with_thinking / assistant_turns
    else:
        reasoning_present = 0.0

    # --- Signal 5: substance (msg count + chars, capped; penalize trivial) ---
    if n_msgs < _MIN_MESSAGES:
        substance = n_msgs / (_MIN_MESSAGES * 4.0)  # hard penalty: <4 msgs -> <=0.19
    else:
        msg_comp = min(1.0, n_msgs / _SUBSTANCE_MSG_SAT)
        char_comp = min(1.0, total_chars / _SUBSTANCE_CHAR_SAT)
        substance = 0.5 * msg_comp + 0.5 * char_comp

    signals = {
        "verified_outcome": round(verified_outcome, 4),
        "self_correction": round(self_correction, 4),
        "tool_density": round(tool_density, 4),
        "reasoning_present": round(reasoning_present, 4),
        "substance": round(substance, 4),
    }

    composite = sum(WEIGHTS[k] * signals[k] for k in WEIGHTS)
    return round(composite, 4), signals


def _print_stats(scores, all_signals, n_emitted, n_total, out_stream=sys.stderr):
    print("=" * 56, file=out_stream)
    print(f"score_dataset stats: {n_total} records read, "
          f"{n_emitted} emitted", file=out_stream)
    if not scores:
        print("(no records scored)", file=out_stream)
        print("=" * 56, file=out_stream)
        return
    lo, hi = min(scores), max(scores)
    avg = sum(scores) / len(scores)
    print(f"composite score: min={lo:.4f} max={hi:.4f} mean={avg:.4f}",
          file=out_stream)

    # Histogram of composite scores in 10 buckets [0,1].
    print("\ncomposite histogram (bucket width 0.1):", file=out_stream)
    buckets = [0] * 10
    for s in scores:
        b = min(9, int(s * 10))
        buckets[b] += 1
    width = max(buckets) or 1
    for i, c in enumerate(buckets):
        lo_b, hi_b = i / 10.0, (i + 1) / 10.0
        bar = "#" * int(round(40 * c / width))
        print(f"  [{lo_b:.1f},{hi_b:.1f}) {c:>4}  {bar}", file=out_stream)

    # Per-signal averages.
    print("\nsignal averages:", file=out_stream)
    for k in WEIGHTS:
        vals = [sig[k] for sig in all_signals]
        a = sum(vals) / len(vals)
        print(f"  {k:<18} avg={a:.4f}  (weight {WEIGHTS[k]:.2f})",
              file=out_stream)
    print("=" * 56, file=out_stream)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score chatml JSONL agent-coding sessions for fine-tuning.")
    ap.add_argument("input", help="input chatml JSONL (one session per line)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="only emit records with composite score >= this")
    ap.add_argument("--out", default=None,
                    help="output file (default: stdout)")
    ap.add_argument("--stats", action="store_true",
                    help="print histogram + signal averages to stderr")
    args = ap.parse_args(argv)

    out = open(args.out, "w") if args.out else sys.stdout
    scores = []
    all_signals = []
    n_total = 0
    n_emitted = 0

    try:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"warning: skipping malformed line {n_total}: {e}",
                          file=sys.stderr)
                    continue
                messages = rec.get("messages", [])
                composite, signals = score_record(messages)
                scores.append(composite)
                all_signals.append(signals)

                if args.min_score is not None and composite < args.min_score:
                    continue

                # Preserve original messages; annotate.
                rec["score"] = composite
                rec["signals"] = signals
                out.write(json.dumps(rec) + "\n")
                n_emitted += 1
    finally:
        if args.out:
            out.close()

    if args.stats:
        _print_stats(scores, all_signals, n_emitted, n_total)

    return 0


if __name__ == "__main__":
    sys.exit(main())
