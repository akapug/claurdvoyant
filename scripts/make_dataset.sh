#!/usr/bin/env bash
# make_dataset — one command from session corpus to a curated, Studio-ready chatml JSONL.
# Pipeline: cv dataset (export + redact) -> score_dataset.py (verified-outcome score) -> filter.
#
# Usage: make_dataset.sh --out final.jsonl [--harness H] [--limit N] [--min-score F] [--no-redact]
# Requires: cv on PATH, scripts/score_dataset.py beside this file, python3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS=""; LIMIT=""; MIN_SCORE="0.5"; OUT=""; REDACT="--redact"
while [ $# -gt 0 ]; do case "$1" in
  --harness) HARNESS="--harness $2"; shift 2;;
  --limit) LIMIT="--limit $2"; shift 2;;
  --min-score) MIN_SCORE="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  --no-redact) REDACT=""; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
[ -n "$OUT" ] || { echo "usage: make_dataset.sh --out FILE [--harness H] [--limit N] [--min-score F] [--no-redact]" >&2; exit 2; }
RAW="$(mktemp /tmp/cv-raw-XXXX.jsonl)"
echo "→ export (cv dataset chatml $REDACT $HARNESS $LIMIT)" >&2
cv dataset --format chatml $REDACT $HARNESS $LIMIT --out "$RAW"
echo "→ score + filter (min-score $MIN_SCORE)" >&2
python3 "$HERE/score_dataset.py" "$RAW" --min-score "$MIN_SCORE" --stats --out "$OUT"
rm -f "$RAW"
echo "✦ curated dataset → $OUT ($(wc -l <"$OUT") records, score>=$MIN_SCORE) — drag into Unsloth Studio" >&2
