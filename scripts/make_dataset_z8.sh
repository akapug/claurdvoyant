#!/usr/bin/env bash
# Runs ENTIRELY on z8 (zero laptop compute). Generates the curated training dataset from session
# dirs rsync'd to z8: cv dataset -> score (high-signal) -> chunk (trainable windows) -> messages-only.
# Usage: make_dataset_z8.sh [out.jsonl] [min_score] [harnesses...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CV="$HOME/.local/bin/cv"; PY="$(command -v python3)"
OUT="${1:-$HOME/datasets/real-dataset.jsonl}"; MINSCORE="${2:-0.5}"; shift 2 2>/dev/null || true
HARNESSES=("${@:-claude}")
mkdir -p "$(dirname "$OUT")" "$HOME/datasets/raw"
PARTS=()
for H in "${HARNESSES[@]}"; do
  echo "→ [$H] cv dataset (capped 24G, redacted)" >&2
  systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \
    "$CV" dataset --format chatml --harness "$H" --redact --min-messages 8 --out "$HOME/datasets/raw/$H-raw.jsonl"
  echo "→ [$H] score >=$MINSCORE" >&2
  "$PY" "$HERE/score_dataset.py" "$HOME/datasets/raw/$H-raw.jsonl" --min-score "$MINSCORE" --out "$HOME/datasets/raw/$H-good.jsonl"
  echo "→ [$H] chunk" >&2
  "$PY" "$HERE/chunk_dataset.py" "$HOME/datasets/raw/$H-good.jsonl" --max-chars 24000 --out "$HOME/datasets/raw/$H-chunked.jsonl" --stats
  PARTS+=("$HOME/datasets/raw/$H-chunked.jsonl")
done
cat "${PARTS[@]}" | shuf > "$HOME/datasets/raw/combined.jsonl"
"$PY" -c "import json
o=open('$OUT','w')
for l in open('$HOME/datasets/raw/combined.jsonl'):
    l=l.strip()
    if l: o.write(json.dumps({'messages':json.loads(l)['messages']})+'\n')
o.close()"
echo "✦ z8 dataset: $OUT ($(wc -l <"$OUT") records)" >&2
