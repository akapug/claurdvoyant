#!/usr/bin/env bash
# Generate the curated dataset ENTIRELY on z8 (keeps the laptop clean — only does IO/orchestration).
# Incrementally rsyncs the local session dirs to z8, then runs the z8-side pipeline.
# Usage: gen_dataset_on_z8.sh [out.jsonl] [min_score] [harnesses...]
set -euo pipefail
echo "→ rsync ~/.claude/projects -> z8 (incremental, compressed)" >&2
rsync -az --delete ~/.claude/projects/ z8:.claude/projects/ 2>&1 | tail -1
echo "→ run pipeline on z8 (cv+score+chunk all on the GPU box)" >&2
ssh z8 "bash ~/dataset-tools/make_dataset_z8.sh ${1:-} ${2:-} ${*:3}"
echo "✦ dataset built on z8 — at ~/datasets/real-dataset.jsonl (scp back or load in Studio)" >&2
