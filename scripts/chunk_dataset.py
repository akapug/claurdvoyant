#!/usr/bin/env python3
"""chunk_dataset — split long chatml sessions into trainable-sized windows.

Long agent sessions (a single Opus arc can be tens of MB / hundreds of thousands of tokens) are
terrible as ONE training example — a trainer truncates them at max_seq_length and wastes the rest.
This splits each `{"messages":[...]}` record into consecutive windows whose total content stays
under --max-chars, cutting ONLY at message boundaries so each window is a coherent slice of the
conversation. A leading system message (if present) is prepended to every window for context.

Pick the high-signal SESSIONS first (score_dataset.py), THEN chunk the keepers — so a window
inherits a verified session's quality. Streaming + stdlib-only; one line at a time (memory-safe
even on a 52 MB record).

Usage:
  chunk_dataset.py <in.jsonl> [--max-chars N] [--out FILE] [--stats]
"""
import argparse, json, sys

def msg_len(m):
    c = m.get("content", "")
    return len(c) if isinstance(c, str) else len(json.dumps(c))

def chunk_record(messages, max_chars):
    """Yield lists-of-messages, each with total content <= max_chars (best-effort, msg-boundary)."""
    if not messages:
        return
    system = messages[0] if messages and messages[0].get("role") == "system" else None
    body = messages[1:] if system else messages[:]
    sys_len = msg_len(system) if system else 0

    cur, cur_len = [], 0
    for m in body:
        ml = msg_len(m)
        # if a single message alone blows the budget, truncate its content to fit
        if ml > max_chars - sys_len:
            if isinstance(m.get("content"), str):
                m = {**m, "content": m["content"][: max(0, max_chars - sys_len - 100)] + "\n…[truncated]"}
                ml = msg_len(m)
        if cur and cur_len + ml > max_chars - sys_len:
            yield ([system] + cur) if system else cur
            cur, cur_len = [], 0
        cur.append(m)
        cur_len += ml
    if cur:
        yield ([system] + cur) if system else cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-chars", type=int, default=24000, help="~6k tokens; window content ceiling")
    ap.add_argument("--out")
    ap.add_argument("--min-msgs", type=int, default=2, help="drop windows with fewer messages")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    out = open(a.out, "w") if a.out else sys.stdout
    sessions = chunks_out = 0
    for line in open(a.input):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception as e:
            print(f"chunk_dataset: skip bad line: {e}", file=sys.stderr)
            continue
        msgs = rec.get("messages", [])
        sessions += 1
        for win in chunk_record(msgs, a.max_chars):
            if len([m for m in win if m.get("role") != "system"]) < a.min_msgs:
                continue
            rec_out = {k: v for k, v in rec.items() if k not in ("messages",)}
            rec_out["messages"] = win
            out.write(json.dumps(rec_out) + "\n")
            chunks_out += 1
    if a.out:
        out.close()
    if a.stats:
        print(f"chunk_dataset: {sessions} sessions -> {chunks_out} windows (max-chars={a.max_chars})", file=sys.stderr)

if __name__ == "__main__":
    main()
