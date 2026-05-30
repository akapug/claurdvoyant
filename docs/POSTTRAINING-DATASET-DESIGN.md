# Posttraining datasets from agent sessions via cv + OpenSession

> Planning doc (akapug fork, branch `docs/posttraining-dataset`). Assessing whether `cv` +
> OpenSession is a viable pipeline for extracting high-quality post-training data from real
> agent-coding sessions — especially deep Opus-4.8 trajectories. Not a build plan yet; a design +
> a feasibility + a risk read. Decisions (esp. legal/provenance) are the maintainer's.

## Thesis

Real, *verified*, *reasoning-visible* agent-coding sessions are some of the highest-signal
post-training data that exists — and `cv` has already done the hard 80%: it parses 17 harness
formats into one typed IR (`Session → Message → Block{Text|Thinking|ToolUse|ToolResult|File|Image}`)
and OpenSession standardizes the wire format. That IR is, almost exactly, the schema you'd design
for agentic SFT / process-reward data. So the question isn't "can we build a parser" (done) — it's
"is the *signal* worth it, and what's the curation + legal layer on top."

Short answer: **yes, with a real curation layer and a hard legal/provenance gate.**

## Why cv is already most of the pipeline

| Dataset need | cv already provides |
|---|---|
| Normalize many agent formats → one schema | `cv-core` IR + 17 harness adapters |
| Portable, shareable record | OpenSession (draft-0.2) export |
| Separate reasoning / tool-calls / results | `Block::Thinking` / `Block::ToolUse` / `Block::ToolResult` |
| Find/slice the corpus | `cv search` (+ index) / `cv ls` |
| Re-home without leaking creds | `cv port` already drops credentials |
| Session digest / labeling primitive | `cv distill` (LLM-backed) |

The gaps are **scoring/curation**, **redaction**, and a **training-format export target** — all
buildable *on* cv as new `convert`/`export` modes, not a new system.

## What makes these sessions unusually valuable (the differentiators)

1. **Verified outcomes, not plausible-looking text.** Many trajectories end in an IRL gate —
   tests pass, build green, a measured before/after, a merged PR. That's a *correctness label*
   most agentic corpora lack (they're synthetic or unlabeled logs).
2. **Visible reasoning.** `Block::Thinking` captures the model's actual chain — premium for
   distilling reasoning into smaller models (vs. answer-only SFT).
3. **Self-correction traces.** The highest-value pattern: plan → wrong premise → external
   correction → research → invert the conclusion → ship. Teaching a small model to *verify before
   claiming* and *recover from its own errors* is exactly what synthetic data underproduces. These
   arcs are dense in real long-horizon sessions.
4. **Tool-use density + real results.** Genuine multi-tool orchestration (shell, edit, git, MCP,
   sub-agents) paired with *real* tool outputs — not fabricated.
5. **Long-horizon continuity.** Multi-hour / multi-session arcs with state carried across
   compaction — rare and hard to synthesize.
6. **Cross-harness, not model-locked.** Because cv normalizes formats, the corpus isn't tied to one
   agent UI; OpenSession makes it a potential *shared standard* (an ecosystem play: others
   contribute sessions in OpenSession → a common high-quality agentic corpus).

## The pipeline (each stage = a cv mode or a thin pass on top)

1. **Extract** — `cv` → OpenSession for every session (search/filter to scope).
2. **Score & filter** — select trajectories worth training on. Signals: ended-in-verified-outcome
   (tests/build/PR), tool-use density, presence of self-correction, length/coherence. Heuristics +
   a judge pass (cv distill can label). Drop low-signal chatter.
3. **Transform** — OpenSession → training shape. Two targets worth building:
   - `--format sft`: chat-style messages (system/user/assistant + tool-call/tool-result) for SFT.
   - `--format trajectory`: step-wise (state, action=tool-call, observation=result, outcome) for
     agentic RL / process-reward modeling, with per-step or per-arc outcome labels.
4. **Redact / decontaminate** — strip secrets (cv port already drops creds — extend to inline
   tokens), private-repo paths, customer code, PII. **Non-negotiable gate.**
5. **Curate** — dedupe, balance by task type, hold out an eval split, label mistakes vs. fixes so
   wrong-then-uncorrected reasoning isn't taught as correct.

## Risks / gates (ember-spirit honesty — these are the real blockers, not the parsing)

- **Provenance / licensing.** Our sessions touch *private* repos (polyana, simbi prod, goodtimes,
  customer code). Verbatim publication is off the table. Two clean uses: (a) **private** dataset to
  train *our own* smaller coding/agentic assistant; (b) a **public** dataset built only from
  sessions over *public/own* code, aggressively redacted. Pick per-use; don't conflate.
- **Model-output terms.** Training a model on another provider's model outputs can hit ToS clauses
  (esp. "don't train a competing model"). For internal, non-competing use the risk is lower; public
  release or competitor-training is a legal decision — **maintainer/Captain call, flagged, not
  assumed.**
- **Correctness ≠ presence.** The model makes mistakes (this very corpus contains a plan built on an
  unverified premise, later corrected; and an indexing path that OOM'd before being fixed). Include
  mistake→correction *as labeled recovery* — never raw wrong-and-unfixed reasoning as a positive.
  This is why stage-2 outcome-labeling is load-bearing.
- **Mode collapse / self-preference.** Distilling a small model purely on one big model's style
  collapses diversity. Mix sources / harnesses (cv's cross-harness normalization helps here).
- **Scale/cost of curation.** The judge/score pass over a large corpus is the real compute cost;
  cv distill on a local model (already wired) keeps it cheap + private.

## Concretely buildable next (on cv, smallest-dose first)

1. `cv export --format sft|trajectory <filter>` — OpenSession → training jsonl (new convert target).
2. An outcome-scorer: detect "verified-ending" sessions (tests-passed / build-green / PR-merged
   heuristics from the tool-result blocks) + a self-correction detector; emit a per-session quality
   score. Local-model judge via the existing distill backend.
3. A redaction pass beyond creds (inline secrets, private paths, PII) with a verify step.
4. A tiny held-out eval harness to measure whether a small model trained on the curated set
   actually improves on agentic-coding tasks (the only honest proof the data is "high quality").

## Verdict

cv + OpenSession turns "we have great sessions somewhere" into a *repeatable extraction pipeline*,
and the verified-outcome + visible-reasoning + self-correction properties make the result a tier
above typical agentic corpora. The engineering is mostly additive (export modes + a scoring/redact
pass). The gating decisions are **legal/provenance** and **public-vs-private intent** — those are
the Captain's to make before any extraction at scale. If green-lit for a *private, internal*
dataset first (lowest risk, immediately useful for a smaller polyana/coding assistant), the build is
small and the eval harness will tell us honestly whether the signal is as good as it looks.
