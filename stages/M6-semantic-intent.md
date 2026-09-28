# M6 — Semantic intent

**Status:** brief.
**Lane:** make/train (owner).

## Goal
High-level intent driving the learned motion manifold: IDLE / LISTEN / NOD / WAVE / POINT / EXPLAIN / SHRUG / TURN_TOWARD_TARGET / …

## What you build / train
- **Vocabulary** — take SORA's existing registered semantic action list as the vocabulary (one vocabulary, two consumers). Freeze the mapping here; SORA integration consumes it later.
- **Conditioning** — intent embedding + intensity + progress/phase; hysteresis held upstream so intent cannot flicker frame-to-frame.
- **Training** — add intent conditioning; test intent-appropriate dynamics and intent-to-intent transitions.

## Gate
Intent produces intent-appropriate dynamics; transitions between intents do not pop; the stability battery still passes.

## Open decisions
Exact vocabulary freeze; how progress/phase is represented.
