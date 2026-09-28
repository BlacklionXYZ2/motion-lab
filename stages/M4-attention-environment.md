# M4 — Attention and environment

**Status:** brief.
**Lane:** make/train (owner).

## Goal
Natural attention: learned allocation of eyes → head → neck → torso participation toward a target, with deterministic gaze correction on top.

## What you build / train
- **Deterministic attention features** as inputs (direction to target, distance, visibility/confidence) — the net never rediscovers geometry.
- **Synthetic target sampling** — where a dataset has no attention targets, sample plausible targets along the observed gaze ray; this lets attention conditioning train on ordinary mocap before hand-labelled data exists.
- **Environment flags** as compact inputs (user present, seated, contacts, distance).
- **Training** — add the features; retrain; test look-at onset/offset behaviour explicitly.

## Gate
Natural look-at onsets/offsets on held-out clips; no stability regression.

## Open decisions
Which environment flags in v1; how confidence is derived.
