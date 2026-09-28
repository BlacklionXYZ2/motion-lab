# M7 — Recurrent MoE

**Status:** brief — last research stage.
**Lane:** make/train (owner).

## Goal
Convert the single recurrent policy into a soft-routed mixture of experts whose experts represent motion *dynamics*, not individual gestures.

## What you build / train
- **Architecture** — 4 experts (general / locomotion / recovery / contact); soft routing `α = softmax(router(x))`; **all experts updated every tick** (dormant-state fix); blend ω before integration — never rotation matrices.
- **Router stability** — router stickiness (previous-α input or low-pass); supervised routing bootstrap (behaviour labels → experts) gradually relaxed to learned mixtures; load-balance/entropy terms.
- **Evaluation** — every earlier gate re-run against the M1 control, plus a dedicated regime-switch test set.

## Gate
≥ M1 control on all metrics; healthy expert utilisation; regime transitions smoother than the control.

## Open decisions
Shared trunk vs independent experts (keep as an experiment); when to relax supervision; expert count (start 4; justify any change).
