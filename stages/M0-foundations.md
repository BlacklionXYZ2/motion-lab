# M0 — Foundations and specs

**Status:** brief — flesh out first.
**Lane:** make/train (owner).

## Goal
Build the math, skeleton and data-format foundations that every later stage consumes. Nothing is trained in this stage.

## What you build
- **`rotate` module** — 6D ↔ rotation-matrix conversions; SO(3) Exp/Log (Rodrigues); integration `R' = R·Exp(Δt·[ω]×)`; angular velocity by finite differencing `ω = Log(R_prevᵀ·R_now)/Δt`.
- **`skeleton` module** — one definition of the canonical skeleton: joint table (VRM Humanoid set), parent indices, rest-pose offsets, per-joint axis limits, per-joint ω_max. Every other module imports this — no duplicated joint lists anywhere.
- **`formats`** — tensor conventions for the canonical state (layout; units: radians/sec, metres; 60 Hz tick), windowing conventions, and the four interface payload drafts (CanonicalState, CognitiveMotionState, MotionCommand, CanonicalPose) as versioned JSON for later promotion into SORA contracts.
- **Tests** — round-trips (6D → R → 6D), Exp/Log inverses, orthonormality preserved over long integration chains, FK against a hand-computed fixture, mirrored-joint symmetry.
- **ONNX export smoke test** — a trivial GRU/MLP exported and re-loaded (onnxruntime, Python only) so the export path is proven early. The Unity intake test belongs to integration (M8).

## Gate
All tests green; the skeleton table reviewed line by line — a wrong rest offset or parent index poisons training, evaluation and IK everywhere.

## Open decisions
- Toes/eyes in v1 or not.
- ω_max values (placeholder constants now; replaced by dataset percentiles in M1).
- Package/module naming.
