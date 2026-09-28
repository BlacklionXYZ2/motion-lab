# M2 — Constraint layer (Python reference)

**Status:** brief.
**Lane:** make/train (owner) — the C# runtime twin is integration (M8), built later by a coding agent.

## Goal
The deterministic pass that makes predicted motion valid: joint limits, ground contact, gaze correction, foot lock. Built here as the reference; the future C# implementation must match it exactly.

## What you build
- **Reference solver** — clamp joint limits; floor/ground penetration fix against the work-area ground plane; simple gaze IK; foot lock while contact flags hold.
- **Differentiable soft proxies for training** — hinge penalty on limits, penetration penalty, gaze error — added to the M1 loss so the model learns to need less correction.
- **Correction telemetry** — per-frame correction magnitude as a health metric.
- **Golden fixtures** — (state → corrected pose) traces exported from real rollouts, for the C# conformance suite later.

## Gate
Rollouts through the solver remain stable; correction magnitudes small and non-persistent; fixtures generated.

## Open decisions
Which constraints in v1 (proposal: limits + ground + gaze; foot lock if it stays simple); how hard corrections may get before they count as a training failure.
