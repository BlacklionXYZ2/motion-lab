# M1 — Motion prior

**Status:** brief — the critical path; the longest stage.
**Lane:** make/train (owner).

## Goal
Train the single recurrent model: motion state → next angular velocities, stable over long autoregressive rollouts. This model is the control that every later stage is compared against.

## What you build
- **Data pipeline** — AMASS + LAFAN1 (licences to confirm) → retarget onto the canonical skeleton → resample to 60 Hz (SLERP rotations, recompute ω) → fixed windows (~1–2 s) with stride → mirror augmentation → subject-disjoint train/val/test splits → contact flags from FK (foot height/velocity thresholds) → normalisation statistics.
- **Model** — pose encoder MLP → GRU (start 1×256; try 2 layers) → decoder MLP → ω, bounded `ω = ω_max · tanh(raw)`, optional root planar velocity head. Target ≈0.5M parameters.
- **Training loop** — teacher forcing → scheduled sampling → multi-step rollout loss with a 1→8→32-step curriculum; input-pose noise; hidden-state burn-in for cold starts; loss terms: geodesic rotation error + angular velocity + FK joint position + acceleration. Per-run config files, seeds, checkpoints, local run log.
- **Eval harness** — ≥60 s autoregressive rollouts: drift-vs-horizon curves, velocity-saturation rate, foot slide, jerk, distribution comparison against the dataset.

## Gate
60 s+ rollouts stable (no divergence, no NaN, no saturation pathology); statistics in-distribution; idle-from-seed motion looks human.

## Open decisions
- Dataset mix and mirror policy; sequence length and batch size.
- 1 vs 2 GRU layers (measure, don't guess).
- Root velocity head in v1 or not.
- Viewer: in-repo mini tool vs deferring to the M8 Unity path — naturalness must be visible to be iterated on, so this needs deciding early in the stage.
