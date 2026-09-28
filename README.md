# motion-lab

Research project for the **SORA motion core**: a learned, recurrent motion controller that replaces stiff procedural animation with natural continuous motion (idle, conversational, gesture, transitions) at 30–60 Hz.

**Status:** stage briefs only — no code yet. Nothing here integrates with SORA until stage M8.

## Division of labour

- **Owner (Oscar): makes and trains** — stages M0–M7.
- **Integration into SORA (M8) and all SORA-repo work:** a coding agent (e.g. GPT-6 Astra) with the owner, later.
- These briefs are written maker/trainer-first: what gets built, what gets trained, with which data, and how you know it worked.

## Contents

- `stages/` — one brief per stage, M0–M8, in order. Each gets fleshed out before its stage begins.
- Programme roadmap (full detail, includes the integration view): `../SORA/docs/plans/MOTION-CORE-ROADMAP_2026-09-28.md` — same stage numbering.

## Invariants that hold across every stage

- Inputs are 6D rotations (first two columns of R); outputs are **angular velocities ω**; integration is `R(t+1) = R_t · Exp(Δt·[ω]×)`. Never Euler angles; never delta-6D output.
- ω is parent-relative (local frame); rotations are rest-pose-relative.
- The ω fed back as input is finite-differenced from the **realised** (post-constraint) pose — never recycled from the model's own raw output.
- The canonical skeleton is the **VRM Humanoid** bone set (no fingers in v1); every dataset retargets onto it.
- The net stays a pure function `f(state, h) → (ω, h')` and must export to ONNX cleanly from day one.
