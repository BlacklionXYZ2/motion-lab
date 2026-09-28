# M8 — Integration into SORA

**Status:** brief — **not the owner's lane.** Performed later by a coding agent (e.g. GPT-6 Astra) with the owner. This file exists so earlier stages keep the right constraints.

## What integration does (for awareness only)
- Export to ONNX → run in Unity (path decided by the integration spike: Sentis, ONNX Runtime plugin, or a hand-rolled GRU).
- Slot behind the existing pose-ownership arbiter: learned core = base layer; gaze / reach / grab / ragdoll = overrides.
- `CognitiveMotionState` rides the existing presentation contract (schema bump), not a parallel channel.
- Root trajectory stays deterministic (existing coordinate authority).
- A/B against the procedural stack in the real app; latency checks alongside llama.cpp and a running game.

## What earlier stages must preserve for this to work
- The net stays a pure function `f(state, h) → (ω, h')` and exports to ONNX cleanly (no exotic ops).
- Interface payloads stay versioned and matched to the roadmap's semantics (CanonicalState / CognitiveMotionState / MotionCommand / CanonicalPose).
- Deterministic-layer behaviour is captured as golden fixtures (M2) so the C# twin can be conformance-tested.
