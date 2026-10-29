# Motion core M0 started in motion-lab: rotation maths, skeleton, formats, contract, ONNX

Model name: **Opus 5.5**. Checkpoint **2026-10-11 21:45:00 +01:00**, Europe/London. Status:
**M0 built; every numeric gate passed; two items open** (the second-agent skeleton review and the
non-gating Unity spike).

The owner asked: "start M0 in motion-lab". This follows
[MOTION-CORE-ROADMAP_2026-10-11](../../SORA/docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md), stage M0.

## Changes

**`~/Desktop/motion-lab`** (the owner's research repo; new files only, existing README and
stage briefs untouched, nothing committed):

- `pyproject.toml`;
- `scripts/setup-env.ps1`: a `.venv` on the owner's ROCm Python via
  `--system-site-packages`; adds onnx, onnxscript and ruff;
- `src/motionlab/`:
  - `rotation.py`, `vrm.py`, `skeleton.py` (with `assets/canonical_skeleton.json`), `fk.py`;
  - `formats.py`, `motion_signal.py`, `export.py`;
- `scripts/build_skeleton.py` and `scripts/m0_gate.py`;
- `specs/motion_signal.v1.schema.json`;
- `tests/`: 29 tests;
- `docs/M0.md`: the conventions and how to run it.

**Decisions made in M0 (open in the old briefs):**

- **Canonical skeleton = her VRM's 21 humanoid body joints:**
  - no upperChest (Seed-san has none; three-spine datasets merge into `chest`);
  - toes in, fingers and eyes out;
  - world-aligned rest frames, with offsets read from `Seed-san.vrm`.
- **ω is in the body frame,** with re-projection onto SO(3) every integration tick.
- **The motion state per tick is 192 numbers;** windows are 2 s with a 0.5 s stride.
- **`MotionSignal` v1 frozen** with the roadmap's defaults for decisions 7 and 8.

**SORA:**

- `docs/verification/MOTION_M0_Opus-5.5_2026-10-11.md` (new);
- the roadmap's status line and M0 status.

## Verification

- **Component:** 29 tests passed; ruff clean.
- **Measured gates (`m0_gate.py`):**
  - 6D round trip 2.4e-7 (limit 1e-5);
  - orthonormality after 100,000 float32 steps 7.2e-7 (limit 1e-4);
  - FK rest pose 0.0 (limit 1e-6);
  - ONNX against PyTorch over 600 ticks 1.7e-6 (limit 1e-5).
- **Found and fixed:**
  - NaN gradients of the geodesic loss at the identity;
  - float32 Log accuracy near π;
  - the exporter splitting the weights into a second file.
- **Not run:** Unity (the runtime spike); the second-agent skeleton review. SORA's own test suite
  wasn't affected (documentation only there).
