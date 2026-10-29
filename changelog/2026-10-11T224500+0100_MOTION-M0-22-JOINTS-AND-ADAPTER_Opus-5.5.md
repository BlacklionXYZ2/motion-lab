# Motion core M0: 22-joint canonical skeleton and the avatar adapter

Model name: **Opus 5.5**. Checkpoint **2026-10-11 22:45:00 +01:00**, Europe/London. Status:
**built and tested in motion-lab; M0 still open** (the second-agent skeleton review and the Unity
spike).

The owner asked: "Instead of training with combined joints, could we use a different model that
has the correct joints? Then missing joints can just be concatenated or trained to be dealt with.
Would an mmd character model work?" (linking Lynae from Wuthering Waves on Sketchfab).

## Changes

**`~/Desktop/motion-lab`** (uncommitted):

- **`skeleton.py`:**
  - 22 joints (the full VRM humanoid body, with `upperChest`); neck and shoulders hang from it;
  - `OPTIONAL_JOINTS` (the VRM's optional bones);
  - placeholder limits for `upperChest`.
- **`scripts/build_skeleton.py`:**
  - checks the hierarchy against the nearest present joint;
  - places a missing `upperChest` at 0.436 along chest → neck (measured on `sister1.0.vrm`);
  - records `avatar_joints` and `synthesised`;
  - asset rebuilt.
- **New `adapter.py`:** `AvatarMap` (fold rules) and `avatar_skeleton(vrm)`.
- **Tests:** a new `tests/test_adapter.py` (the fold rules; Seed-san's orientations exact and
  positions within 4 cm at 20°); skeleton tests updated to 22 joints.
- **`docs/M0.md`:** skeleton and adapter sections; sizes updated.

**SORA:**

- the roadmap's rule 4 (22 joints, adapter) and M0 status;
- an addendum to `docs/verification/MOTION_M0_Opus-5.5_2026-10-11.md`.

## Verification

- **Component:**
  - 32 tests passed; ruff clean;
  - M0 gate re-run: all four pass (ONNX parity 1.5e-6);
  - measured position shift above Seed-san's folded `upperChest`: 1.8 cm, 3.5 cm and 5.2 cm at
    10°, 20° and 30°.
- **Checked, not used:** the Lynae page says CC BY, but it's a reupload of a Wuthering Waves
  (Kuro Games) character credited to another creator. It isn't needed, because the joints come
  from the VRM standard and the datasets.
