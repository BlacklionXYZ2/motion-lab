# Motion MA: the avatar layer and the training environment

Model name: **Opus 5.5**. Checkpoint **2026-10-17 10:00:00 +01:00**, Europe/London. Status:
**MA built and measured; its gate is open** on hair and clothing clipping (A1) and the owner's
judgement of falls.

The owner: "Could you finish the avatar layer and the physics upgrades? The idea is that the
ideal training environment will exist before training begins, everything else can be finished
after that."

## What was built (in `~/Desktop/motion-lab`, uncommitted)

- **`unity-env/`:** a new Unity project (copied settings and packages from `unity-spike/`;
  physics at 120 Hz, TGS solver, enhanced determinism).
- **`Assets/AvatarLayer/`:** the part that ships to SORA at M8.
  - The canonical skeleton in Unity terms.
  - A2: the avatar map (C# twin of `adapter.py`) and fitting (retargeting, hips travel scaled by
    leg length, two-bone leg IK for planted feet).
  - A3: the physics body (articulation links, de Leva masses, anatomical limits, realistic
    torques, the help dial with a usage log, a PhysX-solved balance helper).
  - Desktop surfaces.
  - A4: falling (trigger, reaction, settling, stand-in get-up).
  - The stand-in motion.
- **`Assets/Environment/`:** the test avatars (Alice, Seed-san, and "Odd", Seed-san with long
  legs and short arms), the six 60 s scenarios, the physics ladder, and the A1 clipping probe.
- **`Assets/FromSora/`:** exact copies of SORA's `AvatarSecondaryParts.cs` and
  `AvatarSecondaryMotionTuner.cs`; a test fails if they drift. To update them, run
  `scripts/sync-from-sora.ps1`.
- **`Assets/Tests/`:** 9 PlayMode tests.
- **Scripts:**
  - `scripts/run-unity-env.ps1` runs the tests;
  - `scripts/export_adapter_cases.py` writes the 21 shared adapter cases to
    `tests/fixtures/adapter_cases.json`.
- **Docs:** `docs/MA.md` (new); `README.md` (status, layout, run).

## In SORA

- `docs/verification/MOTION_MA-ENVIRONMENT_Opus-5.5_2026-10-17.md`: the measurements.
- `docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md`: status line and MA status, in place.
- SORA's code and player are unchanged.

## Verification

- Unity PlayMode: **8 passed / 1 failed** (the ladder, on A1 clipping only), 2026-10-17 at about
  09:48.
- motion-lab `pytest`: 35 passed. `ruff` is clean on the new script.
- Kinds of evidence: component and simulation only, with stand-in motion. **Not run:** anything
  in SORA's player, and any owner viewing.

HOW_TO_USE.md: no change (SORA's usage is unchanged).
