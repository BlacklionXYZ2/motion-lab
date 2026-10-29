# motion-lab: each model version self-contained in versions/vN

Model name: **Opus 5.5**. Checkpoint **2026-10-25 11:00:00 +01:00**, Europe/London. Status:
**moved, tested; the v2 data rebuild restarted in its new place**.

The owner: "Put everything to do with v1 in the v1 folder, and everything for v2 in the v2 folder.
This includes data adjustment scripts, training scripts, the actual model architecture and
anything else relating to what produced the model." Then: "there won't be src and scripts
because the work would just sit in the new version's folder, isolated from past versions". Also
"don't delete this model": the v1 model is untouched.

## Layout (motion-lab)

- **`versions/v0/`:** the experiments before v1 (the LAFAN1-only baseline, smoke, load and resume
  tests). These are their runs, config and outputs. There's no code snapshot, because that code
  had changed in place; its README says which code came closest.
- **`versions/v1/`:** everything that produced the owner's kept model.
  - The `motionlab` package, scripts, tests, `configs/combined.json`, v1's converted data, the
    model (`runs/combined`, untouched), its logs and evaluations.
  - Its code is a copy of the working code with v1's retargeting as the default
    (`match_bone_directions=True`), plus v1's LAFAN1 calibration file.
- **`versions/v2/`:** the working version: v2's code (T-pose-relative retargeting), scripts,
  tests, config, data, and the M1 pipeline doc (moved from `docs/M1.md`).
- **The top level** keeps only what is shared:
  - raw downloads (`data/`);
  - docs (with a new `docs/M1.md` index of versions), `specs/`, `stages/`;
  - the Unity projects; `unity-env/tools/` now holds `run-unity-env.ps1`, `sync-to-sora.ps1` and
    `sync-from-sora.ps1`;
  - `setup-env.ps1`.
- **There's no top-level `src/`, `scripts/`, `tests/`, `configs/` or `runs/`.**
- **Nothing was deleted.** Everything was moved; only empty folders and one `__pycache__` were
  removed.

## Other changes

- **Paths:** each version's scripts use their own folder for code, converted data, runs and
  outputs, and the top level for raw data, specs and the Unity spike. Each version has a
  `conftest.py`, so its tests import its own `motionlab`.
- **The `.venv` path file** (`motionlab-src.pth`, pointing at the old `src/`) is retired: replaced
  by a comment, and `setup-env.ps1` no longer writes it. It would have let one version import
  another's code.
- **`pyproject.toml`:** pytest runs one version at a time (`pytest versions\v2`), and ruff treats
  `motionlab` and `scripts` as first-party.
- **`.gitignore`:** ignores `versions/*/data/`.
- **Unity training environment:** its tests find the canonical skeleton and the adapter cases in
  the newest `versions/vN` (`TestAvatars.LatestVersion`). A doc comment in `Canonical.cs` was
  updated and synced into SORA.
- **SORA's `P18PhysicsBodyTests`:** its drift message names `unity-env/tools/sync-to-sora.ps1`.
- **Docs:** motion-lab's README, `docs/M0.md`, `docs/MA.md`, `docs/RUNTIME_SPIKE.md` and
  `data/README.md` name the new paths; there's a README in each version folder.

## Verification

- **Tests:** v1 53 passed (with v1's retargeting); v2 53 passed. Ruff clean across `versions/`.
- **The kept v1 model** loads from `versions/v1` with v1's own code and data (step 27,000; 350
  test clips) and runs a finite rollout. Nothing was written.
- **The SORA sync check:** every avatar-layer file is identical after the sync.
- **The v2 rebuild** starts and runs from `versions/v2`.
- **Not run:** the Unity environment's tests after the path change. The C# changes are small (a
  newest-version lookup and two paths), but they're untested until the next Unity run.

Nothing committed.
