# motion-lab

Research for **SORA's learned motion core**: a recurrent model that turns SORA's semantic state
(`MotionSignal`) into joint rotations for her avatar, replacing the hand-written poses. Nothing
here is part of SORA until stage M8.

- **The plan:** `../SORA/docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md` (stages M0 to M8). It
  supersedes the 28 September README (kept at
  [docs/history/README_2026-09-28.md](docs/history/README_2026-09-28.md)) and the stage briefs
  in `stages/`, which haven't been rewritten yet.
- **Who does what (owner, 2026-10-11):** agents build, train and evaluate. The owner supplies
  dataset licences, GPU time and naturalness judgements.

## Status (2026-10-30)

- **M0 (foundations): complete.** See [docs/M0.md](docs/M0.md).
  - Rotations, the 22-joint canonical skeleton (Alice's proportions), the avatar adapter,
    formats, `MotionSignal` v1 and the ONNX export: 35 tests.
  - The runtime spike is done ([docs/RUNTIME_SPIKE.md](docs/RUNTIME_SPIKE.md)): Sentis on the
    CPU.
  - The skeleton table was reviewed line by line by Gemini 4 Argon (2026-10-16).
- **M1 (the motion prior): training.** Each model version is self-contained in `versions/vN/`
  (owner, 2026-10-25); see [docs/M1.md](docs/M1.md) for the versions and what each produced.
  - **v5 (2026-10-30 onwards)** has the stress battery (`scripts/stress_m1.py`).
    - Freezing is solved: a latent, free-run training and correlated noise.
    - `latent-vel-free7` steers precisely but glides.
    - The foot and ground locks make gliding and hovering impossible: `latent-vel-lock-sup`
      walks on planted feet but doesn't follow commands yet.
    - Status and every run: [versions/v5/README.md](versions/v5/README.md).
- **ARDY** (NVIDIA's model, what SORA's Reload motion model runs): `ardy-runtime/`.
  - Stress-tested on 2026-10-30.
  - The service now anchors long idles and gets her up after a fall (`ardy-runtime/README.md`).
- **MA (the avatar layer and the training environment): built** in `unity-env/`. See
  [docs/MA.md](docs/MA.md): fitting onto any VRM, the physics body, falling, desktop surfaces,
  the seven 60 s scenarios (walking and running included) and the physics ladder. Its code is
  copied into SORA's player with `unity-env/tools/sync-to-sora.ps1`.

## Layout

| Path | What |
|---|---|
| `versions/vN/` | **one model version each, self-contained:** its `motionlab` package (skeleton, rotations, adapter, formats, retargeting, the model), `scripts`, `tests`, `configs`, converted `data`, trained `runs` and `outputs`. New work happens in the newest version's folder; older ones stay as they were |
| `data/` | raw downloads shared by every version (LAFAN1, 100STYLE, CMU, BABEL, Bandai Namco), git-ignored; `data/README.md` lists them |
| `specs/` | `motion_signal.v1.schema.json` |
| `setup-env.ps1` | creates the shared `.venv` |
| `unity-spike/` | the scratch Unity project for the runtime spike (never SORA's) |
| `unity-env/` | the avatar layer and training environment (MA), its own Unity project; `unity-env/tools/` runs its tests and syncs with SORA |
| `tools/` | `power_log.py` logs the PC's CPU and GPU power during a training run; `power_tally.py` totals every run's energy into `outputs/power-tally.md` |
| `outputs/` | generated results (gitignored) |

## Run

```powershell
.\setup-env.ps1                                   # once: .venv on the owner's ROCm PyTorch
.venv\Scripts\python -m pytest versions\v4          # one version's tests (never two at once)
.venv\Scripts\python versions\v4\scripts\m0_gate.py
.\unity-env\tools\run-unity-env.ps1                # MA: the Unity environment's tests and the physics ladder
```
