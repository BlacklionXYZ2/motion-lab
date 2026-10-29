# motion-lab committed and pushed to its own GitHub repository

Model name: **Opus 5.5**. Checkpoint **2026-10-29 12:44 UTC**. Status: **done**.

The owner asked for everything in motion-lab to be committed and pushed to GitHub. motion-lab
had one commit (the 28 September README and stage briefs) and no remote.

## Changes

- **`.gitignore`:** `unity-env/` now ignores the same Unity-generated folders `unity-spike/`
  already did: `Library/`, `Temp/`, `Logs/`, `Builds/`, `UserSettings/`, `obj/`, and the
  generated `*.csproj` and `*.slnx`. Without this the commit would have taken about 750 MB of
  Unity cache (24,700 files, one of 100 MB, over GitHub's file limit). `Assets/`, `Packages/`,
  `ProjectSettings/` and `tools/` are committed.
- **Remote:** a new private repository, `BlacklionXYZ2/motion-lab`, set as `origin`; `master`
  pushed.
- Everything else already ignored stays out: raw data, `external/` (third-party models),
  each version's converted data, runs and outputs, `.venv`s, `*.pt`, `*.onnx`, `*.npz`.

## Verification

- `git add -A --dry-run` before and after the ignore change, and the size of what was staged.
- No tests were run; no code changed.

HOW_TO_USE.md: SORA's page; no change to usage.
