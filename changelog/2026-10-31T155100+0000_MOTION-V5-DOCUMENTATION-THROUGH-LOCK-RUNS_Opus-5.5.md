# Motion core v5: documentation brought up to date (free10 to the lock runs)

Model name: **Opus 5.5**. Checkpoint **2026-10-31 15:51 UTC**. Status: **documented**.

The owner (2026-10-31): "Would you mind adding all the documentation for the new training
runs? The documentation only refers up to free10, while we are past 14."

## Changes

- **`versions/v5/README.md`**, rewritten as one current document:
  - **status:** a table of every v4 problem and its state, the best model for each job, and
    the run-time settings;
  - **memory resets:** no longer recommended for free-run-trained models (they lowered energy
    and the hips);
  - **what v5 changes:** all of it (the latent, the velocity input, correlated noise, the foot
    and ground locks with the flight limit and the cap, growing a model, free and
    counterfactual runs, feedback windows, `keep_checkpoints`, the battery's probes and
    measures);
  - **every run**, latent-b01 to lock-free, with its result;
  - **a selected results table:** v4, free5, free7, free12b, free14, lock-sup;
  - **the tools.**
  - It replaces the old run table, which stopped at free10 (still marked "training") and was
    split by a stray subsection.
- **`docs/M1.md`:** the v5 row says where it stands.
- **`README.md`:** the M1 status says where v5 stands.
- **SORA's `docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md`:** M1 status updated in place (SORA
  changelog, same time).

No code changed. Nothing committed.
