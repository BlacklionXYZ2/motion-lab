# Motion M0: skeleton table reviewed by a second agent; M0 closed

Model name: **Opus 5.5**. Checkpoint **2026-10-16 09:15:00 +01:00**, Europe/London. Status:
**M0 complete**.

## What happened

- The owner chose the reviewer: **Gemini 4 Argon (medium)** through `agy`. It ran read-only in
  plan mode on `~/Desktop/motion-lab`, changed no files, and passed the table
  ([record](../../SORA/docs/verification/MOTION_M0-SKELETON-REVIEW_Gemini-4-Argon_2026-10-16.md)).
- Its two notes are low severity and need no change for M0:
  - `-0.0` values in `canonical_skeleton.json`, from the 180° turn of VRM 0.x models;
  - an adapter fold rule (`parent in target`) that could misbehave only on a skeleton with a
    missing joint below a forward-folded one, which the 22-joint VRM humanoid can't have.

## My cross-check (Opus 5.5)

Recomputed from the JSON with the lab's `.venv` Python, not taken from the review:
- every parent comes before its child;
- each offset's length matches `distance_from_parent_m` (within 0.06 mm);
- left joints are at +X, and the largest left/right mismatch is 0.028 mm;
- the lowest joint (toes) is 0.001 mm above the ground, and the toes are ahead of the feet;
- the synthesised upper chest sits 0.4360 of the way from chest to neck.

`pytest` in motion-lab: 35 passed. The reviewer did not read the VRM files or re-measure the
0.436 ratio on `sister1.0.vrm`; neither did I.

## Files

- New: `docs/verification/MOTION_M0-SKELETON-REVIEW_Gemini-4-Argon_2026-10-16.md` (the review,
  unedited, under its author's name).
- `docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md`: M0 marked complete; the open-gate line replaced.
- `~/Desktop/motion-lab/docs/M0.md`: the open item closed.

HOW_TO_USE.md: no change to usage. Nothing committed.
