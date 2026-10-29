# Motion core M1: the AMASS converter

Model name: **Opus 5.5**. Checkpoint **2026-10-23 18:30:00 +01:00**, Europe/London. Status:
**built and checked against AMASS's sample; the full AMASS isn't downloaded**.

The owner: "build the converter (or does that need amass so you know how to build it?)". Also
asked why the model is about 0.5M parameters rather than 50M; that was answered in the
conversation and noted in motion-lab's `docs/M1.md`. No document had said 50M.

## Changes (motion-lab, uncommitted)

- **`datasets/amass.py`:** SMPL-H onto the canonical skeleton.
  - One-to-one joints, with the Z-up world turned to Y-up.
  - The root scale is fitted from planted feet while the root travels (1.0 for takes that barely
    move).
  - The floor height comes from planted ankles.
- **`scripts/build_amass.py`:** converts AMASS's own folder layout into processed clips with a
  report.
- **`scripts/check_amass_sample.py`:** compares the converter's limbs with the motion-capture
  markers in AMASS's official sample. The sample is downloaded to `data/amass/sample/`
  (git-ignored).
- **`datasets/m1_data.py`:** AMASS's split by sub-dataset (from its paper), and loading from
  several data folders. The training and evaluation scripts accept a list of folders.
- **Tests:** 3 new (47 in total, all passing), including the marker check whenever the sample is
  present.

## Verification

- **Against the sample's markers:**
  - shins, upper arms, forearms and left/right agree within 2 to 12°;
  - the head is a constant 26° off (head-marker slope or SMPL's rest head);
  - nothing is flipped.
- **Scale:** the first scale fit gave 0.55 on the sample. The sample is a standing take, so it
  had nothing to fit from; the scale is now fitted only from travelling frames.
- **Data loading:** a two-step training run loads the same LAFAN1 frames as before.
- **Not run:** a real AMASS build (needs the owner's registration and download), and anything
  with BABEL labels.

Nothing committed.
