# Motion core v5: she turns to face where she goes (free8); penalties can't stop the glide, so a foot lock

Model name: **Opus 5.5**. Checkpoint **2026-10-30 22:45 UTC**. Status: **free8 tested; the foot
lock is built and tested; free9 is training**.

Follows `2026-10-30T192500+0000_MOTION-V5-FREE7-STEERS-BY-GLIDING_Opus-5.5.md`. The usage
limit paused the session again this evening; the runs finished in that time.

## free8 and free8b (from free7, 3,000 steps each, side by side)

- **`latent-vel-free8`:** + `stance` 2 (the slowest foot joint under 0.1 + 0.15 × root speed)
  + `facing` 0.5.
  - Its counterfactual tracking collapsed to 1.35 m/s at step 1,000: standing still was the
    easy way to keep a foot still.
  - It recovered to 0.05 m/s by the end.
- **`latent-vel-free8b`:** rebalanced, track 2, stance 0.5, facing 1.

**Stress battery** (run-time sampling, 16 seeds):

| | recordings | free7 | free8 | free8b |
|---|---|---|---|---|
| Steering: walk 1.2 / run 3 m/s, reached | – | 1.30 / 3.07 | 1.17 / 2.94 | 1.23 / 2.98 |
| **Steering: seconds to face a new heading, 90° / 180°** | – | never / never | **4.4 / 3.7** | **3.6 / 4.3** |
| Gliding while steering | – | 98% | 88% | 97% |
| Gliding in 5-minute walk runs | 30% | 99.5% | 99% | 100% |
| Seeds frozen at 5 min (unknown / walk) | 0 | 0 / 0 | 69% / 100% | 100% / 100% |
| Hovering (idle) | 0.07% | 0 | 6% | **77%** |
| Outside the limits (walk) | 0.5% | 6% | 9% | 18% |
| Turning bias, rad/s | – | 0.006 | 0.033 | 0.07 |

**Reading:**
- **`facing` works:** for the first time she turns to face the way she is sent, within 4.5 s,
  while still tracking the speed.
- **`stance` backfired.** It applies in every free run, so freezing became the cheapest answer
  again, and it never produced stepping. Gliding is unchanged.
- **Neither model is better overall than free7.**

## Why penalties can't stop the glide, and the foot lock (`m1.foot_locked`, `ModelConfig.foot_lock`)

The model outputs the root's velocity separately from the legs, so it can always meet a travel
command by moving the root. Penalties only argue with that one tick at a time.

**The foot lock** is the roadmap's M2 foot lock, built into the step, so it runs the same in
training and at run time:
- **While a foot is near the floor**, the root moves so that foot stays where it was. She can
  then only travel by stepping, and the tracking loss reaches the legs through forward
  kinematics.
- **Choosing the support:**
  - it is a soft minimum (5 mm) over the contact joints' heights;
  - each foot counts by its higher position over the two frames, because a foot leaving the
    floor isn't support. With an even split between two planted feet, the planted one slid
    half the moving one's way, which the first test caught.
- **In flight** the lock fades out between 3 and 8 cm of lift (running flight), and the model's
  own root velocity takes over.
- **Unchanged:** height and facing stay the model's, and it adds no weights.
- **Tests:** 85 pass.
  - Feet on the floor cancel a glide.
  - The weight-bearing foot stays within 1 cm while the other leg swings.
  - In the air the model's velocity applies.

## Next (running)

- **`latent-vel-free9`:** free7 with the foot lock, facing 0.5 and no stance, 3,000 steps.
- **Also measured:** free7 with the lock switched on at run time only, untrained, through the
  battery's new `--foot-lock`. Results in the next entry.

Nothing committed.
