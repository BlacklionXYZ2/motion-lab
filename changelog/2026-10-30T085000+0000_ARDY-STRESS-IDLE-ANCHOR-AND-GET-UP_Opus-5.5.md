# ARDY service: stress-tested; a long idle no longer comes apart, and she gets up after a fall

Model name: **Opus 5.5**. Checkpoint **2026-10-30 08:50 UTC**. Status: **done and verified in
motion-lab** (component and live-model evidence on the owner's GPU; not yet seen in her player).

The owner (2026-10-30): "keep iteratively improving the motion core by research, training and
testing. Your tests should aim to expose unusual or undesirable behaviour and you should then
look to fix it." ARDY's service (`ardy-runtime/service.py`) is what SORA's Reload motion model
runs today, so it got the same stress treatment as motion-lab's own model.

## The stress driver (`ardy-runtime/stress_service.py`)

- **How it drives the service:** `MotionService` runs in-process, stepped chunk by chunk (0.4 s)
  with a signal before every chunk, as SORA's player sends them. That way each signal lands
  exactly where the scenario says.
- **How it scores:** with motion-lab v5's stress measures (`versions/v5/motionlab/stress.py`,
  loaded by path because the service already imports v4's `motionlab`), plus root and joint
  jumps, acceleration against the recordings' 99th percentile, and speed and direction tracking.
- **Scenarios:**
  - turns and reversals, walk then run;
  - walk and idle flickering every chunk;
  - "sit" while asked to travel;
  - 5 m/s, backwards, sideways;
  - starting from a T-pose, a frozen frame, or no seed;
  - no velocity;
  - a 2-minute idle;
  - falling and getting up;
  - emotions;
  - (added with the fix) a real lying pose under idle and under get up, three random seeds
    each.

## What it found (before: `outputs/stress-service-fp16-s4-before-anchor.json`)

**Sound:**
- speed within 1 to 2% of the request and travel direction within 1°;
- seams from a seed under 3°;
- T-pose, frozen-frame and unseeded starts;
- flicker;
- stopping;
- no joint pops in normal use.

**Broken:**
1. **A long idle comes apart.** Over 2 minutes of "idle" at 0 m/s:
   - energy climbed 0.21 → 0.38 → 1.67 rad/s, ending at 4 rad/s;
   - she turned through 8,520° in total;
   - 29% of joint-frames were outside the data's limits;
   - a 60°-per-frame joint pop, and acceleration 12× the recordings' 99th percentile;
   - the head was below the hips 18% of the time, a wrist inside the torso 13%, the legs
     crossed 43%;
   - the hips ended at 1.06 m.

   `idle_test.py` tried variants. **Every one came apart within 30 to 60 s:**
   - the neutral prompt, or no velocity;
   - 8 or 16 frames of history (16 only delayed it);
   - 10 diffusion steps;
   - fp32;
   - three random seeds.

   The hips floated up to 2.4 m or collapsed to 0.1 m.
2. **She never got up.** After a fall, "get up" and then idle left her hips at 0.35 to 0.45 m.
   From a real lying pose, idle and get up both left her down in all six runs (hips 0.11 to
   0.40 m, legs crossed up to 85%). SORA's player sends "fall" and then idle after a physical
   fall.
3. **She doesn't turn to face where she is sent:**
   - asked to reverse, she walks or runs backwards (facing error 166 to 176°);
   - asked to go sideways, she crabs at about 50°.

   SORA's player only ever asks along her facing, so this is latent, but any change that sends
   a world-frame velocity would make her moonwalk.
4. **Weak or failing text:**
   - "run" with no velocity moves at walking speed (1.19 m/s);
   - high-intensity emotions barely change a walk (energy 0.96 to 1.04 rad/s);
   - "fall" only sometimes falls.

## Fixes (`service.py`, `stream.py`, new `guard.py`, `standing_pose.py`)

1. **Idle anchoring (on by default; `--no-anchor` turns it off).**
   - **When:** she has stood still under an idle-like key (idle, unknown, look, stand) at
     0 m/s for 1.6 s, and the newest chunk passes the guard's checks.
   - **What:** that chunk's last frame becomes her anchor. Every replan then puts it, at her
     current spot, as an ARDY full-body keyframe 1 s and 2 s ahead.
   - **Why it doesn't freeze her:** the keyframes move with each replan and are never reached,
     so they pull rather than pose her. Moving, or any other intent, drops the anchor.
   - One keyframe at 2 s held the hips, but the limbs still drifted (up to 45% outside the
     limits under "idle"). Two held every prompt tried.
2. **Getting up.**
   - **When:** asked to get up, or to stand about while her hips are below 0.70 m.
   - **What:** her last good standing pose becomes a single keyframe, turned to her current
     heading. It is fixed in time 2.5 s ahead and comes nearer with each replan, so she gets up
     within it. A target that stays 2 s ahead never lifted her.
   - **Where the pose comes from:** before she has stood still, the seed's last frame if she is
     on her feet. Failing that, `assets/standing_pose.npz`: ARDY's own "stands naturally calmly"
     frame, made once by `standing_pose.py` (hips 0.913 m).
3. **A chunk guard (`guard.py`, `--retries N`, off by default):** a chunk outside the limits,
   out of the hips band, head below hips, or with a pop is drawn again from the same history.
   - It didn't help the idle drift: about 660 draws were rejected and two-thirds of chunks had
     no sound draw, because the drift is systematic, carried in the history.
   - It is kept as an option, and its checks gate when an anchor may be taken.
   - The streamer gained `save`/`restore` for it.

## Verified (`outputs/stress-service-fp16-s4-final.json`, fp16, 4 steps, the owner's GPU, shared with two training runs)

| Scenario | Outside limits | Worst pop, °/frame | Accel / data p99 | Head below hips |
|---|---|---|---|---|
| 2-minute idle | 29% → **3.3%** | 59.5 → **18.4** | 12.3 → **2.5** | 18% → **0** |
| Fall, get up, idle | 24% → **3.5%** | 36.6 → **6.9** | 4.7 → **0.6** | 0 → 0 |
| Lying, idle (3 seeds) | 13 to 16% → **1.3 to 1.8%** | ≤ 7.9 | ≤ 1.04 | 0 |
| Lying, get up (3 seeds) | 16 to 23% → **1.5 to 2.0%** | ≤ 5.0 | ≤ 0.5 | 0 |

- **2-minute idle:** energy by 30 s is now 0.25, 0.20, 0.27 rad/s; total turning 8,520° → 568°;
  the hips end at 0.91 m.
- **From lying:** the hips rise from 0.21 to 0.40 m to 0.905 to 0.912 m in every run, with
  idle as well as with get up.
- **Every other scenario is unchanged** (the anchor doesn't engage while she moves).
- **`test_service.py`:** passes, including `--realtime`. Chunks arrive a median 0.32 s early
  and at worst 0.29 s late, as on 2026-10-27.
- **Load time:** unchanged (1.8 s for the model).

## Not done

- Not seen in her built player, and the owner hasn't judged the get-up or the anchored idle.
- The anchored idle still has short fidgets (energy 0.3 to 0.8 rad/s for a few seconds, up to
  12% outside the limits) that settle by themselves.
- The backwards and sideways walking (3) and the weak text (4) are not fixed. For (3), the
  service could turn the velocity into her facing, or SORA could keep sending it along her
  facing as it does now.

SORA's `HOW_TO_USE.md` notes the behaviour (SORA changelog, same time). Nothing committed.
