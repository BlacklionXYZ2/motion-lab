# MA: the avatar layer and the training environment

Everything between the motion core's canonical pose and a particular VRM avatar, built so the
training and evaluation environment exists before training starts (owner, 2026-10-17). Plan:
`../SORA/docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md`, stage MA. Built by Opus 5.5,
2026-10-17.

## Run it

```powershell
.\unity-env\tools\run-unity-env.ps1               # every test, including the physics ladder (about an hour)
.\unity-env\tools\run-unity-env.ps1 -Filter A3    # one group
.\unity-env\tools\sync-from-sora.ps1              # after SORA's hair and clothing code changes
.\unity-env\tools\sync-to-sora.ps1                # after avatar-layer changes here
.venv\Scripts\python versions\v2\scripts\export_adapter_cases.py   # after adapter.py changes
```

The ladder writes `outputs/ma/ladder.md` (readable) and `outputs/ma/ladder.json` (every
measurement). Unity's log and raw results go to `outputs/unity-env/`.

## Where it lives

`unity-env/` is its own Unity project (6000.6.0f1, UniVRM 0.131.2, physics at 120 Hz with the TGS
solver and enhanced determinism). SORA's player is untouched until stage M8.

| Folder | What |
|---|---|
| `Assets/AvatarLayer/` | The avatar layer itself, the part that ships to SORA at M8 |
| `Assets/Environment/` | Test avatars, the 60 s scenarios, the ladder's measurements |
| `Assets/FromSora/` | Exact copies of SORA's hair and clothing code (A1); SORA's are the source of truth, and a test fails if they drift |
| `Assets/Tests/` | PlayMode tests: A2 fitting, A3 physics body, A4 falling, the ladder |

## The parts

**Canonical skeleton (`Canonical.cs`).** M0's 22 joints in Unity terms. motion-lab is
right-handed with her left at +X; Unity is left-handed with her left at −X, so converting
mirrors X. Rest frames are world-aligned, so a normalized local rotation means the same thing in
the model, the avatar layer and the physics body. A test checks the tables against
`canonical_skeleton.json`.

**A2, fitting (`AvatarMap.cs`, `AvatarFit.cs`).**
- `AvatarMap` is the C# twin of `adapter.py`. It agrees with Python on 21 shared cases
  (`tests/fixtures/adapter_cases.json` in the newest version folder; the Unity tests find it
  there and the canonical skeleton beside it), covering avatars missing every optional joint.
- `AvatarFit` retargets normalized rotations onto the avatar's own bone frames (UniVRM's control
  rig formula), scales hips travel and height by leg length, and pins planted feet with two-bone
  leg IK.

**A3, the physics body (`PhysicsBody.cs`).**
- One articulation link per avatar joint, at the avatar's own size in metres, with real
  gravity. Mapping onto screen pixels is the display's job: in SORA she is 500 pixels tall.
- Masses from de Leva's (1996) female segment proportions, for a body of BMI 21. Light links get
  a floor on rotational inertia, which keeps PhysX articulations stable.
- Colliders: boxes for the trunk and feet, capsules for limbs, a sphere for the head, **fitted to
  the avatar's mesh** (from 2026-10-17). Each part's shape is sized from the rest-pose vertices
  its bones carry (bone matrix times bind pose, so it is right at any scale): capsules to the
  95th percentile of distance from the bone (at most 4 times the proportional guess), boxes to the
  2nd to 98th percentile (at most 3 times). Vertices on spring bones (hair, loose clothing) are
  left out. Each rigid accessory (vertices following a non-humanoid bone that isn't a spring,
  such as Seed-san's robot arm) gets a box of its own on the part it hangs from.
- Only the two thighs collide with each other; every other pair of her own parts is ignored
  (from 2026-10-21: overlapping fitted shapes between feet and shins made them buzz).
- Her shapes use a skin material (static friction 1.0, dynamic 0.9, no bounce), so a limp hand
  stops instead of creeping across the taskbar.
- Anatomical joint limits per axis (the knee bends one way; the elbow one way). Most main bends
  reach M0's placeholder maxima; the knee stops at 150° (M0: 160°, which needs soft tissue the
  shapes don't have; at 160° a fallen body wedged its foot and never settled).
- Motors are PD drives with **realistic peak torques** (N·m per kg of body mass, from hips 2.5 to
  hands 0.15). Stiffness is set so a motor saturates at 20° of error. Unity's drive stiffness is
  per radian (measured: `DriveUnitsAreCalibrated`). Targets are rotation vectors; every test pose
  is reached exactly.
- **Help is a dial defaulting to zero**, and every use is logged (`HelpDial`, `HelpLog`):
  - exempt joints (at full help, nearly rigid, whatever the forces);
  - strength above realistic (up to ten times);
  - a balance helper: a joint from the hips to an invisible anchor at the target, solved by PhysX
    (stable at any strength), at most four times her weight. It can hold any part at any point
    (`HoldAt`): SORA uses that to hang her from where she was grabbed. Unity reports no force for
    joints on articulations, so the log estimates it from the spring law.

**Desktop surfaces (`DesktopSurfaces.cs`).** The taskbar, window tops and screen edges as
colliders, converted from pixels by her height. A surface can be removed (a window closing).

**A4, falling (`FallController.cs`).** Physical, not a preset animation.
1. Trigger: no foot on anything for a moment, a tip past recovery, or a drop. The direction comes
   from her momentum.
2. Reaction, 0.35 s at 60% tone: arms towards the fall, chin tucked, a catch step, a turn.
3. Impact and settling at 15% tone (neck 40%). Passive damping stays at the level of 45% tone,
   and every link has angular damping 1.5 while she settles and rests, so she comes to rest
   rather than flopping. Help is off for the whole fall. She is at rest when every link has
   moved under 5 cm/s, averaged over 0.1 s, for 0.25 s (averaging ignores millimetre contact
   chatter).
   A fall still going after 12 s goes straight to getting up, and after 22 s she is put back on
   her feet, so no fall can last for ever.
4. Getting up: a stand-in (kneel, then stand, with help, logged) until M7 learns it.

**Stand-in motion (`StandInMotion.cs`).** Hand-made poses: stand, walk and run, sit, the fall
reaction, relaxed and kneeling. One gait covers walking and running (`Gait`): the step period
shortens with speed, the share of each stride a foot is down falls from 0.6 to 0.3, and above
that both feet are off the ground briefly (`HasFlight`). M1's model
replaces them through the same `CanonicalPose`; nothing downstream changes.

**Scenarios and the physics ladder (`Scenarios.cs`).** Seven fixed 60 s scenarios (stand, walk,
run, sit, push, fall, get up), each run at the four ladder steps:
1. full help;
2. no exempt joints;
3. realistic motor strength;
4. no balance helper.

The least help each scenario needs is recorded per avatar. Hair and clothing clipping (A1) is
measured at full help: each chain against the body colliders it collides with.

## Test avatars

- **Alice:** SORA's default, VRM 0.x. Read from `../SORA/.local/avatars`, because her licence
  forbids copying her.
- **Seed-san:** VRM 1.0, no upper chest, bundled with SORA.
- **Odd:** Seed-san with 30% longer legs, 25% shorter arms and a 15% longer back, for the swapping
  gate.

## Results (2026-10-22)

The dated records are `../SORA/docs/verification/MOTION_MA-AUDIT-FIXES_Opus-5.5_2026-10-22.md`
(latest) and `MOTION_MA-ENVIRONMENT_Opus-5.5_2026-10-17.md`; the latest run's measurements are in
`outputs/ma/ladder.md`. In short:
- 15 of 17 tests pass. Both failures are Seed-san's fall at full help (2.33 s to settle, gate 2 s:
  her robot-arm hand rolls slowly over its box), and the ladder also fails on A1 clipping (hair,
  skirt and props; waits, by the owner's decision).
- Walking and running pass at realistic motor strength on all three avatars, with no planted-foot
  slide. Odd's run fails at step 2 only (ten times strength, no exempt joints).
- Falls settle in 0.64 to 1.70 s otherwise; the body mesh rests 0 to 6.1 cm into the floor and
  never hovers. Physics costs 0.24 to 0.27 ms a frame.
- Every scenario except falling still needs the balance helper. Balancing is the motion core's job.
- `DiagnosticsTests.RunTrace` (explicit, run by name) traces the start of a run: hips tilt, speed
  and what touches the floor.

**Watch out:** settling after a fall is sensitive to `PassiveDampingTone` and
`SettlingAngularDamping`, not monotonically. 0.45 and 1.5 are the values where the most falls
pass; raising the angular damping to 2.5 made four of twelve fail.
