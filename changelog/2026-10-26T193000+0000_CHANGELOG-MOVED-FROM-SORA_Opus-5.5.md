# motion-lab's changelogs moved out of SORA into motion-lab/changelog

Model name: **Opus 5.5**. Checkpoint **2026-10-26 19:30 UTC**. Status: **done**.

The owner: "Could you move all the motion lab changelogs into the motion lab's own changelog
folder? There's 230 changelogs now and it's going to become more and more challenging to
separate them while they sit in the same folder."

## Changes

- **Moved:** 21 entries, from `SORA/changelog/` to `motion-lab/changelog/`.
  - They cover the work done in motion-lab, 2026-10-11 to 2026-10-26: M0, the training
    environment (MA), M1's pipeline and versions v0 to v4, power logging and ARDY.
  - Two of them were committed in SORA's git and 19 weren't. Nothing was committed or
    removed from git; the move shows in SORA as two deleted tracked files until the owner
    commits.
- **Stayed in SORA:** 20 motion-related entries that changed SORA itself:
  - Alice as the default avatar, wave and walk, and the brain's motion keywords;
  - the physics body in SORA's player and its fixes;
  - the motion-core roadmap edits;
  - the eye rig and avatar layer synced into the player;
  - the hair, avatar height and launch port investigations;
  - two entries by GPT-6 Astra.
- **Links:** three links in moved entries now point to `../../SORA/docs/` (the roadmap and two
  verification records); all three resolve. The moved files are otherwise unchanged, names
  included.
- **References:** `versions/v0/README.md` and `versions/v1/README.md` named three moved entries
  as "SORA's changelog"; they now link here. Nothing in SORA linked to any moved entry.
- **New READMEs and pointers:**
  - `changelog/README.md` here: which folder a change belongs in, and this history;
  - `SORA/changelog/README.md`: a paragraph pointing here (logged in SORA's changelog too, as a
    change to SORA).

SORA's `changelog/` now holds 192 entries plus its README.

Nothing committed.
