"""ARDY motion service: keywords and a target velocity in, canonical 60 Hz poses out.

    .venv\\Scripts\\python service.py [--precision fp16] [--steps 4] [--fast]

SORA's Core starts this as a helper process and talks JSON lines over stdin and stdout (one
object per line). Everything it sends and receives is in motion-lab's **canonical** terms: the
22-joint skeleton, her proportions, right-handed (+Y up, facing +Z, her left +X), local
rotations as quaternions ``[x, y, z, w]``. Unity converts to and from its own frame.

**In** (stdin):
- ``{"type": "seed", "fps": 60, "frames": [{"root": [x, y, z], "rot": [[x, y, z, w] × 22]}, …]}``:
  her actual recent motion (recorded during the handover). Streaming starts from it, so the
  model continues where she is. A seed restarts the stream.
- ``{"type": "start"}``: start streaming without history (she starts standing). Nothing is
  generated before a seed or a start, so a signal sent first applies from the first frame.
- ``{"type": "signal", "key": "walk|cheerful|high", "velocity": [vx, vz] | null}``: what she is
  doing (a phrase-table key; unknown keys fall back to "-|-|-") and, for locomotion, the
  velocity to travel at (m/s, canonical frame). It applies from the next replan (at most 0.4 s).
- ``{"type": "stop"}``.

**Out** (stdout):
- ``{"type": "ready", "model": …, "load_s": …}`` once loaded;
- ``{"type": "chunk", "seq": n, "t0": s, "fps": 60, "frames": [{"root", "rot", "contacts"}…]}``:
  24 frames (0.4 s) at a time, ``t0`` being stream time (s) of the first frame;
- ``{"type": "error", "message": …}`` and ``{"type": "stopped"}``.

It runs at the real-time setting measured on the owner's GPU: a replan every 8 frames (0.4 s),
4 frames of history, 4 diffusion steps by default. It keeps about one chunk ahead of real time.
``--fast`` skips the pacing (tests and evaluation).

**Anchoring (2026-10-30, on by default; ``--no-anchor``):** after 1.6 s standing still under an
idle-like key, her own pose is held as a full-body keyframe ahead of every replan, and when she
is on the ground and asked to get up or stand about, her last standing pose becomes a keyframe
2.5 s ahead. Without it a long idle came apart within a minute and she never got up after a fall
(``stress_service.py``). ``--retries N`` redraws chunks that fail ``guard.py`` (off by default).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# stdout carries the protocol only: libraries that print (NVIDIA's model loading prints its
# settings) write to stderr instead.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import probe  # noqa: E402, I001  (order matters: guard needs the path stream sets)
import retarget  # noqa: E402
from stream import Streamer  # noqa: E402

import guard  # noqa: E402  (after stream, which puts motion-lab's package on the path)

PHRASES = HERE / "assets" / "phrases.npz"
STANDING = HERE / "assets" / "standing_pose.npz"
KEEP = 8  # ARDY frames played between replans (0.4 s at 20 fps)
HISTORY = 4  # ARDY frames of history per replan (NVIDIA's demo default)
LEAD_S = 0.45  # how far ahead of real time to stay
MAX_MOTION = 64  # ARDY frames kept as context (longer than any history used)
OUT_FPS = 60
STILL_INTENTS = {"-", "idle", "look", "stand"}  # idle-like keys that may be anchored
ANCHOR_AFTER = 32  # ARDY frames (1.6 s) of standing still before the anchor is taken
GET_UP_FRAMES = 50  # ARDY frames (2.5 s) she is given to get up
# Only a real standing height is anchored (her rest is 0.894 m; ARDY's idle stands at 0.89 to
# 0.92). The guard's 0.70 m let one run anchor a half crouch after sitting (hips 0.75 m).
ANCHOR_MIN_HIPS = 0.82


def default_standing() -> tuple[torch.Tensor, torch.Tensor] | None:
    """``assets/standing_pose.npz`` (``standing_pose.py``), or None if it hasn't been made."""
    if not STANDING.exists():
        return None
    data = np.load(STANDING)
    return torch.from_numpy(data["posed"]), torch.from_numpy(data["rots"])


class PhraseTable:
    def __init__(self, path: Path = PHRASES) -> None:
        data = np.load(path)
        pairs = zip(data["keys"], data["key_to_prompt"], strict=True)
        self.index = {str(k): int(i) for k, i in pairs}
        self.prompts = [str(p) for p in data["prompts"]]
        self.features = torch.from_numpy(data["features"])

    def feature(self, key: str) -> tuple[torch.Tensor, str]:
        i = self.index.get(key, self.index["-|-|-"])
        return self.features[i].view(1, 1, -1), self.prompts[i]


class MotionService:
    def __init__(self, precision: str, steps: int, retries: int = 0,
                 anchoring: bool = True) -> None:
        started = time.time()
        self.anchoring = anchoring  # idle anchoring (_update_anchor)
        self.still = 0  # ARDY frames she has been standing still under an idle-like key
        self.anchors = 0  # anchors captured
        # get-up target: her own last standing pose, or ARDY's standing_pose.py frame until then
        self.standing: tuple[torch.Tensor, torch.Tensor] | None = default_standing()
        self.getting_up, self.get_ups = False, 0
        self.retries = retries  # redraws of a chunk that fails the guard (next_chunk)
        self.rejected = 0  # draws thrown away
        self.unresolved = 0  # chunks sent although every draw failed (the least bad)
        self.key = "-|-|-"
        self.model = probe.load("onnx-dml", precision)
        self.rep = self.model.motion_rep
        self.names = list(self.rep.skeleton.bone_order_names)
        self.parents = [int(p) for p in self.rep.skeleton.joint_parents]
        self.contact_names = list(self.rep.skeleton.foot_joint_names)
        self.fps = int(self.rep.fps)
        self.phrases = PhraseTable()
        self.streamer = Streamer(self.model, HISTORY, steps)
        self.text, self.prompt = self.phrases.feature("-|-|-")
        self.velocity: tuple[float, float] | None = None
        self.seq = 0
        self.t = 0.0  # stream time of the next frame to send
        self.last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # last canonical frame
        self.load_s = time.time() - started

    # ---------- input ----------

    def seed(self, message: dict) -> None:
        frames = message.get("frames") or []
        fps = int(message.get("fps", OUT_FPS))
        if len(frames) < 2:
            raise ValueError("a seed needs at least two frames")
        rot = np.stack([Rotation.from_quat(f["rot"]).as_matrix() for f in frames])  # (T, 22, 3, 3)
        root = np.asarray([f["root"] for f in frames], dtype=np.float64)
        step = max(1, round(fps / self.fps))
        rot, root = rot[::step], root[::step]  # to ARDY's 20 fps
        local, ardy_root = retarget.from_canonical(rot, root, self.names, self.parents)
        usable = len(local) // self.streamer.patch * self.streamer.patch
        local, ardy_root = local[-usable:], ardy_root[-usable:]
        explicit = self.rep(local_joint_rots=torch.tensor(local[None], dtype=torch.float32),
                            root_positions=torch.tensor(ardy_root[None], dtype=torch.float32),
                            to_normalize=False)
        self.streamer.motion = self.rep.normalize(explicit)
        last_c = (rot[-1:], root[-1:], np.zeros((1, 4), dtype=bool))
        self.last = last_c
        self.still, self.streamer.anchor, self.getting_up = 0, None, False
        self.streamer.anchor_at = None
        # Until she has stood still, the seed's last frame (if she is on her feet) is her
        # get-up target.
        if guard.HIPS_BAND[0] <= float(root[-1, 1]) <= guard.HIPS_BAND[1]:
            self.standing = self._current_pose()
        self.seq, self.t = 0, 0.0

    def signal(self, message: dict) -> None:
        self.key = str(message.get("key", "-|-|-"))
        self.text, self.prompt = self.phrases.feature(self.key)
        velocity = message.get("velocity")
        self.velocity = None if velocity is None else (
            float(velocity[0]) / retarget.ROOT_SCALE, float(velocity[1]) / retarget.ROOT_SCALE)

    # ---------- output ----------

    def _generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One replan: the next KEEP ARDY frames, appended to the stream, as canonical
        (local rotations, root, contacts) at ARDY's frame rate."""
        self.streamer.step(self.text, self.velocity, KEEP)
        if self.streamer.motion.shape[1] > MAX_MOTION:
            self.streamer.motion = self.streamer.motion[:, -MAX_MOTION:]
        out = self.rep.inverse(self.streamer.motion[:, -KEEP:], is_normalized=True)
        return retarget.to_canonical(
            out["global_rot_mats"][0].numpy(), out["root_positions"][0].numpy(),
            out["foot_contacts"][0].numpy(), self.names, self.contact_names)

    def _update_anchor(self, badness: float) -> None:
        """Idle anchoring (2026-10-30, see ``Streamer.anchor``). Once she has stood still under
        an idle-like key for ANCHOR_AFTER ARDY frames and the newest chunk passed the guard, its
        last frame becomes her anchor; moving, or any other intent, drops it."""
        intent = self.key.split("|")[0] or "-"
        stopped = self.velocity is not None and abs(self.velocity[0]) + abs(self.velocity[1]) < 0.05
        hips = float(self.last[1][-1, 1]) if self.last is not None else None
        # below standing height: lying, kneeling, or left half-crouched after sitting
        down = hips is not None and hips < ANCHOR_MIN_HIPS
        streamer = self.streamer
        # Getting up (2026-10-30): asked to get up, or to stand about while she is down (after
        # a fall SORA sends "fall", then idle), ARDY alone left her on the ground (hips 0.35 to
        # 0.45 m for 12 s). Her last good standing pose, turned to her current heading, becomes
        # a keyframe 2 s ahead, so ARDY fills in the getting up.
        if (self.anchoring and stopped and self.standing is not None
                and (intent == "get_up" or (intent in STILL_INTENTS and down))):
            streamer.anchor, streamer.anchor_frames, streamer.anchor_turn = (
                self.standing, (1,), True)
            self.still = 0
            if not self.getting_up:
                self.getting_up, self.get_ups = True, self.get_ups + 1
            if streamer.anchor_at is None or streamer.anchor_at <= streamer.played:
                # (again if the time has come and she is still down: never a target in the past)
                streamer.anchor_at = streamer.played + GET_UP_FRAMES
            return
        if self.getting_up:  # up again (or asked for something else): drop the target
            self.getting_up, streamer.anchor, streamer.anchor_at = False, None, None
        streamer.anchor_frames, streamer.anchor_turn = (1, 21), False
        still = self.anchoring and intent in STILL_INTENTS and stopped
        if not still:
            self.still, streamer.anchor = 0, None
            return
        self.still += KEEP
        upright = hips is not None and hips >= ANCHOR_MIN_HIPS
        if streamer.anchor is None and self.still >= ANCHOR_AFTER and badness == 0 and upright:
            streamer.anchor = self._current_pose()
            self.standing = streamer.anchor
            self.anchors += 1

    def _current_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Her newest frame as ``(posed joints with the root's ground position at the origin,
        global rotations)``, ARDY's frame."""
        out = self.rep.inverse(self.streamer.motion[:, -1:], is_normalized=True)
        posed = out["posed_joints"][0, 0].clone()
        posed[:, [0, 2]] -= posed[self.rep.skeleton.root_idx, [0, 2]].clone()
        return posed, out["global_rot_mats"][0, 0].clone()

    def next_chunk(self) -> dict:
        """Generate the next 0.4 s and return it as canonical 60 Hz frames.

        With ``retries`` (2026-10-30), a chunk that fails :func:`guard.problems` is thrown away
        and drawn again from the same history, up to ``retries`` more times; the least bad draw
        is kept. Left alone, ARDY's stream drifts during a long idle: every variant tried came
        apart within 30 to 60 s (stress_service.py, idle_test.py)."""
        best = None
        for attempt in range(1 + self.retries):
            saved = self.streamer.save()
            local, root, contacts = self._generate()
            score = guard.badness(local, root, self.key, self.last)
            if best is None or score < best[0]:
                best = (score, (local, root, contacts), self.streamer.save())
            if score == 0:
                break
            self.rejected += 1
            if attempt < self.retries:
                self.streamer.restore(saved)
        assert best is not None
        _, (local, root, contacts), state = best
        self.streamer.restore(state)
        if best[0] > 0:
            self.unresolved += 1
        self._update_anchor(best[0])
        if self.last is None:  # no seed: the first frame stands in for "before"
            self.last = (local[:1], root[:1], contacts[:1])
        prev_local, prev_root, prev_contacts = self.last
        both = (np.concatenate((prev_local, local)), np.concatenate((prev_root, root)),
                np.concatenate((prev_contacts, contacts)))
        fine_local, fine_root, fine_contacts = retarget.resample(*both, self.fps, OUT_FPS)
        fine_local, fine_root, fine_contacts = fine_local[1:], fine_root[1:], fine_contacts[1:]
        self.last = (local[-1:], root[-1:], contacts[-1:])
        quats = Rotation.from_matrix(fine_local.reshape(-1, 3, 3)).as_quat().reshape(
            len(fine_local), -1, 4)
        chunk = {"type": "chunk", "seq": self.seq, "t0": round(self.t, 4), "fps": OUT_FPS,
                 "prompt": self.prompt,
                 "frames": [{"root": [round(float(v), 4) for v in fine_root[i]],
                             "rot": np.round(quats[i], 5).tolist(),
                             "contacts": [bool(c) for c in fine_contacts[i]]}
                            for i in range(len(fine_local))]}
        self.seq += 1
        self.t += len(fine_local) / OUT_FPS
        return chunk


def emit(message: dict) -> None:
    PROTOCOL.write(json.dumps(message, separators=(",", ":")) + "\n")
    PROTOCOL.flush()


def read_lines(inbox: queue.Queue) -> None:
    for line in sys.stdin:
        line = line.strip()
        if line:
            inbox.put(line)
    inbox.put(None)  # stdin closed: Core has gone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fast", action="store_true", help="no real-time pacing")
    parser.add_argument("--retries", type=int, default=0,
                        help="redraws of a chunk that fails guard.py's checks")
    parser.add_argument("--no-anchor", action="store_true",
                        help="no idle anchoring (a long idle then drifts)")
    args = parser.parse_args()
    try:
        service = MotionService(args.precision, args.steps, args.retries,
                                anchoring=not args.no_anchor)
    except Exception as exc:
        emit({"type": "error", "message": f"load failed: {type(exc).__name__}: {exc}"[:500]})
        return
    emit({"type": "ready", "model": f"ARDY Core {args.precision}", "steps": args.steps,
          "load_s": round(service.load_s, 2)})
    inbox: queue.Queue = queue.Queue()
    threading.Thread(target=read_lines, args=(inbox,), daemon=True).start()
    clock = None  # wall time at stream time 0
    streaming = False  # nothing is generated before a seed (or "start" without history)
    while True:
        try:
            while True:
                line = inbox.get_nowait()
                if line is None:
                    emit({"type": "stopped"})
                    return
                message = json.loads(line)
                kind = message.get("type")
                if kind == "stop":
                    emit({"type": "stopped"})
                    return
                if kind == "seed":
                    service.seed(message)
                    clock, streaming = None, True
                elif kind == "start":
                    clock, streaming = None, True
                elif kind == "signal":
                    service.signal(message)
                else:
                    emit({"type": "error", "message": f"unknown message type {kind!r}"})
        except queue.Empty:
            pass
        except (ValueError, KeyError, TypeError) as exc:
            emit({"type": "error", "message": f"bad message: {exc}"[:500]})
            continue
        if not streaming:
            time.sleep(0.01)
            continue
        if clock is None:
            clock = time.perf_counter()
        ahead = service.t - (time.perf_counter() - clock)
        if args.fast or ahead < LEAD_S:
            emit(service.next_chunk())
        else:
            time.sleep(min(0.02, ahead - LEAD_S + 0.001))


if __name__ == "__main__":
    main()
