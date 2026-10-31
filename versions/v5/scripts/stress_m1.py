"""Run the stress battery (``motionlab/stress.py``) on a checkpoint.

    ..\\..\\.venv\\Scripts\\python scripts\\stress_m1.py CHECKPOINT [--seeds 16] [--device cpu]

Writes ``outputs/stress-<name>.json`` (``--out`` to choose) and prints it. The checkpoint can be
any version with v4's model interface (v4 or v5).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionlab import m1, stress  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402
from motionlab.formats import FPS  # noqa: E402
from motionlab.motion_signal import MotionSignal  # noqa: E402

S = MotionSignal


def load(checkpoint: Path, device: torch.device):
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = m1.MotionPrior(m1.ModelConfig(**data["model_config"])).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    norm = m1_data.Normaliser(data["normaliser"]["mean"].to(device),
                              data["normaliser"]["std"].to(device))
    return model, norm, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--context", type=int, default=60)
    parser.add_argument("--long-seconds", type=float, default=300.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", nargs="*", help="run only these probes")
    parser.add_argument("--reset", type=float, default=0.0,
                        help="staggered memories rebuilt every S seconds (stress.drive)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="a latent model's prior temperature")
    parser.add_argument("--rho", type=float, default=0.0,
                        help="a latent model's noise correlation per tick")
    parser.add_argument("--foot-lock", action="store_true",
                        help="switch the foot lock on whatever the model was trained with")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    model, norm, ckpt = load(args.checkpoint, device)
    config = ckpt["config"]
    sources = [ROOT / f for f in config["data"]]
    sampler = m1_data.WindowSampler(
        m1_data.load_clips(sources, "test", clean=config.get("clean_data", False),
                           support_frame=config.get("support_frame", False)), device,
        source_weights=config.get("source_weights"))
    gen = torch.Generator().manual_seed(7)
    seed = m1_data.WindowSampler.sample(sampler, args.seeds, args.context + 1, gen, augment=False)
    ref = sampler.sample(64, 20 * FPS, gen, augment=False)
    ref_p = stress.positions(ref.rotations, ref.root)
    toes = ref_p[:, :, [stress.JOINT["leftToes"], stress.JOINT["rightToes"]], 1].flatten()
    toe_floor = float(torch.quantile(toes[::7], 0.005))
    report: dict[str, object] = {"checkpoint": str(args.checkpoint), "step": ckpt["step"],
                                 "seeds": args.seeds, "toe_floor_m": round(toe_floor, 3)}
    rec = stress.summary(ref.rotations, ref.root, toe_floor)
    rec.update({"energy_mean_rad_s": round(float(stress.energy(ref.rotations).mean()), 3),
                "root_speed_mean_m_s": round(float(stress.ground_speed(ref.root).mean()), 3)})
    rec.update({"acceleration_peak_p99_rad_s2": round(float(torch.quantile(
        stress.acceleration_peak(ref.rotations).flatten()[::5], 0.99)), 1)})
    report["recordings"] = rec
    p99 = rec["acceleration_peak_p99_rad_s2"]
    want = set(args.only) if args.only else None
    report["reset_s"] = args.reset
    report["temperature"], report["noise_correlation"] = args.temperature, args.rho
    model.temperature, model.noise_correlation = args.temperature, args.rho
    if args.foot_lock:
        model.config.foot_lock = True
    report["foot_lock"] = model.config.foot_lock

    def drive(*a, **k):
        model.reset_noise()
        return stress.drive(*a, reset=int(args.reset * FPS), **k)

    def probe(name: str) -> bool:
        return want is None or name in want

    def timed(name: str, fn) -> None:
        if not probe(name):
            return
        start = time.time()
        report[name] = fn()
        print(f"{name}: {time.time() - start:.0f} s", flush=True)

    long_ticks = int(args.long_seconds * FPS)

    def long_runs() -> dict[str, object]:
        out = {}
        for label, sig in (("unknown", S()), ("idle", S(intent="idle")),
                           ("walk", S(intent="walk")), ("run", S(intent="run"))):
            d = drive(model, norm, seed, args.context, long_ticks, sig)
            out[label] = stress.summary(d.rotations, d.root, toe_floor)
        if "locomotion" in model.signal_encoders:
            # v5: as SORA drives a velocity model, which always sends a velocity: standing about
            # (0 m/s) and walking ahead at 1.2 m/s, closed loop, for the whole run
            facing = stress.formats.heading(seed.rotations[:, -1, 0])
            for label, sig, speed in (("idle at 0 m/s", S(intent="idle"), 0.0),
                                      ("walk at 1.2 m/s", S(intent="walk"), 1.2)):
                d = drive(model, norm, seed, args.context, long_ticks,
                          controller=stress.go_towards(sig, facing, speed))
                out[label] = stress.summary(d.rotations, d.root, toe_floor)
        return out

    def switches() -> dict[str, object]:
        order = [S(intent="idle"), S(intent="walk"), S(intent="run"), S(intent="walk"),
                 S(intent="sit"), S(intent="idle"), S(intent="wave"), S(intent="run"),
                 S(intent="idle")]
        every = 4 * FPS
        ticks = every * len(order)
        sched = stress.signal_schedule(order, ticks, every, args.seeds)
        d = drive(model, norm, seed, args.context, ticks, sched)
        speed = stress.ground_speed(d.root)
        accel = stress.acceleration_peak(d.rotations)
        per = []
        for k, sig in enumerate(order):
            s = k * every
            per.append({
                "intent": sig.intent,
                "speed_m_s_last_2s": round(float(speed[:, s + 2 * FPS:s + every - 1].mean()), 3),
                "energy_last_2s": round(float(stress.energy(d.rotations)[:, s + 2 * FPS:
                                                                        s + every - 1].mean()), 3),
                "peak_accel_first_0.5s_over_data_p99": round(
                    float(accel[:, max(s - 1, 0):s + FPS // 2].amax(-1).mean()) / p99, 2),
            })
        return {"every_s": 4, "segments": per,
                **stress.body(d.rotations, d.root, toe_floor),
                "outside_limits_share": round(stress.limits_share(d.rotations), 4)}

    def flicker() -> dict[str, object]:
        ticks = 20 * FPS
        out = {}
        for label, pair in (("walk/idle every tick", [S(intent="walk"), S(intent="idle")]),
                            ("run/sit every 3 ticks", [S(intent="run"), S(intent="sit")])):
            every = 1 if "every tick" in label else 3
            d = drive(model, norm, seed, args.context, ticks,
                             stress.signal_schedule(pair, ticks, every, args.seeds))
            steady = drive(model, norm, seed, args.context, ticks, pair[0])
            out[label] = {
                "stable": bool(torch.isfinite(d.rotations).all()),
                "jerk_ratio_vs_steady": round(float(
                    stress.acceleration_peak(d.rotations).mean()
                    / stress.acceleration_peak(steady.rotations).mean()), 2),
                "outside_limits_share": round(stress.limits_share(d.rotations), 4),
                "speed_m_s": round(float(stress.ground_speed(d.root).mean()), 3)}
        return out

    def pushes() -> dict[str, object]:
        out = {}
        for angle in (0.2, 0.5, 1.0):
            g = torch.Generator().manual_seed(11)
            d = drive(model, norm, seed, args.context, 6 * FPS, S(intent="idle"),
                             perturb=stress.push(30, angle, g))
            lim = m1.joint_limits().to(d.rotations.device).without_hips()
            outside = (m1.limit_excess(d.rotations[:, :, 1:], lim) > 0).float().mean(-1)  # (B,T)
            recovered = []
            for b in range(outside.shape[0]):
                after = (outside[b, 31:] < 0.02).nonzero()
                recovered.append(float(after[0]) / FPS if len(after) else float("inf"))
            finite = [x for x in recovered if x != float("inf")]
            out[f"{angle} rad"] = {
                "stable": bool(torch.isfinite(d.rotations).all()),
                "never_recovered_share": round(1 - len(finite) / len(recovered), 3),
                "seconds_to_recover_median": round(float(torch.tensor(finite).median()), 2)
                if finite else None,
                "outside_limits_share_last_2s": round(float(outside[:, -2 * FPS:].mean()), 4),
                "energy_last_2s": round(float(stress.energy(d.rotations)[:, -2 * FPS:].mean()), 3),
                "peak_accel_after_push_over_data_p99": round(
                    float(stress.acceleration_peak(d.rotations)[:, 31:].amax(-1).mean()) / p99, 2),
            }
        return out

    def cold_starts() -> dict[str, object]:
        out = {}
        for label, s in (("t-pose", stress.t_pose(args.seeds, args.context, device)),
                         ("frozen frame", stress.frozen(seed, args.context))):
            for sig_label, sig in (("idle", S(intent="idle")), ("walk", S(intent="walk"))):
                d = drive(model, norm, s, args.context, 20 * FPS, sig)
                summ = stress.summary(d.rotations, d.root, toe_floor)
                out[f"{label}, {sig_label}"] = {
                    k: summ[k] for k in ("stable", "outside_limits_share", "hips_m",
                                         "toe_in_floor_share", "airborne_share",
                                         "head_below_hips_share")}
                out[f"{label}, {sig_label}"]["energy_by_5s"] = stress.segments(
                    stress.energy(d.rotations), 5)
                out[f"{label}, {sig_label}"]["speed_by_5s"] = stress.segments(
                    stress.ground_speed(d.root), 5)
        return out

    def mirror_bias() -> dict[str, object]:
        ticks = 30 * FPS
        out = {}
        for label, sig in (("unknown", S()), ("walk", S(intent="walk"))):
            a = stress.turn_rate(drive(model, norm, seed, args.context, ticks,
                                              sig).rotations).mean(-1)
            b = stress.turn_rate(drive(model, norm, stress.mirrored(seed), args.context,
                                              ticks, sig).rotations).mean(-1)
            out[label] = {"turn_signed_mean_rad_s": round(float(a.mean()), 3),
                          "mirror_turn_signed_mean_rad_s": round(float(b.mean()), 3),
                          "asymmetry_rad_s": round(float((a + b).mean() / 2), 3),
                          "turn_abs_mean_rad_s": round(float(a.abs().mean()), 3)}
        return out

    def steering() -> dict[str, object]:
        """v5, models with a velocity input: walk, run, stop and turn on request, in a loop."""
        if "locomotion" not in model.signal_encoders:
            return {"skipped": "the model has no velocity input"}
        out: dict[str, object] = {}
        facing = stress.formats.heading(seed.rotations[:, -1, 0])
        for label, sig, speed in (("walk 1.2 m/s ahead", S(intent="walk"), 1.2),
                                  ("run 3 m/s ahead", S(intent="run"), 3.0),
                                  ("stand (0 m/s)", S(intent="idle"), 0.0)):
            d = drive(model, norm, seed, args.context, 20 * FPS,
                             controller=stress.go_towards(sig, facing, speed))
            speed_now = stress.ground_speed(d.root)
            summ = stress.summary(d.rotations, d.root, toe_floor)
            out[label] = {"speed_m_s_after_3s": round(float(speed_now[:, 3 * FPS:].mean()), 3),
                          "speed_spread_between_seeds": round(float(
                              speed_now[:, 3 * FPS:].mean(-1).std()), 3),
                          **{k: summ[k] for k in ("energy_by_30s", "glide_share_when_moving",
                                                  "carried_share_when_moving",
                                                  "airborne_share", "legs_crossed_share",
                                                  "outside_limits_share")}}
            out[label]["energy_rad_s"] = round(float(stress.energy(d.rotations).mean()), 3)
        for turn in (90, 180):
            target = facing + math.radians(turn)
            d = drive(model, norm, seed, args.context, 10 * FPS,
                             controller=stress.go_towards(S(intent="walk"), target, 1.2))
            yaw = stress.formats.heading(d.rotations[:, :, 0])
            err = torch.rad2deg(torch.atan2(torch.sin(target.unsqueeze(1) - yaw),
                                            torch.cos(target.unsqueeze(1) - yaw))).abs()
            within = (err < 20).float()
            first = [float((row.nonzero()[0] if row.any() else torch.tensor([len(row)]))[0]) / FPS
                     for row in within]
            travel = (d.root[:, -FPS:, [0, 2]].mean(1) - d.root[:, :FPS, [0, 2]].mean(1))
            travel_yaw = torch.atan2(travel[:, 0], travel[:, 1])
            path_err = torch.rad2deg(torch.atan2(torch.sin(target - travel_yaw),
                                                 torch.cos(target - travel_yaw))).abs()
            out[f"turn {turn} degrees and walk"] = {
                "seconds_to_face_within_20deg_median": round(float(torch.tensor(first).median()),
                                                             2),
                "never_faced_share": round(sum(f >= 10 for f in first) / len(first), 3),
                "facing_error_last_2s_deg": round(float(err[:, -2 * FPS:].mean()), 1),
                "overall_travel_direction_error_deg": round(float(path_err.mean()), 1),
                "glide_share_when_moving": stress.body(d.rotations, d.root, toe_floor)[
                    "glide_share_when_moving"]}
        return out

    timed("steering", steering)
    timed("long_runs", long_runs)
    timed("switches", switches)
    timed("flicker", flicker)
    timed("pushes", pushes)
    timed("cold_starts", cold_starts)
    timed("mirror_bias", mirror_bias)
    name = args.checkpoint.parent.name
    version = args.checkpoint.resolve().parents[2].name
    out = args.out or ROOT / "outputs" / f"stress-{version}-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
