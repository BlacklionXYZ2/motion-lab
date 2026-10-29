"""Train the M1 motion prior (roadmap M1).

    .venv\\Scripts\\python scripts\\train_m1.py configs\\m1\\baseline.json [--steps N] [--name NAME]
    .venv\\Scripts\\python scripts\\train_m1.py --resume runs\\<name>
    .venv\\Scripts\\python scripts\\train_m1.py CONFIG --init-from runs\\<name>\\best.pt

Reads processed clips (``scripts/build_lafan1.py``), trains on the training subjects, validates on
the validation subjects at the longest rollout horizon, and writes ``runs/<name>/``:

- ``config.json`` and ``log.jsonl`` (one line per logged step);
- ``best.pt`` and ``last.pt``: the model, normaliser and config, for evaluation;
- ``state.pt``: everything needed to carry on exactly (model, optimiser, learning-rate schedule,
  random-number states, step), written every ``checkpoint_every`` steps (default 250) and when
  the run pauses.

**Pausing.** Press Ctrl+C, or create a file named ``STOP`` in the run's folder (for a run with no
window). The run saves ``state.pt`` and exits; the ``STOP`` file is renamed ``STOP.handled``.
**Resuming:** ``--resume runs/<name>`` continues from the saved step with the run's own config.
**Starting from a trained model:** ``--init-from <checkpoint>`` begins a new run with that model's
weights and normaliser (its model settings must match the config's), and a fresh optimiser.

The curriculum lists ``[fraction of training, rollout horizon]``: horizon 1 is teacher forcing
(every input recorded); a horizon of ``k`` feeds the model its own output for the last ``k``
ticks of each window. Recorded inputs get Gaussian noise (``input_noise``, in normalised units)
so the model learns to recover from its own small errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import m1  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402


def horizon_at(curriculum: list[list[float]], fraction: float) -> int:
    horizon = int(curriculum[0][1])
    for start, value in curriculum:
        if fraction >= start:
            horizon = int(value)
    return horizon


def evaluate(model: m1.MotionPrior, sampler: m1_data.WindowSampler,
             normaliser: m1_data.Normaliser, config: dict, horizon: int) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(1234)
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for _ in range(max(1, config["validation_windows"] // config["batch"])):
            batch = sampler.sample(config["batch"], config["window"], generator)
            out = m1.rollout(model, batch, normaliser, context=config["window"] - horizon)
            for key, value in m1.losses(out, batch, m1.LossWeights(),
                                        skip=config["warmup_ticks"]).items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
    model.train()
    return {key: value / count for key, value in totals.items()}


def random_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {"sampler": generator.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(generator: torch.Generator, state: dict[str, Any]) -> None:
    generator.set_state(state["sampler"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomically(data: dict[str, Any], path: Path) -> None:
    """Write next to the target, then replace it, so a crash never leaves half a file."""
    partial = path.with_suffix(".partial")
    torch.save(data, partial)
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path, metavar="RUN_FOLDER")
    parser.add_argument("--init-from", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    saved: dict[str, Any] | None = None
    if args.resume:
        if args.config or args.steps or args.name or args.init_from:
            parser.error("--resume uses the run's own config; give nothing else")
        run = args.resume if args.resume.is_absolute() else ROOT / args.resume
        if not (run / "state.pt").exists():
            parser.error(f"{run} has no state.pt to resume from")
        # On the CPU: random-number states must stay there (model and optimiser state move to
        # the device as they are loaded).
        saved = torch.load(run / "state.pt", map_location="cpu", weights_only=False)
        config = saved["config"]
    else:
        if not args.config:
            parser.error("give a config, or --resume a run")
        config = json.loads(args.config.read_text("utf-8"))
        if args.steps:
            config["steps"] = args.steps
        if args.name:
            config["name"] = args.name
        run = ROOT / "runs" / config["name"]
        if (run / "state.pt").exists():
            parser.error(f"{run} already has a run; --resume it or choose another --name")
        if args.init_from:
            config["init_from"] = str(args.init_from)
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    torch.manual_seed(config["seed"])

    sources = config["data"] if isinstance(config["data"], list) else [config["data"]]
    data = [ROOT / folder for folder in sources]
    storage, weights = config.get("storage"), config.get("source_weights")
    train = m1_data.WindowSampler(m1_data.load_clips(data, "train"), device, storage=storage,
                                  source_weights=weights)
    validation = m1_data.WindowSampler(m1_data.load_clips(data, "validation"), device,
                                       storage=storage, source_weights=weights)
    model = m1.MotionPrior(m1.ModelConfig(**config["model"])).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config["steps"])
    generator = torch.Generator().manual_seed(config["seed"])
    first, best, elapsed = 1, float("inf"), 0.0
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        schedule.load_state_dict(saved["schedule"])
        restore_random_state(generator, saved["random"])
        normaliser = m1_data.Normaliser(saved["normaliser"]["mean"], saved["normaliser"]["std"])
        first, best, elapsed = saved["step"] + 1, saved["best"], saved["seconds"]
    elif "init_from" in config:
        source = Path(config["init_from"])
        source = source if source.is_absolute() else ROOT / source
        start = torch.load(source, map_location=device, weights_only=False)
        if start["model_config"] != asdict(model.config):
            raise SystemExit("--init-from: the checkpoint's model settings differ from the "
                             "config's")
        model.load_state_dict(start["model"])
        normaliser = m1_data.Normaliser(start["normaliser"]["mean"], start["normaliser"]["std"])
    else:
        normaliser = m1_data.fit_normaliser(train, seed=config["seed"])
    normaliser = normaliser.to(device)
    weights = m1.LossWeights()
    final_horizon = int(config["curriculum"][-1][1])
    every = config.get("checkpoint_every", 250)
    stop_file = run / "STOP"

    def model_checkpoint(step: int) -> dict[str, Any]:
        return {"model": model.state_dict(), "model_config": asdict(model.config),
                "normaliser": normaliser.state_dict(), "config": config, "step": step}

    def save_state(step: int) -> None:
        save_atomically({**model_checkpoint(step), "optimiser": optimiser.state_dict(),
                         "schedule": schedule.state_dict(), "random": random_state(generator),
                         "best": best, "seconds": elapsed + time.time() - started},
                        run / "state.pt")

    print(f"{config['name']}: {model.parameter_count():,} parameters; train {train.frames:,} "
          f"frames, validation {validation.frames:,}; device {device}"
          + (f"; resuming at step {first}" if saved else ""))
    started = time.time()
    step = first - 1
    try:
        with (run / "log.jsonl").open("a", encoding="utf-8") as log:
            for step in range(first, config["steps"] + 1):
                horizon = horizon_at(config["curriculum"], step / config["steps"])
                batch = train.sample(config["batch"], config["window"], generator)
                out = m1.rollout(model, batch, normaliser, context=config["window"] - horizon,
                                 noise=config["input_noise"],
                                 feedback_gradient=config.get("feedback_gradient", False),
                                 signal_dropout=config.get("signal_dropout", 0.0))
                terms = m1.losses(out, batch, weights, skip=config["warmup_ticks"])
                optimiser.zero_grad(set_to_none=True)
                terms["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      config["gradient_clip"])
                optimiser.step()
                schedule.step()
                if not torch.isfinite(terms["total"]):
                    raise RuntimeError(f"loss became {float(terms['total'])} at step {step}")
                record: dict[str, object] = {
                    "step": step, "horizon": horizon,
                    "seconds": round(elapsed + time.time() - started, 1),
                    "grad_norm": round(float(norm), 4),
                    **{f"train_{k}": round(float(v.detach()), 5) for k, v in terms.items()}}
                if step % config["validate_every"] == 0 or step == config["steps"]:
                    scores = evaluate(model, validation, normaliser, config, final_horizon)
                    record.update({f"validation_{k}": round(v, 5) for k, v in scores.items()})
                    save_atomically(model_checkpoint(step), run / "last.pt")
                    if scores["total"] < best:
                        best = scores["total"]
                        save_atomically(model_checkpoint(step), run / "best.pt")
                    print(f"step {step}: horizon {horizon}, train {float(terms['total']):.4f}, "
                          f"validation {scores['total']:.4f} (rotation "
                          f"{scores['rotation']:.4f} rad, position {scores['position']:.4f} m), "
                          f"{elapsed + time.time() - started:.0f} s")
                if step % 50 == 0 or "validation_total" in record:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if step % every == 0 or step == config["steps"]:
                    save_state(step)
                if stop_file.exists():
                    save_state(step)
                    os.replace(stop_file, run / "STOP.handled")
                    print(f"Paused at step {step} (STOP file). Resume: --resume {run}")
                    return
    except KeyboardInterrupt:
        if step >= first:
            save_state(step)
            print(f"\nPaused at step {step} (Ctrl+C). Resume: --resume {run}")
        return
    print(f"Finished {config['steps']} steps. Best: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
