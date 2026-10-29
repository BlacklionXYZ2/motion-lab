"""Write ``catalog/catalog.json``: every motion engine, version and size motion-lab has, for SORA.

    .venv\\Scripts\\python tools\\motion_catalog.py

SORA's Motion window reads it (``sora/embodiment/motion_catalog.py``, contract
``motion-catalog``). Everything comes from files on disk, so rerun this after training or
evaluating something:

- **ARDY:** ready when the runtime environment, the ONNX networks and NVIDIA's checkpoint are in
  place. Its numbers come from ``ardy-runtime``'s real-time stream summaries (4 diffusion steps,
  4 frames of history, a replan every 8 frames). The preview is the fp16 stream scenario, scaled
  to her height.
- **Custom (our versions):** each ``versions/vN`` with ``runs/combined/best.pt``, sized by its
  parameter count, with its 60 s evaluation (``outputs/eval-combined.json``) and the preview
  from ``tools/version_preview.py``. All are experimental: none has passed M1's gate. Sizes the
  roadmap plans but nothing has trained are listed as planned under the newest version.
- **Recordings:** the reference numbers from the newest evaluation.

Uses only the standard library and NumPy (the ARDY preview is converted from its ``.npz``).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "catalog"
ARDY_RUNTIME = LAB / "ardy-runtime"
ARDY_ONNX = "external/ardy/ARDY-Core-RP-20FPS-Horizon40-ONNX"
ARDY_FILES = [
    "ardy-runtime/.venv/Scripts/python.exe",
    f"{ARDY_ONNX}/directml-mixed-fp16/denoiser.onnx",
    f"{ARDY_ONNX}/directml-mixed-fp16/decoder.onnx",
    "external/ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40/config.yaml",
    "external/ardy/nv-tlabs-ardy/ardy/model/ardy_model.py",
]
PLANNED_SIZES = ("0.3M", "1.5M", "3M", "17.5M")  # the size sweep (owner, 2026-10-25)
ARDY_SCALE = 0.894626 / 0.944  # her standing hips over ARDY's (ardy-runtime/retarget.py)
PREVIEW_FPS = 10
VERSION_NOTES = {
    "v1": "LAFAN1, 100STYLE and CMU with bone-direction retargeting. Smooth but drifts between "
          "the datasets' neutral poses and winds down.",
    "v2": "T-pose-relative retargeting fixed the posture disagreement; drifts into extreme limb "
          "poses over long runs and is about 10× calmer than real motion.",
    "v3": "Clean data, joint-limit losses and longer rollouts fixed the limb drift; even calmer "
          "than v2, and the hips float in some runs.",
    "v4": "Keywords and hips above the support: the most accurate so far, but keywords steer "
          "only weakly (walking at about a third of real speed) and motion winds down.",
}


def size_label(parameters: int) -> str:
    return f"{parameters / 1e6:.1f}M" if parameters < 1e9 else f"{parameters / 1e9:.1f}B"


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


def eval_measures(report: dict) -> dict:
    model = report["model"]
    outside = model.get("outside_data_limits_share", model.get("beyond_limits_share"))
    measures = {
        "outside_limits_share": outside,
        "foot_slide_m_s": model.get("planted_foot_slide_m_s"),
        "jitter_m_s3": model.get("jitter_m_s3"),
        "energy_rad_s": mean(model.get("energy_by_10s_rad_s", [])),
    }
    walk = (report.get("conditions") or {}).get("walk")
    if walk:
        measures["walk_speed_m_s"] = walk.get("root_speed_m_s")
    measures["note"] = ("60 s free run on the test set" + (
        ", limits measured from the data" if "outside_data_limits_share" in model else
        ", limits are the M0 placeholders") + ".")
    return measures


def versions() -> tuple[list[dict], dict]:
    models, recordings = [], {}
    found = sorted((p for p in (LAB / "versions").glob("v*") if re.fullmatch(r"v\d+", p.name)),
                   key=lambda p: int(p.name[1:]))
    newest = None
    for root in found:
        best = root / "runs" / "combined" / "best.pt"
        report_path = root / "outputs" / "eval-combined.json"
        if not best.exists() or not report_path.exists():
            continue
        report = json.loads(report_path.read_text("utf-8"))
        size = size_label(report["parameters"])
        preview = OUT / "previews" / f"{root.name}-{size}.json"
        models.append({
            "version": root.name, "size": size, "parameters": report["parameters"],
            "state": "experimental", "summary": VERSION_NOTES.get(root.name, ""),
            "measures": eval_measures(report),
            "preview": f"previews/{preview.name}" if preview.exists() else None,
            "files": [best.relative_to(LAB).as_posix()],
        })
        rec = report["recordings"]
        recordings = {
            "outside_limits_share": rec.get("outside_data_limits_share",
                                            rec.get("beyond_limits_share")),
            "foot_slide_m_s": rec.get("planted_foot_slide_m_s"),
            "jitter_m_s3": rec.get("jitter_m_s3"),
            "energy_rad_s": mean(rec.get("energy_by_10s_rad_s", [])),
            "walk_speed_m_s": (report.get("recorded_speed_by_intent_m_s") or {}).get("walk"),
            "note": f"The motion-capture test set, measured by {root.name}'s evaluation.",
        }
        newest = root.name
    if newest:
        have = {m["size"] for m in models if m["version"] == newest}
        models += [{"version": newest, "size": size, "state": "planned",
                    "summary": "Planned in the size sweep; not trained yet."}
                   for size in PLANNED_SIZES if size not in have]
    return models, recordings


def ardy_preview(npz: Path) -> str | None:
    if not npz.exists():
        return None
    data = np.load(npz)
    step = int(data["fps"]) // PREVIEW_FPS
    joints = data["posed_joints"][0, ::step] * ARDY_SCALE
    out = OUT / "previews" / "ardy-fp16.json"
    out.write_text(json.dumps({
        "label": "ARDY fp16, 4 steps: walk 20 s, stand 10 s, wave 10 s, run 20 s while turning",
        "fps": PREVIEW_FPS,
        "joint_names": [str(n) for n in data["joint_names"]],
        "joint_parents": [int(p) for p in data["joint_parents"]],
        "frames": [[round(float(c) * 1000) for c in frame.flatten()] for frame in joints],
    }), encoding="utf-8")
    return f"previews/{out.name}"


def ardy_models() -> list[dict]:
    models = []
    for precision in ("fp16", "fp32"):
        tag = f"stream-{precision}-s4-h4-k8-cfg2-seed0"
        summary_path = ARDY_RUNTIME / "outputs" / f"{tag}.json"
        summary = json.loads(summary_path.read_text("utf-8")) if summary_path.exists() else {}
        files = [f.replace("directml-mixed-fp16", "fp32" if precision == "fp32"
                           else "directml-mixed-fp16") for f in ARDY_FILES]
        ready = all((LAB / f).exists() for f in files)
        measures = {
            "outside_limits_share": summary.get("outside_limits_share"),
            "foot_slide_m_s": summary.get("foot_slide_m_s"),
            "jitter_m_s3": summary.get("jitter_m_s3"),
            "energy_rad_s": summary.get("energy_rad_s"),
            "walk_speed_m_s": summary.get("walk_speed_m_s"),
            "step_latency_ms": (summary.get("step_latency_ms") or {}).get("median"),
            "video_memory_gb": 0.7 if precision == "fp16" else None,
            "note": ("60 s streamed scenario on the RX 9070 XT (walk at a 1.2 m/s target, stand, "
                     "wave, run at 2.5 m/s); " + (f"{summary['steps_late']} of {summary['steps']} "
                     "replans missed the 400 ms real-time budget." if summary else
                     "not measured yet.")),
        }
        models.append({
            "version": f"Core {precision}", "size": "192M", "parameters": 192_000_000,
            "state": "ready" if ready else "missing",
            "summary": ("NVIDIA's 192M-parameter real-time model (27-joint Core skeleton, 20 fps), "
                        "steered by our keyword phrases and a target velocity. fp16: less video "
                        "memory, a little jitter at rest." if precision == "fp16" else
                        "The same model at full precision: cleaner at rest, more memory, and "
                        "occasionally slower than real time."),
            "measures": measures,
            "preview": ardy_preview(ARDY_RUNTIME / "outputs" / f"{tag}.npz")
            if precision == "fp16" else None,
            "files": files,
        })
    return models


def main() -> None:
    (OUT / "previews").mkdir(parents=True, exist_ok=True)
    custom, recordings = versions()
    catalog = {
        "version": 1,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "",
        "recordings": recordings,
        "engines": [
            {"id": "ardy", "name": "ARDY (NVIDIA)",
             "summary": "NVIDIA's ARDY, a real-time motion diffusion model (SIGGRAPH 2026), run "
                        "through ONNX Runtime with DirectML on the GPU.",
             "licence": "NVIDIA Open Model License (weights); Apache-2.0 (code)",
             "models": ardy_models()},
            {"id": "custom", "name": "Custom trained",
             "summary": "SORA's own motion models, trained in motion-lab (versions/vN).",
             "licence": "Own work; trained on LAFAN1 (CC BY-NC-ND), 100STYLE (CC BY) and CMU",
             "models": custom},
        ],
    }
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    counts = {e["id"]: len(e["models"]) for e in catalog["engines"]}
    print(f"catalog/catalog.json: {counts}")


if __name__ == "__main__":
    main()
