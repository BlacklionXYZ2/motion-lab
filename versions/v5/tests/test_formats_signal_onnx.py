"""M0 gate: data formats, the frozen MotionSignal, and the ONNX path."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from motionlab import formats, motion_signal, rotation, skeleton
from motionlab.export import StepModel, export, operators, parity

SCHEMA = Path(__file__).resolve().parents[3] / "specs/motion_signal.v2.schema.json"


def test_windows_cover_a_clip_without_running_off_the_end() -> None:
    assert formats.windows(100) == []
    spans = formats.windows(300)
    assert spans[0] == (0, 120) and spans[-1][1] <= 300 and all(e - s == 120 for s, e in spans)


def test_clip_validation_rejects_bad_clips() -> None:
    clip = formats.constant_spin_clip(10, (0.0, 1.0, 0.0), "head")
    clip.validate()
    clip.fps = 30
    with pytest.raises(ValueError, match="60 Hz"):
        clip.validate()
    clip.fps = 60
    clip.rot6d[3, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        clip.validate()


def test_state_omega_comes_from_the_realised_pose() -> None:
    omega = (0.3, -1.2, 2.0)
    clip = formats.constant_spin_clip(30, omega, "leftLowerArm")
    state = formats.motion_state(clip)
    assert state.shape == (29, formats.STATE_SIZE)
    start, _ = formats.STATE_LAYOUT["omega"]
    j = skeleton.INDEX["leftLowerArm"]
    recovered = state[:, start + 3 * j : start + 3 * j + 3]
    assert torch.allclose(recovered, torch.tensor(omega).expand(29, 3), atol=1e-3)


def test_root_velocity_is_in_her_heading_frame() -> None:
    frames = 20
    walk = formats.constant_spin_clip(frames, (0.0, 0.0, 0.0), "head")
    walk.root_position[:, 2] = torch.arange(frames) / formats.FPS  # 1 m/s along +Z
    turned = formats.constant_spin_clip(frames, (0.0, 0.0, 0.0), "head")
    yaw = rotation.so3_exp(torch.tensor([0.0, math.pi / 2, 0.0]))  # face +X instead
    r = rotation.rot6d_to_matrix(turned.rot6d)
    r[:, 0] = yaw
    turned.rot6d = rotation.matrix_to_rot6d(r)
    turned.root_position[:, 0] = torch.arange(frames) / formats.FPS  # 1 m/s along +X
    a, b = (formats.motion_state(c)[:, formats.STATE_LAYOUT["root_velocity"][0]:]
            for c in (walk, turned))
    assert torch.allclose(a, torch.tensor([0.0, 0.0, 1.0]).expand_as(a), atol=1e-4)
    assert torch.allclose(a, b, atol=1e-4)  # forward is forward, whichever way she faces


def test_mirroring_a_clip_twice_gives_it_back() -> None:
    clip = formats.constant_spin_clip(10, (0.4, 0.9, -0.2), "leftUpperArm")
    clip.root_position[:, 0] = 0.25
    clip.contacts = torch.tensor([[1, 0, 0, 1]] * 10, dtype=torch.bool)
    twice = formats.mirror_clip(formats.mirror_clip(clip))
    assert torch.allclose(twice.rot6d, clip.rot6d, atol=1e-6)
    assert torch.allclose(twice.root_position, clip.root_position)
    assert twice.contacts is not None and torch.equal(twice.contacts, clip.contacts)
    once = formats.mirror_clip(clip)
    # The spin moves to the right arm.
    assert not torch.allclose(once.rot6d[:, skeleton.INDEX["rightUpperArm"]],
                              clip.rot6d[:, skeleton.INDEX["rightUpperArm"]])


def test_motion_signal_neutral_features() -> None:
    vector = motion_signal.features(motion_signal.neutral())
    assert vector.shape == (motion_signal.FEATURE_SIZE,) == (47,)  # v5: + 3 locomotion
    named = dict(zip(motion_signal.FEATURE_NAMES, vector.tolist(), strict=True))
    # v2: unknown is all zeros for intent and emotion; idle is a real intent.
    assert named["conversation.idle"] == 1
    assert sum(named[f"intent.{i}"] for i in motion_signal.INTENTS) == 0
    assert sum(named[f"emotion.{e}"] for e in motion_signal.EMOTIONS) == 0
    assert named["emotion.valence"] == named["emotion.arousal"] == 0
    sad = dict(zip(motion_signal.FEATURE_NAMES, motion_signal.features(
        motion_signal.MotionSignal(intent="walk", emotion="sad", intensity="high")).tolist(),
        strict=True))
    assert sad["intent.walk"] == sad["emotion.sad"] == sad["emotion.intensity.high"] == 1
    assert (sad["emotion.valence"], sad["emotion.arousal"]) == pytest.approx(
        motion_signal.EMOTION_ANCHORS["sad"])


def test_label_features_match_features_and_minus_one_is_unknown() -> None:
    intent = torch.tensor([motion_signal.INTENTS.index("wave"), -1])
    emotion = torch.tensor([motion_signal.EMOTIONS.index("cheerful"), -1])
    intensity = torch.tensor([motion_signal.INTENSITIES.index("low"), -1])
    batch = motion_signal.label_features(intent, emotion, intensity)
    assert torch.allclose(batch[0], motion_signal.features(motion_signal.MotionSignal(
        intent="wave", emotion="cheerful", intensity="low")))
    assert torch.allclose(batch[1], motion_signal.features(motion_signal.neutral()))


def test_motion_signal_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="unknown intent"):
        motion_signal.MotionSignal(intent="dance")
    with pytest.raises(ValueError, match="unknown emotion"):
        motion_signal.MotionSignal(emotion="furious")
    with pytest.raises(ValueError, match="unknown intensity"):
        motion_signal.MotionSignal(intensity="extreme")
    with pytest.raises(ValueError, match="pitch"):
        motion_signal.MotionSignal(audio=motion_signal.Audio(rms=0.5, voiced=True, pitch=1.5))


def test_schema_and_python_contract_agree() -> None:
    schema = json.loads(SCHEMA.read_text("utf-8"))
    props = schema["properties"]
    assert tuple(props["intent"]["enum"]) == (*motion_signal.INTENTS, None)
    assert tuple(props["conversation"]["enum"]) == motion_signal.CONVERSATIONS
    assert tuple(props["emotion"]["enum"]) == (*motion_signal.EMOTIONS, None)
    assert tuple(props["intensity"]["enum"]) == (*motion_signal.INTENSITIES, None)
    audio = props["audio"]["oneOf"][1]["properties"]
    assert set(audio) == {"rms", "voiced", "pitch", "onset"}
    assert props["version"]["const"] == motion_signal.VERSION


@pytest.mark.parametrize("layers", [1, 2])
def test_onnx_export_matches_pytorch_without_a_gru_operator(tmp_path: Path, layers: int) -> None:
    torch.manual_seed(0)
    model = StepModel(hidden=128, layers=layers)
    path = export(model, tmp_path / "step.onnx")
    assert not (tmp_path / "step.onnx.data").exists()  # one self-contained file
    # Unity's Sentis has no GRU operator, so the cell is exported as plain arithmetic.
    assert "GRU" not in operators(path)
    assert parity(model, path) < 1e-5


def test_the_onnx_gru_operator_is_still_available(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = StepModel(hidden=64)
    path = export(model, tmp_path / "step.onnx", gru_operator=True)
    assert "GRU" in operators(path)
    assert parity(model, path) < 1e-5
