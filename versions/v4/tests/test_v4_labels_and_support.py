"""v4: keyword labels from the datasets, the support height, and labels through the sampler."""

from __future__ import annotations

import torch

from motionlab import motion_signal, rotation, skeleton
from motionlab.datasets import labels, m1_data
from motionlab.formats import FPS, Clip

INTENT = motion_signal.INTENTS.index


def _standing(frames: int, lift: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rest pose, hips at rest height (+ ``lift`` for the second half), both feet planted."""
    rot6d = rotation.matrix_to_rot6d(torch.eye(3)).expand(frames, skeleton.J, 6).clone()
    root = skeleton.load().offsets[0].expand(frames, 3).clone()
    root[frames // 2:, 1] += lift
    contacts = torch.ones(frames, 4, dtype=torch.bool)
    return rot6d, root, contacts


def test_support_is_the_floor_then_the_box_she_steps_onto() -> None:
    rot6d, root, contacts = _standing(4 * FPS, lift=0.4)
    support = labels.support_height(rot6d, root, contacts)
    assert abs(float(support[:FPS].mean())) < 1e-4
    assert abs(float(support[-FPS:].mean()) - 0.4) < 1e-4
    # So the hips above the support are her rest height on the floor and on the box alike.
    above = root[:, 1] - support
    rest = float(skeleton.load().offsets[0, 1])
    assert abs(float(above[:FPS].mean()) - rest) < 1e-4
    assert abs(float(above[-FPS:].mean()) - rest) < 1e-4


def test_support_holds_while_both_feet_are_in_the_air() -> None:
    rot6d, root, contacts = _standing(4 * FPS)
    root[2 * FPS:2 * FPS + 20, 1] += 0.3  # a jump
    contacts[2 * FPS:2 * FPS + 20] = False
    support = labels.support_height(rot6d, root, contacts)
    assert float(support.abs().max()) < 1e-4


def test_slow_walking_becomes_idle_and_labels_follow_each_source() -> None:
    rot6d, root, contacts = _standing(3 * FPS)
    root[FPS:, 2] = torch.arange(2 * FPS) * 1.2 / FPS  # still for 1 s, then 1.2 m/s
    data = {"rot6d": rot6d, "root_position": root, "contacts": contacts, "source": "lafan1",
            "notes": {"action": "walk"}}
    intent = labels.intent_labels(data, torch.zeros(3 * FPS))
    assert int(intent[10]) == INTENT("idle") and int(intent[-10]) == INTENT("walk")
    data.update(source="100style", labels={"gait": "transitions", "style": "Angry"})
    assert bool((labels.intent_labels(data, torch.zeros(3 * FPS)) == -1).all())
    emotion, intensity = labels.emotion_labels(data)
    assert int(emotion[0]) == motion_signal.EMOTIONS.index("frustrated")
    assert int(intensity[0]) == motion_signal.INTENSITIES.index("high")


def test_babel_priority_picks_the_specific_label_where_segments_overlap() -> None:
    rot6d, root, contacts = _standing(2 * FPS)
    data = {"rot6d": rot6d, "root_position": root, "contacts": contacts, "source": "cmu",
            "labels": {"segments": [
                {"start": 0.0, "end": 2.0, "categories": ["stand"]},
                {"start": 0.5, "end": 1.0, "categories": ["wave", "arm movements"]}]}}
    intent = labels.intent_labels(data, torch.zeros(2 * FPS))
    assert int(intent[10]) == INTENT("idle")
    assert int(intent[45]) == INTENT("wave")


def test_falls_and_getting_up_are_split_by_hips_height() -> None:
    hips = torch.full((6 * FPS,), 0.9)
    hips[2 * FPS:4 * FPS] = 0.2
    intent = labels._falls(hips)
    assert int(intent[2 * FPS - 10]) == INTENT("fall")
    assert int(intent[3 * FPS]) == -1  # lying down
    assert int(intent[4 * FPS + 10]) == INTENT("get_up")


def test_sampler_carries_labels_through_augmentation_and_boosts_rare_intents() -> None:
    def clip(intent: str) -> Clip:
        rot6d, root, contacts = _standing(200)
        tags = torch.tensor([INTENT(intent), -1, -1], dtype=torch.int8).expand(200, 3).clone()
        return Clip(rot6d, root, "cmu", "cmu/01", contacts=contacts, labels=tags)

    sampler = m1_data.WindowSampler([clip("walk"), clip("wave")], rare_intent_boost=20.0)
    assert sampler.weights.tolist() == [1.0, 21.0]
    batch = sampler.sample(64, 50, torch.Generator().manual_seed(0))
    assert batch.labels is not None and batch.labels.shape == (64, 50, 3)
    signal = batch.signal()
    assert signal is not None
    waving = signal[..., motion_signal.INDEX["intent.wave"]].mean()
    assert 0.8 < float(waving) < 1.0  # 21 of 22
