"""M1 pipeline: BVH reading, LAFAN1 retargeting helpers, data sampling, model and rollout."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from motionlab import formats, m1, rotation, skeleton
from motionlab.datasets import bvh, contacts, lafan1, m1_data
from motionlab.export import export, parity

TINY_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Child
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 5 0 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
1 2 3 0 0 0 0 0 0
1 2 3 90 0 0 0 90 0
"""


def test_bvh_reads_hierarchy_channels_and_euler_order() -> None:
    source = bvh.parse(TINY_BVH)
    assert source.names == ("Hips", "Child") and source.parents == (-1, 0)
    assert torch.allclose(source.offsets[1], torch.tensor([10.0, 0, 0], dtype=torch.float64))
    rotations = source.local_rotations()
    assert torch.allclose(rotations[0], torch.eye(3, dtype=torch.float64).expand(2, 3, 3))
    expected = bvh.axis_rotation("Z", torch.tensor(math.pi / 2, dtype=torch.float64))
    assert torch.allclose(rotations[1, 0], expected, atol=1e-9)
    _, positions = bvh.forward(source, rotations, source.root_position())
    # The child sits 10 along the root's rotated +X: (0, 10, 0) after a 90° turn about Z.
    assert torch.allclose(positions[1, 1], torch.tensor([1.0, 12, 3], dtype=torch.float64),
                          atol=1e-9)


def test_rotation_between_maps_one_direction_onto_another() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        a, b = (lafan1._unit(torch.randn(3, dtype=torch.float64)) for _ in range(2))
        r = lafan1._rotation_between(a, b)
        assert torch.allclose(r @ a, b, atol=1e-8)
    a = torch.tensor([1.0, 0, 0], dtype=torch.float64)
    assert torch.allclose(lafan1._rotation_between(a, -a) @ a, -a, atol=1e-8)


def test_resampling_keeps_endpoints_and_interpolates_on_the_geodesic() -> None:
    angle = torch.tensor([0.0, 1.0], dtype=torch.float64)
    local = torch.stack([rotation.so3_exp(torch.tensor([0, a, 0], dtype=torch.float64))
                         for a in angle]).view(2, 1, 3, 3)
    root = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
    out, out_root = lafan1.resample(local, root, source_fps=30, fps=60)
    assert out.shape[0] == 3
    assert torch.allclose(out[1, 0], rotation.so3_exp(torch.tensor([0, 0.5, 0],
                                                                   dtype=torch.float64)),
                          atol=1e-9)
    assert torch.allclose(out_root[1], torch.tensor([1.5, 0, 0], dtype=torch.float64))


def test_take_names_give_action_and_subject() -> None:
    take = lafan1.take_info("fallAndGetUp2_subject3.bvh")
    assert (take.action, take.subject) == ("fallAndGetUp", "lafan1-subject3")


def _walking_clip(frames: int = 40) -> formats.Clip:
    clip = formats.constant_spin_clip(frames, (0.0, 0.8, 0.0), "hips")
    clip.root_position[:, 2] = torch.linspace(0, 1, frames)
    clip.contacts = torch.zeros(frames, 4, dtype=torch.bool)
    return clip


def test_batched_states_match_the_m0_definition() -> None:
    clip = _walking_clip()
    batched = m1_data.states(clip.rot6d.unsqueeze(0), clip.root_position.unsqueeze(0))[0]
    assert torch.allclose(batched, formats.motion_state(clip), atol=1e-5)


def test_augmentation_keeps_heading_relative_quantities() -> None:
    clip = _walking_clip()
    batch = m1_data.Batch(clip.rot6d.unsqueeze(0), clip.root_position.unsqueeze(0),
                          clip.contacts.float().unsqueeze(0))
    turned = m1_data.augment_batch(batch, torch.tensor([False]), torch.tensor([1.1]))
    before = m1_data.states(batch.rot6d, batch.root)
    after = m1_data.states(turned.rot6d, turned.root)
    rot6d_end, omega_end = skeleton.J * 6, skeleton.J * 9
    # Every joint but the root, all angular velocities, and the root's heading-frame velocity.
    assert torch.allclose(before[..., 6:rot6d_end], after[..., 6:rot6d_end], atol=1e-5)
    assert torch.allclose(before[..., rot6d_end:omega_end], after[..., rot6d_end:omega_end],
                          atol=1e-4)
    assert torch.allclose(before[..., omega_end:], after[..., omega_end:], atol=1e-4)
    twice = m1_data.augment_batch(m1_data.augment_batch(batch, torch.tensor([True]),
                                                        torch.tensor([0.0])),
                                  torch.tensor([True]), torch.tensor([0.0]))
    assert torch.allclose(twice.rot6d, batch.rot6d, atol=1e-5)


def test_still_feet_are_planted_and_moving_feet_are_not() -> None:
    still = formats.constant_spin_clip(20, (0.0, 0.0, 0.0), "head")
    assert bool(contacts.contact_flags(still).all())
    moving = _walking_clip()
    moving.root_position[:, 2] = torch.linspace(0, 10, moving.frames)  # 10 m in 0.65 s
    assert not bool(contacts.contact_flags(moving).any())


def _normaliser() -> m1_data.Normaliser:
    return m1_data.Normaliser(torch.zeros(formats.STATE_SIZE), torch.ones(formats.STATE_SIZE))


def test_untrained_model_predicts_no_change_and_rollout_gradients_flow() -> None:
    torch.manual_seed(0)
    for head in ("rotation", "omega"):
        model = m1.MotionPrior(m1.ModelConfig(hidden=32, head=head))
        clip = formats.constant_spin_clip(12, (0.0, 0.0, 0.0), "head")
        batch = m1_data.Batch(clip.rot6d.unsqueeze(0), clip.root_position.unsqueeze(0),
                              torch.zeros(1, 12, 4))
        out = m1.rollout(model, batch, _normaliser(), context=4)
        assert torch.allclose(out.rotations, batch.rotations[:, 2:], atol=1e-5)
        moving = _walking_clip(12)
        moving_batch = m1_data.Batch(moving.rot6d.unsqueeze(0),
                                     moving.root_position.unsqueeze(0), torch.zeros(1, 12, 4))
        with torch.no_grad():
            model.decoder[-1].weight.normal_(0, 0.01)
        terms = m1.losses(m1.rollout(model, moving_batch, _normaliser(), context=6),
                          moving_batch, m1.LossWeights(), skip=2)
        terms["total"].backward()
        assert all(torch.isfinite(v) for v in terms.values())
        assert model.gru.weight_ih_l0.grad is not None
        assert float(model.gru.weight_ih_l0.grad.abs().sum()) > 0


def test_motion_prior_exports_to_onnx_with_matching_outputs(tmp_path: Path) -> None:
    from motionlab.export import StepModel, operators

    torch.manual_seed(0)
    model = m1.MotionPrior(m1.ModelConfig(hidden=64, layers=2))
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.1)
        model.film.weight.normal_(0, 0.1)
    path = export(model, tmp_path / "m1.onnx")  # type: ignore[arg-type]
    assert parity(model, path, ticks=30) < 1e-4  # type: ignore[arg-type]
    # Nothing the runtime spike's model (which ran in Unity's Sentis) didn't already use.
    spike = operators(export(StepModel(hidden=64), tmp_path / "spike.onnx"))
    assert operators(path) <= spike, operators(path) - spike


def test_amass_rest_pose_turns_only_the_world() -> None:
    import numpy as np

    from motionlab.datasets import amass

    local = amass.local_rotations(np.zeros((3, 156)))
    assert torch.allclose(local[:, 0], amass.WORLD.expand(3, 3, 3))
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(local[:, 1:], eye.expand(3, skeleton.J - 1, 3, 3))
    # Z up becomes Y up: the AMASS world's up axis maps onto the canonical up axis.
    assert torch.allclose(amass.WORLD @ torch.tensor([0.0, 0, 1], dtype=torch.float64),
                          torch.tensor([0.0, 1, 0], dtype=torch.float64))


def test_splits_keep_performers_apart() -> None:
    assert m1_data.split_of("lafan1-subject5") == "test"
    assert m1_data.split_of("lafan1-subject2") == "train"
    assert m1_data.split_of("amass/HumanEva/S1") == "validation"
    assert m1_data.split_of("amass/Transitions_mocap/mazen_c3d") == "test"
    assert m1_data.split_of("amass/CMU/01") == "train"


SAMPLE = Path(__file__).resolve().parents[3] / "data/amass/sample/amass_sample.npz"


def test_amass_converter_points_limbs_the_way_the_markers_do() -> None:
    import numpy as np
    import pytest

    from motionlab import fk
    from motionlab.datasets import amass

    if not SAMPLE.exists():
        pytest.skip("AMASS's sample file is not downloaded")
    with np.load(SAMPLE) as data:
        labels = [label.decode() for label in data["marker_labels"]]
        markers = torch.from_numpy(data["marker_data"]).double() @ amass.WORLD.T
        local = amass.local_rotations(data["poses"])
        root, scale = amass.estimate_root(local, torch.from_numpy(data["trans"]).double(),
                                          float(data["mocap_framerate"]))
    assert scale == 1.0  # a standing take: nothing to fit the scale from
    _, positions = fk.forward_kinematics(local, root, skeleton.load(torch.float64))

    def m(*names: str) -> torch.Tensor:
        return torch.stack([markers[:, labels.index(n)] for n in names]).mean(0)

    def worst(a: torch.Tensor, b: torch.Tensor) -> float:
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
        return float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max())

    j = lambda name: positions[:, skeleton.INDEX[name]]  # noqa: E731
    for side, s in (("left", "L"), ("right", "R")):
        assert worst(j(f"{side}Foot") - j(f"{side}LowerLeg"),
                     m(f"{s}ANK") - m(f"{s}KNE", f"{s}KNI")) < 15
        assert worst(j(f"{side}Hand") - j(f"{side}LowerArm"),
                     m(f"{s}OWR", f"{s}IWR") - m(f"{s}ELB", f"{s}IEL")) < 15
    assert worst(j("leftFoot") - j("rightFoot"), m("LANK") - m("RANK")) < 20


def test_signal_groups_cover_every_feature_once_and_neutralising_works() -> None:
    from motionlab import motion_signal

    indices = sorted(i for group in m1.SIGNAL_GROUPS.values() for i in group)
    assert indices == list(range(motion_signal.FEATURE_SIZE))
    assert set(m1.SIGNAL_GROUPS) == {"intent", "conversation", "emotion", "attention", "audio",
                                     "context"}
    waving = motion_signal.features(motion_signal.MotionSignal(
        intent="wave", emotion=motion_signal.Emotion(0.8, 0.9))).expand(64, 5, -1)
    neutral = m1.neutral_signal()
    assert torch.equal(m1.neutralise(waving, 0.0), waving)
    assert torch.equal(m1.neutralise(waving, 1.0), neutral.expand_as(waving))
    some = m1.neutralise(waving, 0.5, torch.Generator().manual_seed(0))
    emotion = m1.SIGNAL_GROUPS["emotion"]
    dropped = (some[:, :, emotion] == neutral[emotion]).all(-1)
    assert bool((dropped == dropped[:, :1]).all())  # one choice per window, not per tick
    assert 10 < int(dropped[:, 0].sum()) < 54  # about half


def test_film_starts_neutral_and_each_signal_reaches_the_model() -> None:
    from motionlab import formats, motion_signal

    torch.manual_seed(0)
    model = m1.MotionPrior(m1.ModelConfig(hidden=32))
    assert float(model.film.weight.abs().sum()) == 0  # no modulation until trained
    state = torch.randn(1, formats.STATE_SIZE)
    neutral = m1.neutral_signal().unsqueeze(0)
    base = model.encode(state, neutral)
    for name, signal in {
        "intent": motion_signal.MotionSignal(intent="wave"),
        "conversation": motion_signal.MotionSignal(conversation="speaking"),
        "emotion": motion_signal.MotionSignal(emotion=motion_signal.Emotion(0.9, 0.9)),
    }.items():
        changed = model.encode(state, motion_signal.features(signal).unsqueeze(0))
        assert not torch.allclose(base, changed), name


def test_new_sources_split_by_subject_or_style_and_stay_put() -> None:
    from motionlab.datasets import cmu, style100

    assert m1_data.split_of("cmu/16") == cmu.split_of_subject("16")
    assert m1_data.split_of("100style/Akimbo") == style100.split_of_style("Akimbo")
    styles = [f"S{i}" for i in range(400)]
    shares = {s: sum(style100.split_of_style(x) == s for x in styles) / 400
              for s in ("train", "validation", "test")}
    assert 0.05 < shares["test"] < 0.15 and 0.05 < shares["validation"] < 0.15


def test_sampler_can_keep_data_in_memory_and_weight_sources() -> None:
    a, b = _walking_clip(80), _walking_clip(80)
    b.source = "never"
    sampler = m1_data.WindowSampler([a, b], "cpu", storage="cpu",
                                    source_weights={"never": 0.0})
    batch = sampler.sample(64, 20, torch.Generator().manual_seed(0), augment=False)
    assert batch.rot6d.dtype == torch.float32 and batch.contacts.dtype == torch.float32
    # Every window comes from the first clip (the second's source has weight 0).
    last_start = a.root_position[-21, 2]
    assert torch.allclose(batch.root[:, 0, 2], batch.root[:, 0, 2].clamp(max=last_start))


TINY_ASF = """:version 1.10
:name tiny
:units
  mass 1.0
  length 1.0
  angle deg
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0
   orientation 0 0 0
:bonedata
  begin
     id 1
     name upper
     direction 1 0 0
     length 2
     axis 0 0 90  XYZ
     dof rx ry rz
  end
  begin
     id 2
     name lower
     direction 1 0 0
     length 1
     axis 0 0 0  XYZ
     dof rz
  end
:hierarchy
  begin
    root upper
    upper lower
  end
"""


def test_asf_amc_rest_pose_and_a_bend() -> None:
    from motionlab.datasets import cmu

    asf = cmu.parse_asf(TINY_ASF)
    assert asf.order == ["root", "upper", "lower"]
    rest = cmu.rest_positions(asf)
    assert torch.allclose(rest["lower"], torch.tensor([2.0, 0, 0], dtype=torch.float64))
    frames = cmu.parse_amc("1\nroot 0 0 0 0 0 0\nupper 0 0 0\nlower 90\n"
                           "2\nroot 0 0 0 0 0 0\nupper 0 0 90\nlower 0\n")
    rotations, starts = cmu.forward(asf, frames)
    # Frame 2: the upper bone turns 90° about Z (its axis frame is turned 90° about Z, so
    # C·Rz·C⁻¹ is still Rz): its end, where the lower bone starts, moves from (2, 0, 0) to
    # (0, 2, 0).
    assert torch.allclose(starts["lower"][0], torch.tensor([2.0, 0, 0], dtype=torch.float64))
    assert torch.allclose(starts["lower"][1], torch.tensor([0.0, 2, 0], dtype=torch.float64),
                          atol=1e-9)
    assert torch.allclose(rotations["lower"][0] @ torch.tensor([1.0, 0, 0], dtype=torch.float64),
                          torch.tensor([0.0, 1, 0], dtype=torch.float64), atol=1e-9)


DATA = Path(__file__).resolve().parents[3] / "data"  # raw downloads, shared


def test_real_cmu_and_100style_takes_match_bone_directions() -> None:
    """v1 retargeting: every canonical bone points exactly where the source's bone points."""
    import zipfile

    import pytest

    from motionlab import fk
    from motionlab.datasets import cmu, retarget, style100
    from motionlab.datasets.lafan1 import BONE_CHILD

    def worst(local: torch.Tensor, root: torch.Tensor, positions: dict, cal, profile) -> float:
        _, p = fk.forward_kinematics(local, root, skeleton.load(torch.float64))
        out = 0.0
        for name, child in BONE_CHILD.items():
            if child is None:
                continue
            a = p[:, skeleton.INDEX[child]] - p[:, skeleton.INDEX[name]]
            b = (positions[profile.position_source[child]]
                 - positions[profile.position_source[name]]) @ cal.world.T
            cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
            out = max(out, float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max()))
        return out

    checked = 0
    if (DATA / "cmu/allasfamc.zip").exists():
        z = zipfile.ZipFile(DATA / "cmu/allasfamc.zip")
        asf = cmu.parse_asf(z.read("all_asfamc/subjects/16/16.asf").decode())
        rots, pos = cmu.forward(asf, cmu.parse_amc(
            z.read("all_asfamc/subjects/16/16_15.amc").decode()))
        rest = cmu.rest_positions(asf)
        cal = retarget.calibrate(rest, [], cmu.PROFILE)
        cal = retarget.Calibration(cal.world, cal.alignment,
                                   retarget.leg_scale(rest, cal.world, cmu.PROFILE))
        local, root = retarget.to_canonical(rots, pos, cal, cmu.PROFILE)
        root = retarget.place_on_floor(local, root, cmu.FPS)
        assert worst(local, root, pos, cal, cmu.PROFILE) < 0.01
        assert 0.8 < float(root[:, 1].median()) < 0.95
        checked += 1
    if (DATA / "100style/100STYLE.zip").exists():
        z = zipfile.ZipFile(DATA / "100style/100STYLE.zip")
        source = bvh.parse(z.read("100STYLE/Neutral/Neutral_FW.bvh").decode())
        cal = retarget.calibrate(style100.rest_positions(source), [source.root_position()[0]],
                                 style100.PROFILE)
        rots, pos = style100.world_and_positions(source)
        local, root = retarget.to_canonical(rots, pos, cal, style100.PROFILE)
        assert worst(local, root, pos, cal, style100.PROFILE) < 0.01
        assert 0.8 < float(root[:, 1].median()) < 0.95
        checked += 1
    if not checked:
        pytest.skip("neither CMU nor 100STYLE is downloaded")
