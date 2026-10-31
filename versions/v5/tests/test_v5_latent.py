"""v5: the per-tick latent (``m1.MotionPrior`` with ``latent > 0``) and the stress battery."""

from __future__ import annotations

import torch

from motionlab import formats, m1, motion_signal, skeleton, stress
from motionlab.datasets import m1_data


def _normaliser() -> m1_data.Normaliser:
    return m1_data.Normaliser(torch.zeros(formats.STATE_SIZE), torch.ones(formats.STATE_SIZE))


def _moving_batch(frames: int = 14) -> m1_data.Batch:
    clip = formats.constant_spin_clip(frames, (0.0, 0.8, 0.0), "hips")
    clip.root_position[:, 2] = torch.linspace(0, 1, frames)
    return m1_data.Batch(clip.rot6d.unsqueeze(0), clip.root_position.unsqueeze(0),
                         torch.zeros(1, frames, 4))


def _trained_deterministic() -> m1.MotionPrior:
    torch.manual_seed(0)
    model = m1.MotionPrior(m1.ModelConfig(hidden=32))
    with torch.no_grad():  # as if trained: the decoder's last layer is no longer zero
        model.decoder[-1].weight.normal_(0, 0.05)
        model.film.weight.normal_(0, 0.05)
    return model


def test_a_latent_model_started_from_a_deterministic_one_matches_it_with_zero_noise() -> None:
    old = _trained_deterministic()
    new = m1.MotionPrior(m1.ModelConfig(hidden=32, latent=8, latent_width=16))
    new.load_deterministic(old.state_dict())
    state = torch.randn(3, formats.STATE_SIZE)
    signal = motion_signal.features(motion_signal.neutral()).expand(3, -1)
    h = torch.randn(1, 3, 32) * 0.1
    with torch.no_grad():
        expected, h_expected = old(state, signal, h)
        raw, h_next, kl = new.step(state, signal, h, eps=torch.zeros(3, 8))
    assert kl is None  # no target: drawn from the prior
    assert torch.allclose(raw, expected, atol=1e-6)
    assert torch.allclose(h_next, h_expected, atol=1e-6)


def test_the_latent_starts_with_no_kl_and_trains_end_to_end() -> None:
    torch.manual_seed(1)
    model = m1.MotionPrior(m1.ModelConfig(hidden=32, latent=8, latent_width=16))
    batch = _moving_batch()
    out = m1.rollout(model, batch, _normaliser(), context=6)
    assert out.kl is not None and out.kl.shape == (1, 12, 8)
    assert float(out.kl.abs().max()) < 1e-6  # prior and posterior both start as N(0, I)
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.01)
        model.posterior[-1].weight.normal_(0, 0.01)
    weights = m1.LossWeights(kl=0.01, kl_free_bits=0.05)
    terms = m1.losses(m1.rollout(model, batch, _normaliser(), context=6), batch, weights, skip=2)
    assert {"kl", "kl_raw"} <= set(terms)
    # free bits: every dimension is charged at least 0.05 nats
    assert float(terms["kl"]) >= 8 * 0.05 - 1e-6
    terms["total"].backward()
    for net in (model.posterior, model.prior, model.latent_in):
        grads = [p.grad for p in net.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert float(model.posterior[-1].weight.grad.abs().sum()) > 0


def test_foot_terms_are_zero_on_the_recording_and_catch_a_gliding_root() -> None:
    model = m1.MotionPrior(m1.ModelConfig(hidden=32))  # untrained: predicts "no change"
    frames = 12
    clip = formats.constant_spin_clip(frames, (0.0, 0.0, 0.0), "head")  # standing still
    still = m1_data.Batch(clip.rot6d.unsqueeze(0), clip.root_position.unsqueeze(0),
                          torch.ones(1, frames, 4))
    weights = m1.LossWeights(foot_height=1.0, foot_skate=1.0)
    terms = m1.losses(m1.rollout(model, still, _normaliser(), context=frames), still, weights)
    assert float(terms["foot_height"]) < 1e-5 and float(terms["foot_skate"]) < 1e-4
    # A model whose root walks forward at 1 m/s with a still pose: the recording's planted
    # feet now glide at the root's speed.
    with torch.no_grad():
        model.decoder[-1].bias[skeleton.J * 6 + 2] = 1.0  # the root's forward velocity, m/s
    terms = m1.losses(m1.rollout(model, still, _normaliser(), context=frames), still, weights)
    assert float(terms["foot_skate"]) > 0.9


def test_prior_rollouts_differ_by_sample_and_validation_can_use_the_prior() -> None:
    torch.manual_seed(2)
    model = m1.MotionPrior(m1.ModelConfig(hidden=32, latent=8, latent_width=16))
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.05)
    batch = _moving_batch()
    with torch.no_grad():
        a = m1.rollout(model, batch, _normaliser(), context=4, use_prior=True)
        b = m1.rollout(model, batch, _normaliser(), context=4, use_prior=True)
    assert a.kl is None
    assert not torch.allclose(a.rotations, b.rotations)


def test_future_velocity_is_in_her_heading_frame() -> None:
    frames = 40
    clip = formats.constant_spin_clip(frames, (0.0, 0.0, 0.0), "head")
    clip.root_position[:, 2] = torch.arange(frames) / 60.0  # 1 m/s along world +Z, facing +Z
    batch = m1_data.Batch(clip.rot6d.unsqueeze(0), clip.root_position.unsqueeze(0),
                          torch.zeros(1, frames, 4))
    v = m1_data.future_velocity(batch, 10)
    assert v.shape == (1, 30, 2)
    assert torch.allclose(v, torch.tensor([0.0, 1.0]).expand_as(v), atol=1e-5)
    # The whole window turned 90°: facing and travel turn together, so in her frame she still
    # walks straight ahead.
    turned = m1_data.augment_batch(batch, torch.tensor([False]), torch.tensor([1.5707963]))
    v = m1_data.future_velocity(turned, 10)
    assert torch.allclose(v, torch.tensor([0.0, 1.0]).expand_as(v), atol=1e-4)
    cut, signal = m1_data.with_velocity(batch, 10)
    assert cut.rot6d.shape[1] == 30 and signal.shape[:2] == (1, 30)
    assert torch.allclose(signal[0, :, motion_signal.LOCOMOTION],
                          torch.tensor([0.0, 1.0, 1.0]).expand(30, 3), atol=1e-5)


def test_a_velocity_model_grown_from_v4_starts_as_v4_and_reads_the_velocity_once_trained() -> None:
    old = _trained_deterministic()
    widths = {**old.config.signal_widths, "locomotion": 16}
    new = m1.MotionPrior(m1.ModelConfig(hidden=32, signal_widths=widths))
    new.load_deterministic(old.state_dict())
    state = torch.randn(2, formats.STATE_SIZE)
    asked = motion_signal.features(motion_signal.MotionSignal(velocity=(0.5, 2.0))).expand(2, -1)
    h = torch.zeros(1, 2, 32)
    with torch.no_grad():
        assert torch.allclose(new(state, asked, h)[0], old(state, asked, h)[0], atol=1e-6)
        new.film.weight.normal_(0, 0.05)  # as if trained
        assert not torch.allclose(new(state, asked, h)[0], old(state, asked, h)[0])


def test_training_draws_velocity_windows_of_the_curriculum_length() -> None:
    from scripts import train_m1

    clips = []
    for k in range(3):
        clip = formats.constant_spin_clip(200, (0.0, 0.1 * k, 0.0), "hips")
        clip.root_position[:, 2] = torch.arange(200) / 60.0 * (k + 1)
        clip.subject = f"lafan1-subject{k + 1}"
        clips.append(clip)
    sampler = m1_data.WindowSampler(clips)
    generator = torch.Generator().manual_seed(0)
    batch, signal = train_m1.draw(sampler, {"batch": 4, "velocity_lookahead": 30}, 72, generator)
    assert batch.rot6d.shape[1] == 72 and signal is not None and signal.shape[:2] == (4, 72)
    assert bool((signal[..., motion_signal.LOCOMOTION[2]] == 1).all())
    speed = signal[..., motion_signal.LOCOMOTION[:2]].norm(dim=-1)
    assert float(speed.min()) > 0.5  # every clip travels at 1 to 3 m/s
    batch, signal = train_m1.draw(sampler, {"batch": 4}, 72, generator)
    assert batch.rot6d.shape[1] == 72 and signal is None  # unlabelled clips, no velocity


def test_plausibility_is_zero_standing_and_catches_hovering_and_a_hand_in_the_body() -> None:
    eye = torch.eye(3).expand(1, 5, skeleton.J, 3, 3).clone()
    root = torch.zeros(1, 5, 3)
    root[..., 1] = float(skeleton.load().offsets[0, 1])  # her rest hips height
    terms = m1.plausibility(eye, root)
    assert all(float(v) == 0 for v in terms.values())  # the T-pose, standing on the floor
    hovering = root.clone()
    hovering[..., 1] += 0.3
    assert abs(float(m1.plausibility(eye, hovering)["ground"]) - (0.3 - m1.FLIGHT_M)) < 1e-4
    sunk = root.clone()
    sunk[..., 1] -= 0.1
    assert float(m1.plausibility(eye, sunk)["ground"]) > 0.07
    # the left upper arm pointing forward and the elbow folded back: the wrist ends on the
    # middle of her chest, about 4 cm from the upper chest joint
    from motionlab import rotation

    hugging = eye.clone()
    hugging[..., skeleton.INDEX["leftUpperArm"], :, :] = rotation.so3_exp(
        torch.tensor([0.0, -1.5708, 0.0]))
    hugging[..., skeleton.INDEX["leftLowerArm"], :, :] = rotation.so3_exp(
        torch.tensor([0.0, -2.6, 0.0]))
    assert float(m1.plausibility(hugging, root)["self_contact"]) > 0.04
    crossed = eye.clone()
    lean = rotation.so3_exp(torch.tensor([0.0, 0.0, -0.6]))  # left leg swung across her front
    crossed[..., skeleton.INDEX["leftUpperLeg"], :, :] = lean
    assert float(m1.plausibility(crossed, root)["leg_cross"]) > 0


def test_free_run_losses_train_the_prior_without_a_recording_to_match() -> None:
    torch.manual_seed(4)
    model = m1.MotionPrior(m1.ModelConfig(hidden=32, latent=8, latent_width=16))
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.05)
    batch = _moving_batch(20)
    weights = m1.LossWeights(ground=1.0, self_contact=1.0, leg_cross=1.0, energy_floor=1.0)
    terms = m1.free_run_losses(model, batch, _normaliser(), weights, context=6)
    assert {"ground", "self_contact", "leg_cross", "energy_floor", "energy_ceiling",
            "free_total"} <= set(terms)
    # the band: a still prediction of moving data costs the floor, never the ceiling
    assert float(terms["energy_ceiling"]) == 0 or float(terms["energy_floor"]) == 0
    assert {"posture", "jerk_ceiling"} <= set(terms)
    terms["free_total"].backward()
    assert float(model.prior[-1].weight.grad.abs().sum()) > 0  # the prior learns from it
    assert model.posterior[-1].weight.grad is None  # the posterior isn't used


def test_tracking_compares_achieved_travel_with_the_command_in_her_frame() -> None:
    frames = 100
    clip = formats.constant_spin_clip(frames, (0.0, 0.0, 0.0), "head")
    clip.root_position[:, 2] = torch.arange(frames) / 60.0  # 1 m/s along +Z, facing +Z
    from motionlab import rotation

    r = rotation.rot6d_to_matrix(clip.rot6d).unsqueeze(0)
    root = clip.root_position.unsqueeze(0)
    exact = motion_signal.features(motion_signal.MotionSignal(velocity=(0.0, 1.0)))
    faster = motion_signal.features(motion_signal.MotionSignal(velocity=(0.0, 2.0)))
    unknown = motion_signal.features(motion_signal.MotionSignal())
    expand = lambda f: f.expand(1, frames + 4, -1)  # noqa: E731
    velocity = m1.step_velocity(root[:, 1:], root[:, :-1])
    r = r[:, 1:]
    assert float(m1._tracking(r, velocity, expand(exact), 0)) < 1e-4
    assert abs(float(m1._tracking(r, velocity, expand(faster), 0)) - 1.0) < 1e-4
    assert float(m1._tracking(r, velocity, expand(unknown), 0)) == 0  # no command, no cost
    # turned 90°, travelling along +X and facing +X: still "straight ahead" in her frame
    turned = m1_data.augment_batch(m1_data.Batch(clip.rot6d.unsqueeze(0), root,
                                                 torch.zeros(1, frames, 4)),
                                   torch.tensor([False]), torch.tensor([1.5707963]))
    turned_velocity = m1.step_velocity(turned.root[:, 1:], turned.root[:, :-1])
    assert float(m1._tracking(turned.rotations[:, 1:], turned_velocity, expand(exact), 0)) < 1e-3


def test_tracking_spreads_its_gradient_over_every_tick() -> None:
    root = torch.zeros(1, 61, 3, requires_grad=True)
    previous = torch.zeros(1, 61, 3)
    eye = torch.eye(3).expand(1, 61, skeleton.J, 3, 3)
    command = motion_signal.features(motion_signal.MotionSignal(velocity=(0.0, 1.0)))
    loss = m1._tracking(eye, m1.step_velocity(root, previous), command.expand(1, 70, -1), 0)
    loss.backward()
    moved = (root.grad[0, :60, 2] != 0).sum()
    assert int(moved) == 60  # every tick of the half second, not one tick at its end


def test_a_world_command_is_given_in_her_frame_with_a_matching_intent() -> None:
    from motionlab import rotation

    features = motion_signal.features(motion_signal.MotionSignal(intent="sit")).expand(3, -1)
    facing_x = rotation.so3_exp(torch.tensor([0.0, 1.5707963, 0.0])).expand(3, 3, 3)  # faces +X
    command = torch.tensor([[2.5, 0.0], [0.0, 1.0], [0.0, 0.0]])  # +X fast, +Z, still
    out = m1.with_command(features, command, facing_x)
    loco = motion_signal.LOCOMOTION
    assert torch.allclose(out[0, loco], torch.tensor([0.0, 2.5, 1.0]), atol=1e-5)  # ahead
    assert torch.allclose(out[1, loco[:2]].abs(), torch.tensor([1.0, 0.0]), atol=1e-5)  # sideways
    idx = motion_signal.INDEX
    assert out[0, idx["intent.run"]] == 1 and out[1, idx["intent.walk"]] == 1
    assert out[2, idx["intent.idle"]] == 1 and out[:, idx["intent.sit"]].eq(0).all()


def test_free_skate_is_zero_standing_and_catches_a_gliding_root() -> None:
    eye = torch.eye(3).expand(1, 10, skeleton.J, 3, 3).clone()
    root = torch.zeros(1, 10, 3)
    root[..., 1] = float(skeleton.load().offsets[0, 1])
    assert float(m1.free_skate(eye, root)) == 0
    root[..., 2] = torch.arange(10) / 60.0  # gliding forward at 1 m/s, feet on the floor
    assert abs(float(m1.free_skate(eye, root)) - 1.0) < 1e-3


def test_counterfactual_commands_can_point_mostly_ahead_of_her() -> None:
    g = torch.Generator().manual_seed(0)
    heading = torch.full((4000,), 1.0)  # facing about 57° from +Z
    cmd = m1.random_commands(4000, g, heading=heading, ahead_share=0.75)
    moving = cmd.norm(dim=-1) > 0
    angle = torch.atan2(cmd[moving, 0], cmd[moving, 1])
    off = torch.atan2(torch.sin(angle - 1.0), torch.cos(angle - 1.0)).abs()
    near = float((off < 1.05).float().mean())  # within 60°
    assert 0.75 < near < 0.92  # about 0.75 × 95% + 0.25 × a third
    uniform = m1.random_commands(4000, g)
    angle = torch.atan2(uniform[:, 0], uniform[:, 1])
    assert float(((angle - 1.0).cos() > 0.5).float().mean()) < 0.45


def test_stance_lets_her_stand_and_catches_a_glide() -> None:
    eye = torch.eye(3).expand(1, 10, skeleton.J, 3, 3).clone()
    root = torch.zeros(1, 10, 3)
    root[..., 1] = float(skeleton.load().offsets[0, 1])
    assert float(m1.stance(eye, root)) == 0
    root[..., 2] = torch.arange(10) / 60.0 * 1.3  # every foot carried along at 1.3 m/s
    assert abs(float(m1.stance(eye, root)) - (1.3 - 0.1 - 0.15 * 1.3)) < 1e-3


def test_facing_wants_her_heading_along_a_fast_command() -> None:
    from motionlab import rotation

    ahead = torch.eye(3).expand(2, 5, skeleton.J, 3, 3)  # facing +Z
    command = torch.tensor([[0.0, 1.0], [0.0, 0.2]])  # +Z at 1 m/s; too slow to count
    assert float(m1.facing(ahead, command, 0)) < 1e-6
    side = torch.tensor([[1.0, 0.0], [0.0, 0.2]])  # +X: a quarter turn away
    assert abs(float(m1.facing(ahead, side, 0)) - 1.0) < 1e-5
    behind = rotation.so3_exp(torch.tensor([0.0, 3.14159265, 0.0])).expand(2, 5, skeleton.J, 3, 3)
    assert abs(float(m1.facing(behind, command, 0)) - 2.0) < 1e-4


def test_the_foot_lock_moves_her_only_by_stepping() -> None:
    from motionlab import rotation

    eye = torch.eye(3).expand(1, skeleton.J, 3, 3).clone()
    root = torch.zeros(1, 3)
    root[:, 1] = float(skeleton.load().offsets[0, 1])
    gliding = root + torch.tensor([[0.0, 0.0, 0.05]])  # the model asks to slide 5 cm
    # Feet on the floor and the pose unchanged: the lock holds her where she is.
    assert torch.allclose(m1.foot_locked(eye, eye, root, gliding)[0], root, atol=1e-6)
    # Swing the right leg about the hip: it leaves the floor, so the left foot bears her and
    # must stay put on the ground plane.
    step = eye.clone()
    step[:, skeleton.INDEX["rightUpperLeg"]] = rotation.so3_exp(torch.tensor([0.3, 0.0, 0.0]))
    feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    before = m1.relative_positions(eye)[:, feet]
    after = m1.relative_positions(step)[:, feet]
    moved = m1.foot_locked(eye, step, root, gliding)[0]
    world_after = after + moved.unsqueeze(1)
    world_before = before + root.unsqueeze(1)
    lowest = int((world_after[0, :, 1]).argmin())
    # whichever foot bears the weight stays put on the ground plane (within the soft minimum)
    support = (world_after - world_before)[0, :, [0, 2]].norm(dim=-1).min()
    assert float(support) < 0.01
    # In the air (every foot 20 cm up), the model's own velocity applies.
    flying = root + torch.tensor([[0.0, 0.2, 0.0]])
    out, air = m1.foot_locked(eye, eye, flying, gliding + torch.tensor([[0.0, 0.2, 0.0]]))
    assert torch.allclose(out[:, [0, 2]], gliding[:, [0, 2]], atol=1e-6)
    assert abs(float(air[0]) - 1 / 60) < 1e-6  # one tick airborne
    # ...but not for long: after 0.3 s up there the lock takes the lowest foot again
    long_up = torch.tensor([m1.AIR_LIMIT_S + 0.01])
    out, air = m1.foot_locked(eye, eye, flying, gliding + torch.tensor([[0.0, 0.2, 0.0]]),
                              long_up)
    assert torch.allclose(out[:, [0, 2]], flying[:, [0, 2]], atol=1e-6)  # held, no glide
    # touching down resets the timer
    assert float(m1.foot_locked(eye, eye, root, gliding, long_up)[1][0]) == 0
    del lowest


def test_the_ground_lock_puts_the_lowest_foot_on_the_floor() -> None:
    eye = torch.eye(3).expand(1, skeleton.J, 3, 3).clone()
    root = torch.zeros(1, 3)
    root[:, 1] = float(skeleton.load().offsets[0, 1])
    # The model keeps its hips 2 cm too high (every foot 2 cm up, fully locked):
    raised = root + torch.tensor([[0.0, 0.02, 0.0]])
    free, _ = m1.foot_locked(eye, eye, raised, raised)
    grounded, _ = m1.foot_locked(eye, eye, raised, raised, ground_height=True)
    assert abs(float(free[0, 1] - raised[0, 1])) < 1e-6  # without it, she hovers
    assert abs(float(grounded[0, 1] - root[0, 1])) < 1e-4  # with it, her feet are down
    # 5 cm up is in the lock's fade band: each tick lowers her part of the way, and a few
    # ticks of the model asking to stay up bring her down
    current = root + torch.tensor([[0.0, 0.05, 0.0]])
    for _ in range(10):
        current, _ = m1.foot_locked(eye, eye, current, current + torch.tensor([[0.0, 0.0, 0.0]]),
                                    ground_height=True)
    assert abs(float(current[0, 1] - root[0, 1])) < 1e-3


def test_counterfactual_free_runs_train_following_a_command() -> None:
    torch.manual_seed(6)
    widths = {**m1.ModelConfig().signal_widths, "locomotion": 8}
    model = m1.MotionPrior(m1.ModelConfig(hidden=32, latent=8, latent_width=16,
                                          signal_widths=widths))
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.05)
    batch = _moving_batch(80)
    weights = m1.LossWeights(track=1.0, ground=1.0, free_skate=1.0)
    terms = m1.counterfactual_losses(model, batch, _normaliser(), weights, context=6,
                                     generator=torch.Generator().manual_seed(0))
    assert {"cf_track", "cf_ground", "cf_skate", "cf_total"} <= set(terms)
    terms["cf_total"].backward()
    assert float(model.signal_encoders["locomotion"][0][0].weight.grad.abs().sum()) > 0


def test_a_feedback_window_carries_gradient_back_through_earlier_own_ticks() -> None:
    torch.manual_seed(7)
    model = m1.MotionPrior(m1.ModelConfig(hidden=16))
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.05)
    batch = _moving_batch(20)

    def first_step_grad(window: int) -> float:
        model.zero_grad()
        out = m1.rollout(model, batch, _normaliser(), context=4, feedback_window=window)
        out.rotations[:, -1].sum().backward()  # only the last prediction's loss
        return float(model.decoder[-1].weight.grad.abs().sum())

    # The last prediction's gradient reaches the decoder through its own tick either way, and
    # through earlier ticks only with a window: more gradient flows.
    assert first_step_grad(8) > first_step_grad(1) > 0
    # Damping keeps the values but shrinks what flows back through the fed-back ticks.
    x = torch.ones(3, requires_grad=True)
    damped = m1._damped(x * 2, 0.3)
    assert torch.allclose(damped, torch.full((3,), 2.0))
    damped.sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), 0.6))


def test_correlated_noise_is_standard_normal_per_tick_but_changes_slowly() -> None:
    torch.manual_seed(5)
    model = m1.MotionPrior(m1.ModelConfig(hidden=8, latent=4, latent_width=8))
    model.noise_correlation = 0.98
    like = torch.zeros(2000, 4)
    draws = [model._correlated_noise(like) for _ in range(200)]
    last = draws[-1]
    assert abs(float(last.std()) - 1.0) < 0.05  # still N(0, 1) at every tick
    step = float((draws[-1] - draws[-2]).std())
    assert step < 0.3  # √(2(1 − ρ)) = 0.2 for ρ = 0.98, against √2 for independent draws
    model.reset_noise()
    assert model._eps is None


def test_stress_drive_holds_the_schedule_and_applies_a_push() -> None:
    torch.manual_seed(3)
    model = _trained_deterministic().eval()
    batch = _moving_batch(8)
    schedule = stress.signal_schedule([motion_signal.MotionSignal(intent="walk"),
                                       motion_signal.MotionSignal(intent="idle")], 10, 5, 1)
    assert schedule.shape == (1, 10, motion_signal.FEATURE_SIZE)
    walk = motion_signal.INDEX["intent.walk"]
    assert schedule[0, :5, walk].eq(1).all() and schedule[0, 5:, walk].eq(0).all()
    calm = stress.drive(model, _normaliser(), batch, 6, 10, schedule)
    pushed = stress.drive(model, _normaliser(), batch, 6, 10, schedule,
                          perturb=stress.push(3, 0.5, torch.Generator().manual_seed(0)))
    assert calm.rotations.shape[1] == 10 and calm.contacts.shape[1] == 10
    assert torch.allclose(calm.rotations[:, :3], pushed.rotations[:, :3])
    assert not torch.allclose(calm.rotations[:, 3], pushed.rotations[:, 3])
    t = stress.t_pose(2, 6)
    assert torch.allclose(t.rotations, torch.eye(3).expand_as(t.rotations))
