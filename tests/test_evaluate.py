"""Checks for the measurement mistakes that changed this study's conclusions."""

import numpy as np
import pytest

from evaluate import Policies, configure_torch, environment_config, ramp_sightlines, run_episode
from hide_and_seek.env_hs import HideAndSeekEnv


@pytest.fixture
def env():
    world = HideAndSeekEnv(**environment_config("room"))
    world.reset(seed=0)
    # A seeker to the right of the room sees through its wall only with elevation.
    world.bodies["seeker_0"].position = (280, 100)
    world.bodies["seeker_1"].position = (280, 130)
    world.bodies["hider_0"].position = (100, 100)
    for i, body in enumerate(world.box_bodies):
        body.position = (450, 450 + 60 * i)
        world.space.reindex_shapes_for_body(body)
    world.ramp_body.position = (280, 100)
    world._update_elevation()
    yield world
    world.close()


def test_sightline_requires_actual_visibility_gain_and_restores_state(env):
    saved = env._elevated_now
    assert ("seeker_0", "hider_0") in ramp_sightlines(env)
    assert env._elevated_now is saved
    env.bodies["hider_0"].position = (350, 100)  # same side, visible without elevation
    assert ramp_sightlines(env) == []
    assert env._elevated_now is saved


def test_dormant_seekers_do_not_count(env):
    env.active_seekers = 1
    env._elevated_now = {"seeker_1"}
    assert ramp_sightlines(env) == []


def test_sightline_checks_second_hider():
    world = HideAndSeekEnv(**environment_config("two-hiders"))
    try:
        world.reset(seed=0)
        world.bodies["seeker_0"].position = (200, 70)
        world.bodies["hider_0"].position = (250, 70)  # visible without the ramp
        world.bodies["hider_1"].position = (70, 70)  # behind the wall
        for i, body in enumerate(world.box_bodies):
            body.position = (450, 450 + 60 * i)
            world.space.reindex_shapes_for_body(body)
        world._elevated_now = {"seeker_0"}
        assert ramp_sightlines(world) == [("seeker_0", "hider_1")]
    finally:
        world.close()


def test_xray_off_does_not_report_a_sightline_gain(env):
    env.ramp_xray = False
    assert ramp_sightlines(env) == []


def test_fixed_seed_replays_independently_of_episode_order():
    configure_torch()
    policies = Policies("two-hiders")
    first = run_episode("two-hiders", policies, 4)
    run_episode("two-hiders", policies, 17, active_seekers=1)
    replay = run_episode("two-hiders", policies, 4)
    assert first == replay
    assert first["play_steps"] == 216
    assert 0 <= first["ramp_sightline_fraction"] <= first["near_ramp_fraction"] <= 1
    assert first["ramp_max_displacement_px"] >= first["ramp_net_displacement_px"]


def test_default_evaluation_has_no_spawn_assists(env):
    env.reset(seed=0, options={"ramp_active": True, "active_seekers": 2})
    assert env.box_lock_owner == [None, None]
    assert env.ramp_lock_owner is None
    assert not env.ramp_climb
    assert env.ramp_xray
    obs, _, _, _, infos = env.step({n: np.zeros(3) for n in env.possible_agents})
    assert all(info["in_prep"] for info in infos.values())
    assert all(np.isfinite(value).all() for value in obs.values())
