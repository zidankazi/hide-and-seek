"""Evaluate saved policies without training spawn assists. See --help for options."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hide_and_seek.env_hs import HideAndSeekEnv
from hide_and_seek.ppo_recurrent import ActorCriticLSTM

ROOT = Path(__file__).resolve().parent
PRESETS = {
    "room": {
        "prefix": "hs_lstm",
        "layout": "room",
        "n_hiders": 1,
        "n_seekers": 2,
        "seeker_speed_mult": 1.0,
    },
    "tight-room": {
        "prefix": "hs_roomt",
        "layout": "roomt",
        "n_hiders": 1,
        "n_seekers": 2,
        "seeker_speed_mult": 1.0,
    },
    "two-hiders": {
        "prefix": "hs_2v2",
        "layout": "roomt",
        "n_hiders": 2,
        "n_seekers": 2,
        "seeker_speed_mult": 1.4,
    },
}


def environment_config(preset):
    return {
        **{k: v for k, v in PRESETS[preset].items() if k != "prefix"},
        "ramp": True,
        "max_steps": 360,
        "lock_mode": "level",
        "box_mass": 2,
        "door_box_size": 72,
        "ramp_climb": False,
        "ramp_xray": True,
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_torch():
    # These small CPU policies run faster with one thread; keep the inference path fixed.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class Policies:
    def __init__(self, preset, final=False):
        config = environment_config(preset)
        env = HideAndSeekEnv(**config)
        agent = env.possible_agents[0]
        obs_dim = env.observation_space(agent).shape[0]
        act_dim = env.action_space(agent).shape[0]
        env.close()
        self.nets, self.paths = {}, {}
        suffix = "_final" if final else ""
        for team in ("hider", "seeker"):
            path = ROOT / "checkpoints" / f"{PRESETS[preset]['prefix']}_{team}{suffix}.pt"
            # Fail on a missing requested checkpoint; never silently substitute weights.
            state = torch.load(path, map_location="cpu", weights_only=True)
            net = ActorCriticLSTM(obs_dim, act_dim, hidden=state["enc.0.weight"].shape[0])
            net.load_state_dict(state)
            net.eval()
            self.nets[team], self.paths[team] = net, path
        self.hidden = {}

    def reset(self, env):
        self.hidden = {n: self.nets[env.team[n]].init_hidden() for n in env.possible_agents}

    @torch.no_grad()
    def actions(self, env, obs, disable_hiders=False):
        actions = {}
        for name in env.possible_agents:
            if disable_hiders and env.team[name] == "hider":
                actions[name] = np.zeros(3, dtype=np.float32)
                continue
            tensor = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
            mean, _, self.hidden[name] = self.nets[env.team[name]].step(tensor, self.hidden[name])
            actions[name] = mean.squeeze(0).numpy()
        return actions


def ramp_sightlines(env):
    """Return visible seeker/hider pairs that become occluded without elevation.

    This is an instantaneous LOS comparison, not a second episode or an estimate of
    the policy's performance without a ramp. Ignore dormant seekers in the control.
    """
    saved = env._elevated_now
    pairs = []
    try:
        for seeker in env.teams["seeker"][: env.active_seekers]:
            if seeker not in saved:
                continue
            for hider in env.teams["hider"]:
                env._elevated_now = saved
                visible = env._visible(seeker, env.bodies[hider])
                env._elevated_now = set()
                if visible and not env._visible(seeker, env.bodies[hider]):
                    pairs.append((seeker, hider))
    finally:
        env._elevated_now = saved
    return pairs


def run_episode(preset, policies, seed, active_seekers=None, disable_hiders=False):
    # Fresh physics state per episode makes a seed replayable in isolation, including GIFs.
    env = HideAndSeekEnv(**environment_config(preset))
    try:
        obs, _ = env.reset(
            seed=seed,
            options={
                "ramp_active": True,
                "active_seekers": active_seekers or env.n_seekers,
            },
        )
        policies.reset(env)
        ramp_start = np.array(env.ramp_body.position)
        max_displacement = 0.0
        play = hidden = near = sight = 0
        while env.agents:
            obs, _, _, _, infos = env.step(policies.actions(env, obs, disable_hiders))
            info = infos[env.teams["hider"][0]]
            displacement = float(np.linalg.norm(np.array(env.ramp_body.position) - ramp_start))
            max_displacement = max(max_displacement, displacement)
            if not info["in_prep"]:
                play += 1
                hidden += not info["seeker_sees_hider"]
                near += any(
                    s in env._elevated_now for s in env.teams["seeker"][: env.active_seekers]
                )
                sight += bool(ramp_sightlines(env))
        return {
            "seed": seed,
            "play_steps": play,
            "hidden_fraction": hidden / play,
            "barricade_at_end": bool(info["doorway_barricaded"]),
            "near_ramp_fraction": near / play,
            "ramp_sightline_fraction": sight / play,
            "ramp_locked_at_end": info["ramp_lock_owner"] == "hider",
            "ramp_net_displacement_px": displacement,
            "ramp_max_displacement_px": max_displacement,
        }
    finally:
        env.close()


def summarize(episodes):
    return {
        key: float(np.mean([ep[key] for ep in episodes]))
        for key in episodes[0]
        if key not in ("seed", "play_steps")
    }


def evaluate(preset, episodes, seed_start=0, final=False, primary_only=False):
    configure_torch()
    policies = Policies(preset, final)
    started = time.perf_counter()
    conditions = {"trained": {}}
    if not primary_only:
        conditions.update(
            {
                "one_active_seeker": {"active_seekers": 1},
                "zero_action_hiders": {"disable_hiders": True},
            }
        )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "environment": environment_config(preset),
        "protocol": {
            "episodes_per_condition": episodes,
            "seed_start": seed_start,
            "seed_stop_exclusive": seed_start + episodes,
            "actions": "policy means on CPU",
            "torch_threads": 1,
            "fresh_environment_per_episode": True,
            "training_assists": False,
            "checkpoint_selection": "final" if final else "save-best",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "pymunk", "gymnasium", "pettingzoo")
            },
        },
        "checkpoints": {
            team: {"file": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for team, path in policies.paths.items()
        },
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "hide_and_seek/env_hs.py",
                "hide_and_seek/ppo_recurrent.py",
                "evaluate.py",
                "uv.lock",
            )
        },
        "conditions": {},
    }
    for name, kwargs in conditions.items():
        rows = []
        for seed in range(seed_start, seed_start + episodes):
            rows.append(run_episode(preset, policies, seed, **kwargs))
            if len(rows) % 50 == 0:
                print(f"{preset}: {name} {len(rows)}/{episodes}", file=sys.stderr, flush=True)
        report["conditions"][name] = {"mean": summarize(rows), "episodes": rows}
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="two-hiders")
    parser.add_argument("--episodes", type=positive_int, default=200)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="skip the one-seeker and zero-action-hider conditions",
    )
    parser.add_argument("--output", type=Path, help="save metadata and per-episode results as JSON")
    args = parser.parse_args()
    if args.seed_start < 0:
        parser.error("--seed-start must be nonnegative")
    report = evaluate(args.preset, args.episodes, args.seed_start, args.final, args.primary_only)
    for name, result in report["conditions"].items():
        print(f"\n{name} ({args.episodes} episodes)")
        for key, value in result["mean"].items():
            formatted = f"{value:.2f} px" if key.endswith("_px") else f"{value:.2%}"
            print(f"  {key}: {formatted}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
