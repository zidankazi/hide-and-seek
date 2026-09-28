"""Replay a saved policy at a given seed and write a GIF."""

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PIL import Image, ImageDraw

from evaluate import (
    PRESETS,
    Policies,
    configure_torch,
    environment_config,
    ramp_sightlines,
)
from hide_and_seek.env_hs import HideAndSeekEnv


def render(preset, seed, output, final=False):
    configure_torch()
    random.seed(seed)  # renderer particles only; environment uses its own seeded RNG
    policies = Policies(preset, final)
    env = HideAndSeekEnv(**environment_config(preset), render_mode="human")
    frames = []
    try:
        obs, _ = env.reset(
            seed=seed, options={"ramp_active": True, "active_seekers": env.n_seekers}
        )
        policies.reset(env)
        for step in range(env.MAX_STEPS):
            obs, _, _, _, _ = env.step(policies.actions(env, obs))
            if step % 3:
                continue
            env.render()
            # Only draw sightlines that disappear when elevation is removed.
            if step >= env.PREP_STEPS:
                for seeker, hider in ramp_sightlines(env):
                    start = env.renderer._to_screen(env.bodies[seeker].position)
                    end = env.renderer._to_screen(env.bodies[hider].position)
                    pygame.draw.line(env.renderer.screen, (200, 65, 55), start, end, 3)
            pixels = np.transpose(pygame.surfarray.array3d(env.renderer.screen), (1, 0, 2))
            frame = Image.fromarray(pixels).resize((560, 624), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (560, 674), (248, 248, 251))
            canvas.paste(frame, (0, 50))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 8), f"{preset} | seed {seed}", fill=(35, 40, 50))
            draw.text(
                (16, 27),
                "Red line: a sightline gained from elevation",
                fill=(80, 85, 95),
            )
            frames.append(canvas)
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True
    )
    print(f"Saved {output}: {preset}, seed {seed}, {len(frames)} frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="two-hiders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("media/demo.gif"))
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.output.suffix.lower() != ".gif":
        parser.error("--output must end in .gif")
    render(args.preset, args.seed, args.output, args.final)


if __name__ == "__main__":
    main()
