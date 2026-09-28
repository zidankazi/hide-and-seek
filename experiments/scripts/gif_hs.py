"""
Render a trained policy to an animated GIF — headless, for the writeup's stage montage.

Auto-detects the checkpoint architecture (feed-forward ppo_continuous vs recurrent
ppo_recurrent) so it works across every stage's policies. Runs several episodes, scores
each by the behavior you want to showcase (--want), and writes the best-matching episode
as a GIF. No display needed (offscreen SDL).

Note: --want=elev scores *genuine* ramp use (steps where standing on the ramp is what
reveals the hider), not the env's `seeker_elevated` flag, which only means "within
RAMP_USE_DIST of the ramp" and fires even when the elevation reveals nothing.

Usage:
  python gif_hs.py <prefix> [--final|--best] [--layout=room|roomt] [--nh=1] [--ns=2]
                   [--speed=1.0] [--want=barricade|rlock|elev|hidden|any]
                   [--episodes=30] [--out=stage.gif] [--stride=2] [--scale=0.6] [--fps=30]

Examples (the montage):
  python gif_hs.py hs_mega1v1 --layout=room --nh=1 --ns=1 --want=hidden --out=1_evasion.gif
  python gif_hs.py hs_lstm    --layout=room  --nh=1 --ns=2 --want=elev      --out=2_ramp.gif
  python gif_hs.py hs_roomt   --layout=roomt --nh=1 --ns=2 --want=barricade --out=3_barricade.gif
  python gif_hs.py hs_2v2     --layout=roomt --nh=2 --ns=2 --speed=1.4 --want=rlock --out=4_arc.gif
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import sys
import numpy as np
import pygame
import torch
from PIL import Image

from hide_and_seek.env_hs import HideAndSeekEnv


def arg(name, default, cast=str):
    hit = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), None)
    return cast(hit) if hit is not None else default


PREFIX = next((a for a in sys.argv[1:] if not a.startswith("--")), "hs_2v2")
SUFFIX = "" if "--best" in sys.argv else "_final"
LAYOUT = arg("layout", "roomt")
NH, NS = arg("nh", 1, int), arg("ns", 2, int)
SPEED = arg("speed", 1.0, float)
WANT = arg("want", "any")
EPISODES = arg("episodes", 30, int)
OUT = arg("out", f"{PREFIX}.gif")
STRIDE = arg("stride", 2, int)          # keep every Nth frame (physics is 60fps; GIF needn't be)
SCALE = arg("scale", 0.6, float)
FPS = arg("fps", 30, int)
CROP = arg("crop", None)                # "wx0,wy0,wx1,wy1" world bbox to zoom into (else full arena)
LABEL = arg("label", "")                # caption burned into a bar at the top of every frame
SIGHTLINE = "--sightline" in sys.argv   # draw the over-the-wall sightline elevation buys
TARGET_W = arg("width", 460, int)       # output width in px (height follows the crop aspect)

env = HideAndSeekEnv(layout=LAYOUT, ramp=True, max_steps=360, lock_mode="level",
                     n_hiders=NH, n_seekers=NS, box_mass=2, door_box_size=72,
                     seeker_speed_mult=SPEED, render_mode="human")
obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
act_dim = env.action_space(env.possible_agents[0]).shape[0]
H0 = env.teams["hider"][0]

# ---- load policies, auto-detecting architecture from the checkpoint keys ----
def ckpt(team):
    p = f"{PREFIX}_{team}{SUFFIX}.pt"
    if SUFFIX and not os.path.exists(p):
        p = f"{PREFIX}_{team}.pt"          # fall back to save-best if no _final exists
    return p

sd0 = torch.load(ckpt("hider"))
IS_LSTM = "enc.0.weight" in sd0
if IS_LSTM:
    from hide_and_seek.ppo_recurrent import ActorCriticLSTM
    def build(sd):
        ac = ActorCriticLSTM(obs_dim, act_dim, hidden=sd["enc.0.weight"].shape[0])
        ac.load_state_dict(sd); ac.eval(); return ac
else:
    from hide_and_seek.ppo_continuous import ActorCritic
    def build(sd):
        ac = ActorCritic(obs_dim, act_dim, hidden=sd["shared.0.weight"].shape[0])
        ac.load_state_dict(sd); ac.eval(); return ac

nets = {"hider": build(sd0),
        "seeker": build(torch.load(ckpt("seeker")))}
print(f"[gif] {PREFIX}{SUFFIX} arch={'LSTM' if IS_LSTM else 'FF'} layout={LAYOUT} "
      f"{NH}v{NS} speed={SPEED} want={WANT}")

_hc = {}
def reset_hidden():
    for n in env.possible_agents:
        _hc[n] = nets[env.team[n]].init_hidden() if IS_LSTM else None

def act(name, o):
    t = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if IS_LSTM:
            mean, _, _hc[name] = nets[env.team[name]].step(t, _hc[name])
        else:
            mean, _ = nets[env.team[name]](t)
    return mean.squeeze(0).numpy()


# world bbox -> pixel crop box, using the live renderer's arena geometry
_CROPBOX = None
if CROP:
    R = env.renderer
    wx0, wy0, wx1, wy1 = (float(v) for v in CROP.split(","))
    sc = R.arena_size / 600.0
    _CROPBOX = (int(R.arena_x + wx0 * sc), int(R.arena_y + wy0 * sc),
                int(R.arena_x + wx1 * sc), int(R.arena_y + wy1 * sc))


def grab():
    """Current renderer surface -> (H, W, 3) uint8, cropped to the zoom box if set."""
    a = np.transpose(pygame.surfarray.array3d(env.renderer.screen), (1, 0, 2))
    if _CROPBOX:
        x0, y0, x1, y1 = _CROPBOX
        a = a[y0:y1, x0:x1]
    return a


def decisive_seekers():
    """Seekers using the ramp *right now* to see over a wall.

    env's `seeker_elevated` only says a seeker is within RAMP_USE_DIST of the ramp, which
    is true whenever it stands near it — including when elevation reveals nothing. Here we
    require that the elevation is what grants the sightline: the seeker sees the hider now
    and would NOT see it un-elevated. That is the behavior rung 2 actually claims.
    """
    hb = env.bodies[H0]
    saved = env._elevated_now
    out = []
    for s in env.teams["seeker"][:env.active_seekers]:
        if s not in saved or not env._visible(s, hb):
            continue
        env._elevated_now = set()               # same query, un-elevated
        try:
            blocked = not env._visible(s, hb)
        finally:
            env._elevated_now = saved
        if blocked:
            out.append(s)
    return out


def draw_sightline(seekers):
    """Draw the over-the-wall sightline the elevation bought, so the GIF shows the mechanic.

    Without this the frame is just a seeker standing on a green wedge — the viewer can't
    tell it is seeing through the interior wall. Overlay only; env/renderer untouched.
    """
    if not SIGHTLINE or not seekers:
        return
    R = env.renderer
    sc = R.arena_size / 600.0
    to_px = lambda p: (int(R.arena_x + p.x * sc), int(R.arena_y + p.y * sc))
    hp = to_px(env.bodies[H0].position)
    for s in seekers:
        pygame.draw.line(R.screen, (223, 90, 79), to_px(env.bodies[s].position), hp, 3)
        pygame.draw.circle(R.screen, (223, 90, 79), hp, 9, 2)


def score(info, hid_frac):
    if WANT == "barricade":  return float(info["doorway_barricaded"])
    if WANT == "rlock":      return float(info["ramp_lock_owner"] == "hider")
    if WANT == "elev":       return info.get("_elev_frac", 0.0)
    if WANT == "hidden":     return hid_frac
    if WANT == "chase":      # a dynamic near-miss: hider moves a lot AND a seeker gets close
        return info.get("_moved", 0.0) / 1000.0 - info.get("_mind", 1e9) / 300.0
    if WANT == "arc":        # richest tool-use episode: weight the rare rungs highest
        return (1.0 * float(info["doorway_barricaded"])
                + 1.0 * info.get("_elev_frac", 0.0)
                + 5.0 * float(info["ramp_lock_owner"] == "hider"))
    return 1.0


best = {"score": -1.0, "frames": None}
for ep in range(EPISODES):
    obs, _ = env.reset(seed=1000 + ep, options={"ramp_active": True, "active_seekers": NS})
    reset_hidden()
    frames, play, hid, ele, prox = [], 0, 0, 0, 0
    mind, moved, prevh = 1e9, 0.0, None
    info = None
    while env.agents:
        actions = {n: act(n, obs[n]) for n in env.possible_agents}
        obs, rew, terms, truncs, infos = env.step(actions)
        info = infos[H0]
        dec = [] if info["in_prep"] else decisive_seekers()
        if not info["in_prep"]:
            play += 1
            hid += (not info["seeker_sees_hider"])
            prox += info["seeker_elevated"]      # merely near the ramp
            ele += bool(dec)                     # actually seeing over a wall because of it
            hp = env.bodies[H0].position
            for s_ in env.teams["seeker"][:env.active_seekers]:
                mind = min(mind, (env.bodies[s_].position - hp).length)
            if prevh is not None:
                moved += (hp - prevh).length
            prevh = hp
        env.render()
        draw_sightline(dec)
        frames.append(grab())
    info["_elev_frac"] = ele / play if play else 0.0
    info["_prox_frac"] = prox / play if play else 0.0
    info["_mind"] = mind
    info["_moved"] = moved
    s = score(info, hid / play if play else 0.0)
    if s > best["score"]:
        best = {"score": s, "frames": frames, "ep": ep}
    print(f"  ep {ep}: score={s:.2f} (best={best['score']:.2f})"
          f"  [ramp used {info['_elev_frac']:.0%}, near-ramp {info['_prox_frac']:.0%}]")

# ---- assemble the winning episode into a GIF (zoom already applied in grab) ----
from PIL import ImageDraw, ImageFont
BAR_H = 30 if LABEL else 0
_font = None
if LABEL:
    for fp in ("/System/Library/Fonts/Helvetica.ttc",
               "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            _font = ImageFont.truetype(fp, 15); break
        except Exception:
            pass
    if _font is None:
        _font = ImageFont.load_default()


def finish(frame):
    im = Image.fromarray(frame)
    h = int(im.height * TARGET_W / im.width)
    im = im.resize((TARGET_W, h), Image.BILINEAR)
    if LABEL:
        canvas = Image.new("RGB", (TARGET_W, h + BAR_H), (26, 30, 38))
        canvas.paste(im, (0, BAR_H))
        d = ImageDraw.Draw(canvas)
        d.text((10, 7), LABEL, fill=(238, 241, 247), font=_font)
        im = canvas
    return im.convert("P", palette=Image.ADAPTIVE)


imgs = [finish(f) for f in best["frames"][::STRIDE]]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], loop=0,
             duration=int(1000 / FPS), optimize=True, disposal=2)
env.close()
print(f"[gif] wrote {OUT}: episode {best['ep']} (want={WANT} score={best['score']:.2f}), "
      f"{len(imgs)} frames")
