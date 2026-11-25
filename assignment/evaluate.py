"""Evaluate a trained agent, record a video, or dump a per-step trace.

    python evaluate.py runs/q1_baseline/seed0                       # 20 episodes, same env as training
    python evaluate.py runs/q1_baseline/seed0 --faulty              # zero-shot on the faulty robot (Q3)
    python evaluate.py runs/q1_baseline/seed0 --video videos/Q1_baseline.mp4
    python evaluate.py runs/q3_faulty/seed0 --trace traces/Q3_faulty.csv

Evaluates best_model.zip by default (use --final for the network at the end of training).
Results are written next to the model as eval_<robot>[_final].json (and printed).
Evaluation uses the greedy policy (deterministic=True) and fixed seeds 10000, 10001, ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rl_assignment.envs import make_env
from rl_assignment.variant import Variant

EVAL_SEED0 = 10_000


def run_episode(model, env, seed: int, trace: list | None = None, frames: list | None = None):
    obs, _ = env.reset(seed=seed)
    ret, length, xs, done = 0.0, 0, [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        x, x_dot, th, th_dot = env.unwrapped.state
        if trace is not None:
            trace.append(dict(t=length, x=x, x_dot=x_dot, theta=th, theta_dot=th_dot, action=int(action)))
        if frames is not None:
            frames.append(env.render())
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r); length += 1; xs.append(float(env.unwrapped.state[0]))
        if trace is not None:
            trace[-1]["reward"] = float(r)
        done = term or trunc
    return ret, length, np.array(xs)


def evaluate_model(model, variant: Variant, faulty=False, reward="default", action_history=0, episodes=20, seed0=EVAL_SEED0) -> dict:
    env = make_env(variant, faulty=faulty, reward=reward, action_history=action_history)
    rets, lens, dists = [], [], []
    for i in range(episodes):
        r, n, xs = run_episode(model, env, seed0 + i)
        rets.append(r); lens.append(n)
        dists.append(float(np.mean(np.abs(xs - variant.target_x))))  # Q4 metric: mean |x - x*| over the episode
    env.close()
    return {
        "fingerprint": variant.fingerprint, "episodes": episodes, "seeds": [seed0, seed0 + episodes - 1],
        "reward": reward, "faulty": faulty,
        "returns": rets, "lengths": lens,
        "mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
        "mean_length": float(np.mean(lens)), "std_length": float(np.std(lens)),
        "target_x": variant.target_x, "mean_abs_x_minus_target": float(np.mean(dists)),
    }


def main():
    from stable_baselines3 import DQN
    from rl_assignment.agents import DQNNoTarget
    from rl_assignment.variant import make_variant

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory containing model.zip and config.json")
    ap.add_argument("--episodes", type=int, default=20)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--faulty", action="store_true", help="evaluate on the faulty robot")
    g.add_argument("--nominal", action="store_true", help="evaluate on the nominal robot")
    ap.add_argument("--reward", default=None, help="reward to report (default: the one used in training)")
    ap.add_argument("--final", action="store_true", help="evaluate model.zip (end of training) instead of best_model.zip")
    ap.add_argument("--video", help="save one episode as .mp4 (or .gif)")
    ap.add_argument("--trace", help="save one episode step-by-step as CSV")
    a = ap.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    variant = make_variant(cfg["student_id"])
    faulty = True if a.faulty else False if a.nominal else cfg["faulty"]
    reward = a.reward or cfg["reward"]
    cls = DQNNoTarget if cfg.get("algo") == "DQNNoTarget" else DQN
    model_file = "model.zip" if a.final or not (run / "best_model.zip").exists() else "best_model.zip"
    model = cls.load(run / model_file, device="cpu")

    hist = cfg.get("action_history", 0)
    res = evaluate_model(model, variant, faulty=faulty, reward=reward, action_history=hist, episodes=a.episodes)
    tag = ("faulty" if faulty else "nominal") + ("_final" if model_file == "model.zip" else "")
    res["evaluated_on"] = "faulty" if faulty else "nominal"
    res["model"] = model_file
    (run / f"eval_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"{run}/{model_file} on {res['evaluated_on']} robot ({reward} reward): return {res['mean_return']:.1f} ± {res['std_return']:.1f}, "
          f"length {res['mean_length']:.0f} ± {res['std_length']:.0f}, mean |x - x*| {res['mean_abs_x_minus_target']:.2f}"
          f"  -> {run / f'eval_{tag}.json'}")

    if a.trace:
        env = make_env(variant, faulty=faulty, reward=reward, action_history=hist)
        tr: list = []
        run_episode(model, env, EVAL_SEED0, trace=tr)
        env.close()
        Path(a.trace).parent.mkdir(parents=True, exist_ok=True)
        with open(a.trace, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tr[0].keys())); w.writeheader(); w.writerows(tr)
        print(f"trace ({len(tr)} steps) -> {a.trace}   target x* = {variant.target_x:+.2f}")

    if a.video:
        env = make_env(variant, faulty=faulty, reward=reward, action_history=hist, render_mode="rgb_array")
        frames: list = []
        ret, n, _ = run_episode(model, env, EVAL_SEED0, frames=frames)
        env.close()
        path = save_video(frames, a.video)
        print(f"video ({n} steps, return {ret:.1f}) -> {path}")


def save_video(frames, path: str, fps: int = 50) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    try:
        imageio.mimsave(path, frames, fps=fps)
        return path
    except Exception:  # no ffmpeg available: fall back to GIF
        Path(path).unlink(missing_ok=True)
        gif = str(Path(path).with_suffix(".gif"))
        imageio.mimsave(gif, frames[::2], duration=1000 * 2 / fps, loop=0)
        print("(ffmpeg not available, saved a GIF instead)")
        return gif


if __name__ == "__main__":
    main()
