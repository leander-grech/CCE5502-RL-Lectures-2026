"""Train DQN agents on your personal CartPole variant.

Examples
--------
    python train.py --name q1_baseline --seeds 0 1 2
    python train.py --name q2_no_target --ablation no-target --seeds 0 1 2
    python train.py --name q3_faulty --faulty --seeds 0 1 2
    python train.py --name q3_faulty_hist --faulty --action-history 4 --seeds 0 1 2
    python train.py --name q4_naive --reward naive --seeds 0 1
    python train.py --name lr_test --hp learning_rate=1e-4 gamma=0.95 --seeds 0
    python train.py --name big_net --net 256 256

Each seed is written to runs/<name>/seed<k>/ :
    config.json   everything needed to reproduce the run (incl. your variant fingerprint)
    monitor.csv   one line per training episode (return, length, time)
    tb/           TensorBoard logs      ->  tensorboard --logdir runs
    model.zip       network at the end of training
    best_model.zip  network with the best periodic evaluation score (checked every 10k steps)
    evals/          the periodic evaluations (also in TensorBoard under eval/)
    eval.json       20 evaluation episodes of best_model.zip with the greedy policy on the training robot
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from rl_assignment.agents import ABLATIONS, DEFAULT_HP
from rl_assignment.envs import make_env
from rl_assignment.rewards import REWARDS
from rl_assignment.student import load_variant
from evaluate import evaluate_model


def parse_hp(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def train_one(args, variant, seed: int) -> Path:
    out = Path("runs") / args.name / f"seed{seed}"
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} already exists — pick another --name or pass --overwrite")
        shutil.rmtree(out)  # start clean: no stale TensorBoard logs or eval files from the old run
    out.mkdir(parents=True)
    try:
        return _train(args, variant, seed, out)
    except BaseException:  # crash or Ctrl-C: remove the incomplete run so the same command can be re-run
        shutil.rmtree(out, ignore_errors=True)
        print(f"run failed or was interrupted — removed the incomplete folder {out}", file=sys.stderr)
        raise


def _train(args, variant, seed: int, out: Path) -> Path:
    algo_cls, ablation_hp = ABLATIONS[args.ablation]
    hp = {**DEFAULT_HP, **ablation_hp, **parse_hp(args.hp)}
    if args.net:
        hp["policy_kwargs"] = dict(net_arch=args.net)

    env_kw = dict(faulty=args.faulty, reward=args.reward, action_history=args.action_history)
    env = Monitor(make_env(variant, **env_kw), filename=str(out / "monitor.csv"))
    env.reset(seed=seed)
    model = algo_cls("MlpPolicy", env, seed=seed, verbose=0, device="cpu", tensorboard_log=str(out / "tb"), **hp)

    config = {
        "student_id": variant.student_id,
        "fingerprint": variant.fingerprint,
        "name": args.name, "seed": seed, "timesteps": args.timesteps,
        "faulty": args.faulty, "reward": args.reward, "ablation": args.ablation, "algo": algo_cls.__name__,
        "action_history": args.action_history,
        "hyperparameters": {k: (str(v) if k == "policy_kwargs" else v) for k, v in hp.items()},
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "gymnasium": gymnasium.__version__,
                     "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "git": git_state(),
        "command": " ".join(sys.argv),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    t0 = time.time()
    eval_cb = EvalCallback(Monitor(make_env(variant, **env_kw)), n_eval_episodes=5, eval_freq=10_000,
                           best_model_save_path=str(out), log_path=str(out / "evals"), deterministic=True, verbose=0)
    model.learn(total_timesteps=args.timesteps, tb_log_name="dqn", progress_bar=False, callback=eval_cb)
    model.save(out / "model.zip")
    best = out / "best_model.zip"
    if not best.exists():  # training shorter than one evaluation period
        model.save(best)
    model = algo_cls.load(best, device="cpu")
    config["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config["wall_time_s"] = round(time.time() - t0, 1)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    res = evaluate_model(model, variant, episodes=20, **env_kw)
    res["evaluated_on"] = "faulty" if args.faulty else "nominal"
    res["model"] = "best_model.zip"
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.name} seed {seed}] {config['wall_time_s']:.0f}s — best model: eval return {res['mean_return']:.1f} ± {res['std_return']:.1f}, "
          f"length {res['mean_length']:.0f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run group name, e.g. q1_baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--faulty", action="store_true", help="train on the robot with the faulty actuator")
    ap.add_argument("--reward", default="default", choices=list(REWARDS), help="reward design registered in REWARDS in rl_assignment/rewards.py")
    ap.add_argument("--action-history", type=int, default=0, metavar="K", help="append the last K actions to the observation")
    ap.add_argument("--ablation", default="none", choices=list(ABLATIONS))
    ap.add_argument("--hp", nargs="*", default=[], help="hyper-parameter overrides, e.g. learning_rate=1e-4 gamma=0.95")
    ap.add_argument("--net", type=int, nargs="+", help="hidden layer sizes of the Q-network, e.g. --net 64 64")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    variant = load_variant()
    torch.set_num_threads(1)
    for s in args.seeds:
        train_one(args, variant, s)


if __name__ == "__main__":
    main()
