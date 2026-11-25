# CCE5502 RL assignment 2026 — starter kit

DQN on **your personal CartPole** (physics, sensor noise, a faulty actuator and a target position all
generated from your student number). Read the assignment brief first; this file is the technical how-to.

## Install

```bash
git clone https://github.com/leander-grech/CCE5502-RL-Lectures-2026.git
cd CCE5502-RL-Lectures-2026/assignment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_student.py <your UM student number>        # once; prints your variant card (e.g. 123456M)
python train.py --name smoke --timesteps 5000           # sanity check (~30 s)
```

No GPU needed — everything runs on CPU (and is faster there for networks this small).

## Commands

| What | Command |
|---|---|
| Train 3 seeds | `python train.py --name q1_baseline --seeds 0 1 2` |
| Ablations (Q2) | `--ablation no-target` / `--ablation no-replay` |
| Faulty robot (Q3) | `--faulty` · add the last K commands to the observation: `--action-history K` |
| Reward design (Q4) | `--reward naive` / `--reward target` (edit `rl_assignment/rewards.py`) |
| Hyper-parameters | `--hp learning_rate=1e-4 gamma=0.95` · `--net 128 128` · `--timesteps 50000` |
| Evaluate | `python evaluate.py runs/q1_baseline/seed0 [--faulty or --nominal]` |
| Video | `python evaluate.py runs/q1_baseline/seed0 --video videos/Q1_best.mp4` |
| Per-step trace | `python evaluate.py runs/q1_baseline/seed0 --trace traces/Q1.csv` |
| Learning curves | `python plot_runs.py runs/q1_baseline runs/q2_no_target:"no target" -o figures/Q2.png` |
| TensorBoard | `tensorboard --logdir runs` |
| **Submit** | save your report as `report.pdf` here, then `python make_submission.py Firstname_Lastname` → upload the ZIP to the VLE |

Tip: seeds are independent, so you can run them in parallel in separate terminals
(`--seeds 0`, `--seeds 1`, …) if your laptop has spare cores.

## What a run folder contains

```text
runs/<name>/seed<k>/
├── config.json      settings, versions, git commit, variant fingerprint, wall time
├── monitor.csv      every training episode: return r, length l, time t
├── tb/              TensorBoard logs (loss, exploration rate, episode reward …)
├── model.zip        the Q-network at the end of training
├── best_model.zip   the best Q-network of the periodic evaluations (every 10k steps)
├── evals/           those periodic evaluations
└── eval.json        20 greedy evaluation episodes of best_model.zip (seeds 10000–10019) on the training robot
```

These files are your evidence — keep them all and submit them.

## Files you may edit

`rl_assignment/rewards.py` (Q4) — and anything else you like for extra investigations, as long as you
document it. **Do not** change `variant.py` or the physics in `envs.py` for the main questions:
submissions are re-evaluated with the official environment code.

## Troubleshooting

- `No student ID set` → run `python setup_student.py <id>` in this `assignment/` folder.
- Linux: `pip install -r requirements.txt` downloads the GPU build of PyTorch (several GB). To save the space and
  time, install the CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Video fails with an ffmpeg error → a GIF is saved instead; that's fine to submit.
- `pygame` errors when rendering → `pip install "gymnasium[classic-control]"`.
- Anything else → post the **full text** of the error on the VLE forum.
