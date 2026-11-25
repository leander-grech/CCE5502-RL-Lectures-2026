# CCE5502 RL Assignment 2026 — ⟨Firstname Lastname⟩ ⟨student number⟩

> Keep these headings (they match the assignment sheet). Maximum 8 pages excluding the appendix.
> Export the finished report as `report.pdf` in the `assignment/` folder, then run `python make_submission.py Firstname_Lastname`.
> Put the source file next to every number, e.g. `(runs/q1_baseline/seed*/eval.json)`.

## Part A — Baseline

**A1** Variant card:
```text
(paste the output of `python setup_student.py <id>`)
```

**A2** Figure: learning curves, 3 seeds (`figures/A_curves.png`)

**A3**

| seed | mean episode length ± std | file |
|---|---|---|
| 0 | | `runs/q1_baseline/seed0/eval.json` |
| 1 | | |
| 2 | | |

**A4** Description of the learning curves:

**A5** One observation explained with a lecture idea:

**A6** Video `videos/A_best.mp4`, trace plot `figures/A_trace.png`, description:

**A7** Is the noisy observation still a good (Markov) state?

## Part B — What makes DQN work?

**B1** Figure: baseline vs no-target vs no-replay (`figures/B_ablation.png`)

**B2**

| version | mean episode length ± std (3 seeds) | files |
|---|---|---|
| baseline | | |
| no-target | | |
| no-replay | | |

**B3** No target network — what happened, why, TensorBoard evidence:

**B4** No replay buffer — what happened, why:

**B5** Predictions P2 and P5:

## Part C — The broken motor

**C1**

| agents | robot | mean episode length ± std | file |
|---|---|---|---|
| baseline, seeds 0–2 | nominal | | |
| baseline, seeds 0–2 | faulty | | |

Percentage drop: ___ %

**C2** What goes wrong, with times (`videos/C_zeroshot.mp4`, `traces/...`), and why:

**C3** Figure: faulty-robot training vs baseline (`figures/C_retrain.png`)

**C4** Is the observation still Markov with a delayed motor? What is missing?

**C5** K = __ because … ; figure comparing C3 and C5:

## Part D — Design a reward

**D1** What the naive reward teaches, and why that is optimal for it:

**D2** My reward (formula, idea), results table, trace plot with x* (`figures/D_trace.png`):

| reward | mean episode length | mean abs(x − x*) | file |
|---|---|---|---|
| naive | | | |
| mine | | | |

**D3** A reward that did not work, or a way to cheat mine:

## Part E — Reflection

**E1** Predictions P1–P5 vs what happened:

**E2** AI-use statement (tools, what for, one example prompt — or "none"):

## Appendix (optional)
