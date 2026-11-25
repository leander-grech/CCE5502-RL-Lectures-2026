"""Reward wrappers for Part D (reward design).

Goal: keep the pole balanced AND keep the cart close to YOUR target position x*
(printed by `python -m rl_assignment.variant`).

Rules
-----
* You may only change the reward, not the dynamics or the termination condition.
* Read the true state with `self.env.unwrapped.state`  ->  (x, x_dot, theta, theta_dot).
* Register every design you try in REWARDS below, and train it with
      python train.py --name q4_mydesign --reward mydesign
* Keep the designs that failed! Explaining a failed reward is worth as much as a working one.
"""
from __future__ import annotations

import gymnasium as gym


class NaiveDistancePenalty(gym.RewardWrapper):
    """Attempt 0 (given): penalise the distance to the target at every step.

    Train it, watch it, and explain in your report what the agent learns and why.
    """

    def __init__(self, env, variant):
        super().__init__(env)
        self.target_x = variant.target_x

    def reward(self, reward):
        x = self.env.unwrapped.state[0]
        return -abs(x - self.target_x)


class TargetPositionReward(gym.RewardWrapper):
    """YOUR design. Replace the body of `reward` (and add more classes if you iterate)."""

    def __init__(self, env, variant):
        super().__init__(env)
        self.target_x = variant.target_x

    def reward(self, reward):
        x, x_dot, theta, theta_dot = self.env.unwrapped.state
        # ------------------------------------------------------------------
        # TODO(student): design a reward. `reward` is the original +1 per step.
        # ------------------------------------------------------------------
        raise NotImplementedError("Q4: implement TargetPositionReward.reward in rl_assignment/rewards.py")


REWARDS = {
    "default": None,          # original CartPole reward: +1 per step while upright
    "naive": NaiveDistancePenalty,
    "target": TargetPositionReward,
}
