"""Environment construction: standard CartPole-v1 + your personal variant.

    make_env(variant)                     nominal robot (your physics + sensor noise)
    make_env(variant, faulty=True)        same robot with the faulty actuator (delay + weak side)
    make_env(variant, action_history=4)   observation also contains the agent's last 4 actions
    make_env(variant, reward="target")    Q4: your reward wrapper from rewards.py
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from .rewards import REWARDS
from .variant import Variant

LEFT, RIGHT = 0, 1  # CartPole actions: 0 pushes the cart left, 1 pushes it right


class VariantPhysics(gym.Wrapper):
    """Overwrites the physical constants of the underlying CartPoleEnv."""

    def __init__(self, env: gym.Env, variant: Variant):
        super().__init__(env)
        u = env.unwrapped
        u.length, u.masspole, u.masscart, u.force_mag = variant.pole_length, variant.pole_mass, variant.cart_mass, variant.force_mag
        u.total_mass = u.masspole + u.masscart
        u.polemass_length = u.masspole * u.length


class FaultyActuator(gym.Wrapper):
    """The motor executes every command `fault_delay` control steps late, and one push
    direction only delivers a fraction of the nominal force.

    At the start of an episode the motor is still executing an alternating left/right
    pattern, so the first `fault_delay` commands have no immediate effect."""

    def __init__(self, env: gym.Env, variant: Variant):
        super().__init__(env)
        self.nominal = env.unwrapped.force_mag
        self.delay = variant.fault_delay
        self.weak_action = LEFT if variant.fault_side == "left" else RIGHT
        self.strength = variant.fault_strength
        self.pending: deque = deque()

    def reset(self, **kwargs):
        self.pending = deque(i % 2 for i in range(self.delay))
        return self.env.reset(**kwargs)

    def step(self, action):
        self.pending.append(int(action))
        executed = self.pending.popleft()
        self.env.unwrapped.force_mag = self.nominal * (self.strength if executed == self.weak_action else 1.0)
        return self.env.step(executed)


class ActionHistory(gym.Wrapper):
    """Appends the agent's last k actions (as −1 = left / +1 = right) to the observation."""

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        low = np.concatenate([env.observation_space.low, -np.ones(k)]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, np.ones(k)]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.history: deque = deque(maxlen=k)

    def _obs(self, obs):
        return np.concatenate([obs, np.array(self.history, dtype=np.float32)]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.extend([0.0] * self.k)
        return self._obs(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.history.append(1.0 if int(action) == RIGHT else -1.0)
        return self._obs(obs), r, term, trunc, info


class ObservationNoise(gym.ObservationWrapper):
    """Gaussian sensor noise on every observation (the true state is unaffected)."""

    def __init__(self, env: gym.Env, std: float):
        super().__init__(env)
        self.std = std
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 12345)
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        if self.std <= 0:
            return obs
        return (obs + self.rng.normal(0.0, self.std, size=obs.shape)).astype(np.float32)


def make_env(variant: Variant, faulty: bool = False, reward: str = "default", action_history: int = 0,
             render_mode: str | None = None) -> gym.Env:
    env = gym.make("CartPole-v1", render_mode=render_mode)  # 500-step time limit (truncation)
    env = VariantPhysics(env, variant)
    if faulty:
        env = FaultyActuator(env, variant)
    if reward not in REWARDS:
        raise ValueError(f"unknown reward {reward!r}: register it in REWARDS in rl_assignment/rewards.py "
                         f"(known: {', '.join(REWARDS)})")
    if REWARDS[reward] is not None:
        env = REWARDS[reward](env, variant)
    env = ObservationNoise(env, variant.obs_noise_std)
    if action_history:
        env = ActionHistory(env, action_history)
    return env
