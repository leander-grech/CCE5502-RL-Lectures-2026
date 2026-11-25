"""DQN configuration and the two ablations used in Q2."""
from __future__ import annotations

from stable_baselines3 import DQN

# The same settings as the DQN example in Lecture 2.
DEFAULT_HP = dict(
    learning_rate=1e-3,
    batch_size=32,
    buffer_size=50_000,
    learning_starts=1_000,
    gamma=0.99,
    train_freq=4,                # one gradient step every 4 environment steps
    gradient_steps=1,
    target_update_interval=500,  # θ⁻ ← θ every 500 environment steps
    exploration_fraction=0.1,    # ε decays 1.0 → 0.02 over the first 10% of training
    exploration_final_eps=0.02,
    policy_kwargs=dict(net_arch=[64, 64]),
)


class DQNNoTarget(DQN):
    """DQN *without* a target network: the target is recomputed with the online
    network after every single gradient step (θ⁻ = θ at all times)."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        for _ in range(gradient_steps):
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            super().train(gradient_steps=1, batch_size=batch_size)


ABLATIONS = {
    # name: (class, hyper-parameter overrides)
    "none": (DQN, {}),
    "no-target": (DQNNoTarget, {}),
    # The buffer only holds the most recent `batch_size` transitions: no re-use of old experience,
    # and every minibatch is a short stretch of highly correlated, consecutive steps.
    "no-replay": (DQN, {"buffer_size": DEFAULT_HP["batch_size"]}),
}
