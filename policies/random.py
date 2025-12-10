"""
Random policy for baseline comparison.
"""

from typing import Dict
import numpy as np
import gymnasium as gym

from .base import BasePolicy


class RandomPolicy(BasePolicy):
    """
    Random policy that samples uniformly from action space.
    Useful as a baseline for comparison.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(observation_space, action_space)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Randomly sample action from action space."""
        self.total_steps += 1
        return self.action_space.sample()

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """Random policy has no learning, returns empty metrics."""
        return {}

    def save(self, path: str):
        """Random policy has no parameters to save."""
        pass

    def load(self, path: str):
        """Random policy has no parameters to load."""
        pass
