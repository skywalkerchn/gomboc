"""
Base policy class for online RL algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym


class BasePolicy(ABC):
    """
    Base class for all online RL policies.

    All policies should implement:
    - select_action: Choose action given observation
    - update: Update policy based on experience
    - save/load: Serialize policy
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Args:
            observation_space: Observation space of the environment
            action_space: Action space of the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.total_steps = 0

    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action given observation.

        Args:
            observation: Current observation
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """
        Update policy based on experience tuple.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            terminated: Whether episode terminated
            truncated: Whether episode was truncated

        Returns:
            Dictionary of training metrics (e.g., loss, etc.)
        """
        pass

    def save(self, path: str):
        """Save policy to file."""
        raise NotImplementedError("save() not implemented")

    def load(self, path: str):
        """Load policy from file."""
        raise NotImplementedError("load() not implemented")

    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics for logging."""
        return {"total_steps": self.total_steps}
