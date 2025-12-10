"""
Deep Q-Network (DQN) policy implementation.
"""

from collections import deque
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BasePolicy


class QNetwork(nn.Module):
    """Q-network for DQN."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Tuple):
        """Add transition to buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        obs, actions, rewards, next_obs, dones = zip(*batch)

        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNPolicy(BasePolicy):
    """
    Deep Q-Network (DQN) policy with experience replay and target network.

    Reference: Mnih et al. "Human-level control through deep reinforcement learning" (2015)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        learning_starts: int = 1000,
        device: str = "cpu",
    ):
        """
        Args:
            observation_space: Observation space
            action_space: Action space (must be Discrete)
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Number of steps to decay epsilon
            buffer_size: Replay buffer capacity
            batch_size: Batch size for updates
            target_update_freq: Frequency (in steps) to update target network
            learning_starts: Number of steps before learning starts
            device: Device to run on ("cpu" or "cuda")
        """
        super().__init__(observation_space, action_space)

        assert isinstance(action_space, gym.spaces.Discrete), "DQN only supports Discrete action space"

        self.obs_dim = np.prod(observation_space.shape)
        self.action_dim = action_space.n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.device = torch.device(device)

        # Q-networks
        self.q_network = QNetwork(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Metrics
        self.last_loss = 0.0

    def _get_epsilon(self) -> float:
        """Get current epsilon based on linear decay schedule."""
        progress = min(1.0, self.total_steps / self.epsilon_decay)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        self.total_steps += 1

        if training:
            self.epsilon = self._get_epsilon()

        # Epsilon-greedy exploration
        if training and np.random.rand() < self.epsilon:
            return self.action_space.sample()

        # Greedy action
        obs_tensor = torch.FloatTensor(observation.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax(dim=1).item()

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """Update Q-network using experience replay."""
        # Add to replay buffer
        done = terminated or truncated
        self.replay_buffer.push(
            (observation.flatten(), action, reward, next_observation.flatten(), done)
        )

        # Don't update until we have enough samples
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return {"loss": 0.0, "epsilon": self.epsilon}

        # Sample batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute current Q-values
        q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_batch).max(dim=1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = loss.item()

        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": self.last_loss, "epsilon": self.epsilon}

    def save(self, path: str):
        """Save Q-network to file."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        """Load Q-network from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]

    def get_stats(self) -> Dict[str, float]:
        """Get policy statistics."""
        return {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "loss": self.last_loss,
            "buffer_size": len(self.replay_buffer),
        }
