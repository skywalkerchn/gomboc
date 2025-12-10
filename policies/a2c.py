"""
Advantage Actor-Critic (A2C) policy implementation.
"""

from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .base import BasePolicy


class ActorCritic(nn.Module):
    """Actor-Critic network for A2C."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extraction
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.actor(x)
        value = self.critic(x)

        return action_logits, value

    def get_action_and_value(self, x: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and get log prob, entropy, and value.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        action_logits, value = self.forward(x)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy, value


class A2CPolicy(BasePolicy):
    """
    Advantage Actor-Critic (A2C) policy with synchronous updates.

    Reference: Mnih et al. "Asynchronous Methods for Deep Reinforcement Learning" (2016)
    Note: This is the synchronous version (A2C) without multiple parallel actors.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_dim: int = 128,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        device: str = "cpu",
    ):
        """
        Args:
            observation_space: Observation space
            action_space: Action space (must be Discrete)
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_steps: Number of steps to accumulate before update
            device: Device to run on ("cpu" or "cuda")
        """
        super().__init__(observation_space, action_space)

        assert isinstance(action_space, gym.spaces.Discrete), "A2C only supports Discrete action space"

        self.obs_dim = np.prod(observation_space.shape)
        self.action_dim = action_space.n
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.device = torch.device(device)

        # Actor-Critic network
        self.ac_network = ActorCritic(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.ac_network.parameters(), lr=lr, eps=1e-5, alpha=0.99)

        # Rollout buffer for n-step returns
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_log_probs = []
        self.rollout_dones = []

        # For tracking last action's log prob and value
        self.last_log_prob = None
        self.last_value = None

        # Metrics
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using current policy."""
        self.total_steps += 1

        obs_tensor = torch.FloatTensor(observation.flatten()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, entropy, value = self.ac_network.get_action_and_value(obs_tensor)

        # Store for later update
        self.last_log_prob = log_prob.item()
        self.last_value = value.item()

        return action

    def _compute_returns(self, rewards, values, dones, next_value) -> np.ndarray:
        """
        Compute n-step returns.

        Returns:
            returns: Discounted returns
        """
        returns = np.zeros_like(rewards)
        R = next_value

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        return returns

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """Update policy using A2C algorithm."""
        # Add to rollout buffer
        done = terminated or truncated
        self.rollout_obs.append(observation.flatten())
        self.rollout_actions.append(action)
        self.rollout_rewards.append(reward)
        self.rollout_values.append(self.last_value)
        self.rollout_log_probs.append(self.last_log_prob)
        self.rollout_dones.append(done)

        # Update only when we have n_steps or episode ends
        if len(self.rollout_obs) < self.n_steps and not done:
            return {
                "policy_loss": self.last_policy_loss,
                "value_loss": self.last_value_loss,
                "entropy": self.last_entropy,
            }

        # Convert to numpy arrays
        obs_batch = np.array(self.rollout_obs)
        actions_batch = np.array(self.rollout_actions)
        rewards_batch = np.array(self.rollout_rewards)
        values_batch = np.array(self.rollout_values)
        log_probs_batch = np.array(self.rollout_log_probs)
        dones_batch = np.array(self.rollout_dones)

        # Compute next value for bootstrap
        next_obs_tensor = torch.FloatTensor(next_observation.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.ac_network(next_obs_tensor)
            next_value = next_value.item() * (1 - done)  # Zero out if terminal

        # Compute returns
        returns = self._compute_returns(rewards_batch, values_batch, dones_batch, next_value)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        actions_tensor = torch.LongTensor(actions_batch).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        values_tensor = torch.FloatTensor(values_batch).to(self.device)

        # Get current policy outputs
        action_logits, values = self.ac_network(obs_tensor)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()

        # Compute advantages
        advantages = returns_tensor - values_tensor

        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Store metrics
        self.last_policy_loss = policy_loss.item()
        self.last_value_loss = value_loss.item()
        self.last_entropy = entropy.item()

        # Clear rollout buffer
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_log_probs = []
        self.rollout_dones = []

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
        }

    def save(self, path: str):
        """Save policy to file."""
        torch.save({
            "ac_network": self.ac_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint["ac_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]

    def get_stats(self) -> Dict[str, float]:
        """Get policy statistics."""
        return {
            "total_steps": self.total_steps,
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "buffer_size": len(self.rollout_obs),
        }
