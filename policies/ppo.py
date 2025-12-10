"""
Proximal Policy Optimization (PPO) policy implementation.
"""

from typing import Dict, List, Tuple
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .base import BasePolicy


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

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


class PPOBuffer:
    """Rollout buffer for PPO."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def push(self, obs, action, reward, value, log_prob, done):
        """Add transition to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self) -> Tuple:
        """Get all transitions and clear buffer."""
        data = (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones),
        )
        self.clear()
        return data

    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        return len(self.observations)


class PPOPolicy(BasePolicy):
    """
    Proximal Policy Optimization (PPO) policy with clipped objective.

    Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        rollout_length: int = 2048,
        device: str = "cpu",
    ):
        """
        Args:
            observation_space: Observation space
            action_space: Action space (must be Discrete)
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs to update policy per rollout
            minibatch_size: Minibatch size for updates
            rollout_length: Length of rollout before update
            device: Device to run on ("cpu" or "cuda")
        """
        super().__init__(observation_space, action_space)

        assert isinstance(action_space, gym.spaces.Discrete), "PPO only supports Discrete action space"

        self.obs_dim = np.prod(observation_space.shape)
        self.action_dim = action_space.n
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_length = rollout_length
        self.device = torch.device(device)

        # Actor-Critic network
        self.ac_network = ActorCritic(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = PPOBuffer()

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

    def _compute_gae(self, rewards, values, dones, next_value) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        """Update policy using PPO algorithm."""
        # Add to buffer
        done = terminated or truncated
        self.buffer.push(
            observation.flatten(),
            action,
            reward,
            self.last_value,
            self.last_log_prob,
            done
        )

        # Update only when buffer is full
        if len(self.buffer) < self.rollout_length:
            return {
                "policy_loss": self.last_policy_loss,
                "value_loss": self.last_value_loss,
                "entropy": self.last_entropy,
            }

        # Get rollout data
        obs_batch, actions_batch, rewards_batch, values_batch, log_probs_batch, dones_batch = \
            self.buffer.get()

        # Compute next value for GAE
        next_obs_tensor = torch.FloatTensor(next_observation.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.ac_network(next_obs_tensor)
            next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards_batch, values_batch, dones_batch, next_value)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        actions_tensor = torch.LongTensor(actions_batch).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs_batch).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Update policy for multiple epochs
        for _ in range(self.update_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(obs_batch))

            for start in range(0, len(obs_batch), self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                # Get current policy outputs
                action_logits, values = self.ac_network(mb_obs)
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss with PPO clipping
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)

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
            "buffer_size": len(self.buffer),
        }
