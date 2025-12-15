#!/usr/bin/env python3
"""
CartPole environment with noise blocks overlay for video recording.
Can be easily adapted to other Gymnasium environments (MuJoCo, MetaWorld, etc.)
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from uuid import uuid4
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Import RL policies
from policies import RandomPolicy, DQNPolicy, PPOPolicy, A2CPolicy, BasePolicy


# ============================================================================
# Part 1: NoiseBlock data structure
# ============================================================================

@dataclass
class NoiseBlock:
    """
    Represents a single noise block (a *field patch*) with position, velocity, field vector, and lifetime.

    Important semantics:
    - (ax, ay) represent the **field vector** carried by this patch (used for visualization + optional force-field).
    - The patch itself moves with (vx, vy). We intentionally do NOT use (ax, ay) to accelerate the patch,
      otherwise it looks like "the block is accelerating" rather than "the block is a noise/force field region".
    """
    x: float  # Current center x coordinate (pixels)
    y: float  # Current center y coordinate (pixels)
    vx: float  # Velocity in x direction
    vy: float  # Velocity in y direction
    ax: float  # Acceleration in x direction
    ay: float  # Acceleration in y direction
    size: int  # Block side length (pixels)
    color: np.ndarray  # RGB color (3,)
    age: int  # Number of frames this block has existed
    max_age: int  # Maximum lifetime in frames
    is_static: bool  # Whether this block moves or stays fixed


# ============================================================================
# Part 2: NoiseField - manages all noise blocks
# ============================================================================

class NoiseField:
    """
    Manages a collection of noise blocks with random spawning, physics updates,
    and rendering on top of frames.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        max_blocks: int = 20,
        spawn_prob: float = 0.05,
        static_ratio: float = 0.3,
        min_block_size: int = 10,
        max_block_size: int = 40,
        min_lifetime_frames: int = 30,
        max_lifetime_frames: int = 150,
        ax_global: float = 0.0,
        ay_global: float = 30.0,
        random_acceleration: bool = True,
        horizontal_only: bool = False,
        acc_magnitude_range: Tuple[float, float] = (10.0, 50.0),
        spawn_y_center: Optional[float] = None,
        spawn_y_half_range: Optional[float] = None,
    ):
        """
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second (for physics updates)
            max_blocks: Maximum number of blocks that can exist simultaneously
            spawn_prob: Probability of spawning a new block each frame
            static_ratio: Ratio of static blocks (0.3 = 30% static, 70% dynamic)
            min_block_size: Minimum block size in pixels
            max_block_size: Maximum block size in pixels
            min_lifetime_frames: Minimum block lifetime in frames
            max_lifetime_frames: Maximum block lifetime in frames
            ax_global: Global acceleration field in x direction (used if random_acceleration=False)
            ay_global: Global acceleration field in y direction (used if random_acceleration=False)
            random_acceleration: If True, each block gets random acceleration direction
            horizontal_only: If True, acceleration only in horizontal direction (ay=0)
            acc_magnitude_range: Range of acceleration magnitude for random accelerations
            spawn_y_center: If set (pixels), spawn blocks around this y center instead of anywhere on screen
            spawn_y_half_range: Half range (pixels). If set with spawn_y_center, spawn y ~ Uniform[center-half, center+half]
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps

        self.max_blocks = max_blocks
        self.spawn_prob = spawn_prob
        self.static_ratio = static_ratio

        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.min_lifetime_frames = min_lifetime_frames
        self.max_lifetime_frames = max_lifetime_frames

        self.ax_global = ax_global
        self.ay_global = ay_global
        self.random_acceleration = random_acceleration
        self.horizontal_only = horizontal_only
        self.acc_magnitude_range = acc_magnitude_range

        self.spawn_y_center = spawn_y_center
        self.spawn_y_half_range = spawn_y_half_range

        self.blocks: List[NoiseBlock] = []

    def _spawn_random_block(self) -> NoiseBlock:
        """Spawn a new random noise block within frame boundaries."""
        size = np.random.randint(self.min_block_size, self.max_block_size + 1)
        is_static = np.random.rand() < self.static_ratio

        # Random position (ensure block stays within frame)
        x = np.random.uniform(size / 2, self.width - size / 2)
        y_low = size / 2
        y_high = self.height - size / 2
        if (
            self.spawn_y_center is not None
            and self.spawn_y_half_range is not None
            and self.spawn_y_half_range >= 0
        ):
            band_low = float(self.spawn_y_center) - float(self.spawn_y_half_range)
            band_high = float(self.spawn_y_center) + float(self.spawn_y_half_range)
            # Clamp band to valid range for this size
            y0 = max(y_low, band_low)
            y1 = min(y_high, band_high)
            if y1 > y0:
                y = np.random.uniform(y0, y1)
            else:
                # Band too narrow for this block size -> fall back to full range
                y = np.random.uniform(y_low, y_high)
        else:
            y = np.random.uniform(y_low, y_high)

        # Initial velocity
        if is_static:
            vx, vy = 0.0, 0.0
        else:
            # Small random perturbation for dynamic blocks
            vx = np.random.uniform(-50, 50)
            vy = np.random.uniform(-50, 50)

        # Acceleration
        if self.random_acceleration:
            # Random acceleration direction with random magnitude
            acc_magnitude = np.random.uniform(self.acc_magnitude_range[0], self.acc_magnitude_range[1])

            if self.horizontal_only:
                # Only horizontal acceleration
                ax = np.random.choice([-1, 1]) * acc_magnitude
                ay = 0.0
            else:
                # Random direction in 2D
                angle = np.random.uniform(0, 2 * np.pi)
                ax = acc_magnitude * np.cos(angle)
                ay = acc_magnitude * np.sin(angle)
        else:
            # Use global acceleration field
            ax = self.ax_global
            ay = self.ay_global

        # Random color
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)

        # Random lifetime
        max_age = np.random.randint(self.min_lifetime_frames, self.max_lifetime_frames + 1)

        return NoiseBlock(
            x=x, y=y, vx=vx, vy=vy, ax=ax, ay=ay,
            size=size, color=color, age=0, max_age=max_age,
            is_static=is_static
        )

    def _update_blocks(self):
        """Update all blocks: age, position, velocity. Remove expired/out-of-bounds blocks."""
        blocks_to_keep = []

        for block in self.blocks:
            block.age += 1

            # Check if block has expired
            if block.age > block.max_age:
                continue

            # Update position and velocity for dynamic blocks
            if not block.is_static:
                # Update position (patch drifts with its own velocity; field vector (ax, ay) is NOT used here)
                block.x += block.vx * self.dt
                block.y += block.vy * self.dt

                # Check if block is out of bounds
                half_size = block.size / 2
                if (block.x + half_size < 0 or block.x - half_size > self.width or
                    block.y + half_size < 0 or block.y - half_size > self.height):
                    continue  # Block flew out of screen

            blocks_to_keep.append(block)

        self.blocks = blocks_to_keep

    def _spawn_new_blocks(self):
        """Randomly spawn new blocks based on spawn probability and max blocks limit."""
        if len(self.blocks) < self.max_blocks and np.random.rand() < self.spawn_prob:
            new_block = self._spawn_random_block()
            self.blocks.append(new_block)

    def tick(self):
        """
        Advance noise simulation by one environment step/frame:
        - update existing blocks (age/physics/out-of-bounds)
        - possibly spawn new blocks
        """
        self._update_blocks()
        self._spawn_new_blocks()

    def _draw_blocks(self, frame: np.ndarray) -> np.ndarray:
        """Draw all blocks onto the frame with field-direction arrows."""
        import cv2

        # Pass 1: draw all block squares (so later blocks don't overwrite arrows)
        for block in self.blocks:
            # Calculate block bounding box
            x0 = int(round(block.x - block.size / 2))
            x1 = int(round(block.x + block.size / 2))
            y0 = int(round(block.y - block.size / 2))
            y1 = int(round(block.y + block.size / 2))

            # Clip to frame boundaries
            x0 = max(0, min(x0, self.width))
            x1 = max(0, min(x1, self.width))
            y0 = max(0, min(y0, self.height))
            y1 = max(0, min(y1, self.height))

            # Draw block
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1, :] = block.color

        # Pass 2: draw all field arrows on top (so they're always visible)
        for block in self.blocks:
            if block.ax == 0 and block.ay == 0:
                continue

            center_x = int(round(block.x))
            center_y = int(round(block.y))

            # Keep arrow endpoints reasonable and visible:
            # length increases with field magnitude (not inverse), and can extend beyond the block.
            ax = float(block.ax)
            ay = float(block.ay)
            acc_magnitude = float(np.sqrt(ax * ax + ay * ay))
            if acc_magnitude < 1e-8:
                continue

            dir_x = ax / acc_magnitude
            dir_y = ay / acc_magnitude

            min_len = 10.0
            max_len = max(20.0, float(block.size) * 1.5)
            arrow_len = float(np.clip(acc_magnitude * 0.6, min_len, max_len))

            end_x = int(round(center_x + dir_x * arrow_len))
            end_y = int(round(center_y + dir_y * arrow_len))

            # Choose contrasting color for arrow (white or black)
            avg_color = float(np.mean(block.color))
            arrow_color = (255, 255, 255) if avg_color < 128 else (0, 0, 0)

            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (end_x, end_y),
                arrow_color,
                thickness=2,
                tipLength=0.3,
            )

        return frame

    def update_and_draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Main entry point: update all blocks and draw them on the frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3), uint8

        Returns:
            Modified frame with noise blocks drawn on top
        """
        self.tick()
        frame = self._draw_blocks(frame)
        return frame


# ============================================================================
# Part 3: Reward wrappers and NoiseOverlayWrapper
# ============================================================================

class ExplorationRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds exploration reward to encourage the agent to move around.
    Tracks cart position and rewards visiting new areas.
    """

    def __init__(
        self,
        env: gym.Env,
        exploration_weight: float = 0.1,
        position_bins: int = 50,
        position_range: tuple = (-2.4, 2.4),
    ):
        """
        Args:
            env: Base environment
            exploration_weight: Weight for exploration reward component
            position_bins: Number of bins to discretize position space
            position_range: Range of cart position (min, max)
        """
        super().__init__(env)
        self.exploration_weight = exploration_weight
        self.position_bins = position_bins
        self.position_range = position_range

        # Track visited positions
        self.visited_bins = set()
        self.position_history = []

        # Calculate bin width
        self.bin_width = (position_range[1] - position_range[0]) / position_bins

    def _get_position_bin(self, position: float) -> int:
        """Convert continuous position to discrete bin index."""
        normalized = (position - self.position_range[0]) / (self.position_range[1] - self.position_range[0])
        bin_idx = int(normalized * self.position_bins)
        return max(0, min(self.position_bins - 1, bin_idx))

    def _calculate_exploration_reward(self, obs: np.ndarray) -> float:
        """
        Calculate exploration reward based on:
        1. Visiting new positions (novelty)
        2. Position variance (encouraging movement)
        """
        # Extract cart position (first element in CartPole observation)
        cart_position = float(obs[0])
        self.position_history.append(cart_position)

        # Novelty reward: bonus for visiting new bins
        current_bin = self._get_position_bin(cart_position)
        novelty_reward = 0.0
        if current_bin not in self.visited_bins:
            self.visited_bins.add(current_bin)
            novelty_reward = 1.0

        # Movement reward: encourage position variance
        if len(self.position_history) >= 10:
            recent_positions = self.position_history[-10:]
            position_variance = np.var(recent_positions)
            movement_reward = np.clip(position_variance * 5.0, 0.0, 1.0)
        else:
            movement_reward = 0.0

        # Combine rewards
        exploration_reward = (novelty_reward + movement_reward) * self.exploration_weight
        return exploration_reward

    def reset(self, **kwargs):
        """Reset environment and exploration tracking."""
        self.visited_bins = set()
        self.position_history = []
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step environment and add exploration reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add exploration reward
        exploration_reward = self._calculate_exploration_reward(obs)
        modified_reward = reward + exploration_reward

        # Store reward components in info
        info['original_reward'] = reward
        info['exploration_reward'] = exploration_reward
        info['total_reward'] = modified_reward

        return obs, modified_reward, terminated, truncated, info


class NoiseOverlayWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that overlays noise blocks on top of rendered frames.
    """

    def __init__(
        self,
        env: gym.Env,
        noise_field: NoiseField,
        converter=None,
        show_cart_marker: bool = True,
        cart_marker_radius: int = 4,
    ):
        """
        Args:
            env: Base Gymnasium environment
            noise_field: NoiseField instance to manage noise blocks
            converter: Optional CoordinateConverter (for CartPole marker)
            show_cart_marker: Whether to draw a marker for cart position
            cart_marker_radius: Radius (pixels) for cart marker
        """
        super().__init__(env)
        self.noise_field = noise_field
        self.converter = converter
        self.show_cart_marker = show_cart_marker
        self.cart_marker_radius = cart_marker_radius

    def render(self):
        """Override render to add noise overlay."""
        frame = self.env.render()
        if frame is None:
            return None
        # Only draw. Noise simulation is advanced in step() wrappers.
        frame = self.noise_field._draw_blocks(frame)

        # Optional: draw cart marker (blue dot) at the cart's current x position.
        if self.show_cart_marker and self.converter is not None:
            try:
                import cv2

                state = getattr(self.env.unwrapped, "state", None)
                if state is not None and len(state) >= 1:
                    cart_x = float(state[0])
                    cart_x_px = int(round(float(self.converter.cart_to_pixel_x(cart_x))))
                    cart_y_px = int(round(float(self.converter.cart_y_pixel)))
                    # Blue in OpenCV BGR
                    cv2.circle(frame, (cart_x_px, cart_y_px), int(self.cart_marker_radius), (255, 0, 0), -1)
            except Exception:
                pass
        return frame


class NoiseFieldStepWrapper(gym.Wrapper):
    """
    Advance NoiseField on every env.step(), so noise exists during training even without rendering.
    """

    def __init__(self, env: gym.Env, noise_field: NoiseField):
        super().__init__(env)
        self.noise_field = noise_field

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.noise_field.tick()
        return obs, reward, terminated, truncated, info


# ============================================================================
# Part 4: Environment setup and headless display
# ============================================================================

def ensure_headless_display(width: int = 1280, height: int = 720):
    """
    Start virtual display (Xvfb) for headless environments.
    Prints warning if fails but doesn't exit.
    """
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(width, height))
        display.start()
        print(f"[OK] Virtual display started: {width}x{height}")
    except ImportError:
        warnings.warn("pyvirtualdisplay not available. Skipping virtual display setup.")
    except Exception as e:
        warnings.warn(f"Failed to start virtual display: {e}")


def make_env(
    env_name: str = "CartPole-v1",
    render_mode: str = "rgb_array",
    noise_field_kwargs: Optional[dict] = None,
    max_episode_steps: Optional[int] = None,
    enable_exploration_reward: bool = False,
    exploration_weight: float = 0.1,
    enable_plan_b: bool = False,
    enable_plan_b_reward: bool = True,
    k_nearest: int = 5,
    collision_penalty: float = -5.0,
    distance_weight: float = 0.1,
    enable_force_field: bool = False,
    field_strength: float = 1.0,
    field_pole_strength: float = 0.0,
    show_cart_marker: bool = True,
) -> gym.Env:
    """
    Create Gymnasium environment with noise overlay and optional exploration reward.

    Args:
        env_name: Name of the Gymnasium environment
        render_mode: Render mode (should be "rgb_array" for video recording)
        noise_field_kwargs: Keyword arguments for NoiseField initialization
        max_episode_steps: Maximum steps per episode (None = use default)
        enable_exploration_reward: Whether to add exploration reward
        exploration_weight: Weight for exploration reward (if enabled)
        enable_plan_b: Whether to enable Plan B (noise avoidance with features)
        k_nearest: Number of nearest blocks to include in observation (Plan B)
        collision_penalty: Penalty for collision with noise blocks (Plan B)
        distance_weight: Weight for distance reward (Plan B)

    Returns:
        Environment wrapped with NoiseOverlayWrapper and optionally ExplorationRewardWrapper
    """
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)

    # Add exploration reward wrapper first (before noise overlay)
    if enable_exploration_reward:
        env = ExplorationRewardWrapper(env, exploration_weight=exploration_weight)

    # Get frame dimensions from environment.
    # IMPORTANT: rendering classic-control envs may require pygame; avoid hard dependency for training.
    env.reset()  # Reset returns (obs, info)

    width = getattr(env.unwrapped, "screen_width", None)
    height = getattr(env.unwrapped, "screen_height", None)

    if width is None or height is None:
        try:
            test_frame = env.render()
            if test_frame is not None:
                height, width = test_frame.shape[:2]
        except Exception as e:
            warnings.warn(f"Render unavailable for frame size detection ({e}). Falling back to defaults.")

    if width is None or height is None:
        # Fallback dimensions
        width, height = 640, 480
        warnings.warn(f"Could not determine frame size, using default {width}x{height}")

    # Coordinate converter is useful for Plan B features, the force-field, and cart marker.
    converter = None
    if enable_plan_b or enable_force_field or env_name.lower().startswith("cartpole"):
        from noise_avoidance.wrappers import CoordinateConverter

        converter = CoordinateConverter(
            frame_width=width,
            frame_height=height,
            cart_range=(-2.4, 2.4),
        )

    # Create noise field
    if noise_field_kwargs is None:
        noise_field_kwargs = {}
    # If using strict x+y in-square force-field, default to spawning blocks near the cart's y line (±2 pixels)
    if enable_force_field:
        noise_field_kwargs = dict(noise_field_kwargs)
        noise_field_kwargs.setdefault("spawn_y_center", float(converter.cart_y_pixel) if converter is not None else None)
        noise_field_kwargs.setdefault("spawn_y_half_range", 2.0)

    noise_field = NoiseField(width=width, height=height, **noise_field_kwargs)

    # Advance noise on every step (so it exists during training even without render calls).
    # If we install a CartPole dynamics wrapper that ticks noise itself, we must avoid double-tick.
    if enable_force_field and env_name.lower().startswith("cartpole"):
        # Apply field *inside* CartPole dynamics (pre-integration). Must happen BEFORE ObservationWrappers
        # like NoiseFeaturesWrapper, otherwise we'd return 4D obs and bypass the 19D feature augmentation.
        from noise_avoidance.wrappers import CartPoleNoiseForceFieldDynamicsWrapper

        env = CartPoleNoiseForceFieldDynamicsWrapper(
            env,
            noise_field=noise_field,
            converter=converter,
            cart_strength=field_strength,
            pole_strength=field_pole_strength,
            max_episode_steps=max_episode_steps,
        )
    else:
        env = NoiseFieldStepWrapper(env, noise_field)

    # Plan B: Add noise features to observation and reward shaping
    if enable_plan_b:
        from noise_avoidance.wrappers import (
            NoiseFeaturesWrapper,
            NoiseAvoidanceRewardWrapper,
        )

        # Add noise features to observation
        env = NoiseFeaturesWrapper(
            env,
            noise_field=noise_field,
            converter=converter,
            k_nearest=k_nearest,
        )

        # Add noise avoidance reward (optional)
        if enable_plan_b_reward:
            env = NoiseAvoidanceRewardWrapper(
                env,
                noise_field=noise_field,
                converter=converter,
                collision_penalty=collision_penalty,
                distance_weight=distance_weight,
                terminate_on_collision=False,
            )

    # Wrap with noise overlay for rendering
    env = NoiseOverlayWrapper(env, noise_field, converter=converter, show_cart_marker=show_cart_marker)

    return env


# ============================================================================
# Part 5: Policy creation
# ============================================================================

def create_policy(
    policy_name: str,
    env: gym.Env,
    device: str = "cpu",
    **kwargs,
) -> BasePolicy:
    """
    Create RL policy based on name.

    Args:
        policy_name: Name of policy ("random", "dqn", "ppo", "a2c")
        env: Environment
        device: Device to run on ("cpu" or "cuda")
        **kwargs: Additional policy-specific arguments

    Returns:
        Initialized policy
    """
    policy_name = policy_name.lower()

    if policy_name == "random":
        return RandomPolicy(env.observation_space, env.action_space)
    elif policy_name == "dqn":
        return DQNPolicy(env.observation_space, env.action_space, device=device, **kwargs)
    elif policy_name == "ppo":
        return PPOPolicy(env.observation_space, env.action_space, device=device, **kwargs)
    elif policy_name == "a2c":
        return A2CPolicy(env.observation_space, env.action_space, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown policy: {policy_name}. Choose from: random, dqn, ppo, a2c")


# ============================================================================
# Part 6: Training with online RL
# ============================================================================

def train_with_policy(
    env_name: str = "CartPole-v1",
    policy_name: str = "random",
    steps: int = 10000,
    seed: int = 0,
    noise_field_kwargs: Optional[dict] = None,
    save_path: Optional[str] = None,
    log_interval: int = 1000,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    enable_exploration_reward: bool = False,
    exploration_weight: float = 0.1,
    enable_plan_b: bool = False,
    enable_plan_b_reward: bool = True,
    k_nearest: int = 5,
    collision_penalty: float = -5.0,
    distance_weight: float = 0.1,
    enable_force_field: bool = False,
    field_strength: float = 1.0,
    field_pole_strength: float = 0.0,
    show_cart_marker: bool = True,
):
    """
    Train policy on environment with noise blocks.

    Args:
        env_name: Name of the Gymnasium environment
        policy_name: Name of policy to use
        steps: Number of training steps
        seed: Random seed
        noise_field_kwargs: Keyword arguments for NoiseField
        save_path: Path to save trained policy (if provided)
        log_interval: Steps between logging
        device: Device to run on
        max_episode_steps: Maximum steps per episode (None = use default)
        enable_exploration_reward: Whether to add exploration reward
        exploration_weight: Weight for exploration reward
        enable_plan_b: Whether to enable Plan B (noise avoidance features)
        k_nearest: Number of nearest blocks (Plan B)
        collision_penalty: Collision penalty (Plan B)
        distance_weight: Distance reward weight (Plan B)
    """
    # Create environment with noise overlay
    env = make_env(
        env_name,
        render_mode="rgb_array",
        noise_field_kwargs=noise_field_kwargs,
        max_episode_steps=max_episode_steps,
        enable_exploration_reward=enable_exploration_reward,
        exploration_weight=exploration_weight,
        enable_plan_b=enable_plan_b,
        enable_plan_b_reward=enable_plan_b_reward,
        k_nearest=k_nearest,
        collision_penalty=collision_penalty,
        distance_weight=distance_weight,
        enable_force_field=enable_force_field,
        field_strength=field_strength,
        field_pole_strength=field_pole_strength,
        show_cart_marker=show_cart_marker,
    )

    # Create policy
    policy = create_policy(policy_name, env, device=device)

    print(f"\nTraining {policy_name.upper()} on {env_name}...")
    print(f"Total steps: {steps}")
    print(f"Device: {device}")

    # Training loop
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    episode_reward = 0.0
    episode_count = 0
    episode_steps = 0

    for t in range(steps):
        # Select action
        action = policy.select_action(obs, training=True)

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        episode_steps += 1

        # Update policy
        metrics = policy.update(obs, action, reward, next_obs, terminated, truncated)

        # Check if episode ended
        if terminated or truncated:
            total_reward += episode_reward
            episode_count += 1

            if episode_count % 300 == 0:
                avg_reward = total_reward / episode_count
                print(f"Episode {episode_count}: reward={episode_reward:.2f}, avg_reward={avg_reward:.2f}, steps={episode_steps}")

            episode_reward = 0.0
            episode_steps = 0
            obs, info = env.reset()
        else:
            obs = next_obs

        # Log metrics
        if (t + 1) % log_interval == 0:
            stats = policy.get_stats()
            print(f"\nStep {t+1}/{steps}:")
            print(f"  Episodes: {episode_count}")
            print(f"  Avg reward: {total_reward / max(1, episode_count):.2f}")
            for key, value in stats.items():
                if key != "total_steps":
                    print(f"  {key}: {value:.4f}")

    env.close()

    # Save policy if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        policy.save(save_path)
        print(f"\n[OK] Policy saved to: {save_path}")

    print(f"\n[OK] Training complete!")
    print(f"  Total episodes: {episode_count}")
    print(f"  Average reward: {total_reward / max(1, episode_count):.2f}")

    return policy


# ============================================================================
# Part 7: Video recording with policy
# ============================================================================

def record_video_with_policy(
    env_name: str = "CartPole-v1",
    policy_name: str = "random",
    outdir: str = "videos",
    prefix: str = "env_with_noise",
    steps: int = 1000,
    seed: int = 0,
    noise_field_kwargs: Optional[dict] = None,
    policy_path: Optional[str] = None,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    enable_exploration_reward: bool = False,
    exploration_weight: float = 0.1,
    enable_plan_b: bool = False,
    enable_plan_b_reward: bool = True,
    k_nearest: int = 5,
    collision_penalty: float = -5.0,
    distance_weight: float = 0.1,
    enable_force_field: bool = False,
    field_strength: float = 1.0,
    field_pole_strength: float = 0.0,
    record_only_first_episode: bool = False,
    record_every_n_episodes: int = 1,
    video_length: int = 0,
    record_episodes: int = 0,
    min_episode_reward_to_save: float = 0.0,
):
    """
    Record video of environment with noise blocks overlay using a trained policy.

    Args:
        env_name: Name of the Gymnasium environment
        policy_name: Name of policy to use
        outdir: Output directory for videos
        prefix: Prefix for video filename
        steps: Number of steps to run
        seed: Random seed
        noise_field_kwargs: Keyword arguments for NoiseField
        policy_path: Path to load pre-trained policy (if provided)
        device: Device to run on
        max_episode_steps: Maximum steps per episode (None = use default)
        enable_exploration_reward: Whether to add exploration reward
        exploration_weight: Weight for exploration reward
        enable_plan_b: Whether to enable Plan B (noise avoidance features)
        k_nearest: Number of nearest blocks (Plan B)
        collision_penalty: Collision penalty (Plan B)
        distance_weight: Distance reward weight (Plan B)
    """
    from gymnasium.wrappers import RecordVideo

    # Create output directory
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # If we want to conditionally keep videos based on episode reward, record into a temp folder
    # and only move kept episodes into outdir. This avoids issues with overwriting same filenames
    # (set-diff snapshots can't detect overwrites).
    use_temp_video_dir = float(min_episode_reward_to_save) > 0.0
    temp_video_dir = outdir_path / f".tmp_recordings_{uuid4().hex}"
    video_folder = temp_video_dir if use_temp_video_dir else outdir_path
    if use_temp_video_dir:
        temp_video_dir.mkdir(parents=True, exist_ok=True)

    # Create environment with noise overlay
    env = make_env(
        env_name,
        render_mode="rgb_array",
        noise_field_kwargs=noise_field_kwargs,
        max_episode_steps=max_episode_steps,
        enable_exploration_reward=enable_exploration_reward,
        exploration_weight=exploration_weight,
        enable_plan_b=enable_plan_b,
        enable_plan_b_reward=enable_plan_b_reward,
        k_nearest=k_nearest,
        collision_penalty=collision_penalty,
        distance_weight=distance_weight,
        enable_force_field=enable_force_field,
        field_strength=field_strength,
        field_pole_strength=field_pole_strength,
        show_cart_marker=True,
    )

    # Wrap with RecordVideo
    if record_only_first_episode:
        episode_trigger = lambda ep: ep == 0  # noqa: E731
    else:
        n = max(1, int(record_every_n_episodes))
        episode_trigger = (lambda ep, n=n: (ep % n) == 0)  # noqa: E731

    env = RecordVideo(
        env,
        video_folder=str(video_folder),
        name_prefix=prefix,
        episode_trigger=episode_trigger,
        video_length=max(0, int(video_length)),
    )

    # Create policy
    policy = create_policy(policy_name, env, device=device)

    # Load pre-trained weights if provided
    if policy_path:
        policy.load(policy_path)
        print(f"Loaded policy from: {policy_path}")

    def _try_unlink(p: Path, retries: int = 10, delay_s: float = 0.05):
        for _ in range(retries):
            try:
                p.unlink(missing_ok=True)
                return True
            except Exception:
                time.sleep(delay_s)
        return False

    def _try_move(src: Path, dst: Path, retries: int = 10, delay_s: float = 0.05):
        for _ in range(retries):
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    _try_unlink(dst)
                shutil.move(str(src), str(dst))
                return True
            except Exception:
                time.sleep(delay_s)
        return False

    # Run environment with policy
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    episode_count = 0
    episode_reward = 0.0

    # Track newly created video files per episode so we can conditionally delete/move them.
    def _snapshot_video_files() -> set:
        try:
            return set(video_folder.glob(f"{prefix}-episode-*.*"))
        except Exception:
            return set()

    prev_files = _snapshot_video_files()

    for t in range(steps):
        action = policy.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        episode_reward += float(reward)

        if terminated or truncated:
            # IMPORTANT: Gymnasium's RecordVideo typically finalizes/writes the episode video on reset().
            # Snapshot BEFORE reset, then reset (flush), then snapshot AFTER reset to get the files produced
            # by the just-finished episode.
            files_before_reset = _snapshot_video_files()

            obs, info = env.reset()

            files_after_reset = _snapshot_video_files()
            new_files = files_after_reset - files_before_reset

            if use_temp_video_dir:
                keep = episode_reward >= float(min_episode_reward_to_save)
                moved = 0
                deleted = 0

                if keep:
                    for p in new_files:
                        target = outdir_path / p.name
                        if _try_move(p, target):
                            moved += 1
                else:
                    for p in new_files:
                        if _try_unlink(p):
                            deleted += 1

                print(
                    f"[record] episode_end reward={episode_reward:.2f} "
                    f"new_files={len(new_files)} action={'KEEP' if keep else 'DROP'} "
                    f"moved={moved} deleted={deleted}"
                )
            else:
                if float(min_episode_reward_to_save) > 0.0 and episode_reward < float(min_episode_reward_to_save):
                    for p in new_files:
                        _try_unlink(p)
            episode_count += 1
            episode_reward = 0.0
            prev_files = _snapshot_video_files()
            if int(record_episodes) > 0 and episode_count >= int(record_episodes):
                break

    env.close()

    # Best-effort cleanup of temp folder if used and now empty
    if use_temp_video_dir:
        try:
            # Do NOT delete remaining files here; that can destroy kept episodes if moves failed.
            # Only remove the temp dir if it's already empty.
            if not any(temp_video_dir.iterdir()):
                temp_video_dir.rmdir()
        except Exception:
            pass

    print(f"\n[OK] Video recording complete!")
    print(f"  Output: {outdir}")
    print(f"  Prefix: {prefix}")
    print(f"  Policy: {policy_name}")
    print(f"  Steps: {steps}")
    print(f"  Episodes: {episode_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Avg reward per episode: {total_reward / max(1, episode_count):.2f}")


# ============================================================================
# Part 8: Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train/record Gymnasium environment with noise blocks and RL policies"
    )

    # Mode selection
    parser.add_argument("--mode", type=str, default="record", choices=["train", "record"],
                        help="Mode: 'train' to train policy, 'record' to record video")

    # Environment settings
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment name")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="Maximum steps per episode (None = use environment default)")

    # Policy settings
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "dqn", "ppo", "a2c"],
                        help="RL policy to use")
    parser.add_argument("--policy-path", type=str, default=None,
                        help="Path to load/save policy weights")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run on")

    # Training settings
    parser.add_argument("--log-interval", type=int, default=10000,
                        help="Steps between logging (training mode)")

    # Reward settings
    parser.add_argument("--enable-exploration-reward", action="store_true",
                        help="Enable exploration reward to encourage movement")
    parser.add_argument("--exploration-weight", type=float, default=0.1,
                        help="Weight for exploration reward (default: 0.1)")

    # Plan B settings (noise avoidance with features)
    parser.add_argument("--plan-b", action="store_true",
                        help="Enable Plan B: noise avoidance with low-dim features (obs becomes 19D)")
    parser.add_argument("--no-plan-b-reward", action="store_true",
                        help="Plan B features only: disable collision/distance reward shaping (no 'collision' concept)")
    parser.add_argument("--k-nearest", type=int, default=5,
                        help="Number of nearest noise blocks to track (Plan B)")
    parser.add_argument("--collision-penalty", type=float, default=-5.0,
                        help="Penalty for collision with noise blocks (Plan B)")
    parser.add_argument("--distance-reward-weight", type=float, default=0.1,
                        help="Weight for distance reward from blocks (Plan B)")

    # Physical force-field settings (optional, works with --plan-b)
    parser.add_argument("--force-field", action="store_true",
                        help="Enable force field: noise blocks apply horizontal acceleration to the cart when nearby")
    parser.add_argument("--field-strength", type=float, default=1.0,
                        help="Overall strength multiplier for block acceleration applied to the cart")
    parser.add_argument("--field-pole-strength", type=float, default=0.0,
                        help="Additional strength multiplier applied to pole angular velocity disturbance (0 = disabled)")

    # Output settings
    parser.add_argument("--outdir", type=str, default="videos",
                        help="Output directory for videos (record mode)")
    parser.add_argument("--prefix", type=str, default="env_with_noise",
                        help="Video filename prefix (record mode)")
    parser.add_argument("--record-only-first-episode", action="store_true",
                        help="Record only the first episode (record mode)")
    parser.add_argument("--record-every-n-episodes", type=int, default=1,
                        help="Record every N episodes (record mode). Ignored if --record-only-first-episode is set")
    parser.add_argument("--video-length", type=int, default=0,
                        help="Fixed video length in steps (0 = full episode) (record mode)")
    parser.add_argument("--record-episodes", type=int, default=0,
                        help="Stop recording after N episodes (0 = no limit, controlled by --steps) (record mode)")
    parser.add_argument("--min-episode-reward-to-save", type=float, default=0.0,
                        help="If > 0, discard (delete) recorded episode videos whose total episode reward is below this threshold")
    parser.add_argument("--no-cart-marker", action="store_true",
                        help="Disable drawing blue dot marker for cart x position on rendered frames")

    # Noise field parameters
    parser.add_argument("--max-blocks", type=int, default=20,
                        help="Maximum number of noise blocks")
    parser.add_argument("--spawn-prob", type=float, default=0.05,
                        help="Probability of spawning a block each frame")
    parser.add_argument("--spawn-y-band-half", type=float, default=None,
                        help="If set, spawn blocks only near the cart y-line: y in [cart_y - band, cart_y + band] (pixels). Example: 2.0")
    parser.add_argument("--static-ratio", type=float, default=0.3,
                        help="Ratio of static blocks (0.0-1.0)")
    parser.add_argument("--min-block-size", type=int, default=10,
                        help="Minimum block size in pixels")
    parser.add_argument("--max-block-size", type=int, default=40,
                        help="Maximum block size in pixels")
    parser.add_argument("--min-lifetime", type=int, default=30,
                        help="Minimum block lifetime in frames")
    parser.add_argument("--max-lifetime", type=int, default=150,
                        help="Maximum block lifetime in frames")
    parser.add_argument("--ax", type=float, default=0.0,
                        help="Global acceleration in x direction (used when --no-random-acceleration)")
    parser.add_argument("--ay", type=float, default=30.0,
                        help="Global acceleration in y direction (used when --no-random-acceleration)")
    parser.add_argument("--no-random-acceleration", action="store_true",
                        help="Use global acceleration field instead of random per-block accelerations")
    parser.add_argument("--horizontal-only", action="store_true",
                        help="Restrict accelerations to horizontal direction only (no vertical component)")
    parser.add_argument("--acc-min", type=float, default=10.0,
                        help="Minimum acceleration magnitude for random accelerations")
    parser.add_argument("--acc-max", type=float, default=50.0,
                        help="Maximum acceleration magnitude for random accelerations")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second (for physics)")

    # Display settings
    parser.add_argument("--no-headless", action="store_true",
                        help="Don't start virtual display (use existing display)")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Start headless display if needed
    if not args.no_headless:
        ensure_headless_display()

    # Prepare noise field parameters
    noise_field_kwargs = {
        "fps": args.fps,
        "max_blocks": args.max_blocks,
        "spawn_prob": args.spawn_prob,
        "static_ratio": args.static_ratio,
        "min_block_size": args.min_block_size,
        "max_block_size": args.max_block_size,
        "min_lifetime_frames": args.min_lifetime,
        "max_lifetime_frames": args.max_lifetime,
        "ax_global": args.ax,
        "ay_global": args.ay,
        "random_acceleration": not args.no_random_acceleration,
        "horizontal_only": args.horizontal_only,
        "acc_magnitude_range": (args.acc_min, args.acc_max),
    }
    if args.spawn_y_band_half is not None:
        # Center will be filled in make_env() from cart_y_pixel; we only pass the half-range here.
        noise_field_kwargs["spawn_y_half_range"] = float(args.spawn_y_band_half)

    print(f"Environment: {args.env}")
    print(f"Policy: {args.policy.upper()}")
    print(f"Noise configuration:")
    print(f"  Max blocks: {args.max_blocks}")
    print(f"  Spawn probability: {args.spawn_prob}")
    print(f"  Static ratio: {args.static_ratio}")
    if args.no_random_acceleration:
        print(f"  Acceleration: ({args.ax}, {args.ay}) [global field]")
    else:
        print(f"  Acceleration: random, magnitude={args.acc_min}-{args.acc_max}")
        if args.horizontal_only:
            print(f"  Direction: horizontal only")
    if args.enable_exploration_reward:
        print(f"Exploration reward: ENABLED (weight={args.exploration_weight})")
    if args.plan_b:
        print(f"Plan B (noise avoidance): ENABLED")
        print(f"  K nearest blocks: {args.k_nearest}")
        if args.no_plan_b_reward:
            print(f"  Plan B reward: DISABLED (features only)")
        else:
            print(f"  Collision penalty: {args.collision_penalty}")
            print(f"  Distance reward weight: {args.distance_reward_weight}")
        print(f"  Observation dimension: 4 + {args.k_nearest}*3 = {4 + args.k_nearest * 3}")
        if args.force_field:
            print(f"  Force field: ENABLED (radius = block size, cart_strength={args.field_strength}, pole_strength={args.field_pole_strength})")

    # Run based on mode
    if args.mode == "train":
        train_with_policy(
            env_name=args.env,
            policy_name=args.policy,
            steps=args.steps,
            seed=args.seed,
            noise_field_kwargs=noise_field_kwargs,
            save_path=args.policy_path,
            log_interval=args.log_interval,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            enable_exploration_reward=args.enable_exploration_reward,
            exploration_weight=args.exploration_weight,
            enable_plan_b=args.plan_b,
            enable_plan_b_reward=(args.plan_b and (not args.no_plan_b_reward)),
            k_nearest=args.k_nearest,
            collision_penalty=args.collision_penalty,
            distance_weight=args.distance_reward_weight,
            enable_force_field=args.force_field,
            field_strength=args.field_strength,
            field_pole_strength=args.field_pole_strength,
            show_cart_marker=(not args.no_cart_marker),
        )
    elif args.mode == "record":
        record_video_with_policy(
            env_name=args.env,
            policy_name=args.policy,
            outdir=args.outdir,
            prefix=args.prefix,
            steps=args.steps,
            seed=args.seed,
            noise_field_kwargs=noise_field_kwargs,
            policy_path=args.policy_path,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            enable_exploration_reward=args.enable_exploration_reward,
            exploration_weight=args.exploration_weight,
            enable_plan_b=args.plan_b,
            enable_plan_b_reward=(args.plan_b and (not args.no_plan_b_reward)),
            k_nearest=args.k_nearest,
            collision_penalty=args.collision_penalty,
            distance_weight=args.distance_reward_weight,
            enable_force_field=args.force_field,
            field_strength=args.field_strength,
            field_pole_strength=args.field_pole_strength,
            record_only_first_episode=args.record_only_first_episode,
            record_every_n_episodes=args.record_every_n_episodes,
            video_length=args.video_length,
            record_episodes=args.record_episodes,
            min_episode_reward_to_save=args.min_episode_reward_to_save,
            # marker is part of env creation
        )


if __name__ == "__main__":
    main()
