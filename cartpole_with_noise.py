#!/usr/bin/env python3
"""
CartPole environment with noise blocks overlay for video recording.
Can be easily adapted to other Gymnasium environments (MuJoCo, MetaWorld, etc.)
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
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
    """Represents a single noise block with position, velocity, acceleration, and lifetime."""
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
            ax_global: Global acceleration field in x direction
            ay_global: Global acceleration field in y direction
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

        self.blocks: List[NoiseBlock] = []

    def _spawn_random_block(self) -> NoiseBlock:
        """Spawn a new random noise block within frame boundaries."""
        size = np.random.randint(self.min_block_size, self.max_block_size + 1)
        is_static = np.random.rand() < self.static_ratio

        # Random position (ensure block stays within frame)
        x = np.random.uniform(size / 2, self.width - size / 2)
        y = np.random.uniform(size / 2, self.height - size / 2)

        # Initial velocity
        if is_static:
            vx, vy = 0.0, 0.0
        else:
            # Small random perturbation for dynamic blocks
            vx = np.random.uniform(-50, 50)
            vy = np.random.uniform(-50, 50)

        # Acceleration from global field
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
                # Update velocity
                block.vx += block.ax * self.dt
                block.vy += block.ay * self.dt

                # Update position
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

    def _draw_blocks(self, frame: np.ndarray) -> np.ndarray:
        """Draw all blocks onto the frame."""
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

        return frame

    def update_and_draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Main entry point: update all blocks and draw them on the frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3), uint8

        Returns:
            Modified frame with noise blocks drawn on top
        """
        self._update_blocks()
        self._spawn_new_blocks()
        frame = self._draw_blocks(frame)
        return frame


# ============================================================================
# Part 3: NoiseOverlayWrapper - Gymnasium wrapper
# ============================================================================

class NoiseOverlayWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that overlays noise blocks on top of rendered frames.
    """

    def __init__(self, env: gym.Env, noise_field: NoiseField):
        """
        Args:
            env: Base Gymnasium environment
            noise_field: NoiseField instance to manage noise blocks
        """
        super().__init__(env)
        self.noise_field = noise_field

    def render(self):
        """Override render to add noise overlay."""
        frame = self.env.render()
        if frame is None:
            return None
        # Add noise blocks on top of the frame
        frame = self.noise_field.update_and_draw(frame)
        return frame


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
        print(f"✓ Virtual display started: {width}x{height}")
    except ImportError:
        warnings.warn("pyvirtualdisplay not available. Skipping virtual display setup.")
    except Exception as e:
        warnings.warn(f"Failed to start virtual display: {e}")


def make_env(
    env_name: str = "CartPole-v1",
    render_mode: str = "rgb_array",
    noise_field_kwargs: Optional[dict] = None,
) -> gym.Env:
    """
    Create Gymnasium environment with noise overlay.

    Args:
        env_name: Name of the Gymnasium environment
        render_mode: Render mode (should be "rgb_array" for video recording)
        noise_field_kwargs: Keyword arguments for NoiseField initialization

    Returns:
        Environment wrapped with NoiseOverlayWrapper
    """
    env = gym.make(env_name, render_mode=render_mode)

    # Get frame dimensions from environment
    # For CartPole and most envs, we can render once to get shape
    frame = env.reset()[0]  # Reset returns (obs, info)
    env.render()  # Force render to initialize
    test_frame = env.render()
    if test_frame is not None:
        height, width = test_frame.shape[:2]
    else:
        # Fallback dimensions
        width, height = 640, 480
        warnings.warn(f"Could not determine frame size, using default {width}x{height}")

    # Create noise field
    if noise_field_kwargs is None:
        noise_field_kwargs = {}
    noise_field = NoiseField(width=width, height=height, **noise_field_kwargs)

    # Wrap environment
    env = NoiseOverlayWrapper(env, noise_field)

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
    """
    # Create environment with noise overlay
    env = make_env(env_name, render_mode="rgb_array", noise_field_kwargs=noise_field_kwargs)

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

            if episode_count % 10 == 0:
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
        print(f"\n✓ Policy saved to: {save_path}")

    print(f"\n✓ Training complete!")
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
    """
    from gymnasium.wrappers import RecordVideo

    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Create environment with noise overlay
    env = make_env(env_name, render_mode="rgb_array", noise_field_kwargs=noise_field_kwargs)

    # Wrap with RecordVideo
    env = RecordVideo(env, video_folder=outdir, name_prefix=prefix)

    # Create policy
    policy = create_policy(policy_name, env, device=device)

    # Load pre-trained weights if provided
    if policy_path:
        policy.load(policy_path)
        print(f"Loaded policy from: {policy_path}")

    # Run environment with policy
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    episode_count = 0

    for t in range(steps):
        action = policy.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            obs, info = env.reset()
            episode_count += 1

    env.close()

    print(f"\n✓ Video recording complete!")
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
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Steps between logging (training mode)")

    # Output settings
    parser.add_argument("--outdir", type=str, default="videos",
                        help="Output directory for videos (record mode)")
    parser.add_argument("--prefix", type=str, default="env_with_noise",
                        help="Video filename prefix (record mode)")

    # Noise field parameters
    parser.add_argument("--max-blocks", type=int, default=20,
                        help="Maximum number of noise blocks")
    parser.add_argument("--spawn-prob", type=float, default=0.05,
                        help="Probability of spawning a block each frame")
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
                        help="Global acceleration in x direction")
    parser.add_argument("--ay", type=float, default=30.0,
                        help="Global acceleration in y direction")
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
    }

    print(f"Environment: {args.env}")
    print(f"Policy: {args.policy.upper()}")
    print(f"Noise configuration:")
    print(f"  Max blocks: {args.max_blocks}")
    print(f"  Spawn probability: {args.spawn_prob}")
    print(f"  Static ratio: {args.static_ratio}")
    print(f"  Acceleration: ({args.ax}, {args.ay})")

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
        )


if __name__ == "__main__":
    main()
