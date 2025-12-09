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
# Part 5: Video recording
# ============================================================================

def record_video_with_noise(
    env_name: str = "CartPole-v1",
    outdir: str = "videos",
    prefix: str = "env_with_noise",
    steps: int = 1000,
    seed: int = 0,
    noise_field_kwargs: Optional[dict] = None,
):
    """
    Record video of environment with noise blocks overlay.

    Args:
        env_name: Name of the Gymnasium environment
        outdir: Output directory for videos
        prefix: Prefix for video filename
        steps: Number of steps to run
        seed: Random seed
        noise_field_kwargs: Keyword arguments for NoiseField
    """
    from gymnasium.wrappers import RecordVideo

    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Create environment with noise overlay
    env = make_env(env_name, render_mode="rgb_array", noise_field_kwargs=noise_field_kwargs)

    # Wrap with RecordVideo
    env = RecordVideo(env, video_folder=outdir, name_prefix=prefix)

    # Run environment with random policy
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    episode_count = 0

    for t in range(steps):
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            obs, info = env.reset()
            episode_count += 1

    env.close()

    print(f"\n✓ Video recording complete!")
    print(f"  Output: {outdir}")
    print(f"  Prefix: {prefix}")
    print(f"  Steps: {steps}")
    print(f"  Episodes: {episode_count}")
    print(f"  Total reward: {total_reward:.2f}")


# ============================================================================
# Part 6: Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Record Gymnasium environment with noise blocks overlay"
    )

    # Environment settings
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment name")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    # Output settings
    parser.add_argument("--outdir", type=str, default="videos",
                        help="Output directory for videos")
    parser.add_argument("--prefix", type=str, default="env_with_noise",
                        help="Video filename prefix")

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

    # Record video
    print(f"Recording {args.env} with noise blocks...")
    print(f"Configuration:")
    print(f"  Max blocks: {args.max_blocks}")
    print(f"  Spawn probability: {args.spawn_prob}")
    print(f"  Static ratio: {args.static_ratio}")
    print(f"  Acceleration: ({args.ax}, {args.ay})")

    record_video_with_noise(
        env_name=args.env,
        outdir=args.outdir,
        prefix=args.prefix,
        steps=args.steps,
        seed=args.seed,
        noise_field_kwargs=noise_field_kwargs,
    )


if __name__ == "__main__":
    main()
