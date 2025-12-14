#!/usr/bin/env python3
"""
Test script for Plan B (noise features wrapper).

This script demonstrates how to use the NoiseFeaturesWrapper and
NoiseAvoidanceRewardWrapper to enable noise avoidance learning.

Usage:
    python test_plan_b.py
"""

import sys
import numpy as np
import gymnasium as gym

# Import noise components
sys.path.insert(0, '.')
from cartpole_with_noise import NoiseField, NoiseOverlayWrapper, NoiseFieldStepWrapper
from noise_avoidance.wrappers import (
    CoordinateConverter,
    NoiseFeaturesWrapper,
    NoiseAvoidanceRewardWrapper,
)


def create_env_plan_b(
    k_nearest: int = 5,
    max_blocks: int = 10,
    collision_penalty: float = -5.0,
    distance_weight: float = 0.1,
    render_mode: str = "rgb_array",
):
    """
    Create CartPole environment with Plan B wrappers.

    Args:
        k_nearest: Number of nearest blocks in observation
        max_blocks: Maximum number of noise blocks
        collision_penalty: Penalty for hitting blocks
        distance_weight: Weight for distance reward
        render_mode: Render mode for visualization

    Returns:
        Wrapped environment ready for training
    """
    # Base environment
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # Get frame dimensions without requiring pygame render()
    env.reset()
    frame_width = getattr(env.unwrapped, "screen_width", 600)
    frame_height = getattr(env.unwrapped, "screen_height", 400)

    print(f"Frame size: {frame_width}x{frame_height}")

    # Create noise field
    noise_field = NoiseField(
        width=frame_width,
        height=frame_height,
        max_blocks=max_blocks,
        spawn_prob=0.05,
        static_ratio=0.3,
        random_acceleration=True,
        horizontal_only=False,
    )

    # Create coordinate converter
    converter = CoordinateConverter(
        frame_width=frame_width,
        frame_height=frame_height,
        cart_range=(-2.4, 2.4),
    )

    # Wrap with noise features
    env = NoiseFieldStepWrapper(env, noise_field)
    env = NoiseFeaturesWrapper(
        env,
        noise_field=noise_field,
        converter=converter,
        k_nearest=k_nearest,
    )

    # Wrap with noise avoidance reward
    env = NoiseAvoidanceRewardWrapper(
        env,
        noise_field=noise_field,
        converter=converter,
        collision_penalty=collision_penalty,
        distance_weight=distance_weight,
        terminate_on_collision=False,  # Don't end episode on collision
    )

    # Optionally wrap with noise overlay for rendering
    env = NoiseOverlayWrapper(env, noise_field)

    return env


def test_observation_space():
    """Test that observation space has correct dimensions."""
    print("\n" + "="*60)
    print("Test 1: Observation Space")
    print("="*60)

    env = create_env_plan_b(k_nearest=5)

    print(f"Observation space: {env.observation_space}")
    print(f"Expected shape: (4 + 5*3,) = (19,)")

    obs, info = env.reset()
    print(f"Actual observation shape: {obs.shape}")
    print(f"Observation sample: {obs}")

    # Check components
    print(f"\nCartPole state: {obs[:4]}")
    print(f"Noise features: {obs[4:]}")

    assert obs.shape == (19,), f"Expected shape (19,), got {obs.shape}"
    print("\n[OK] Test passed!")

    env.close()


def test_feature_extraction():
    """Test that noise features are extracted correctly."""
    print("\n" + "="*60)
    print("Test 2: Feature Extraction")
    print("="*60)

    env = create_env_plan_b(k_nearest=3, max_blocks=5)
    obs, info = env.reset()

    print(f"Initial observation: {obs}")
    print(f"  CartPole state: {obs[:4]}")
    print(f"  Noise features: {obs[4:]}")

    # Run a few steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        cart_x = obs[0]
        noise_feat = obs[4:].reshape(3, 3)  # (k_nearest, 3)

        print(f"\nStep {step+1}:")
        print(f"  Cart position: {cart_x:.3f}")
        print(f"  Noise blocks (rel_x, rel_vx, size):")
        for i, feat in enumerate(noise_feat):
            if feat[0] < 900:  # Not padding
                print(f"    Block {i+1}: rel_x={feat[0]:.3f}, "
                      f"rel_vx={feat[1]:.3f}, size={feat[2]:.3f}")
            else:
                print(f"    Block {i+1}: [no block]")

        if terminated or truncated:
            break

    print("\n[OK] Test passed!")
    env.close()


def test_reward_shaping():
    """Test that reward shaping works correctly."""
    print("\n" + "="*60)
    print("Test 3: Reward Shaping")
    print("="*60)

    env = create_env_plan_b(
        k_nearest=5,
        max_blocks=15,
        collision_penalty=-10.0,
        distance_weight=0.2,
    )

    obs, info = env.reset()

    total_reward = 0.0
    collision_count = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Check if reward was modified (not just 1.0)
        if abs(reward - 1.0) > 1e-6:
            print(f"Step {step+1}: modified reward = {reward:.3f}")
            if reward < -5:
                collision_count += 1
                print(f"  -> Collision detected!")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Collisions: {collision_count}")
    print(f"Collision rate: {info.get('collision_rate', 0):.3f}")

    print("\n[OK] Test passed!")
    env.close()


def test_coordinate_conversion():
    """Test coordinate conversion utilities."""
    print("\n" + "="*60)
    print("Test 4: Coordinate Conversion")
    print("="*60)

    converter = CoordinateConverter(
        frame_width=600,
        frame_height=400,
        cart_range=(-2.4, 2.4),
    )

    # Test center conversion
    center_pixel = 300
    center_cart = converter.pixel_to_cart_x(center_pixel)
    print(f"Center pixel {center_pixel} -> cart {center_cart:.3f} (expected 0.0)")
    assert abs(center_cart) < 1e-6, "Center should map to 0.0"

    # Test left edge
    left_pixel = 0
    left_cart = converter.pixel_to_cart_x(left_pixel)
    print(f"Left pixel {left_pixel} -> cart {left_cart:.3f} (expected -2.4)")
    assert abs(left_cart - (-2.4)) < 1e-6

    # Test right edge
    right_pixel = 600
    right_cart = converter.pixel_to_cart_x(right_pixel)
    print(f"Right pixel {right_pixel} -> cart {right_cart:.3f} (expected 2.4)")
    assert abs(right_cart - 2.4) < 1e-6

    # Test round-trip
    test_cart = 1.2
    test_pixel = converter.cart_to_pixel_x(test_cart)
    test_cart_back = converter.pixel_to_cart_x(test_pixel)
    print(f"\nRound-trip: cart {test_cart} -> pixel {test_pixel:.1f} -> "
          f"cart {test_cart_back:.3f}")
    assert abs(test_cart - test_cart_back) < 1e-6

    print("\n[OK] Test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Plan B Implementation")
    print("="*60)

    test_coordinate_conversion()
    test_observation_space()
    test_feature_extraction()
    test_reward_shaping()

    print("\n" + "="*60)
    print("All tests passed! [OK]")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

