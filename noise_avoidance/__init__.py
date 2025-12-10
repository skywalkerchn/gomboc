"""
Noise avoidance module for Gömböc project.

Provides wrappers and utilities for:
- Converting noise blocks to low-dimensional features (Plan B)
- Converting observations to pixel space (Plan A)
- Reward shaping for noise avoidance
"""

from .wrappers import (
    CoordinateConverter,
    NoiseFeaturesWrapper,
    NoiseAvoidanceRewardWrapper,
)

__all__ = [
    "CoordinateConverter",
    "NoiseFeaturesWrapper",
    "NoiseAvoidanceRewardWrapper",
]
