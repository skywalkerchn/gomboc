"""
Gymnasium wrappers for noise avoidance.

Includes:
- CoordinateConverter: Convert between pixel and CartPole coordinates
- NoiseFeaturesWrapper: Extract noise block features into observation (Plan B)
- NoiseAvoidanceRewardWrapper: Add collision penalties and distance rewards
"""

from typing import List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ============================================================================
# Part 1: Coordinate System Conversion
# ============================================================================

class CoordinateConverter:
    """
    Convert between pixel coordinates and CartPole physical coordinates.

    CartPole coordinate system:
        - cart position x ∈ [-2.4, 2.4]
        - screen center = x = 0
        - cart at bottom of screen (y ≈ frame_height * 0.8)

    Pixel coordinate system:
        - x ∈ [0, frame_width]
        - y ∈ [0, frame_height]  (0 = top, height = bottom)
        - screen center = (width/2, height/2)

    Usage:
        converter = CoordinateConverter(frame_width=600, frame_height=400)
        cart_x = converter.pixel_to_cart_x(300)  # Center pixel → 0.0
        cart_size = converter.pixel_to_cart_size(20)  # 20 pixels → ~0.16
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        cart_range: Tuple[float, float] = (-2.4, 2.4),
        cart_y_ratio: float = 0.8,
    ):
        """
        Args:
            frame_width: Width of rendered frame in pixels
            frame_height: Height of rendered frame in pixels
            cart_range: CartPole cart position range (min, max)
            cart_y_ratio: Vertical position of cart as fraction of frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cart_min, self.cart_max = cart_range
        self.cart_range_total = self.cart_max - self.cart_min
        self.cart_y_pixel = frame_height * cart_y_ratio

    def pixel_to_cart_x(self, pixel_x: float) -> float:
        """
        Convert pixel x coordinate to CartPole cart coordinate.

        Args:
            pixel_x: X position in pixels (0 = left edge, width = right edge)

        Returns:
            Cart position in CartPole coordinates (typically -2.4 to 2.4)
        """
        # Normalize to [0, 1]
        normalized = pixel_x / self.frame_width
        # Map to cart range
        return self.cart_min + normalized * self.cart_range_total

    def cart_to_pixel_x(self, cart_x: float) -> float:
        """
        Convert CartPole cart coordinate to pixel x coordinate.

        Args:
            cart_x: Cart position in CartPole coordinates

        Returns:
            X position in pixels
        """
        # Normalize to [0, 1]
        normalized = (cart_x - self.cart_min) / self.cart_range_total
        # Map to pixel range
        return normalized * self.frame_width

    def pixel_to_cart_v(self, pixel_vx: float) -> float:
        """
        Convert pixel velocity to CartPole velocity.

        Args:
            pixel_vx: Velocity in pixels per frame

        Returns:
            Velocity in CartPole units per frame
        """
        return pixel_vx * (self.cart_range_total / self.frame_width)

    def pixel_to_cart_size(self, pixel_size: float) -> float:
        """
        Convert pixel size to CartPole length unit.

        Args:
            pixel_size: Size in pixels

        Returns:
            Size in CartPole units
        """
        return pixel_size * (self.cart_range_total / self.frame_width)


# ============================================================================
# Part 2: Noise Features Wrapper (Plan B)
# ============================================================================

class NoiseFeaturesWrapper(gym.ObservationWrapper):
    """
    Augment observation with noise block features (Plan B).

    This wrapper adds low-dimensional features about nearby noise blocks
    to the original CartPole state. The agent can use these features to
    learn avoidance behavior with an MLP policy.

    Observation space:
        Original: [cart_x, cart_v, pole_theta, pole_omega]  (4D)
        New: [cart_x, cart_v, pole_theta, pole_omega,
              rel_x_1, rel_vx_1, size_1,    # Nearest block
              rel_x_2, rel_vx_2, size_2,    # 2nd nearest
              ...
              rel_x_K, rel_vx_K, size_K]    # K-th nearest
        Total: 4 + K*3 dimensions

    Features for each block:
        - rel_x: Relative x distance (block_x - cart_x) in CartPole units
        - rel_vx: Relative x velocity (block_vx - cart_vx)
        - size: Block size in CartPole units (affects collision radius)

    If fewer than K blocks exist, remaining slots are filled with:
        [999.0, 0.0, 0.0]  # Special "no block" marker

    Usage:
        env = gym.make("CartPole-v1")
        env = NoiseFeaturesWrapper(env, noise_field, k_nearest=5)
    """

    def __init__(
        self,
        env: gym.Env,
        noise_field,  # NoiseField instance
        converter: CoordinateConverter,
        k_nearest: int = 5,
        no_block_marker: float = 999.0,
    ):
        """
        Args:
            env: Base Gymnasium environment (should be CartPole or compatible)
            noise_field: NoiseField instance managing noise blocks
            converter: CoordinateConverter for coordinate transformations
            k_nearest: Number of nearest blocks to include in observation
            no_block_marker: Special value to indicate "no block" in padding
        """
        super().__init__(env)
        self.noise_field = noise_field
        self.converter = converter
        self.k_nearest = k_nearest
        self.no_block_marker = no_block_marker

        # Extend observation space
        # Original CartPole: Box(4,)
        orig_space = env.observation_space
        assert isinstance(orig_space, spaces.Box), "Only Box observation space supported"
        assert orig_space.shape == (4,), "Expected CartPole observation space (4,)"

        # Noise features: [rel_x, rel_vx, size] * k_nearest
        noise_feat_dim = k_nearest * 3

        # Set reasonable bounds for noise features
        # rel_x, rel_vx can be negative (block to the left)
        noise_low = np.full(noise_feat_dim, -10.0, dtype=np.float32)
        noise_high = np.full(noise_feat_dim, 10.0, dtype=np.float32)

        # Allow special marker values
        noise_low[::3] = -no_block_marker  # rel_x positions
        noise_high[::3] = no_block_marker

        # Concatenate original and noise features
        self.observation_space = spaces.Box(
            low=np.concatenate([orig_space.low, noise_low]),
            high=np.concatenate([orig_space.high, noise_high]),
            dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Augment CartPole observation with noise block features.

        Args:
            obs: Original observation [cart_x, cart_v, pole_theta, pole_omega]

        Returns:
            Augmented observation with noise features appended
        """
        # Extract cart state
        cart_x = float(obs[0])
        cart_v = float(obs[1])

        # Get current noise blocks
        blocks = self.noise_field.blocks

        # Compute features for nearest K blocks
        noise_features = self._extract_noise_features(cart_x, cart_v, blocks)

        # Concatenate
        return np.concatenate([obs, noise_features]).astype(np.float32)

    def _extract_noise_features(
        self,
        cart_x: float,
        cart_v: float,
        blocks: List,
    ) -> np.ndarray:
        """
        Extract features of K nearest noise blocks.

        Args:
            cart_x: Current cart position in CartPole coordinates
            cart_v: Current cart velocity
            blocks: List of NoiseBlock instances

        Returns:
            Array of shape (k_nearest * 3,) containing block features
        """
        if len(blocks) == 0:
            # No blocks: fill all with "no block" marker
            return self._get_padding(self.k_nearest)

        # Compute distance to each block and sort
        block_distances = []
        for block in blocks:
            # Convert block position to CartPole coordinates
            block_x_cart = self.converter.pixel_to_cart_x(block.x)
            rel_x = block_x_cart - cart_x
            distance = abs(rel_x)

            block_distances.append((distance, block, rel_x))

        # Sort by distance (closest first)
        block_distances.sort(key=lambda t: t[0])

        # Extract features for K nearest blocks
        features = []
        for i in range(min(self.k_nearest, len(block_distances))):
            _, block, rel_x = block_distances[i]

            # Convert block velocity to CartPole units
            block_vx_cart = self.converter.pixel_to_cart_v(block.vx)
            rel_vx = block_vx_cart - cart_v

            # Convert block size to CartPole units
            block_size_cart = self.converter.pixel_to_cart_size(block.size)

            features.extend([rel_x, rel_vx, block_size_cart])

        # Pad if we have fewer than K blocks
        if len(blocks) < self.k_nearest:
            padding = self._get_padding(self.k_nearest - len(blocks))
            features = np.concatenate([features, padding])

        return np.array(features, dtype=np.float32)

    def _get_padding(self, n_blocks: int) -> np.ndarray:
        """
        Generate padding for missing blocks.

        Args:
            n_blocks: Number of blocks to pad

        Returns:
            Array of padding values
        """
        # Each block: [no_block_marker, 0.0, 0.0]
        padding = np.zeros(n_blocks * 3, dtype=np.float32)
        padding[::3] = self.no_block_marker  # rel_x positions
        return padding


# ============================================================================
# Part 3: Noise Avoidance Reward Wrapper
# ============================================================================

class NoiseAvoidanceRewardWrapper(gym.RewardWrapper):
    """
    Add reward shaping for noise avoidance.

    Reward components:
        1. Original task reward (CartPole balance): r_original
        2. Collision penalty: r_collision (negative when cart hits block)
        3. Distance reward: r_distance (positive when far from blocks)

    Total reward:
        r_total = r_original + collision_penalty * I(collision)
                  + distance_weight * clip(min_distance, 0, max_dist)

    Where:
        - I(collision): 1 if collision detected, 0 otherwise
        - min_distance: Distance to nearest block
        - clip: Clips distance to [0, max_dist] for reward normalization

    Usage:
        env = gym.make("CartPole-v1")
        env = NoiseAvoidanceRewardWrapper(
            env, noise_field, converter,
            collision_penalty=-5.0,
            distance_weight=0.1,
            terminate_on_collision=True  # End episode on collision
        )
    """

    def __init__(
        self,
        env: gym.Env,
        noise_field,
        converter: CoordinateConverter,
        collision_penalty: float = -5.0,
        distance_weight: float = 0.1,
        max_reward_distance: float = 2.0,
        terminate_on_collision: bool = False,
        cart_half_width: float = 0.25,
    ):
        """
        Args:
            env: Base environment
            noise_field: NoiseField managing blocks
            converter: CoordinateConverter
            collision_penalty: Penalty added when collision detected (should be negative)
            distance_weight: Weight for distance reward (encourages staying away)
            max_reward_distance: Maximum distance for reward clipping
            terminate_on_collision: If True, end episode on collision
            cart_half_width: Half-width of cart for collision detection
        """
        super().__init__(env)
        self.noise_field = noise_field
        self.converter = converter
        self.collision_penalty = collision_penalty
        self.distance_weight = distance_weight
        self.max_reward_distance = max_reward_distance
        self.terminate_on_collision = terminate_on_collision
        self.cart_half_width = cart_half_width

        # Statistics
        self.num_collisions = 0
        self.total_steps = 0

    def reward(self, reward: float) -> float:
        """
        Augment reward with noise avoidance terms.

        Args:
            reward: Original reward from environment

        Returns:
            Modified reward
        """
        self.total_steps += 1

        # Get current cart position from environment state
        # CartPole state: [x, x_dot, theta, theta_dot]
        cart_x = float(self.env.unwrapped.state[0])

        # Check collision
        collision = self._check_collision(cart_x)

        if collision:
            self.num_collisions += 1
            r_collision = self.collision_penalty

            # Optionally terminate episode
            if self.terminate_on_collision:
                # Signal termination
                self.env.unwrapped.terminated = True

        else:
            r_collision = 0.0

        # Distance reward (dense reward for staying away from blocks)
        min_distance = self._get_min_distance(cart_x)
        r_distance = self.distance_weight * np.clip(
            min_distance, 0.0, self.max_reward_distance
        )

        # Total reward
        total_reward = reward + r_collision + r_distance

        return total_reward

    def _check_collision(self, cart_x: float) -> bool:
        """
        Check if cart collides with any noise block.

        Collision detection:
            - 1D horizontal overlap: |cart_x - block_x| < (cart_width + block_width) / 2
            - Vertical overlap: block bottom >= cart top

        Args:
            cart_x: Cart position in CartPole coordinates

        Returns:
            True if collision detected
        """
        blocks = self.noise_field.blocks

        if len(blocks) == 0:
            return False

        # Cart bounding box in CartPole coordinates
        cart_left = cart_x - self.cart_half_width
        cart_right = cart_x + self.cart_half_width

        # Cart vertical position in pixels (approximate)
        cart_y_top = self.converter.cart_y_pixel - 20  # Cart height ~ 40 pixels

        for block in blocks:
            # Block bounding box in CartPole coordinates
            block_x_cart = self.converter.pixel_to_cart_x(block.x)
            block_half_width = self.converter.pixel_to_cart_size(block.size / 2)
            block_left = block_x_cart - block_half_width
            block_right = block_x_cart + block_half_width

            # Check horizontal overlap
            horizontal_overlap = not (cart_right < block_left or cart_left > block_right)

            if not horizontal_overlap:
                continue

            # Check vertical overlap (block reaches cart height)
            block_bottom = block.y + block.size / 2
            vertical_overlap = block_bottom >= cart_y_top

            if vertical_overlap:
                return True

        return False

    def _get_min_distance(self, cart_x: float) -> float:
        """
        Get minimum horizontal distance to any block.

        Args:
            cart_x: Cart position

        Returns:
            Distance to nearest block (float('inf') if no blocks)
        """
        blocks = self.noise_field.blocks

        if len(blocks) == 0:
            return float('inf')

        min_dist = float('inf')
        for block in blocks:
            block_x_cart = self.converter.pixel_to_cart_x(block.x)
            distance = abs(cart_x - block_x_cart)
            min_dist = min(min_dist, distance)

        return min_dist

    def reset(self, **kwargs):
        """Reset and print statistics."""
        obs, info = self.env.reset(**kwargs)

        # Print statistics from previous episode
        if self.total_steps > 0:
            collision_rate = self.num_collisions / self.total_steps
            info['collision_rate'] = collision_rate

        # Reset counters
        self.num_collisions = 0
        self.total_steps = 0

        return obs, info


class NoiseForceFieldWrapper(gym.Wrapper):
    """
    Apply an external horizontal acceleration to the CartPole cart based on nearby noise blocks.

    Concept:
    - Each noise block carries an "acceleration vector" (block.ax, block.ay) in pixel units.
    - We interpret block.ax as a source of horizontal acceleration field.
    - When the cart is within a configurable radius of a block (in x), it receives extra acceleration.

    Notes:
    - This modifies both the returned observation (x_dot) and env.unwrapped.state[1] to stay consistent.
    - It is CartPole-specific (assumes state = [x, x_dot, theta, theta_dot]).
    """

    def __init__(
        self,
        env: gym.Env,
        noise_field,
        converter: CoordinateConverter,
        strength: float = 1.0,
        pole_strength: float = 0.0,
        max_abs_acc: float = 50.0,
        max_abs_pole_acc: float = 50.0,
    ):
        super().__init__(env)
        self.noise_field = noise_field
        self.converter = converter
        self.strength = float(strength)
        self.pole_strength = float(pole_strength)
        self.max_abs_acc = float(max_abs_acc)
        self.max_abs_pole_acc = float(max_abs_pole_acc)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # CartPole state (Gymnasium classic-control): [x, x_dot, theta, theta_dot]
        state = getattr(self.env.unwrapped, "state", None)
        if state is None or len(state) < 2:
            return obs, reward, terminated, truncated, info

        cart_x = float(state[0])
        cart_x_pixel = float(self.converter.cart_to_pixel_x(cart_x))
        cart_y_pixel = float(self.converter.cart_y_pixel)

        # Convert pixel acceleration to CartPole units (same scale as positions).
        scale = self.converter.cart_range_total / self.converter.frame_width  # cart_unit per pixel

        tau = float(getattr(self.env.unwrapped, "tau", 0.02))

        extra_acc = 0.0
        extra_pole_acc = 0.0
        hits = 0
        for block in getattr(self.noise_field, "blocks", []):
            # Strict in-square check in pixel space (matches the visual square exactly)
            half = float(block.size) / 2.0
            if abs(cart_x_pixel - float(block.x)) > half:
                continue
            if abs(cart_y_pixel - float(block.y)) > half:
                continue

            # Inside the square => apply full field (no falloff)
            block_acc_cart = float(block.ax) * scale
            extra_acc += block_acc_cart
            # Optionally also perturb pole angular velocity (torque-like disturbance).
            # Use the same field sign/magnitude by default; scale separately.
            extra_pole_acc += block_acc_cart
            hits += 1

        extra_acc = float(np.clip(extra_acc * self.strength, -self.max_abs_acc, self.max_abs_acc))
        extra_pole_acc = float(
            np.clip(extra_pole_acc * self.pole_strength, -self.max_abs_pole_acc, self.max_abs_pole_acc)
        )

        if abs(extra_acc) > 1e-12 or abs(extra_pole_acc) > 1e-12:
            # Update x_dot / theta_dot in both observation and internal state for consistency
            try:
                obs = np.array(obs, dtype=np.float32, copy=True)
                obs[1] = float(obs[1]) + extra_acc * tau
                if obs.shape[0] >= 4:
                    obs[3] = float(obs[3]) + extra_pole_acc * tau
            except Exception:
                pass

            try:
                self.env.unwrapped.state[1] = float(self.env.unwrapped.state[1]) + extra_acc * tau
                if len(self.env.unwrapped.state) >= 4:
                    self.env.unwrapped.state[3] = float(self.env.unwrapped.state[3]) + extra_pole_acc * tau
            except Exception:
                pass

        info = dict(info)
        info["noise_forcefield_extra_acc_x"] = extra_acc
        info["noise_forcefield_extra_acc_theta"] = extra_pole_acc
        info["noise_forcefield_strength"] = self.strength
        info["noise_forcefield_pole_strength"] = self.pole_strength
        info["noise_forcefield_hits"] = hits

        return obs, reward, terminated, truncated, info


class CartPoleNoiseForceFieldDynamicsWrapper(gym.Wrapper):
    """
    CartPole-specific force-field wrapper that applies the noise field *inside the true dynamics*.

    Why this exists:
    - Modifying observation/state AFTER env.step() does NOT affect the physics integration of that step.
    - Here we implement the CartPole equations of motion directly (same as classic control),
      adding external force / external pole angular acceleration BEFORE integration.

    Field activation (strict square in pixel space):
    - The cart "effective point" is (cart_x_pixel, cart_y_pixel).
    - A block is active iff that point lies inside its square [x±size/2, y±size/2].
    """

    def __init__(
        self,
        env: gym.Env,
        noise_field,
        converter: CoordinateConverter,
        cart_strength: float = 1.0,
        pole_strength: float = 0.0,
        max_abs_cart_acc: float = 50.0,
        max_abs_pole_acc: float = 50.0,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env)
        self.noise_field = noise_field
        self.converter = converter
        self.cart_strength = float(cart_strength)
        self.pole_strength = float(pole_strength)
        self.max_abs_cart_acc = float(max_abs_cart_acc)
        self.max_abs_pole_acc = float(max_abs_pole_acc)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def _field_hits_and_acc(self, cart_x: float) -> tuple[float, float, int]:
        cart_x_pixel = float(self.converter.cart_to_pixel_x(cart_x))
        cart_y_pixel = float(self.converter.cart_y_pixel)

        # pixel -> cart units
        scale = self.converter.cart_range_total / self.converter.frame_width

        cart_acc = 0.0
        pole_acc = 0.0
        hits = 0
        for block in getattr(self.noise_field, "blocks", []):
            half = float(block.size) / 2.0
            if abs(cart_x_pixel - float(block.x)) > half:
                continue
            if abs(cart_y_pixel - float(block.y)) > half:
                continue

            block_acc_cart = float(block.ax) * scale
            cart_acc += block_acc_cart
            pole_acc += block_acc_cart
            hits += 1

        cart_acc = float(np.clip(cart_acc * self.cart_strength, -self.max_abs_cart_acc, self.max_abs_cart_acc))
        pole_acc = float(np.clip(pole_acc * self.pole_strength, -self.max_abs_pole_acc, self.max_abs_pole_acc))
        return cart_acc, pole_acc, hits

    def step(self, action):
        unwrapped = self.env.unwrapped

        # Must be CartPole-like
        state = getattr(unwrapped, "state", None)
        if state is None or len(state) != 4:
            # Fallback: at least tick noise and delegate
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.noise_field.tick()
            info = dict(info)
            info["noise_forcefield_note"] = "fallback_delegated_step"
            return obs, reward, terminated, truncated, info

        x, x_dot, theta, theta_dot = map(float, state)

        # Compute field accelerations based on current effective point BEFORE integration
        extra_cart_acc, extra_pole_acc, hits = self._field_hits_and_acc(x)

        # Base action force (CartPole is discrete left/right)
        force_mag = float(getattr(unwrapped, "force_mag", 10.0))
        force = force_mag if int(action) == 1 else -force_mag

        # Convert extra "cart acceleration" into an equivalent force term on the cart mass.
        # F = m * a
        masscart = float(getattr(unwrapped, "masscart", 1.0))
        extra_force = masscart * extra_cart_acc
        total_force = force + extra_force

        gravity = float(getattr(unwrapped, "gravity", 9.8))
        masspole = float(getattr(unwrapped, "masspole", 0.1))
        total_mass = float(getattr(unwrapped, "total_mass", masspole + masscart))
        length = float(getattr(unwrapped, "length", 0.5))  # actually half the pole's length
        polemass_length = float(getattr(unwrapped, "polemass_length", masspole * length))
        tau = float(getattr(unwrapped, "tau", 0.02))

        costheta = float(np.cos(theta))
        sintheta = float(np.sin(theta))

        temp = (total_force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Add pole disturbance as an extra angular acceleration term
        thetaacc = thetaacc + extra_pole_acc

        # Integrate
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc

        unwrapped.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Termination
        x_threshold = float(getattr(unwrapped, "x_threshold", 2.4))
        theta_threshold_radians = float(getattr(unwrapped, "theta_threshold_radians", 12 * 2 * np.pi / 360))
        terminated = bool(
            x < -x_threshold
            or x > x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians
        )

        # Truncation (time limit) - to avoid relying on TimeLimit wrapper we may have bypassed
        self._elapsed_steps += 1
        truncated = False
        if self.max_episode_steps is not None and self._elapsed_steps >= int(self.max_episode_steps):
            truncated = True

        # Reward (classic CartPole)
        reward = 1.0

        # Tick noise once per step (advance patches for next step)
        self.noise_field.tick()

        obs = np.array(unwrapped.state, dtype=np.float32)
        info = {
            "noise_forcefield_hits": hits,
            "noise_forcefield_extra_acc_x": extra_cart_acc,
            "noise_forcefield_extra_acc_theta": extra_pole_acc,
            "noise_forcefield_cart_strength": self.cart_strength,
            "noise_forcefield_pole_strength": self.pole_strength,
        }
        return obs, reward, terminated, truncated, info
