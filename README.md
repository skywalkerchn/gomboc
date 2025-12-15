# Gömböc

Gymnasium environment video recorder with physics-based noise blocks overlay.

## Demo

<video src="https://github.com/skywalkerchn/gomboc/raw/cartpole_noiseblock/demo_ppo.mp4" controls width="640"></video>

*PPO agent trained on CartPole with dynamic noise field blocks. The agent learns to balance the pole while navigating through randomly spawning noise blocks with physics-based force fields.*

## Features

- **Multiple RL Policies**: Support for DQN, PPO, A2C, and Random policies
- **Dynamic Noise Blocks**: Physics-based noise blocks with customizable acceleration fields
- **Force Field System**: Noise blocks can physically affect CartPole dynamics (cart position and pole angular velocity)
- **Noise Avoidance (Plan B)**: Low-dimensional features for noise proximity detection and avoidance reward shaping
- **Advanced Video Recording**: Conditional episode saving based on reward thresholds, flexible recording strategies
- **Visualization**: Real-time cart position markers and force field direction arrows
- **Customizable Spawning**: Control noise block spawn zones (e.g., near cart y-coordinate)
- **Training & Recording Modes**: Train policies or record videos with trained models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Basic usage with CartPole:

```bash
python cartpole_with_noise.py --steps 1000
```

## Usage Examples

### Train a DQN policy with noise blocks
```bash
python cartpole_with_noise.py --mode train --policy dqn --steps 10000 --policy-path models/dqn.pt
```

### Train with Plan B (noise avoidance features)
```bash
python cartpole_with_noise.py --mode train --policy dqn --steps 5000 --plan-b --policy-path models/dqn_planb.pt --max-blocks 15 --spawn-prob 0.15
```

### Train with force field enabled
```bash
python cartpole_with_noise.py --mode train --policy ppo --steps 10000 --plan-b --force-field --field-strength 1.0 --field-pole-strength 0.5 --policy-path models/ppo_forcefield.pt
```

### Record video with trained policy
```bash
python cartpole_with_noise.py --mode record --policy dqn --policy-path models/dqn_planb.pt --steps 500 --plan-b --max-blocks 15 --spawn-prob 0.15 --prefix "dqn_planb"
```

### Record only high-quality episodes (reward > 400)
```bash
python cartpole_with_noise.py --mode record --policy ppo --policy-path models/ppo.pt --record-episodes 20 --min-episode-reward-to-save 400 --prefix "ppo_best"
```

### Spawn noise blocks only near cart
```bash
python cartpole_with_noise.py --mode record --policy random --spawn-y-band-half 2.0 --max-blocks 20 --steps 500
```

## Command-Line Arguments

### Environment Settings
- `--env`: Gymnasium environment name (default: CartPole-v1)
- `--steps`: Number of steps to run (default: 1000)
- `--seed`: Random seed (default: 0)

### Output Settings
- `--outdir`: Output directory for videos (default: videos)
- `--prefix`: Video filename prefix (default: env_with_noise)

### Noise Block Parameters
- `--max-blocks`: Maximum number of blocks (default: 20)
- `--spawn-prob`: Spawn probability per frame (default: 0.05)
- `--static-ratio`: Ratio of static blocks 0-1 (default: 0.3)
- `--min-block-size`: Minimum block size in pixels (default: 10)
- `--max-block-size`: Maximum block size in pixels (default: 40)
- `--min-lifetime`: Minimum lifetime in frames (default: 30)
- `--max-lifetime`: Maximum lifetime in frames (default: 150)

### Physics Parameters
- `--ax`: Global acceleration in x direction (default: 0.0)
- `--ay`: Global acceleration in y direction (default: 30.0)
- `--fps`: Frames per second for physics (default: 30.0)

### Display Settings
- `--no-headless`: Don't start virtual display

## How It Works

1. **NoiseBlock**: Data structure representing a single noise block with position, velocity, acceleration, color, and lifetime
2. **NoiseField**: Manages all noise blocks, handles spawning, physics updates, and rendering
3. **NoiseOverlayWrapper**: Gymnasium wrapper that overlays noise on rendered frames
4. **Physics Simulation**: Blocks follow acceleration field with velocity integration
5. **Random Spawning**: New blocks spawn randomly with configurable probability

## Adapting to Other Environments

Simply change the `--env` argument:

```bash
# MuJoCo environments
python cartpole_with_noise.py --env Hopper-v4 --steps 2000

# Box2D environments
python cartpole_with_noise.py --env LunarLander-v2 --steps 1500
```

The script automatically detects frame dimensions from the environment.

## Video Output

Videos are saved to the `videos/` directory (or path specified by `--outdir`) as MP4 files with the specified prefix.
