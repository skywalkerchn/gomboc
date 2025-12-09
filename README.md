# Gömböc

Gymnasium environment video recorder with physics-based noise blocks overlay.

## Features

- Record videos of any Gymnasium environment (CartPole, MuJoCo, MetaWorld, etc.)
- Overlay dynamic noise blocks with physics simulation
- Support for both static and dynamic noise blocks
- Configurable acceleration field
- Fully customizable via command-line arguments

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

### Basic recording
```bash
python cartpole_with_noise.py --env CartPole-v1 --steps 1000
```

### More noise blocks with stronger gravity
```bash
python cartpole_with_noise.py --max-blocks 50 --ay 100 --spawn-prob 0.1
```

### Static blocks only (no physics)
```bash
python cartpole_with_noise.py --static-ratio 1.0 --ax 0 --ay 0
```

### Use with other environments
```bash
python cartpole_with_noise.py --env Pendulum-v1 --steps 500
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
