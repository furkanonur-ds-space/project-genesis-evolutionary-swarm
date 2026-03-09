# Project Genesis - Neuro-Evolutionary Swarm Simulation

Project Genesis is a Python/PyGame simulation where autonomous agents evolve over generations to survive and gather food.

Each agent is controlled by a lightweight neural network implemented with NumPy. The population evolves using a Genetic Algorithm (selection, crossover, mutation).

## Features

- Vector-based agent physics (position, velocity, acceleration)
- Health and survival mechanics
- Sensor-driven neural control
- Pure NumPy feed-forward neural network (no deep learning frameworks)
- Genetic Algorithm with elitism and mutation
- Real-time PyGame visualization at 60 FPS
- Generation-based evolution loop

## Architecture

### `main.py`
- Simulation loop and rendering
- Population lifecycle and generation transitions
- UI overlay:
  - `Generation: X`
  - `Alive: Y`
  - `Best Fitness: Z`

### `agent.py`
- Agent dynamics and behavior
- Inputs to brain:
  - normalized distance to nearest food
  - angle to nearest food relative to heading
  - normalized health level
- Outputs from brain:
  - thrust (speed)
  - steering (turn)
- Health logic:
  - starts at `100`
  - decays by `0.5` per frame
  - eating food gives `+20`
  - dies when health `<= 0`

### `brain.py`
- Feed-forward network: `3 -> 8 -> 2`
- Activations: `tanh`
- DNA representation:
  - flattened array of all weights and biases
- Supports:
  - clone
  - crossover
  - mutation (Gaussian noise)

### `food.py`
- Food spawn and respawn logic
- Nearest-food query for agent sensors
- Rendering of pellets

### `genetic_algorithm.py`
- Fitness function:
  - `fitness = time_alive + (food_eaten * 50)`
- Selection:
  - top `20%` elites
- Reproduction:
  - elites copied forward
  - others produced via crossover of elite parents
- Mutation:
  - `5%` per-gene mutation probability (Gaussian noise)

## Requirements

- Python 3.10+
- `numpy`
- `pygame` API (works via `pygame-ce` in this workspace)

## Quick Start (Windows PowerShell)

From `C:\Projeler\Genesis`:

```powershell
cd .\project_genesis
C:\Projeler\Genesis\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pygame-ce
python main.py
```

Notes:
- In this environment, `pygame-ce` is used and imported as `pygame`.
- If PowerShell blocks script activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Alternative Setup Script

A bootstrap script is included:

```powershell
.\setup_project_genesis.ps1
```

It creates:
- project folder
- virtual environment
- `requirements.txt`
- source files

Then install dependencies and run as shown above.

## Controls

- Close the window to stop simulation.

## Simulation Lifecycle

- Starts with:
  - `50` agents
  - `30` food pellets
- Generation ends when:
  - all agents die, or
  - frame limit is reached (`1800` frames)
- New generation is created from evolved brains.

## Important Current Behavior

- Evolution is **in-memory only**.
- Closing the app resets progress.
- Every new run starts from **Generation 1**.

If needed, checkpoint persistence (`save/load` evolved brains) can be added in a future update.

## Troubleshooting

### `can't open file ...\main.py`
You are likely running from the wrong directory.

Use either:

```powershell
cd C:\Projeler\Genesis\project_genesis
python main.py
```

or:

```powershell
C:\Projeler\Genesis\.venv\Scripts\python.exe C:\Projeler\Genesis\project_genesis\main.py
```

### Yellow import warnings in VS Code
- Ensure selected interpreter is:
  - `C:\Projeler\Genesis\.venv\Scripts\python.exe`
- Install dependencies in that same environment:

```powershell
pip install numpy pygame-ce
```

- If warnings persist, run `Developer: Reload Window` in VS Code.
