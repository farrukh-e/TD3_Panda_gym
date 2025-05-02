# TD3 Sample Efficiency Analysis in Panda Robot Environment

This project evaluates the sample efficiency of the Twin Delayed Deep Deterministic Policy Gradient (TD3) against three of its modifications: TD3 + Hindsight Experience Replay (HER), TD3 + Imitation Learning and combination of both for robotic manipulation tasks using the panda-gym environment.

## Project Overview

The goal of this project is to evaluate sample efficiency and max achievable reward of following implementaitons:

- **TD3**: Baseline 
- **TD3+HER**: TD3 enhanced with Hindsight Experience Replay
- **TD3+Imitation**: TD3 augmented with human/agent demonstrations
- **TD3+Imitation+HER**: TD3 augmented with both

## Environment

We use [panda-gym](https://github.com/qgallouedec/panda-gym), a set of goal-oriented robotic environments with a Franka Emika Panda robot in PyBullet. Tasks include:
- Reaching targets
- Pushing objects
- Pick and place operations

TODO: Finalize the list of tasks
## Installation

```bash
# Clone the repository

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage


## Methodology

Our evaluation focuses on sample efficiency by:
1. Measuring learning progress at fixed training sample intervals
2. Comparing performance across algorithms using metrics like:
    - Success rate
    - Average return
    - Sample efficiency ratio (performance/samples used)
3. Testing generalization to slight variations of tasks

## Results

Results will be visualized in the `results/` directory, including:
- Learning curves showing sample efficiency\

TODO: Finalize
## References

- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [HER Paper](https://arxiv.org/abs/1707.01495)
- [Panda-Gym](https://panda-gym.readthedocs.io/en/latest/)