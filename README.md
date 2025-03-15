# Model-Agnostic Meta-Learning (MAML) - Minimal Reproduction

This repository contains a minimal working implementation of the MAML algorithm from the paper [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)](https://arxiv.org/abs/1703.03400). The implementation focuses specifically on the point navigation environment, which is one of the original few-shot reinforcement learning experiments from the paper.

## Environment

We are using the Point Navigation environment from the original paper. In this environment:
- The agent controls a 2D point mass that must navigate to a target position
- Each task has a different goal position that the agent must reach
- MAML learns a policy that can quickly adapt to new goal positions with minimal gradient updates

## Results

The results of running the meta-reinforcement learning algorithm on the point navigation environment are visualized in the `maml_point_results.png` image. This shows the learning curves demonstrating how a MAML-trained policy can quickly adapt to new navigation goals compared to training from scratch or standard pre-training.

To visualize your results after running the training, use:
```bash
python plot_maml_results.py
```

This script loads the training progress from the CSV file generated during training and creates plots showing:
1. Pre-adaptation vs post-adaptation performance
2. Improvement factor over training iterations

## Setup Instructions (Ubuntu 22.04)

### Using Conda Environment File

The easiest way to set up the environment is using the provided environment.yml file:

```bash
# Create the conda environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate maml_rl_fixed
```

### Manual Setup (Alternative)

If you prefer to set up the environment manually:

1. **Create a conda environment with Python 3.5**:
```bash
conda create -n maml_rl python=3.5
conda activate maml_rl
```

2. **Install TensorFlow 1.4.0** (critical for compatibility):
```bash
pip install tensorflow==1.4.0
```

3. **Install other dependencies**:
```bash
pip install numpy scipy matplotlib gym mujoco-py
pip install joblib==0.9.4 python-dateutil pandas
pip install path.py mako flask h5py scikit-learn
```

## Running the Experiment

1. **Clone the repository**:
```bash
git clone https://github.com/[your-username]/maml_rl.git
cd maml_rl
```

2. **Run the point navigation experiment**:
```bash
cd maml_examples
python maml_trpo_point.py
```

3. **Visualize the results**:
```bash
cd ..  # Return to main directory
python plot_maml_results.py
```

## Notes for Reproducibility

- This codebase requires TensorFlow 1.4.0 specifically. Using newer versions will result in errors.
- Python 3.5 is required for compatibility with the dependencies.
- The code may produce warnings about deprecated NumPy functions which can be safely ignored.
- Training takes approximately 10-20 minutes depending on your hardware.

## Minimal Working Implementation

The core MAML implementation for reinforcement learning is contained in:
- `sandbox/rocky/tf/algos/maml_trpo.py`: Implementation of MAML with TRPO
- `maml_examples/point_env_randgoal.py`: The 2D point navigation environment
- `maml_examples/maml_trpo_point.py`: Script to run the point navigation experiment
- `plot_maml_results.py`: Script to visualize training results

This implementation demonstrates the essential components of MAML:
1. Meta-learning across a distribution of tasks
2. Fast adaptation with a few gradient steps
3. Comparison between pre-adaptation and post-adaptation performance
```