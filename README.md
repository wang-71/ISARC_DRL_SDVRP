# Attention-Based DRL for SDVRP in Robotic Connection Operations

This repository provides a reproducible implementation of an attention-based deep reinforcement learning (DRL) approach for task sequence optimization in gantry-robot connection operations, formulated as a Split-Delivery Vehicle Routing Problem (SDVRP). The DRL policy is trained using RL4CO and benchmarked against Gurobi (exact), Genetic Algorithm (GA), and a Nearest Neighbor (NN) heuristic.

<img width="340" height="500" alt="Screenshot 2026-01-15 190148" src="https://github.com/user-attachments/assets/8e307dd0-7173-41af-98a7-f25a9a461d83" />

## Environment Requirements

- Python: **3.12**
- RL4CO: **0.6.0**
- Gurobi: **13.0.0** (required for exact-solver benchmarking)
- PyTorch: installed automatically via dependencies (GPU optional but recommended)

> Note: Gurobi requires a valid license. Please install and activate Gurobi following the official instructions.

## Repository Structure

- `1_Train.ipynb`  
  Training notebook: trains the attention-based DRL model on generated SDVRP instances.

- `2_DRL_Results.ipynb`  
  Inference notebook: loads a trained checkpoint and generates DRL solutions/results.

- `3_Benchmark_Gurobi_GA_NN.ipynb`  
  Benchmark notebook: runs three baselines (Gurobi / GA / Nearest Neighbor) and compares them with DRL.

- `simple_stru_sampler.py`  
  Instance generator (task layout + demand/capacity rules) for the structural-assembly SDVRP setting.

## Installation

### 1) Create a Python 3.12 environment (conda)
```bash
conda create -n rl4co_sdvrp python=3.12 -y
conda activate rl4co_sdvrp
```
### 2) Install RL4CO
```bash
pip install rl4co==0.6.0
```
### 3)(Optional) Install Gurobi for exact-solver benchmarking
Verify:
```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

Quick Start
A) Train DRL model

Run 1_Train.ipynb.

B) Run DRL inference (checkpoint)

Run 2_DRL_Results.ipynb and set the checkpoint path near the top of the notebook (e.g., ckpt_path = "checkpoints/xxx.ckpt").

C) Benchmark vs Gurobi / GA / NN

Run 3_Benchmark_Gurobi_GA_NN.ipynb.

Reproducibility (minimal notes)

Instance generation is seed-controlled (see notebooks).

Structural layout and demand rules are defined in simple_stru_sampler.py.

For fair comparison, use the same dataset/seed across all methods.

Outputs

Typical outputs include route visualizations (DRL/Gurobi/GA/NN), total route length (objective), and runtime statistics.
