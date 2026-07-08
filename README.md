# HSAC — Learning to build covering structures with continuous adjustments

Code for the paper **"Learning to build covering structures with continuous adjustments"**
by Gabriel Vallat, Maryam Kamgarpour and Stefana Parascho.

We introduce **Hybrid Soft Actor-Critic (HSAC)**, an extension of SAC to *mixed*
(parameterized) action spaces, and apply it to plan-free robotic construction. Rather than
following a predefined plan, the agent generates a construction sequence adaptively as the
structure is built: at each step it selects **which block** to add (a discrete choice) and
**where** to place it (a continuous `(x, y, α)` pose). The structure is represented as a
heterogeneous graph with per-block reference frames, making the state invariant to
translation, permutation, and vertical rotation.

The central technical idea is to attach candidate actions to the state graph with
**unidirectional edges**, structurally enforcing that the Q-value of one discrete action is
independent of the continuous parameters of the others. This removes the over-parameterization
that plagues naïve DQN extensions to hybrid action spaces and enables sample-efficient learning
despite a costly physics-based stability check.

We validate HSAC against **HPPO** (hybrid-PPO), and deploy a trained policy on a physical
two-robot setup (two ABB GoFa arms + a Zivid 3D camera) that builds a spanning arch from
3D-printed blocks in closed loop.

---

## Repository layout

The repository is organized as four importable top-level Python packages plus the experiment
scripts:

```
hsac/              HSAC algorithm — agent, policies, Q/value functions, replay buffers, training loop
hppo/              Baseline hybrid-PPO implementation, mirroring the HSAC architecture
masongraph_envs/   Gym-style construction environments (ColumnMasonGraph is used by the paper)
  tools/           Action <-> tensor converters, HER, rendering helpers
simulator/
  cont3Dsimulator/ Construction simulator & physics
    batchsim*.py   Batched simulator (NumPy / Warp backends)
    physics/       Rigid Block Analysis (RBA) stability model, Gurobi-based solver
    placement/     Block placement geometry (Warp kernels)
    state_rep/     Graph state representations (bframegraph, ftbframegraph, ...)
    objectives/    Coverage objective
    render/        Polyscope-based rendering (structure, forces, graph, actions)
    utils/         Geometry helpers
    data/          Block meshes (.obj) and example structures (arch, dome, ...)
    tests/         Placement / physics / objective sanity checks
experiments/
  config.json      Base hyperparameters (environment + algorithm)
  HSACvsHPPO/      Learning-curve comparison (Fig. 6)
  HyperHSAC/       Hyperparameter robustness study (Fig. 7): base/large/lesstrain/deterministic
  DiffEnv/         Environment sweeps (Fig. 8): friction μ and #discrete-actions N
```

Each `experiments/*/` folder contains a `run_exp_*.py` script and a matching `exp_*.run`
SLURM batch file. The import names used throughout the code map directly to these directories:
`hsac`, `hppo`, `simulator`, and `masongraph_envs`.

## The algorithm at a glance

- **State** — a labeled heterogeneous graph `G = {N, E}` with block nodes, action-candidate
  nodes, and objective (coverage-polygon) nodes. Edges carry relative transforms between block
  frames rather than global coordinates. See appendix A of the paper for the full labeling; the
  representations live in `simulator/cont3Dsimulator/state_rep/` (the base config uses `bframegraph`).
- **Networks** — [Transformer-convolution](https://arxiv.org/abs/2009.03509) GNNs map the graph
  to per-action features, which fully-connected heads turn into the discrete policy `π_d(a|s)`,
  the continuous Gaussian policy `π_c(u|a,s)`, the value function, and the two Q-functions.
- **Learning** — SAC-style objectives with the policy cost **split** into separate discrete and
  continuous terms so that entropy is controlled independently for each (dynamic `α_c`, `α_d`).
  This is what stops the discrete policy from collapsing to a near-deterministic choice.
- **Simulation** — [Warp](https://github.com/nvidia/warp) parallelizes contact detection across
  environments; [Gurobi](https://www.gurobi.com) solves the Rigid Block Analysis stability LP;
  states/actions are stored as [PyTorch Geometric](https://pyg.org) graphs for batched GPU
  training.

## Requirements

- Python 3.11
- [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pyg.org)
- [NVIDIA Warp](https://github.com/nvidia/warp) (`warp-lang`) — CUDA GPU strongly recommended
- [Gurobi](https://www.gurobi.com) + `gurobipy` (license required for the RBA stability solver)
- `numpy`, `scipy`, `shapely`, `trimesh`, `matplotlib`, `piqp`
- [`gymnasium`](https://gymnasium.farama.org) for the environment API
- [`polyscope`](https://polyscope.run) for rendering
- [`wandb`](https://wandb.ai) for experiment logging (runs **offline by default** — no account needed)

## Installation

```bash
# from the repository root
pip install -r requirements.txt        # dependencies
pip install -e .                        # make hsac / hppo / simulator / masongraph_envs importable
```

Gurobi and NVIDIA Warp are installed as ordinary Python packages, but Gurobi needs a valid
license and Warp needs a CUDA-capable GPU for realistic training throughput.

Installing with `pip install -e .` is optional: every `run_exp_*.py` script also prepends the
repository root to `sys.path`, so the packages resolve even without an editable install.

## Running experiments

Each experiment is a standalone script that can be launched from anywhere:

```bash
python experiments/HSACvsHPPO/run_exp_sac.py     # HSAC (Fig. 6)
python experiments/HSACvsHPPO/run_exp_ppo.py     # HPPO baseline (Fig. 6)
python experiments/HyperHSAC/run_exp_base.py     # hyperparameter study (Fig. 7)
python experiments/DiffEnv/run_exp_N10.py        # environment sweep (Fig. 8)
```

A run script (e.g. `experiments/HSACvsHPPO/run_exp_sac.py`) does the following:

1. loads `experiments/config.json` (override with the `HSAC_CONFIG` environment variable),
2. initializes Warp on CUDA and a `wandb` run,
3. builds the `ColumnMasonGraph` environment,
4. constructs the `SAC` agent from the environment's state/action metadata, and
5. calls the training loop, saving policies periodically to `./models/`.

The base setup uses 16 parallel environments, a discount factor of 0.9, blocks with slopes
γ ∈ {5°, 10°, 20°}, and a friction coefficient of 0.3 (see `experiments/config.json` and
Tables I–II in the paper).

### Experiment logging (Weights & Biases)

Logging defaults to **offline** mode, so no W&B account or API key is required — runs are
written to a local `wandb/` directory. To sync to your own account instead:

```bash
wandb login                 # once, with your own key
WANDB_MODE=online WANDB_PROJECT=my-project python experiments/HSACvsHPPO/run_exp_sac.py
```

`WANDB_MODE` and `WANDB_PROJECT` override the defaults; alternatively set `"online": true` in
`experiments/config.json`.

### Cluster (SLURM)

The `experiments/*/exp_*.run` files are example SLURM batch scripts. Submit them from the
repository root (e.g. `sbatch experiments/HSACvsHPPO/exp_sac.run`) and adjust the partition,
Gurobi module, and environment activation for your site.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{vallat_hsac,
  title   = {Learning to build covering structures with continuous adjustments},
  author  = {Vallat, Gabriel and Kamgarpour, Maryam and Parascho, Stefana},
}
```

## Acknowledgment

Robot closed-loop control (path/motion planning and force control) builds on code by Jingwen
Wang. Supported by NCCR Automation (Swiss National Science Foundation, grant 51NF40 225155) and
the EPFL AI Center.

## License

MIT — see [LICENSE](LICENSE).
