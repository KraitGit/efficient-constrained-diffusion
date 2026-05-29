# Efficient Diffusion Models under Nonconvex Equality and Inequality Constraints via Landing

Official PyTorch implementation of **"Efficient Diffusion Models under Nonconvex Equality and Inequality Constraints via Landing"** (ICML 2026).

This repository implements constrained diffusion models on feasible sets with equality and inequality constraints. It includes overdamped and underdamped landing samplers for sphere, mesh/SDF, special orthogonal group, molecular, and robot-planning experiments.

## Contents

- Training code for OLLA, OLLA-P, ULLA, and ULLA-P samplers.
- Ready-to-run constrained diffusion experiments on spherical datasets, mesh/SDF data, SO(10), alanine dipeptide, and 7-DOF robot arm trajectories.
- Included datasets and SDF constraint assets for reproducing the released configurations.

## Repository Layout

```text
.
|-- main.py                  # training entry point
|-- configs/                 # Hydra experiment configs
|-- runners/                 # experiment-specific training and evaluation loops
|-- manifolds/               # constraint and manifold definitions
|-- models/                  # score network architectures
|-- src/                     # sampling, SDE, loss, and utility code
|-- scripts/                 # lightweight helper scripts
|-- data/                    # experiment datasets
`-- constraint/              # SDF constraint models
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate cdiffusion
```

The default `environment.yaml` installs PyTorch with a CUDA wheel. If your machine needs a different CUDA/PyTorch build, edit the PyTorch lines in `environment.yaml` before creating the environment. The official selector is:

```text
https://pytorch.org/get-started/locally/
```

Check that PyTorch can see CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## Data

The datasets and constraint assets needed by the configs are included under:

```text
data/
constraint/model/
```

SDF experiments use files such as:

```text
constraint/model/bunny_whole_sdf.pt
constraint/model/spot_whole_sdf.pt
data/bunny/bunny_mesh_simple.ply
data/spot/spot_mesh_simple.ply
```

## Training

Single GPU:

```bash
python main.py experiment=volcano gpu=0
```

Use a specific sampler:

```bash
python main.py experiment=volcano sample.sampler=OLLA
python main.py experiment=volcano sample.sampler=OLLA-P
python main.py experiment=volcano sample.sampler=ULLA
python main.py experiment=volcano sample.sampler=ULLA-P
```

Multi-GPU training with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 main.py experiment=volcano
```

Generate samples from a trained run:

```bash
python main.py experiment=volcano if_train=False if_sample=True \
  load_model_path=model.pt save_prefix=<prefix> seed=<seed> now=<timestamp>
```

Outputs are written to:

```text
results/<manifold>/<dataset>/<save_prefix>-<seed>-<timestamp>-<sampler>/
```

## Experiments

Available experiment configs:

```text
volcano
earthquake
flood
fire
bunny_eigfn049
bunny_eigfn099
spot_eigfn049
spot_eigfn099
SO10_3w
SO10_5w
dipeptide
robot
```

Examples:

```bash
python main.py experiment=bunny_eigfn099 sample.sampler=ULLA-P
python main.py experiment=SO10_5w sample.sampler=OLLA
python main.py experiment=dipeptide sample.sampler=ULLA
python main.py experiment=robot sample.sampler=ULLA
```

## Citing

If you use this codebase or benchmark experiments, please cite:

```bibtex
@inproceedings{jeon2026efficientdiffusion,
  title     = {Efficient Diffusion Models under Nonconvex Equality and Inequality Constraints via Landing},
  author    = {Jeon, Kijung and Muehlebach, Michael and Tao, Molei},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
}
```

## License

This project is released under the MIT License.
