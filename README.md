# UASPL Public Code

This repository contains the public code for UASPL-related experiments. It is
organized into two independent subprojects:

- `uaspl_pic/`: image classification reproduction code for CIFAR-10,
  FashionMNIST, MNIST, and SVHN.
- `uaspl25/`: tabular classification code using CSV datasets.

Each subproject has its own README, dependencies, data instructions, pretrained
model requirements, and run commands.

## Directory Structure

```text
public_code/
├── uaspl_pic/       # UASPL_pic image-classification experiments
└── uaspl25/         # UASPL tabular-classification experiments
```

## Subprojects

### `uaspl_pic`

`uaspl_pic` contains the public reproduction code for `UASPL_pic` on four image
classification datasets:

```text
cifar10, FashionMNIST, MNIST, SVHN
```

Main files:

```text
uaspl_pic/UASPL_pic.py
uaspl_pic/run_pic.py
uaspl_pic/requirements.txt
```

Typical run:

```bash
cd uaspl_pic
python run_pic.py --profile full
```

Single-seed reproducibility check:

```bash
cd uaspl_pic
CUDA_VISIBLE_DEVICES=6 python run_pic.py \
  --datasets cifar10 \
  --profile full \
  --seeds 83105 \
  --output-dir result/repro_check_cifar10_83105 \
  --no-resume
```

For full details, see:

```text
uaspl_pic/README.md
uaspl_pic/README_zh.md
```

### `uaspl25`

`uaspl25` contains a concise implementation of UASPL for tabular classification
datasets in CSV format.

Main files:

```text
uaspl25/UASPL.py
uaspl25/run.py
uaspl25/requirements.txt
```

Typical run:

```bash
cd uaspl25
python run.py --dataset Caesarian.csv
```

Quick smoke test:

```bash
cd uaspl25
python run.py \
  --dataset Caesarian.csv \
  --num_rounds 1 \
  --num_epochs 1 \
  --inner_epochs 1 \
  --device cpu
```

For full details, see:

```text
uaspl25/README.md
uaspl25/README_zh.md
```

## Installation

The two subprojects use separate dependency files. Install dependencies inside
the subproject you want to run:

```bash
cd uaspl_pic
pip install -r requirements.txt
```

or:

```bash
cd uaspl25
pip install -r requirements.txt
```

`uaspl_pic` was checked with Python 3.6.13, PyTorch 1.10.2, torchvision 0.11.3,
and CUDA 11.3. `uaspl25` was checked with Python 3.11 and a CPU PyTorch
environment. See the subproject README files for exact environment notes.

## Data and Pretrained Models

`uaspl_pic` uses torchvision datasets and ResNet18 pretrained checkpoints. By
default, checkpoints are expected under:

```text
uaspl_pic/param/ResNet18/
```

`uaspl25` uses CSV datasets under:

```text
uaspl25/data/
```

and pretrained checkpoints under:

```text
uaspl25/param/uci_pre_model/
```

## Recommended Reading Order

1. Read this root README to choose the correct subproject.
2. Open the selected subproject README.
3. Install that subproject's dependencies.
4. Check its data and pretrained model paths.
5. Run the provided smoke test before launching full experiments.
