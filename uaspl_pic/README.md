# UASPL_pic Reproduction Code

This repository contains the public reproduction code for `UASPL_pic` on four
image classification datasets:

- `cifar10`
- `FashionMNIST`
- `MNIST`
- `SVHN`

The main method is implemented in `UASPL_pic.py`. The experiment entry point is
`run_pic.py`.

## Project Structure

```text
.
├── UASPL_pic.py              # UASPL_pic method implementation
├── run_pic.py                # Reproduction runner
├── Net/                      # ResNet18 model definition and initializer
├── utils/                    # Data loading, metrics, EDL loss, path helpers
├── param/ResNet18/           # Optional pretrained model files
└── requirements.txt          # Python dependencies
```

## Environment


```bash
conda create -n uaspl python=3.6.13
conda activate uaspl
pip install -r requirements.txt
```

If you use an existing environment, make sure it provides PyTorch,
torchvision, NumPy, scikit-learn, and Pillow.

The reproduced run was checked on the following server environment:

```text
Python: 3.6.13
PyTorch: 1.10.2
torchvision: 0.11.3
CUDA: 11.3
```

## Data

Datasets are loaded through torchvision with `download=True`. By default, data
will be stored under:

```text
data/
```

If the folder does not exist, torchvision will download the datasets
automatically. If you already have the datasets, place them in torchvision's
standard layout under `data/`.

## Pretrained Models

Self-paced training starts from pretrained ResNet18 classifiers. By default,
the runner expects files in:

```text
param/ResNet18/<dataset>_<seed>_pre_model.pth
```

Example:

```text
param/ResNet18/cifar10_83105_pre_model.pth
```

The expected default seeds are:

```text
83105 20839 94652 72600 32712
```

You can also provide another pretrained-model directory:

```bash
python run_pic.py --pre-model-dir /path/to/pretrained_models
```

The filenames in that directory should still follow:

```text
<dataset>_<seed>_pre_model.pth
```

## Final Settings

Default full-run settings:

```text
num_rounds   = 5
num_epochs   = 6
inner_epochs = 15
batch_size   = 64
model        = ResNet18
activation   = exp
```

The runner also supports `--act softplus` and `--act relu`, but the reproduced
final results use `exp`.

## Run Experiments

Run the full four-dataset reproduction:

```bash
python run_pic.py --profile full
```

Run a single dataset:

```bash
python run_pic.py --datasets cifar10 --profile full
```

Run one seed for a quick reproducibility check:

```bash
python run_pic.py \
  --datasets cifar10 \
  --profile full \
  --seeds 83105 \
  --output-dir result/repro_check_cifar10_83105 \
  --no-resume
```

Run on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=6 python run_pic.py \
  --datasets cifar10 \
  --profile full \
  --seeds 83105 \
  --output-dir result/repro_check_cifar10_83105 \
  --no-resume
```

## Profiles

`run_pic.py` provides three runtime profiles:

| Profile | Seeds | Outer epochs | Inner epochs | Batch size |
| --- | ---: | ---: | ---: | ---: |
| `quick` | 1 | 6 | 3 | 64 |
| `mid` | 3 | 6 | 8 | 64 |
| `full` | 5 | 6 | 15 | 64 |

The default profile is `full`.

## Outputs

By default, results are saved under:

```text
result/run_pic/<timestamp>/
```

Each run writes:

- `runs.csv`: per-dataset, per-seed metrics
- `summary.csv`: mean and standard deviation across seeds
- `report.json`: command-line arguments and final dataset configurations

The printed per-seed metrics include:

```text
accuracy, precision, recall, f1
```

## Reproducibility Check

For `cifar10` with seed `83105`, the expected final test accuracy is:

```text
0.9347
```

The expected epoch-wise test accuracy is:

```text
0.6599|0.6990|0.8982|0.9060|0.9306|0.9347
```

Small differences may occur across different PyTorch, CUDA, or cuDNN versions,
but a correctly configured environment should produce closely matching results.
