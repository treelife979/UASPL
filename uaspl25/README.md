# UASPL

中文说明见 [README_zh.md](README_zh.md).

This repository contains a concise implementation of `UASPL`, an evidential self-paced learning method for tabular classification.

## Environment

Python 3.11 is recommended. Install the dependency versions listed in `requirements.txt`. This configuration has been verified with:

- OS: Windows 10, build `10.0.26200`
- Processor: `Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`
- Python: `3.11.9` from Anaconda
- NumPy: `1.26.4`
- pandas: `2.2.3`
- scikit-learn: `1.5.1`
- PyTorch: `2.4.1+cpu`
- CUDA: disabled

If you need GPU acceleration, install the CUDA-enabled PyTorch build that matches your local GPU driver. Keep the other dependency versions consistent with `requirements.txt`.

## Project Layout

```text
run.py                  experiment entry point
UASPL.py                UASPL training and evaluation method
Net.py                  MLP model definition and optimizer initialization
utils/
  cal_uncertainty.py    Dirichlet evidence and uncertainty utilities
  edl_loss.py           EDL MSE and KL loss utilities
  get_noise_label.py    optional label-noise injection
  global_seed.py        random seed setup
  load_dataset.py       CSV dataset loader
  metrics.py            metric aggregation helpers
data/                   CSV datasets
param/uci_pre_model/    pretrained checkpoints for reproducible runs
```

## Data Format

Place CSV files under `data/`.

The loader expects:

- first column: class label
- remaining columns: numeric features

The default dataset is:

```text
data/Caesarian.csv
```

## Pretrained Checkpoints

This version of `run.py` expects pretrained model checkpoints to exist under:

```text
param/uci_pre_model/
```

For round `i`, the expected file name is:

```text
{dataset_name}_{i}_pre_model.pth
```

For example:

```text
param/uci_pre_model/Caesarian.csv_0_pre_model.pth
```

## Install

```bash
conda create -n uaspl python=3.11
conda activate uaspl
pip install -r requirements.txt
```

Example command using an existing conda environment named `pytorch`:

```bash
conda run -n pytorch python run.py --dataset Caesarian.csv --num_rounds 1 --num_epochs 1 --inner_epochs 1 --device cpu
```

## Run

Default run:

```bash
python run.py
```

Run Caesarian explicitly:

```bash
python run.py --dataset Caesarian.csv
```

Quick smoke test:

```bash
python run.py --dataset Caesarian.csv --num_rounds 1 --num_epochs 1 --inner_epochs 1 --device cpu
```

Run another dataset:

```bash
python run.py --dataset wine.csv
```

Run with label noise:

```bash
python run.py --dataset Caesarian.csv --corruption_type uniform --corruption_ratio 0.2
python run.py --dataset Caesarian.csv --corruption_type asn --corruption_ratio 0.2
```

## Main Arguments

- `--dataset`: CSV file name under `data/`; default is `Caesarian.csv`
- `--lr`: learning rate; default is `0.1`
- `--num_epochs`: outer UASPL epochs; default is `6`
- `--inner_epochs`: inner training epochs per outer epoch; default is `150`
- `--num_rounds`: random rounds; default is `50`
- `--test-size`: test split ratio; default is `0.5`
- `--device`: force `cpu` or `cuda`; by default, CUDA is used when available
- `--corruption_type`: optional label noise type, `uniform` or `asn`
- `--corruption_ratio`: label noise ratio; default is `0.0`

## Reproducibility Notes

- Round seeds are fixed as `2025 + i`.
- Data splitting uses `train_test_split(X, y, test_size=args.test_size, random_state=seed)`.
- Feature standardization is fitted on the training split only.
- Evidence activation is ReLU.
