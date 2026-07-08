# UASPL

本仓库提供 `UASPL` 方法的精简可运行源码。该方法用于表格分类任务。

English version: [README.md](README.md)

## 环境

建议使用 Python 3.11，并安装 `requirements.txt` 中列出的依赖版本。该配置已在以下环境中验证通过：

- 操作系统：Windows 10，build `10.0.26200`
- 处理器：`Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`
- Python：`3.11.9`，Anaconda
- NumPy：`1.26.4`
- pandas：`2.2.3`
- scikit-learn：`1.5.1`
- PyTorch：`2.4.1+cpu`
- CUDA：未启用

如果需要使用 GPU，请根据本机显卡驱动安装对应的 CUDA 版 PyTorch；其余依赖版本建议保持与 `requirements.txt` 一致。

## 项目结构

```text
run.py                  实验入口
UASPL.py                UASPL 训练与评估方法
Net.py                  MLP 网络结构与优化器初始化
utils/
  cal_uncertainty.py    Dirichlet evidence 与不确定性计算
  edl_loss.py           EDL MSE 与 KL 损失
  get_noise_label.py    可选标签噪声注入
  global_seed.py        随机种子设置
  load_dataset.py       CSV 数据集读取
  metrics.py            指标均值和标准差统计
data/                   CSV 数据集
param/uci_pre_model/    用于复现实验的预训练 checkpoint
```

## 数据格式

CSV 文件放在 `data/` 目录下。

读取器默认约定：

- 第 1 列：类别标签
- 第 2 列到最后 1 列：数值特征

默认数据集是：

```text
data/Caesarian.csv
```

## 预训练 Checkpoint

本版本的 `run.py` 默认从以下目录读取已经存在的预训练模型：

```text
param/uci_pre_model/
```

第 `i` 轮实验对应的 checkpoint 文件名为：

```text
{dataset_name}_{i}_pre_model.pth
```

例如：

```text
param/uci_pre_model/Caesarian.csv_0_pre_model.pth
```

## 安装

```bash
conda create -n uaspl python=3.11
conda activate uaspl
pip install -r requirements.txt
```

也可以使用本地已有的 `pytorch` 环境运行：

```bash
conda run -n pytorch python run.py --dataset Caesarian.csv --num_rounds 1 --num_epochs 1 --inner_epochs 1 --device cpu
```

## 运行

默认运行：

```bash
python run.py
```

显式运行 Caesarian 数据集：

```bash
python run.py --dataset Caesarian.csv
```

快速测试：

```bash
python run.py --dataset Caesarian.csv --num_rounds 1 --num_epochs 1 --inner_epochs 1 --device cpu
```

运行其他数据集：

```bash
python run.py --dataset wine.csv
```

加入标签噪声：

```bash
python run.py --dataset Caesarian.csv --corruption_type uniform --corruption_ratio 0.2
python run.py --dataset Caesarian.csv --corruption_type asn --corruption_ratio 0.2
```

## 主要参数

- `--dataset`：`data/` 下的 CSV 文件名，默认 `Caesarian.csv`
- `--lr`：学习率，默认 `0.1`
- `--num_epochs`：UASPL 外层 epoch 数，默认 `6`
- `--inner_epochs`：每个外层 epoch 内部训练轮数，默认 `150`
- `--num_rounds`：随机实验轮数，默认 `50`
- `--test-size`：测试集比例，默认 `0.5`
- `--device`：强制使用 `cpu` 或 `cuda`；默认优先使用可用 CUDA
- `--corruption_type`：可选标签噪声类型，支持 `uniform` 或 `asn`
- `--corruption_ratio`：标签噪声比例，默认 `0.0`

## 可复现性说明

- 每轮随机种子固定为 `2025 + i`。
- 数据划分方式为：`train_test_split(X, y, test_size=args.test_size, random_state=seed)`。
- 特征标准化只使用训练集统计量。
- evidence 激活函数为 ReLU。

## requirements.txt 为什么很少

本仓库源码直接使用的第三方依赖只有：

```text
numpy
pandas
scikit-learn
torch
```

`argparse`、`os`、`platform`、`sys`、`time`、`pathlib`、`random` 等都是 Python 标准库，不需要写进 `requirements.txt`。其他底层依赖会由上述包自动安装。
