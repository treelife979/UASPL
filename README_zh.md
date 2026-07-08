# UASPL 公开代码

本仓库包含 UASPL 相关实验的公开代码，分为两个相互独立的子项目：

- `uaspl_pic/`：图像分类复现实验，包含 CIFAR-10、FashionMNIST、MNIST 和 SVHN。
- `uaspl25/`：表格 CSV 分类实验代码。

每个子项目都有自己的 README、依赖文件、数据说明、预训练模型路径和运行命令。

## 目录结构

```text
public_code/
├── uaspl_pic/       # UASPL_pic 图像分类实验
└── uaspl25/         # UASPL 表格分类实验
```

## 子项目说明

### `uaspl_pic`

`uaspl_pic` 是 `UASPL_pic` 在四个图像分类数据集上的公开复现实验代码：

```text
cifar10, FashionMNIST, MNIST, SVHN
```

主要文件：

```text
uaspl_pic/UASPL_pic.py
uaspl_pic/run_pic.py
uaspl_pic/requirements.txt
```

典型运行命令：

```bash
cd uaspl_pic
python run_pic.py --profile full
```

单 seed 复现检查：

```bash
cd uaspl_pic
CUDA_VISIBLE_DEVICES=6 python run_pic.py \
  --datasets cifar10 \
  --profile full \
  --seeds 83105 \
  --output-dir result/repro_check_cifar10_83105 \
  --no-resume
```

详细说明见：

```text
uaspl_pic/README.md
uaspl_pic/README_zh.md
```

### `uaspl25`

`uaspl25` 是 UASPL 在表格分类数据集上的简洁实现，数据集使用 CSV 格式。

主要文件：

```text
uaspl25/UASPL.py
uaspl25/run.py
uaspl25/requirements.txt
```

典型运行命令：

```bash
cd uaspl25
python run.py --dataset Caesarian.csv
```

快速冒烟测试：

```bash
cd uaspl25
python run.py \
  --dataset Caesarian.csv \
  --num_rounds 1 \
  --num_epochs 1 \
  --inner_epochs 1 \
  --device cpu
```

详细说明见：

```text
uaspl25/README.md
uaspl25/README_zh.md
```

## 安装依赖

两个子项目使用各自的依赖文件。请进入需要运行的子项目后安装依赖：

```bash
cd uaspl_pic
pip install -r requirements.txt
```

或者：

```bash
cd uaspl25
pip install -r requirements.txt
```

`uaspl_pic` 的复现检查环境为 Python 3.6.13、PyTorch 1.10.2、
torchvision 0.11.3 和 CUDA 11.3。`uaspl25` 使用 Python 3.11 和 CPU 版
PyTorch 环境检查。更具体的环境说明见各子项目 README。

## 数据和预训练模型

`uaspl_pic` 使用 torchvision 图像数据集，并依赖 ResNet18 预训练模型。默认预训练
模型目录为：

```text
uaspl_pic/param/ResNet18/
```

`uaspl25` 使用 CSV 数据集，默认数据目录为：

```text
uaspl25/data/
```

其预训练模型默认放在：

```text
uaspl25/param/uci_pre_model/
```

## 推荐阅读顺序

1. 先阅读根目录 README，确定需要运行哪个子项目。
2. 打开对应子项目的 README。
3. 安装该子项目的依赖。
4. 检查数据和预训练模型路径。
5. 先运行冒烟测试，再运行完整实验。
