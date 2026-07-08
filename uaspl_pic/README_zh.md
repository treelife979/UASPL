# UASPL_pic 复现实验代码

本项目提供 `UASPL_pic` 方法在四个图像分类数据集上的公开复现实验代码：

- `cifar10`
- `FashionMNIST`
- `MNIST`
- `SVHN`

方法主体在 `UASPL_pic.py` 中，实验运行入口是 `run_pic.py`。

## 项目结构

```text
.
├── UASPL_pic.py              # UASPL_pic 方法实现
├── run_pic.py                # 复现实验入口
├── Net/                      # ResNet18 模型定义和初始化
├── utils/                    # 数据加载、指标、EDL loss、路径工具
├── param/ResNet18/           # 可选的预训练模型文件
└── requirements.txt          # Python 依赖
```

## 环境配置

```bash
conda create -n uaspl python=3.6.13
conda activate uaspl
pip install -r requirements.txt
```

如果使用已有环境，需要保证安装了 PyTorch、torchvision、NumPy、
scikit-learn 和 Pillow。

本项目的复现检查使用的服务器环境如下：

```text
Python: 3.6.13
PyTorch: 1.10.2
torchvision: 0.11.3
CUDA: 11.3
```

## 数据集

数据集通过 torchvision 加载，并设置了 `download=True`。默认数据目录是：

```text
data/
```

如果该目录不存在，torchvision 会自动下载数据集。如果已经有数据集，也可以按
torchvision 的标准目录结构放到 `data/` 下。

## 预训练模型

自步训练从预训练的 ResNet18 分类模型开始。默认预训练模型路径是：

```text
param/ResNet18/<dataset>_<seed>_pre_model.pth
```

例如：

```text
param/ResNet18/cifar10_83105_pre_model.pth
```

默认使用的五个随机种子是：

```text
83105 20839 94652 72600 32712
```

也可以手动指定其他预训练模型目录：

```bash
python run_pic.py --pre-model-dir /path/to/pretrained_models
```

该目录下的文件名仍需满足：

```text
<dataset>_<seed>_pre_model.pth
```

## 最终实验配置

默认 full 设置如下：

```text
num_rounds   = 5
num_epochs   = 6
inner_epochs = 15
batch_size   = 64
model        = ResNet18
activation   = exp
```

## 运行实验

运行四个数据集的完整复现实验：

```bash
python run_pic.py --profile full
```

只运行一个数据集：

```bash
python run_pic.py --datasets cifar10 --profile full
```

运行单个 seed 做复现检查：

```bash
python run_pic.py \
  --datasets cifar10 \
  --profile full \
  --seeds 83105 \
  --output-dir result/repro_check_cifar10_83105 \
  --no-resume
```

指定 GPU 运行：

```bash
CUDA_VISIBLE_DEVICES=6 python run_pic.py \
  --datasets cifar10 \
  --profile full \
  --seeds 83105 \
  --output-dir result/repro_check_cifar10_83105 \
  --no-resume
```

## 输出文件

默认结果目录是：

```text
result/run_pic/<timestamp>/
```

每次运行会生成：

- `runs.csv`：每个数据集、每个 seed 的指标
- `summary.csv`：多 seed 的均值和标准差
- `report.json`：命令行参数和每个数据集的最终配置

每个 seed 结束时会打印四个指标：

```text
accuracy, precision, recall, f1
```
