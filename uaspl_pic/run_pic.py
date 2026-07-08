import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from UASPL_pic import UASPL_pic
from utils.path_utils import resolve_project_path
from utils.get_Dataloader import get_Dataloader
from utils.load_data import get_dataset


DEFAULT_SEEDS = [83105, 20839, 94652, 72600, 32712]  # 随机生成的默认种子
PUBLIC_DATASETS = ["cifar10", "FashionMNIST", "MNIST", "SVHN"]

# 四个数据集的 UASPL_pic 配置。
FINAL_CONFIG_BY_DATASET = {
    "cifar10": {
        "name": "uaspl_pic_010_005",
        "bili": 0.1,
        "bili_final": 0.05,
    },
    "FashionMNIST": {
        "name": "uaspl_pic_010_const_cor025",
        "bili": 0.1,
        "bili_final": 0.1,
        "cor_reg_xishu": 0.25,
    },
    "MNIST": {
        "name": "uaspl_pic_010_const_cor15",
        "bili": 0.1,
        "bili_final": 0.1,
        "cor_reg_xishu": 1.5,
    },
    "SVHN": {
        "name": "uaspl_pic_010_005",
        "bili": 0.1,
        "bili_final": 0.05,
    },
}


def infer_num_classes(dataset_name):
    """当前四个图像数据集均为 10 类分类。"""
    if dataset_name in PUBLIC_DATASETS:
        return 10
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_profile_defaults(profile):
    """不同运行配置：quick 用于冒烟测试，full 对应正式 5-seed 设置。"""
    if profile == "quick":
        return {"num_rounds": 1, "num_epochs": 6, "inner_epochs": 3, "batch_size": 64}
    if profile == "mid":
        return {"num_rounds": 3, "num_epochs": 6, "inner_epochs": 8, "batch_size": 64}
    if profile == "full":
        return {"num_rounds": 5, "num_epochs": 6, "inner_epochs": 15, "batch_size": 64}
    raise ValueError(f"Unknown profile: {profile}")


def complete_config(raw_config):
    """补齐 UASPL_pic 所需的默认字段，并把 name 转成内部 tag。"""
    config = {
        "tag": raw_config["name"],
        "cor_reg_xishu": 1.0,
    }
    config.update(raw_config)
    config.pop("name")
    return config


def pre_model_path(args, dataset_name, seed):
    """定位预训练模型。默认使用公开包内置的 ResNet18 权重目录。"""
    if args.pre_model_dir is not None:
        base_dir = Path(args.pre_model_dir)
    else:
        base_dir = ROOT_DIR / "param" / args.model_name
    return base_dir / f"{dataset_name}_{seed}_pre_model.pth"

def run_one(dataset_name, raw_config, seed, args, device):
    """运行单个 dataset + seed，并返回一行可写入 CSV 的结果。"""
    num_classes = infer_num_classes(dataset_name)
    model_path = pre_model_path(args, dataset_name, seed)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing pretrained model: {model_path}")

    # 加载公开数据集，并构造可复现的顺序 train_loader。
    train_dataset, test_dataset = get_dataset(dataset_name, seed)
    shuffled_train_dataset, train_loader, test_loader = get_Dataloader(
        train_dataset,
        test_dataset,
        num_classes,
        args.batch_size,
        args.num_workers,
        args.imbalanced_factor,
        args.corruption_type,
        args.corruption_ratio,
        seed,
    )

    # 将数据集最终配置传给 UASPL_pic
    loss_config = dict(raw_config)
    config_name = loss_config["name"]
    method_config = complete_config(loss_config)

    result = UASPL_pic(
        model_path,
        shuffled_train_dataset,
        train_loader,
        test_loader,
        args.lr,
        num_classes,
        args.inner_epochs,
        args.num_epochs,
        args.batch_size,
        args.model_name,
        dataset_name,
        seed,
        args.act,
        device,
        False,
        False,
        method_config,
    )

    accuracy, precision, recall, f1, epoch_acc = result

    row = {
        "dataset": dataset_name,
        "config": config_name,
        "seed": seed,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "epoch_acc": "|".join(f"{value:.4f}" for value in epoch_acc),
        "config_json": json.dumps(loss_config, sort_keys=True),
    }
    return row


def aggregate(rows):
    """按 dataset + config 汇总多 seed 的均值和标准差。"""
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["config"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dataset, config), group_rows in grouped.items():
        item = {"dataset": dataset, "config": config, "runs": len(group_rows)}
        metric_keys = [
            key
            for key in group_rows[0].keys()
            if key not in {"dataset", "config", "seed", "epoch_acc", "config_json"}
        ]
        for key in metric_keys:
            values = [float(row[key]) for row in group_rows if row.get(key) not in [None, ""]]
            if values:
                item[f"{key}_mean"] = statistics.mean(values)
                item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        item["config_json"] = group_rows[0]["config_json"]
        summary.append(item)
    return summary


def write_csv(path, rows):
    """写 CSV；runs.csv 和 summary.csv 都复用这个函数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path):
    """支持 resume：若 runs.csv 已存在，则跳过已经完成的 seed。"""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_key(row):
    return (row["dataset"], row["config"], str(row["seed"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run UASPL_pic on the four datasets.")
    parser.add_argument("--datasets", nargs="+", default=PUBLIC_DATASETS, choices=PUBLIC_DATASETS)
    parser.add_argument("--profile", choices=["quick", "mid", "full"], default="full")
    parser.add_argument("--model-name", default="ResNet18")
    parser.add_argument("--act", choices=["exp", "softplus", "relu"], default="exp")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--pre-model-dir", default=None)
    parser.add_argument("--imbalanced-factor", type=float, default=None)
    parser.add_argument("--corruption-type", choices=["none", "uniform", "flip1", "flip2"], default="none")
    parser.add_argument("--corruption-ratio", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    # 根据 profile 填充没有显式传入的训练轮数、inner epochs 和 batch size。
    defaults = get_profile_defaults(args.profile)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.seeds is None:
        args.seeds = DEFAULT_SEEDS[: args.num_rounds]
    else:
        args.num_rounds = len(args.seeds)
    if args.corruption_type == "none":
        args.corruption_type = None

    start = time.time()
    run_id = time.strftime("%Y%m%d%H%M%S", time.localtime(start))
    if args.output_dir is None:
        output_dir = ROOT_DIR / "result" / "run_pic" / run_id
    else:
        output_dir = resolve_project_path(ROOT_DIR, args.output_dir)
    args._output_dir = output_dir
    rows_path = output_dir / "runs.csv"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"root={ROOT_DIR}")
    print(f"device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"datasets={args.datasets}")
    print(f"profile={args.profile}, seeds={args.seeds}")
    print(
        f"num_epochs={args.num_epochs}, inner_epochs={args.inner_epochs}, "
        f"batch_size={args.batch_size}"
    )

    rows = read_existing_rows(rows_path) if args.resume else []
    completed = {run_key(row) for row in rows}
    if rows:
        print(f"resume={args.resume}, loaded completed runs={len(rows)} from {rows_path}")

    for dataset_name in args.datasets:
        config = FINAL_CONFIG_BY_DATASET[dataset_name]
        config_name = config["name"]
        print(f"\n******* dataset={dataset_name} | config={config_name} *******")
        for idx, seed in enumerate(args.seeds, start=1):
            key = (dataset_name, config_name, str(seed))
            if key in completed:
                print(f"\n========== skip completed {idx}/{len(args.seeds)}, seed={seed} ==========")
                continue
            print(f"\n========== run {idx}/{len(args.seeds)}, seed={seed} ==========")
            row = run_one(dataset_name, config, seed, args, device)
            rows.append(row)
            completed.add(key)
            write_csv(rows_path, rows)
            write_csv(summary_path, aggregate(rows))
            print(
                f"{dataset_name} {config_name} seed={seed}: "
                f"acc={row['accuracy']:.4f}, precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, f1={row['f1']:.4f}"
            )

    summary = aggregate(rows)
    write_csv(rows_path, rows)
    write_csv(summary_path, summary)

    report = {
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "runs_csv": str(rows_path),
        "summary_csv": str(summary_path),
        "final_config_by_dataset": FINAL_CONFIG_BY_DATASET,
        "elapsed_minutes": (time.time() - start) / 60.0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n================ UASPL_pic public run summary ================")
    for item in sorted(summary, key=lambda x: x["dataset"]):
        line = (
            f"{item['dataset']:<14} {item['config']:<28} "
            f"acc={item['accuracy_mean']:.4f}+/-{item['accuracy_std']:.4f}"
        )
        print(line)

    print(f"\nSaved runs: {rows_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    print(f"Total running time: {(time.time() - start) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
