import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print(platform.processor())
root_dir = os.path.dirname(os.path.abspath(__file__))
print(root_dir)
sys.path.append(root_dir)

from UASPL import UASPL
from utils.get_noise_label import add_noise
from utils.global_seed import set_global_seed
from utils.load_dataset import load_dataset
from utils.metrics import mean_and_std


def build_parser(corruption_list):
    parser = argparse.ArgumentParser(description="=== running! ===")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=6, help="outer UASPL epochs")
    parser.add_argument("--inner_epochs", type=int, default=150, help="inner training epochs")
    parser.add_argument("--num_rounds", type=int, default=50, help="random experiment rounds")
    parser.add_argument("--corruption_type", type=str, default=None, choices=corruption_list)
    parser.add_argument("--corruption_ratio", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default="Caesarian.csv", help="CSV file name under ./data")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="force device")
    parser.add_argument("--test-size", type=float, default=0.5, help="test split ratio")
    return parser


def main():
    run_start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start_time))
    print(f"Start time: {start_time_str}")

    parser = build_parser(["uniform", "asn"])
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}\n")

    dataset_name = args.dataset
    method_name = "UASPL"
    print(args)

    X, y = load_dataset(dataset_name)
    if X is not None and y is not None:
        print(f"Dataset {dataset_name} loaded successfully.")
    else:
        print(f"Dataset {dataset_name} failed to load.")

    print(f"method={method_name}, evidence_activation=relu")

    all_accuracies, all_precisions, all_recalls, all_f1s = [], [], [], []

    for i in range(args.num_rounds):
        seed = 2025 + i
        set_global_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=seed
        )
        num_classes = np.max(y_train) + 1

        if args.corruption_type is not None and args.corruption_ratio > 0:
            if i == 0:
                print(f"Add {args.corruption_type} noise, ratio: {args.corruption_ratio}")
            y_train = add_noise(
                y_train,
                args.corruption_type,
                args.corruption_ratio,
                num_classes,
                seed=seed,
            )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        # UASPL starts from an existing cross-entropy pretrained checkpoint.
        base_dir = Path(root_dir) / "param/uci_pre_model"
        cross_model_path = base_dir / f"{dataset_name}_{i}_pre_model.pth"
        base_dir.mkdir(parents=True, exist_ok=True)

        accuracy, precision, recall, f1 = UASPL(
            X_train_tensor,
            y_train_tensor,
            X_test_tensor,
            y_test_tensor,
            args.num_epochs,
            args.inner_epochs,
            cross_model_path,
            args.lr,
            device,
        )

        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    mean_accuracy, std_accuracy = mean_and_std(all_accuracies)
    mean_precision, std_precision = mean_and_std(all_precisions)
    mean_recall, std_recall = mean_and_std(all_recalls)
    mean_f1, std_f1 = mean_and_std(all_f1s)

    result_message = (
        f"{method_name + ':':<25} "
        f"Average Accuracy: {mean_accuracy:.4f}+/-{std_accuracy:.4f}, "
        f"Average Precision: {mean_precision:.4f}+/-{std_precision:.4f}, "
        f"Average Recall: {mean_recall:.4f}+/-{std_recall:.4f}, "
        f"Average F1: {mean_f1:.4f}+/-{std_f1:.4f}"
    )

    print(result_message)

    run_end_time = time.time()
    elapsed_time = run_end_time - run_start_time
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_end_time))
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"Start time: {start_time_str}")
    print(f"End time: {end_time_str}")
    print(f"Total running time: {hours}h {minutes}min {seconds:.4f}s")


if __name__ == "__main__":
    main()
