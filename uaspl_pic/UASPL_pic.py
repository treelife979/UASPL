import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, Subset

# 将项目根目录加入 Python 搜索路径，保证从任意位置运行入口文件时都能导入本项目模块。
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from Net.initial_model import initial_model
from utils.cal_uncertainty import cal_alpha, cal_cor_reg_update, cal_uncertainty
from utils.edl_loss import cal_kl, edl_mse_loss_single
from utils.evaluate import evaluate_model
from utils.global_seed import seed_worker, set_global_seed



DEFAULT_CONFIG = {
    "tag": "UASPL_pic",
    "bili": 0.1,              # KL 项初始系数
    "bili_final": 0.1,        # KL 项最终系数
    "cor_reg_xishu": 1.0,     # correlation regularization 系数
}


def make_config(**overrides):
    """合并默认配置和数据集特定配置。"""
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config


def to_sample_vector(x, num_samples):
    """将标量、矩阵或 batch 级 loss 统一整理为样本级向量 [B]。"""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 0:
        return x.repeat(num_samples)
    if x.numel() == num_samples:
        return x.reshape(-1)
    return x.reshape(num_samples, -1).mean(dim=1)


def minmax_norm(x, eps=1e-8):
    """对样本级向量做 min-max 归一化，用于构造可比较的选样得分。"""
    x = x.reshape(-1)
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_edl_loss_vector(alpha, y_one_hot, num_samples):
    """计算样本级 EDL-MSE 主损失。"""
    err, var_term = edl_mse_loss_single(alpha, y_one_hot)
    return to_sample_vector(err + var_term, num_samples)


def compute_selected_ratio(epoch):
    """自步学习选样比例：前期少选，后期逐步放宽，最高不超过全量样本。"""
    return min(1.0, 0.25 + 0.15 * epoch)


def compute_epoch_bili(config, epoch, epochs):
    """按当前 epoch 计算 KL 项系数；bili 与 bili_final 相同时自然等价于常数。"""
    progress = epoch / max(1, epochs - 1)
    return config["bili"] + (config["bili_final"] - config["bili"]) * progress


def compute_weighted_kl_loss(kl_base, is_wrong, u):
    """计算 uncertainty-aware KL。
    正确样本使用 u 作为 KL 权重；错误样本使用 1-u 作为 KL 权重。
    """
    kl_weight = (1.0 - is_wrong) * u + is_wrong * (1.0 - u)
    return kl_weight * kl_base


def compute_selection_score(total_loss, u, ratio):
    """构造 UASPL 选样得分。
    score 越小，样本越优先被选中。前期主要看训练损失，后期逐步提高
    (1-u) 项的权重，使模型优先选择损失较低且仍有证据挖掘空间的样本。
    """
    loss_norm = minmax_norm(total_loss.detach())
    one_minus_u_norm = minmax_norm((1.0 - u.detach()).clamp(0.0, 1.0))
    return (1.0 - ratio) * loss_norm + ratio * one_minus_u_norm


def UASPL_pic(
    pre_model,
    train_dataset,
    train_loader,
    test_loader,
    lr,
    num_classes,
    inner_epochs,
    epochs,
    batch_size,
    model_name,
    dataset_name,
    SEED,
    act="exp",
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_model=False,
    sample_info=False,
    method_config=None,
):
    """图像数据集 UASPL 主函数。
    1. 用当前模型评估全部训练样本，计算样本级 loss、uncertainty 和选样得分。
    2. 按得分从小到大选出当前比例的样本。
    3. 在被选样本上训练若干 inner epochs，然后在测试集上评估。
    """
    config = make_config(**(method_config or {}))
    set_global_seed(SEED)

    # selected_loader 使用单独 generator，保证每次相同 seed 下选中样本的打乱顺序可复现。
    generator = torch.Generator()
    generator.manual_seed(SEED)

    model, optimizer = initial_model(model_name, dataset_name, num_classes, lr)
    model.load_state_dict(torch.load(pre_model, map_location=device))
    model.to(device)

    # 本实现依赖 train_loader 顺序遍历 train_dataset，以便按 batch 顺序写回样本级统计量。
    num_total_samples = len(train_dataset)
    if len(train_loader.dataset) != num_total_samples:
        raise ValueError("train_loader.dataset and train_dataset must have the same length.")
    if not isinstance(train_loader.sampler, SequentialSampler):
        raise ValueError("UASPL_pic expects train_loader to iterate train_dataset sequentially.")

    all_test_accuracies = []
    sample_stats = {
        "selected_ratio": [],
        "num_selected": [],
        "score_gate_ratio": [],
        "mean_loss": [],
        "mean_u": [],
        "mean_score": [],
        "selected_indices": [],
    }
    test_metrics = None

    for epoch in range(epochs):
        if epoch == epochs - 4 or epoch == epochs - 2:
            lr = lr / 10
            for group in optimizer.param_groups:
                group["lr"] = lr

        ratio = (epoch + 1) / epochs
        bili = compute_epoch_bili(config, epoch, epochs)
        cor_reg_xishu = config["cor_reg_xishu"]

        # 阶段一：遍历全量训练集，计算每个样本的训练损失和 uncertainty。
        model.eval()
        with torch.no_grad():
            all_loss = torch.empty(num_total_samples, device=device)
            all_u = torch.empty(num_total_samples, device=device)

            offset = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size_actual = len(targets)
                batch_indices = slice(offset, offset + batch_size_actual)
                offset += batch_size_actual

                outputs = model(inputs)
                alpha = cal_alpha(outputs, act)
                pred = torch.argmax(outputs, dim=1)
                is_wrong = (pred != targets).float()
                u = to_sample_vector(cal_uncertainty(alpha, num_classes), len(targets))

                y_one_hot = F.one_hot(targets.long(), num_classes=int(num_classes)).float()
                y_one_hot = y_one_hot.to(device)

                edl_loss = compute_edl_loss_vector(alpha, y_one_hot, len(targets))
                kl_base = to_sample_vector(cal_kl(alpha, targets, num_classes), len(targets))
                kl_loss = compute_weighted_kl_loss(kl_base, is_wrong, u)
                cor_reg = to_sample_vector(
                    cal_cor_reg_update(outputs, y_one_hot, u.unsqueeze(-1).detach()),
                    len(targets),
                )

                total_loss = edl_loss + bili * kl_loss + cor_reg_xishu * cor_reg
                all_loss[batch_indices] = to_sample_vector(total_loss, len(targets)).detach()
                all_u[batch_indices] = u.detach()

            if offset != num_total_samples:
                raise RuntimeError(
                    f"train_loader yielded {offset} samples, expected {num_total_samples}."
                )

            # 阶段二：根据 UASPL score 做 hard selection。
            score = compute_selection_score(all_loss, all_u, ratio)
            selected_ratio = compute_selected_ratio(epoch)
            num_selected = max(1, min(int(selected_ratio * num_total_samples), num_total_samples))
            _, sorted_indices = torch.sort(score, descending=False)
            selected_indices = sorted_indices[:num_selected]

            if sample_info:
                sample_stats["selected_ratio"].append(float(selected_ratio))
                sample_stats["num_selected"].append(int(num_selected))
                sample_stats["score_gate_ratio"].append(float(ratio))
                sample_stats["mean_loss"].append(float(all_loss.mean().item()))
                sample_stats["mean_u"].append(float(all_u.mean().item()))
                sample_stats["mean_score"].append(float(score.mean().item()))
                sample_stats["selected_indices"].append(selected_indices.detach().cpu().tolist())

            selected_dataset = Subset(train_dataset, selected_indices.detach().cpu().tolist())
            selected_loader = DataLoader(
                selected_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                worker_init_fn=seed_worker,
                generator=generator,
            )

        # 阶段三：只使用当前轮被选中的样本更新模型。
        for _ in range(inner_epochs):
            model.train()
            for inputs, targets in selected_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                alpha = cal_alpha(outputs, act)
                pred = torch.argmax(outputs, dim=1)
                is_wrong = (pred != targets).float()
                u = to_sample_vector(cal_uncertainty(alpha, num_classes), len(targets))

                y_one_hot = F.one_hot(targets.long(), num_classes=int(num_classes)).float()
                y_one_hot = y_one_hot.to(device)

                edl_loss = compute_edl_loss_vector(alpha, y_one_hot, len(targets))
                kl_base = to_sample_vector(cal_kl(alpha, targets, num_classes), len(targets))
                kl_loss = compute_weighted_kl_loss(kl_base, is_wrong, u)
                cor_reg = to_sample_vector(
                    cal_cor_reg_update(outputs, y_one_hot, u.unsqueeze(-1).detach()),
                    len(targets),
                )

                loss_vec = edl_loss + bili * kl_loss + cor_reg_xishu * cor_reg
                loss = loss_vec.mean()
                loss.backward()
                optimizer.step()

        # 每轮 self-paced 训练后评估一次，epoch_acc 会写入 runs.csv。
        test_metrics = evaluate_model(model, test_loader, device)
        train_metrics = evaluate_model(model, train_loader, device)
        all_test_accuracies.append(test_metrics[0])

        print(f"epoch{epoch + 1}: Train Accuracy: {train_metrics[0]:.4f}")
        print(f"epoch{epoch + 1}: [{config['tag']}] Test Accuracy: {test_metrics[0]:.4f}")

    if test_metrics is None:
        raise RuntimeError("epochs must be greater than 0.")

    result = (
        test_metrics[0],
        test_metrics[1],
        test_metrics[2],
        test_metrics[3],
        all_test_accuracies,
    )
    if sample_info:
        result = result + (sample_stats,)
    if save_model:
        result = result + (model, optimizer)
    return result
