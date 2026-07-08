import torch
import torch.nn.functional as F


def edl_mse_loss_single(alpha, targets_one_hot):
    """样本级 EDL-MSE 损失，返回误差项和方差项。"""
    strength = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum((targets_one_hot - alpha / strength) ** 2, dim=1, keepdim=True)
    var_term = torch.sum(
        alpha * (strength - alpha) / (strength * strength * (strength + 1)),
        dim=1,
        keepdim=True,
    )
    return err.squeeze(), var_term.squeeze()


def cal_kl(alpha, true_labels, num_classes):
    """计算 Dirichlet KL 项。"""
    y = F.one_hot(true_labels, num_classes=num_classes).float()
    tilde_alpha = y + (1 - y) * alpha
    ones = torch.ones_like(tilde_alpha, dtype=torch.float32, device=alpha.device)
    sum_tilde_alpha = torch.sum(tilde_alpha, dim=1, keepdim=True)

    first_term = (
        torch.lgamma(sum_tilde_alpha)
        - torch.lgamma(tilde_alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (tilde_alpha - ones)
        * (torch.digamma(tilde_alpha) - torch.digamma(sum_tilde_alpha))
    ).sum(dim=1, keepdim=True)

    return (first_term + second_term).squeeze(1)
