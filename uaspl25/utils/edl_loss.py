import torch
import torch.nn.functional as F

from utils.cal_uncertainty import cal_alpha


def edl_mse_loss_single(output,targets):
    alpha = cal_alpha(output)
    s = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum((targets - (alpha / s)) ** 2, dim=1, keepdim=True)
    var_term = torch.sum(
        alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)

    return err.squeeze(), var_term.squeeze()


def kl_divergence(alpha, num_classes):
    ones = torch.ones((1, num_classes), dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1, keepdim=True)
    return first_term + second_term


def cal_kl(outputs, true_labels, num_classes):
    alpha = cal_alpha(outputs)
    y = F.one_hot(true_labels.long(), num_classes=num_classes).float().to(outputs.device)
    tilde_alpha = y + (1.0 - y) * alpha
    return kl_divergence(tilde_alpha, num_classes).squeeze(1)
