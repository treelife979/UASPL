import torch
import torch.nn.functional as F


def cal_alpha(outputs, act):
    """将模型输出转换为 Dirichlet 参数 alpha。"""
    if act == "exp":
        evidence = torch.exp(outputs)
    elif act == "softplus":
        evidence = F.softplus(outputs)
    elif act == "relu":
        evidence = F.relu(outputs)
    else:
        raise ValueError(f"不支持的激活函数类型 act_type={act} !")

    return evidence + 1


def cal_uncertainty(alpha, num_classes):
    """计算vacuity/uncertainty：u = K / sum(alpha)。"""
    s = torch.sum(alpha, 1, keepdim=True)
    uncertainty = num_classes / s
    return uncertainty.squeeze()


def cal_cor_reg_update(output, true_label, u):
    """只对真实类别的负输出施加不确定性感知正则。"""
    output_correct = output * true_label
    output_correct = torch.clamp_max(output_correct, 0.0)
    cor_reg = torch.sum(u * output_correct, dim=1, keepdim=True)
    return -cor_reg.squeeze(1)
