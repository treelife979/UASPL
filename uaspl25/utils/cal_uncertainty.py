import torch
import torch.nn.functional as F


def evidence_from_outputs(outputs):
    return F.relu(outputs)

def cal_alpha(outputs):
    return evidence_from_outputs(outputs) + 1.0


def cal_uncertainty(outputs, num_classes):
    alpha = cal_alpha(outputs)
    S = torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = num_classes / S
    return uncertainty.squeeze(1)
