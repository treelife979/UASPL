import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Net import initial_model
from utils.cal_uncertainty import cal_uncertainty
from utils.edl_loss import cal_kl, edl_mse_loss_single


def _load_state_dict(model, model_path, device):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


def _select_easy_samples(loss, x_train, y_train, epoch):
    selected_ratio = min(1.0, 0.25 + 0.15 * epoch)
    num_selected = max(1, int(selected_ratio * len(x_train)))
    _, indices = loss.sort(descending=False)
    selected_indices = indices[:num_selected]
    return x_train[selected_indices], y_train[selected_indices]


def uaspl_loss(outputs, labels, num_classes):
    y_one_hot = F.one_hot(labels, num_classes=int(num_classes)).float().to(outputs.device)
    err, var_term = edl_mse_loss_single(outputs, y_one_hot)
    edl_loss = err + var_term

    uncertainty = cal_uncertainty(outputs, num_classes)
    pred = torch.argmax(outputs, dim=1)
    is_wrong = (pred != labels).float()
    kl = cal_kl(outputs, labels, num_classes)
    adaptive_kl_loss = is_wrong * (1.0 - uncertainty) * kl + (1.0 - is_wrong) * uncertainty * kl
    return edl_loss + adaptive_kl_loss


def UASPL(
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs,
    inner_epochs,
    model_path,
    lr,
    device=None
):
    model, optimizer = initial_model(X_train, y_train, lr)
    _load_state_dict(model, model_path, device)
    model = model.to(device)

    num_classes = int(torch.max(y_train).item()) + 1

    for epoch in range(num_epochs):
        model.train()
        anneal = epoch / max(num_epochs - 1, 1)

        with torch.no_grad():
            outputs = model(X_train)
            sample_loss = uaspl_loss(outputs, y_train, num_classes)
            selected_X_train, selected_y_train = _select_easy_samples(
                sample_loss,
                X_train,
                y_train,
                epoch
            )

        for _ in range(inner_epochs):
            optimizer.zero_grad()
            outputs = model(selected_X_train)
            loss = uaspl_loss(outputs, selected_y_train, num_classes).mean()
            loss.backward()
            optimizer.step()
            if loss.item() < 1e-4:
                break

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = torch.argmax(outputs, dim=1)

        y_true = y_test.detach().cpu().numpy()
        y_pred = predicted.detach().cpu().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return accuracy, precision, recall, f1
