import torch


def compute_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    if preds.numel() == 0 or labels.numel() == 0:
        return confmat

    preds = preds.view(-1).to(torch.long).cpu()
    labels = labels.view(-1).to(torch.long).cpu()

    valid = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[valid]
    labels = labels[valid]

    if preds.numel() == 0:
        return confmat

    indices = labels * num_classes + preds
    bincount = torch.bincount(indices, minlength=num_classes * num_classes)
    return bincount.reshape(num_classes, num_classes)


def compute_miou_from_confmat(confmat: torch.Tensor) -> float:
    confmat = confmat.to(torch.float32)

    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    denom = tp + fp + fn

    valid = denom > 0
    if valid.sum() == 0:
        return float("nan")

    iou = torch.zeros_like(tp)
    iou[valid] = tp[valid] / denom[valid]
    return iou[valid].mean().item()