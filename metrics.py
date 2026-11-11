import torch

CLASS_NAMES = [
    "background", "skin", "left_brow", "right_brow", "left_eye",
    "right_eye", "nose", "upper_lip", "inner_mouth", "lower_lip", "hair",
]


@torch.no_grad()
def evaluate(model, loader, device, num_classes=11):
    """
    Runs the model on a loader and returns segmentation metrics.
    Uses the fused output d0 (the first model output).
    Mean F1 / mean IoU exclude background (class 0), as is usual for LaPa.
    """
    model.eval()
    conf = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)[0].argmax(dim=1)

        idx = labels.reshape(-1) * num_classes + preds.reshape(-1)
        conf += torch.bincount(idx, minlength=num_classes * num_classes)

    conf = conf.reshape(num_classes, num_classes).double()   # rows = truth, cols = prediction

    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp

    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    iou = tp / (tp + fp + fn).clamp(min=1)

    return {
        "mean_f1": f1[1:].mean().item(),
        "mean_iou": iou[1:].mean().item(),
        "pixel_acc": (tp.sum() / conf.sum()).item(),
        "per_class_f1": f1.tolist(),
        "per_class_iou": iou.tolist(),
    }
