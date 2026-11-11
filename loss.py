import torch
import torch.nn as nn


# Main loss for 11-class face parsing
criterion = nn.CrossEntropyLoss()


def multi_output_loss(outputs, labels):
    """
    U2NET gives 7 outputs:
    d0, d1, d2, d3, d4, d5, d6

    Each output:
    [batch, 11, height, width]

    Labels:
    [batch, height, width]
    """

    d0, d1, d2, d3, d4, d5, d6 = outputs

    loss0 = criterion(d0, labels)
    loss1 = criterion(d1, labels)
    loss2 = criterion(d2, labels)
    loss3 = criterion(d3, labels)
    loss4 = criterion(d4, labels)
    loss5 = criterion(d5, labels)
    loss6 = criterion(d6, labels)

    # same idea as original U2NET:
    # supervise every side output
    loss = (
        loss0 +
        loss1 +
        loss2 +
        loss3 +
        loss4 +
        loss5 +
        loss6
    )

    return loss