import os
import csv
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import U2NET
from dataset import LaPaDataset, transform
from loss import multi_output_loss
from metrics import evaluate


DATA_ROOT = "/content/LaPa" if os.path.exists("/content/LaPa") else "LaPa"         
BATCH_SIZE = 16
EPOCHS = 20
TRAIN_SAMPLES = 9000       # use None for the full dataset
LR = 0.001

CKPT_DIR = "checkpoints"
LAST_CKPT = os.path.join(CKPT_DIR, "last.pth")   
BEST_CKPT = os.path.join(CKPT_DIR, "best.pth")   
LOG_FILE = os.path.join(CKPT_DIR, "log.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

os.makedirs(CKPT_DIR, exist_ok=True)


# Data

train_dataset = LaPaDataset(f"{DATA_ROOT}/train/images", f"{DATA_ROOT}/train/labels", transform,
                              max_samples=TRAIN_SAMPLES)
val_dataset = LaPaDataset(f"{DATA_ROOT}/val/images", f"{DATA_ROOT}/val/labels", transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))



# Model + optimizer


model = U2NET(in_ch=3, out_ch=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)



# Resume from latest checkpoint (if any)


start_epoch = 0      # epochs already completed
best_f1 = 0.0

if os.path.exists(LAST_CKPT):
    ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]
    best_f1 = ckpt.get("best_f1", 0.0)
    print(f"Resumed from {LAST_CKPT} (completed epochs: {start_epoch}, best val F1: {best_f1:.4f})")
else:
    print("No checkpoint found, starting from scratch")
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mean_f1", "val_mean_iou", "val_pixel_acc"])

if start_epoch >= EPOCHS:
    print(f"Already trained {start_epoch} epochs. Increase EPOCHS above {start_epoch} to continue.")


# Training loop


for epoch in range(start_epoch, EPOCHS):

    model.train()
    running_loss = 0.0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = multi_output_loss(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)

    # ---- validation ----
    m = evaluate(model, val_loader, DEVICE, NUM_CLASSES)

    print(
        f"Epoch {epoch+1} | train loss {epoch_loss:.4f} | "
        f"val mean F1 {m['mean_f1']:.4f} | val mIoU {m['mean_iou']:.4f} | "
        f"pixel acc {m['pixel_acc']:.4f}"
    )

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([epoch + 1, f"{epoch_loss:.4f}", f"{m['mean_f1']:.4f}",
                                f"{m['mean_iou']:.4f}", f"{m['pixel_acc']:.4f}"])

    # ---- keep the best model ----
    if m["mean_f1"] > best_f1:
        best_f1 = m["mean_f1"]
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"  new best model saved (val mean F1 {best_f1:.4f})")

    # ---- full checkpoint for resuming  ----
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_f1,
    }, LAST_CKPT + ".tmp")
    os.replace(LAST_CKPT + ".tmp", LAST_CKPT)


print("Training finished!")
