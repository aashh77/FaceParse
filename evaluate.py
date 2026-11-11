"""Final evaluation of the best model on the LaPa TEST split."""
import os
import json
import torch
from torch.utils.data import DataLoader

from model import U2NET
from dataset import LaPaDataset, transform
from metrics import evaluate, CLASS_NAMES

DATA_ROOT = "/content/LaPa" if os.path.exists("/content/LaPa") else "LaPa"
NUM_CLASSES = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = LaPaDataset(f"{DATA_ROOT}/test/images", f"{DATA_ROOT}/test/labels", transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

model = U2NET(in_ch=3, out_ch=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best.pth", map_location=DEVICE))

m = evaluate(model, test_loader, DEVICE, NUM_CLASSES)

print(f"\nTest images: {len(test_dataset)}")
print(f"Mean F1 (excl. background): {m['mean_f1']:.4f}")
print(f"Mean IoU (excl. background): {m['mean_iou']:.4f}")
print(f"Pixel accuracy: {m['pixel_acc']:.4f}\n")
print(f"{'class':<14}{'F1':>8}{'IoU':>8}")
for name, f1, iou in zip(CLASS_NAMES, m["per_class_f1"], m["per_class_iou"]):
    print(f"{name:<14}{f1:>8.4f}{iou:>8.4f}")

with open("checkpoints/test_results.json", "w") as f:
    json.dump(m, f, indent=2)
print("\nSaved checkpoints/test_results.json")
