import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 128


class LaPaDataset(Dataset):

    def __init__(self, image_dir, label_dir, transform=None, max_samples=None):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform


        label_map = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(label_dir)
            if f.endswith(".png")
        }
        self.pairs = []
        skipped = []

        for f in sorted(os.listdir(image_dir)):
            if not (f.endswith(".jpg") or f.endswith(".png")):
                continue
            stem = os.path.splitext(f)[0]
            if stem in label_map:
                self.pairs.append((f, label_map[stem]))
            else:
                skipped.append(f)

        if skipped:
            print(f"[dataset] skipped {len(skipped)} image(s) with no matching label, e.g. {skipped[:3]}")

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        assert len(self.pairs) > 0, f"No image/label pairs found in {image_dir}"


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):

        img_name, label_name = self.pairs[idx]

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, label_name))

        # nearest-neighbour keeps class ids intact
        label = label.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        label = torch.from_numpy(np.array(label, dtype=np.int64))

        if self.transform:
            image = self.transform(image)

        return image, label


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
