# Face Parsing with U²-Net

A U²-Net model trained to assign each pixel of a face image to one of 11 classes: background, skin, left and right eyebrow, left and right eye, nose, upper lip, inner mouth, lower lip and hair. The model is implemented in PyTorch, trained from scratch on the LaPa dataset, and run on a free Google Colab T4 GPU.

This repository contains the baseline version. It is kept intentionally simple to provide a measured reference point for later improvements.

## Results

Evaluation on the full LaPa test split (2,000 images), using the checkpoint with the highest validation F1.

| Metric | Score |
|---|---|
| Mean F1 (10 face classes, background excluded) | 89.4% |
| Mean IoU (same classes) | 81.1% |
| Pixel accuracy | 97.6% |

Per-class results:

| Class | F1 (%) | IoU (%) |
|---|---|---|
| background | 98.8 | 97.7 |
| skin | 96.7 | 93.7 |
| left brow | 86.7 | 76.6 |
| right brow | 87.7 | 78.0 |
| left eye | 88.4 | 79.2 |
| right eye | 88.1 | 78.8 |
| nose | 95.4 | 91.1 |
| upper lip | 84.1 | 72.5 |
| inner mouth | 85.8 | 75.1 |
| lower lip | 86.3 | 76.0 |
| hair | 94.6 | 89.7 |

Large regions (skin, nose, hair) are segmented with high accuracy. The lowest scores are on small, thin structures: the lips and eyebrows. At 128×128 input resolution these classes cover only a few pixels in width, which limits the detail the model can recover.

Raw metrics are stored in `results/baseline/test_results.json` and training curves in `results/baseline/curves.png`.

## Experimental Setup

| Item | Value |
|---|---|
| Model | U²-Net, 11 output channels, trained from scratch (no pretrained weights) |
| Input size | 128×128 |
| Training data | First 9,000 of 18,176 LaPa training images |
| Validation / test data | 2,000 / 2,000 images, used in full |
| Loss | Cross-entropy on the fused output and all six side outputs, summed |
| Optimizer | Adam, learning rate 0.001, batch size 16 |
| Epochs | 20 |
| Hardware | Google Colab, Tesla T4 |

A subset of the training data was used to keep each run short on a free GPU. Validation F1 is computed after every epoch and the best-scoring checkpoint is retained, so the reported test score does not simply reflect the final epoch. The test split was evaluated once, after training.

## Usage

The notebook `run_baseline.ipynb` runs the full pipeline on Colab with a GPU runtime: it mounts Google Drive, downloads LaPa, trains and evaluates.

To run the scripts directly, arrange LaPa as `LaPa/{train,val,test}/{images,labels}` and run:

```
python train.py       # trains; writes checkpoints/ (best.pth, last.pth, log.csv)
python evaluate.py    # evaluates checkpoints/best.pth on the test split
```

Training resumes automatically from `checkpoints/last.pth` when it exists, which allows work to continue after a Colab disconnect. Delete the `checkpoints` folder to start from scratch. To train on all 18,176 images, set `TRAIN_SAMPLES = None` in `train.py`.

## Repository Structure

```
model.py       U²-Net (RSU blocks and full network)
dataset.py     LaPa loader; pairs images and labels by filename
loss.py        multi-output cross-entropy loss
metrics.py     confusion-matrix based F1, IoU and pixel accuracy
train.py       training loop with validation and checkpoint resume
evaluate.py    final test-set evaluation
results/       baseline metrics and training curves
```

