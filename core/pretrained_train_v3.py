import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import roc_auc_score, roc_curve

# --------------------------------------------------
# Project root
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.custom_dataset import CustomDataset

# --------------------------------------------------
# Config
# --------------------------------------------------
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 6

HEAD_LR = 3e-4
LAYER4_LR = 1e-4
LAYER3_LR = 5e-5
WEIGHT_DECAY = 1e-4

# --------------------------------------------------
# Dataset
# --------------------------------------------------
train_dataset = CustomDataset(ROOT / "preprocessed" / "train")
val_dataset   = CustomDataset(ROOT / "preprocessed" / "val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    persistent_workers=True
)

# --------------------------------------------------
# Class balance
# --------------------------------------------------
pos = sum(label == 1 for _, label in train_dataset.samples)
neg = sum(label == 0 for _, label in train_dataset.samples)
POS_WEIGHT = neg / max(pos, 1)

print(f"POS_WEIGHT: {POS_WEIGHT:.2f}")

# --------------------------------------------------
# Model
# --------------------------------------------------
class ResNetDeepfake(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze mid + high level features
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

model = ResNetDeepfake().to(DEVICE)

# --------------------------------------------------
# Loss & Optimizer
# --------------------------------------------------
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE)
)

optimizer = optim.AdamW(
    [
        {"params": model.backbone.layer3.parameters(), "lr": LAYER3_LR},
        {"params": model.backbone.layer4.parameters(), "lr": LAYER4_LR},
        {"params": model.backbone.fc.parameters(), "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

# --------------------------------------------------
# Training
# --------------------------------------------------
best_val_auc = 0.0
early_stop = 0
best_threshold = 0.5

print("Training ResNet18 Deepfake Detector (v3)")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("Training...")

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if step % 20 == 0:
            print(f"  Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    scheduler.step()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

    
print("Training complete.")
print(f"Best Val AUC: {best_val_auc:.4f}")
print(f"Best Threshold: {best_threshold:.3f}")
