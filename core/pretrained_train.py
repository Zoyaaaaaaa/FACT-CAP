# train_pretrained_resnet.py
# Deepfake Detection using Pretrained ResNet18 (CPU-only)
# Logs train/val loss and accuracy with early stopping

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# -------------------------------------------------------------------
# Ensure project root is on sys.path
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.custom_dataset import CustomDataset

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 1.2

# -------------------------------------------------------------------
# Dataset & DataLoader
# -------------------------------------------------------------------
# train_dataset = CustomDataset("preprocessed/train")
# val_dataset   = CustomDataset("preprocessed/val")
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

# -------------------------------------------------------------------
# Pretrained Model
# -------------------------------------------------------------------
class ResNetDeepfake(nn.Module):
    def __init__(self, dropout=0.5, freeze_backbone=True):
        super().__init__()

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)

model = ResNetDeepfake(dropout=0.5, freeze_backbone=True).to(DEVICE)

# -------------------------------------------------------------------
# Loss & Optimizer
# -------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# -------------------------------------------------------------------
# Training Loop with Logging & Early Stopping
# -------------------------------------------------------------------
best_val_acc = 0.0
early_stop_counter = 0

print("Starting training with pretrained ResNet18 (CPU)...")

for epoch in range(EPOCHS):
    # ---------------- Training ----------------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        train_correct += (preds.squeeze() == labels.squeeze()).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            val_correct += (preds.squeeze() == labels.squeeze()).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ---------------- Early Stopping ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_resnet18_deepfake.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete.")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print("Best model saved as best_resnet18_deepfake.pth")
#Epoch [10/10] | Train Loss: 0.4754 | Train Acc: 0.7979 | Val Loss: 0.5818 | Val Acc: 0.7353
#Training complete.
#Best Validation Accuracy: 0.7487