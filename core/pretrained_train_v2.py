# train_resnet18_improved.py
# Improved Deepfake Detection with Pretrained ResNet18 (CPU)
# - Partial unfreeze
# - Proper class weighting
# - ROC-AUC logging
# - Cosine LR scheduler
# - Better regularization

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

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
EPOCHS = 15
PATIENCE = 4

HEAD_LR = 3e-4
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
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
# Compute POS_WEIGHT properly
# -------------------------------------------------------------------
pos = sum(label == 1 for _, label in train_dataset.samples)
neg = sum(label == 0 for _, label in train_dataset.samples)
POS_WEIGHT = neg / max(pos, 1)

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
class ResNetDeepfake(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only layer4
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)

model = ResNetDeepfake(dropout=DROPOUT).to(DEVICE)

# -------------------------------------------------------------------
# Loss & Optimizer
# -------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE)
)

optimizer = optim.AdamW(
    [
        {"params": model.backbone.layer4.parameters(), "lr": BACKBONE_LR},
        {"params": model.backbone.fc.parameters(), "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

# -------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------
best_val_auc = 0.0
early_stop_counter = 0

print("Training ResNet18 (improved setup, CPU only)")

for epoch in range(EPOCHS):
    # -------------------- Training --------------------
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
        preds = (torch.sigmoid(outputs) > 0.5)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    scheduler.step()

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # -------------------- Validation --------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_auc = roc_auc_score(all_labels, all_probs)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    # -------------------- Early Stopping --------------------
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_resnet18_deepfake_v2.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete.")
print(f"Best Validation AUC: {best_val_auc:.4f}")

# Epoch [14/15] | Train Loss: 0.0047 | Train Acc: 0.9984 | Val Loss: 0.6558 | Val Acc: 0.8433 | Val AUC: 0.9418
# Epoch [15/15] | Train Loss: 0.0069 | Train Acc: 0.9977 | Val Loss: 0.5464 | Val Acc: 0.8580 | Val AUC: 0.9431
# Training complete.
# Best Validation AUC: 0.9431