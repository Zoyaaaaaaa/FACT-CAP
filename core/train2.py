# improved using residual methods and deeper architecture

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# -------------------------------------------------
# Ensure project root is on sys.path
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.custom_dataset import CustomDataset

# -------------------------------------------------
# Configuration
# -------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cpu")  # explicit CPU

# -------------------------------------------------
# Dataset & DataLoader
# -------------------------------------------------
train_dataset = CustomDataset("preprocessed/train")
val_dataset   = CustomDataset("preprocessed/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=False
)

# -------------------------------------------------
# Residual Block
# -------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

# -------------------------------------------------
# Improved CNN Model
# -------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            nn.Dropout2d(0.2)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            nn.Dropout2d(0.3)
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            nn.Dropout2d(0.4)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)

# -------------------------------------------------
# Initialize Model, Loss, Optimizer, Scheduler
# -------------------------------------------------
model = CNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

# -------------------------------------------------
# Training Loop
# -------------------------------------------------
print("Starting training...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).squeeze()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
        f"precision: {correct}/{total}"
    )

# -------------------------------------------------
# Save Model
# -------------------------------------------------
torch.save(model.state_dict(), "deepfake_cnn_residual_cpu.pth")
print("\nModel saved as deepfake_cnn_residual_cpu.pth")
