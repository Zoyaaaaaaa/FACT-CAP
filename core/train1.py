import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so top-level packages (e.g. `preprocessing`) can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.custom_dataset import CustomDataset

# -----------------------
# Configuration
# -----------------------
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cpu")  # explicit CPU

# -----------------------
# Dataset & DataLoader
# -----------------------
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

# -----------------------
# Model (simple CNN)
# -----------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN().to(DEVICE)

# -----------------------
# Loss & Optimizer
# -----------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Training loop
# -----------------------
print("Starting training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
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
    print(f"Completed epoch {epoch+1}/{EPOCHS}")
    avg_train_loss = running_loss / len(train_loader)

    # -------------------
    # Validation
    # -------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "deepfake_cnn_cpuv1.pth")
print("Model saved as deepfake_cnn_cpuv1.pth")

# Epoch [10/10] Train Loss: 0.5759 Val Acc: 0.6567
# Model saved as deepfake_cnn_cpuv1.pth