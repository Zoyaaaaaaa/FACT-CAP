import torch
import torch .nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import torch.optim as optim 
import optuna
from sklearn.metrics import accuracy_score
# Ensure project root is on sys.path so top-level packages (e.g. `preprocessing`) can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   
from preprocessing.custom_dataset import CustomDataset
# -----------------------
# Configuration
# -----------------------
BATCH_SIZE = 16
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
# Model (ResNet)
# -----------------------       
class CNN(nn.Module):
    def __init__(self, dropout_p):
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
            nn.Dropout2d(dropout_p),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

EPOCHS = 15

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    pos_weight_val = trial.suggest_float("pos_weight", 1.0, 1.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = CNN(dropout).to(DEVICE)

    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    best_val_acc = 0.0
    patience = 4
    counter = 0

    for epoch in range(EPOCHS):
        model.train()

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                preds_all.extend(preds.flatten())
                labels_all.extend(labels.numpy())

        val_acc = accuracy_score(labels_all, preds_all)

        trial.report(val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_trial.params)
print("Best validation accuracy:", study.best_value)
# Save the best model
best_params = study.best_trial.params
best_model = CNN(best_params["dropout"]).to(DEVICE)
pos_weight = torch.tensor([best_params["pos_weight"]]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(
    best_model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    num_workers=4

)
val_loader = DataLoader(
    val_dataset,
    batch_size=best_params["batch_size"],
    shuffle=False,
    num_workers=4
)
for epoch in range(EPOCHS):
    best_model.train()

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(best_model(images), labels)
        loss.backward()
        optimizer.step()

# Save the best model
torch.save(best_model.state_dict(), "best_deepfake_cnn_cpu_hpt.pth")        
print("Best model saved as best_deepfake_cnn_cpu_hpt.pth")