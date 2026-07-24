import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Dataset PyTorch

# DIR_EXP = Path("/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/First_Test_Daninha_Boa_Vista")
DIR_EXP = Path(DIR_EXP)

TRAIN_DIR = DIR_EXP / "Train_Norm"
VAL_DIR   = DIR_EXP / "Val_Norm"
TEST_DIR  = DIR_EXP / "Test_Norm"


class MultispectralWeedDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None, image_size=(224, 224)):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        if class_to_idx is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        self.samples = []

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name

            for file_path in cls_dir.glob("*.npy"):
                label = self.class_to_idx[cls_name]
                self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        img = np.load(file_path)  # (H, W, 5)
        img = torch.tensor(img, dtype=torch.float32)

        # (H, W, 5) -> (5, H, W)
        img = img.permute(2, 0, 1)

        # adiciona batch temporário: (1, 5, H, W)
        img = img.unsqueeze(0)

        # resize
        img = F.interpolate(
            img,
            size=self.image_size,
            mode="bilinear",
            align_corners=False
        )

        # remove batch: (5, 224, 224)
        img = img.squeeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return img, label

# 2. DataLoaders

train_dataset = MultispectralWeedDataset(TRAIN_DIR)
class_to_idx = train_dataset.class_to_idx

val_dataset = MultispectralWeedDataset(
    VAL_DIR,
    class_to_idx=class_to_idx
)

test_dataset = MultispectralWeedDataset(
    TEST_DIR,
    class_to_idx=class_to_idx
)

idx_to_class = {v: k for k, v in class_to_idx.items()}

print("Classes:")
for k, v in class_to_idx.items():
    print(v, k)

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

batch_size = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

#======================================================================
# 3. Modelo pequeno

class SmallMultispectralCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

#======================================================================
# 4. Treinamento

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

num_classes = len(class_to_idx)

model = SmallMultispectralCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 30

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_labels, all_preds


best_val_acc = 0.0
best_model_path = DIR_EXP / "best_small_cnn_multispectral.pth"

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    print(
        f"Epoch [{epoch+1:02d}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("Melhor modelo salvo.")

#======================================================================
# 5. Avaliação final em Val e Test

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)

val_loss, val_acc, val_labels, val_preds = evaluate(
    model, val_loader, criterion, device
)

test_loss, test_acc, test_labels, test_preds = evaluate(
    model, test_loader, criterion, device
)

print("\nVAL")
print("Loss:", val_loss)
print("Accuracy:", val_acc)

print("\nTEST")
print("Loss:", test_loss)
print("Accuracy:", test_acc)

# 6. Relatório de classificação

target_names = [idx_to_class[i] for i in range(num_classes)]

print("\nClassification Report - VAL")
print(classification_report(
    val_labels,
    val_preds,
    target_names=target_names,
    zero_division=0
))

print("\nClassification Report - TEST")
print(classification_report(
    test_labels,
    test_preds,
    target_names=target_names,
    zero_division=0
))

# 7. Matriz de confusão

print("\nConfusion Matrix - VAL")
print(confusion_matrix(val_labels, val_preds))

print("\nConfusion Matrix - TEST")
print(confusion_matrix(test_labels, test_preds))