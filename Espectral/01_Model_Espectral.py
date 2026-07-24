import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import argparse
from time import sleep

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#======================================================================

print("\n\t\t -- Model -- ")

#======================================================================
# Argparse

parser = argparse.ArgumentParser()
parser.add_argument("--BAND", type=int, required=True)
args = parser.parse_args()

BAND = args.BAND

print(f"\n BAND: {BAND}\n")
sleep(4)

# x = 1/0

#============================================================

# 1. Dataset PyTorch

experiment_name = "01_spectral_analysis_08_07"
experiment_type = "Spectral"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/{experiment_type}/{experiment_name}"

# BAND = 1

DIR_EXP_BAND = Path(DIR_EXP + f"/Band_0{BAND}")

TRAIN_DIR = DIR_EXP_BAND / "Train_Norm"
VAL_DIR   = DIR_EXP_BAND / "Val_Norm"
TEST_DIR  = DIR_EXP_BAND / "Test_Norm"

#============================================================
# Dataset Class

# class MultispectralWeedDataset(Dataset):
#     def __init__(self, root_dir, class_to_idx=None):
#         self.root_dir = Path(root_dir)

#         self.classes = sorted([
#             d.name for d in self.root_dir.iterdir()
#             if d.is_dir()
#         ])

#         if class_to_idx is None:
#             self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
#         else:
#             self.class_to_idx = class_to_idx

#         self.samples = []

#         for cls_name in self.classes:
#             cls_dir = self.root_dir / cls_name

#             for file_path in cls_dir.glob("*.npy"):
#                 label = self.class_to_idx[cls_name]
#                 self.samples.append((file_path, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         file_path, label = self.samples[idx]

#         img = np.load(file_path).astype(np.float32)  # (H, W, 5)

#         # numpy -> torch sem cópia extra grande
#         img = torch.from_numpy(img)

#         # (H, W, 5) -> (5, H, W)
#         img = img.permute(2, 0, 1)

#         label = torch.tensor(label, dtype=torch.long)

#         return img, label
    

class SingleBandWeedDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None):
        self.root_dir = Path(root_dir)

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        if class_to_idx is None:
            self.class_to_idx = {
                cls_name: i for i, cls_name in enumerate(self.classes)
            }
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

        # (H, W)
        img = np.load(file_path).astype(np.float32)

        # numpy -> torch
        img = torch.from_numpy(img)

        # (H, W) -> (1, H, W)
        img = img.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return img, label
    
#------------------------------------------------------------------------

# 2. DataLoaders

train_dataset = SingleBandWeedDataset(TRAIN_DIR)
class_to_idx = train_dataset.class_to_idx

val_dataset = SingleBandWeedDataset(
    VAL_DIR,
    class_to_idx=class_to_idx
)

test_dataset = SingleBandWeedDataset(
    TEST_DIR,
    class_to_idx=class_to_idx
)

idx_to_class = {v: k for k, v in class_to_idx.items()}

print("Classes:")
for k, v in class_to_idx.items():
    print(v, k)

print("\nTrain:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

#======================================================================
# 3. Modelo pequeno

# class SmallMultispectralCNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(5, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((1, 1))
#         )

#         self.classifier = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.flatten(1)
#         x = self.classifier(x)
#         return x


class SmallSingleBandCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
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
# Cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

#======================================================================
# Model & Config

num_classes = len(class_to_idx)

model = SmallSingleBandCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 40

#======================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        imgs = F.interpolate(
            imgs,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        optimizer.zero_grad(set_to_none=True)

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
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs = F.interpolate(
                imgs,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_labels, all_preds

#------------------------------------------------------------------------
# Train

best_val_acc = 0.0
best_model_path = DIR_EXP_BAND / "best_small_cnn_multispectral.pth"
loss_train_list = []
loss_val_list = []

acc_train_list = []
acc_val_list = []

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

    loss_train_list.append(train_loss)
    loss_val_list.append(val_loss)

    acc_train_list.append(train_acc)
    acc_val_list.append(val_acc)


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("Melhor modelo salvo.")

df_loss = pd.DataFrame()
df_loss["train_loss"] = loss_train_list
df_loss["val_loss"] = loss_val_list
df_loss["train_acc"] = acc_train_list
df_loss["val_acc"] = acc_val_list

df_loss[["train_loss","val_loss"]].plot()
df_loss[["train_acc","val_acc"]].plot()

loss_path = DIR_EXP_BAND / "df_loss.csv"
df_loss.to_csv(loss_path, index=False)


#======================================================================
# 5. Avaliação final em Val e Test

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

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

#======================================================================
# 6. Relatório de classificação

target_names = [idx_to_class[i] for i in range(num_classes)]

print("\nClassification Report - VAL")
val_report = classification_report(
    val_labels,
    val_preds,
    target_names=target_names,
    zero_division=0
)
print(val_report)

print("\nClassification Report - TEST")
test_report = classification_report(
    test_labels,
    test_preds,
    target_names=target_names,
    zero_division=0
)

print(test_report)

with open(DIR_EXP_BAND / "report_VAL.txt", "w", encoding="utf-8") as f:
    f.write(val_report)

with open(DIR_EXP_BAND / "report_TEST.txt", "w", encoding="utf-8") as f:
    f.write(test_report)

#======================================================================
# 7. Matriz de confusão

print("\nConfusion Matrix - VAL")
val_conf_mat = confusion_matrix(val_labels, val_preds)
val_conf_mat = pd.DataFrame(val_conf_mat, index=list(idx_to_class.values()), columns=list(idx_to_class.values()))
val_conf_mat.to_csv(DIR_EXP_BAND / "conf_mat_VAL.csv")
print(val_conf_mat)

print("\nConfusion Matrix - TEST")
test_conf_mat = confusion_matrix(test_labels, test_preds)
test_conf_mat = pd.DataFrame(test_conf_mat, index=list(idx_to_class.values()), columns=list(idx_to_class.values()))
test_conf_mat.to_csv(DIR_EXP_BAND / "conf_mat_TEST.csv")

print(test_conf_mat)