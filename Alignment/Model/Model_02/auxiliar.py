from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

from time import sleep
from copy import deepcopy

#======================================================================
#======================================================================

def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
    print(data.shape)


#======================================================================

def classification_metrics_dataframe(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    class_names=None,
    zero_division=0
) -> pd.DataFrame:
    """
    Calcula métricas de classificação e retorna um DataFrame
    com uma única linha.

    Parameters
    ----------
    y_real : np.ndarray
        Classes verdadeiras, com formato (n_amostras,).

    y_pred : np.ndarray
        Classes preditas, com formato (n_amostras,).

    class_names : dict, list ou tuple, opcional
        Nomes das classes.

        Pode ser um dicionário no formato:
            {0: "classe_A", 1: "classe_B"}

        Ou uma lista:
            ["classe_A", "classe_B"]

        Caso não seja informado, serão utilizados os próprios
        valores das classes.

    zero_division : int ou float, padrão=0
        Valor usado quando precision ou recall não puderem ser
        calculados por ausência de amostras ou predições.

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha contendo as métricas gerais
        e as métricas de cada classe.
    """

    # --------------------------------------------------------------
    # Validação e padronização das entradas
    # --------------------------------------------------------------
    y_real = np.asarray(y_real).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_real.size == 0:
        raise ValueError("y_real não pode estar vazio.")

    if y_pred.size == 0:
        raise ValueError("y_pred não pode estar vazio.")

    if y_real.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_real e y_pred precisam ter o mesmo número de elementos. "
            f"Recebido: {y_real.shape[0]} e {y_pred.shape[0]}."
        )

    # Inclui classes presentes em y_real ou y_pred
    labels = np.unique(
        np.concatenate([y_real, y_pred])
    )

    # --------------------------------------------------------------
    # Define os nomes das classes
    # --------------------------------------------------------------
    if class_names is None:
        label_to_name = {
            label: str(label)
            for label in labels
        }

    elif isinstance(class_names, dict):
        label_to_name = {
            label: str(class_names.get(label, label))
            for label in labels
        }

    elif isinstance(class_names, (list, tuple)):
        label_to_name = {}

        for label in labels:
            try:
                label_to_name[label] = str(class_names[int(label)])
            except (IndexError, TypeError, ValueError):
                label_to_name[label] = str(label)

    else:
        raise TypeError(
            "class_names deve ser None, dict, list ou tuple."
        )

    # --------------------------------------------------------------
    # Métricas gerais
    # --------------------------------------------------------------
    metrics = {
        "n_amostras": y_real.shape[0],

        "acuracia": accuracy_score(
            y_real,
            y_pred
        ),

        "acuracia_balanceada": balanced_accuracy_score(
            y_real,
            y_pred
        ),

        "precision_macro": precision_score(
            y_real,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=zero_division
        ),

        "precision_micro": precision_score(
            y_real,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=zero_division
        ),

        "precision_weighted": precision_score(
            y_real,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=zero_division
        ),

        "recall_macro": recall_score(
            y_real,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=zero_division
        ),

        "recall_micro": recall_score(
            y_real,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=zero_division
        ),

        "recall_weighted": recall_score(
            y_real,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=zero_division
        ),

        "f1_macro": f1_score(
            y_real,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=zero_division
        ),

        "f1_micro": f1_score(
            y_real,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=zero_division
        ),

        "f1_weighted": f1_score(
            y_real,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=zero_division
        ),

        "cohen_kappa": cohen_kappa_score(
            y_real,
            y_pred
        ),

        "matthews_corrcoef": matthews_corrcoef(
            y_real,
            y_pred
        )
    }

    # --------------------------------------------------------------
    # Métricas por classe
    # --------------------------------------------------------------
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            y_real,
            y_pred,
            labels=labels,
            average=None,
            zero_division=zero_division
        )
    )

    for label, precision, recall, f1, n_samples in zip(
        labels,
        precision_per_class,
        recall_per_class,
        f1_per_class,
        support
    ):
        class_name = label_to_name[label]

        # Evita espaços e caracteres pouco convenientes nas colunas
        class_name = (
            class_name
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
        )

        metrics[f"precision__{class_name}"] = precision
        metrics[f"recall__{class_name}"] = recall
        metrics[f"f1__{class_name}"] = f1
        metrics[f"support__{class_name}"] = int(n_samples)

    return pd.DataFrame([metrics])

#======================================================================
# Dataset Class

class MultispectralWeedDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None):
        self.root_dir = Path(root_dir)

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

        img = np.load(file_path).astype(np.float32)  # (H, W, 5)

        # numpy -> torch sem cópia extra grande
        img = torch.from_numpy(img)

        # (H, W, 5) -> (5, H, W)
        img = img.permute(2, 0, 1)

        label = torch.tensor(label, dtype=torch.long)

        return img, label



#======================================================================
# Dataloader

def get_labels(loader):
    labels = []

    for _, y in loader:
        labels.append(y)

    return torch.cat(labels).numpy()

#======================================================================
# Model

class SmallMultispectralCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=5,
        image_size=(224, 224),
        device=None
    ):
        super().__init__()

        self.image_size = image_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.features = nn.Sequential(
            self._conv_block(input_channels, 16),
            nn.MaxPool2d(2),

            self._conv_block(16, 32),
            nn.MaxPool2d(2),

            self._conv_block(32, 64),
            nn.MaxPool2d(2),

            self._conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

        self.history = {}
        self.best_epoch = None
        self.best_val_acc = None

        self.to(self.device)

    #----------------------------------------------------
    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)

        return self.classifier(x)

    #----------------------------------------------------
    def _prepare_batch(self, images, labels):
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if images.shape[-2:] != self.image_size:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bilinear",
                align_corners=False
            )

        return images, labels

    #----------------------------------------------------
    def _run_epoch(self, loader, criterion, optimizer=None):
        is_training = optimizer is not None

        self.train(is_training)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            for images, labels in loader:
                images, labels = self._prepare_batch(images, labels)

                if is_training:
                    optimizer.zero_grad(set_to_none=True)

                outputs = self(images)
                loss = criterion(outputs, labels)

                if is_training:
                    loss.backward()
                    optimizer.step()

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                total_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

                total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples

        return epoch_loss, epoch_acc

    #----------------------------------------------------
    def fit(
        self,
        train_loader,
        val_loader,
        epochs=35,
        learning_rate=1e-3,
        criterion=None,
        optimizer=None,
        save_path=None,
        restore_best=True,
        verbose=True
    ):
        criterion = criterion or nn.CrossEntropyLoss()

        optimizer = optimizer or torch.optim.Adam(
            self.parameters(),
            lr=learning_rate
        )

        save_path = Path(save_path) if save_path else None

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        self.best_val_acc = float("-inf")
        self.best_epoch = None

        best_weights = None

        if verbose:
            print("\n\t--- Start Training ---\n")
            print(f"Device: {self.device}\n")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                train_loader,
                criterion,
                optimizer
            )

            val_loss, val_acc = self._run_epoch(
                val_loader,
                criterion
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                best_weights = deepcopy(self.state_dict())

                if save_path:
                    torch.save(best_weights, save_path)

                    if verbose:
                        print("Melhor modelo salvo.")

        if restore_best and best_weights is not None:
            self.load_state_dict(best_weights)

        if verbose:
            print(
                f"\nMelhor época: {self.best_epoch} | "
                f"Val Acc: {self.best_val_acc:.4f}"
            )

        return self
    
    #----------------------------------------------------   
    def predict(self, loader):

        self.eval()

        predictions = []
        probabilities = []

        with torch.no_grad():
            for images, labels in loader:
                images, _ = self._prepare_batch(images, labels)

                outputs = self(images)

                probs = torch.softmax(outputs, dim=1)

                predictions.append(
                    probs.argmax(dim=1).cpu()
                )

                probabilities.append(
                    probs.cpu()
                )

        predictions = torch.cat(predictions).numpy()
        probabilities = torch.cat(probabilities).numpy()

        return predictions, probabilities

    #----------------------------------------------------
    def predict_with_labels(self, loader):

        self.eval()

        y_true = []
        predictions = []
        probabilities = []

        with torch.no_grad():
            for images, labels in loader:

                images, labels = self._prepare_batch(images, labels)

                outputs = self(images)
                probs = torch.softmax(outputs, dim=1)

                y_true.append(labels.cpu())
                predictions.append(probs.argmax(dim=1).cpu())
                probabilities.append(probs.cpu())

        y_true = torch.cat(y_true).numpy()
        predictions = torch.cat(predictions).numpy()
        probabilities = torch.cat(probabilities).numpy()

        return y_true, predictions, probabilities


#======================================================================
# 