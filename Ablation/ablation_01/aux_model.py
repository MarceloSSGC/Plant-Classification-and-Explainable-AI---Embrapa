from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

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

print(f"\n\033[100;40m\t     --- Auxiliar Model RUN ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# Metrics

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
#======================================================================

def fill_band_with_value(img_5b, band, value=0):
    if not isinstance(band, (int, np.integer)) or band not in range(img_5b.shape[-1]):
        raise ValueError("band not in range(img_5b.shape[-1])")

    img_5b_tr = np.copy(img_5b)

    img_5b_tr[:, :, band] = value

    return img_5b_tr

#----------------------------------------------------------------------

def define_T(band, value):

    def func(img_5b):
        return fill_band_with_value(img_5b, band, value)

    return func

#======================================================================
#======================================================================
# Dataset Class - Multiclass (single-label)

class WeedDataset_Transform(Dataset):
    def __init__(self, root_dir, transform=None, expected_bands=None):
        """
        root_dir: diretório (TRAIN_DIR, VAL_DIR ou TEST_DIR)
                  contendo uma subpasta por classe, e dentro de cada uma,
                  os arquivos .npy das imagens (H, W, N), onde N é o número
                  de bandas/canais resultante do pré-processamento.
        transform: transformações opcionais a serem aplicadas no tensor da imagem
        expected_bands: (opcional) número de bandas esperado (N). Se informado,
                  cada amostra é validada contra esse valor ao ser carregada,
                  levantando erro em caso de inconsistência. Se None, o número
                  de bandas é inferido automaticamente a partir do primeiro
                  arquivo do dataset e fica disponível em self.num_bands.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        # mapa nome da classe -> índice
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]

            for file_path in cls_dir.glob("*.npy"):
                self.samples.append((file_path, cls_idx, cls_name))

        if not self.samples:
            raise RuntimeError(f"Nenhum arquivo .npy encontrado em {self.root_dir}")

        # Determina o número de bandas (N) a partir do primeiro arquivo,
        # ou usa/valida contra o valor informado em expected_bands.
        first_file = self.samples[0][0]
        inferred_bands = np.load(first_file, mmap_mode="r").shape[-1]

        if expected_bands is not None and inferred_bands != expected_bands:
            raise ValueError(
                f"Número de bandas inconsistente: esperado {expected_bands}, "
                f"encontrado {inferred_bands} em {first_file}"
            )

        self.num_bands = expected_bands if expected_bands is not None else inferred_bands

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, cls_idx, cls_name = self.samples[idx]

        img = np.load(file_path).astype(np.float32)  # (H, W, N)

        if self.transform is not None:
            img = self.transform(img)

        if img.shape[-1] != self.num_bands:
            raise ValueError(
                f"{file_path} tem {img.shape[-1]} bandas, esperado {self.num_bands}"
            )

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)  # (N, H, W)


        y_i = torch.tensor(cls_idx, dtype=torch.long)  # índice da classe (0..30)
        c_i = cls_name                                  # nome da espécie
        n_i = file_path.stem                             # nome do arquivo, ex: "IMG_0035__rotation_180"

        return img, y_i, c_i, n_i

    
#======================================================================
# Dataset Class - Multiclass (single-label)

class MultispectralWeedDataset(Dataset):
    def __init__(self, root_dir, transform=None, expected_bands=None):
        """
        root_dir: diretório (TRAIN_DIR, VAL_DIR ou TEST_DIR)
                  contendo uma subpasta por classe, e dentro de cada uma,
                  os arquivos .npy das imagens (H, W, N), onde N é o número
                  de bandas/canais resultante do pré-processamento.
        transform: transformações opcionais a serem aplicadas no tensor da imagem
        expected_bands: (opcional) número de bandas esperado (N). Se informado,
                  cada amostra é validada contra esse valor ao ser carregada,
                  levantando erro em caso de inconsistência. Se None, o número
                  de bandas é inferido automaticamente a partir do primeiro
                  arquivo do dataset e fica disponível em self.num_bands.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        # mapa nome da classe -> índice
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]

            for file_path in cls_dir.glob("*.npy"):
                self.samples.append((file_path, cls_idx, cls_name))

        if not self.samples:
            raise RuntimeError(f"Nenhum arquivo .npy encontrado em {self.root_dir}")

        # Determina o número de bandas (N) a partir do primeiro arquivo,
        # ou usa/valida contra o valor informado em expected_bands.
        first_file = self.samples[0][0]
        inferred_bands = np.load(first_file, mmap_mode="r").shape[-1]

        if expected_bands is not None and inferred_bands != expected_bands:
            raise ValueError(
                f"Número de bandas inconsistente: esperado {expected_bands}, "
                f"encontrado {inferred_bands} em {first_file}"
            )

        self.num_bands = expected_bands if expected_bands is not None else inferred_bands

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, cls_idx, cls_name = self.samples[idx]

        img = np.load(file_path).astype(np.float32)  # (H, W, N)

        if img.shape[-1] != self.num_bands:
            raise ValueError(
                f"{file_path} tem {img.shape[-1]} bandas, esperado {self.num_bands}"
            )

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)  # (N, H, W)

        if self.transform is not None:
            img = self.transform(img)

        y_i = torch.tensor(cls_idx, dtype=torch.long)  # índice da classe (0..30)
        c_i = cls_name                                  # nome da espécie
        n_i = file_path.stem                             # nome do arquivo, ex: "IMG_0035__rotation_180"

        return img, y_i, c_i, n_i

# #======================================================================
# #======================================================================
# # # Modelo - Multiclass (single-label)

# def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#     )


# #======================================================================


# class MulticlassSmallCNN(nn.Module):
#     def __init__(self, in_channels=5, num_classes=31, base_channels=32):
#         super().__init__()

#         self.features = nn.Sequential(
#             conv_block(in_channels, base_channels),            # 5   -> 32
#             conv_block(base_channels, base_channels * 2),       # 32  -> 64
#             conv_block(base_channels * 2, base_channels * 4),   # 64  -> 128
#             conv_block(base_channels * 4, base_channels * 8),   # 128 -> 256
#         )

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(base_channels * 8, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x)
#         x = torch.flatten(x, 1)
#         logits = self.classifier(x)  # sem softmax -> CrossEntropyLoss
#         return logits

#     # ------------------------------------------------------------------
#     # Treino

#     def fit(
#         self,
#         train_loader,
#         val_loader,
#         epochs=30,
#         lr=1e-3,
#         weight_decay=1e-4,
#         device="cuda",
#         checkpoint_path="best_model.pt",
#         patience=None,
#         verbose=True,
#     ):
#         self.to(device)
#         optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
#         criterion = nn.CrossEntropyLoss()

#         print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

#         metric_names = [
#             "loss", "acc", "balanced_acc",
#             "f1_macro", "f1_micro",
#             "precision_macro", "recall_macro",
#             "kappa",
#         ]
#         history = {}
#         for m in metric_names:
#             history[f"train_{m}"] = []
#             history[f"val_{m}"] = []

#         best_val_f1_macro = -1.0
#         best_state = None
#         epochs_no_improve = 0

#         for epoch in range(1, epochs + 1):
#             # ---- passo de otimização (treino) ----
#             self.train()
#             for imgs, y_is, c_is, n_is in train_loader:
#                 imgs, y_is = imgs.to(device), y_is.to(device)

#                 optimizer.zero_grad()
#                 logits = self(imgs)
#                 loss = criterion(logits, y_is)
#                 loss.backward()
#                 optimizer.step()

#             # ---- avaliação em treino e validação (mesma métrica, mesmo critério) ----
#             train_metrics = self._evaluate(train_loader, criterion, device)
#             val_metrics = self._evaluate(val_loader, criterion, device)

#             for m in metric_names:
#                 history[f"train_{m}"].append(train_metrics[m])
#                 history[f"val_{m}"].append(val_metrics[m])

#             if verbose:
#                 print(
#                     f"\n[Epoch {epoch:03d}/{epochs}] "
#                     f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
#                     f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
#                     f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
#                 )

#             # ---- checkpoint do melhor modelo (critério: F1 macro na validação) ----
#             if val_metrics["f1_macro"] > best_val_f1_macro:
#                 best_val_f1_macro = val_metrics["f1_macro"]
#                 best_state = deepcopy(self.state_dict())
#                 torch.save(best_state, checkpoint_path)
#                 epochs_no_improve = 0
#                 if verbose:
#                     print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_f1_macro={best_val_f1_macro:.4f})")
#             else:
#                 epochs_no_improve += 1

#             # ---- early stopping opcional ----
#             if patience is not None and epochs_no_improve >= patience:
#                 if verbose:
#                     print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
#                 break

#         # recarrega os melhores pesos encontrados durante o treino
#         if best_state is not None:
#             self.load_state_dict(best_state)

#         return history

#     # ------------------------------------------------------------------
#     # Avaliação interna (usada no fit, tanto para train quanto para val)

#     @torch.no_grad()
#     def _evaluate(self, loader, criterion, device):
#         self.eval()
#         running_loss = 0.0
#         n_samples = 0
#         all_preds, all_true = [], []

#         for imgs, y_is, c_is, n_is in loader:
#             imgs, y_is = imgs.to(device), y_is.to(device)

#             logits = self(imgs)
#             loss = criterion(logits, y_is)
#             running_loss += loss.item() * imgs.size(0)
#             n_samples += imgs.size(0)

#             preds = torch.argmax(logits, dim=1)

#             all_preds.append(preds.cpu())
#             all_true.append(y_is.cpu())

#         avg_loss = running_loss / n_samples

#         preds = torch.cat(all_preds).numpy()
#         true = torch.cat(all_true).numpy()

#         metrics = {
#             "loss": avg_loss,
#             "acc": float(np.mean(preds == true)),
#             "balanced_acc": balanced_accuracy_score(true, preds),
#             "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
#             "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
#             "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
#             "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
#             "kappa": cohen_kappa_score(true, preds),
#         }

#         return metrics

#     # ------------------------------------------------------------------
#     # Predição

#     @torch.no_grad()
#     def predict(self, loader, device="cuda"):
#         self.to(device)
#         self.eval()

#         all_probs, all_preds, all_true = [], [], []
#         all_species, all_names = [], []

#         for imgs, y_is, c_is, n_is in loader:
#             imgs = imgs.to(device)
#             logits = self(imgs)
#             probs = torch.softmax(logits, dim=1)
#             preds = torch.argmax(probs, dim=1)

#             all_probs.append(probs.cpu())
#             all_preds.append(preds.cpu())
#             all_true.append(y_is)
#             all_species.extend(c_is)
#             all_names.extend(n_is)

#         return {
#             "probs": torch.cat(all_probs).numpy(),
#             "preds": torch.cat(all_preds).numpy(),
#             "true": torch.cat(all_true).numpy(),
#             "species": all_species,
#             "filenames": all_names,
#         }

# #======================================================================
