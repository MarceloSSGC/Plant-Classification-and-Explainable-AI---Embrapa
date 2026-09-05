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
