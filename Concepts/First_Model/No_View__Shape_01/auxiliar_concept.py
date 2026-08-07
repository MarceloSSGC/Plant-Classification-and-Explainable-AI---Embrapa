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
# Dataset Class - Multioutput (multilabel)

class MultispectralWeedMultilabelDataset(Dataset):
    def __init__(self, root_dir, df_shape, label_cols=None):
        """
        root_dir: diretório (TRAIN_DIR, VAL_DIR ou TEST_DIR)
        df_shape: DataFrame com a coluna 'especie' + colunas de rótulo (0/1)
        label_cols: lista de colunas de rótulo. Se None, assume que são
                    todas as colunas de df_shape exceto a primeira (especie)
        """
        self.root_dir = Path(root_dir)

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        if label_cols is None:
            label_cols = df_shape.columns[1:].tolist()
        self.label_cols = label_cols

        # mapa espécie -> vetor multirrótulo (float32, para BCEWithLogitsLoss)
        species_col = df_shape.columns[0]
        self.species_to_label = {
            row[species_col]: row[label_cols].to_numpy(dtype=np.float32)
            for _, row in df_shape.iterrows()
        }

        # sanity check: todas as pastas de classe precisam ter rótulo em df_shape
        missing = [c for c in self.classes if c not in self.species_to_label]
        if missing:
            raise ValueError(f"Espécies sem rótulo em df_shape: {missing}")

        self.samples = []

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            label_vec = self.species_to_label[cls_name]

            for file_path in cls_dir.glob("*.npy"):
                self.samples.append((file_path, label_vec, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_vec, cls_name = self.samples[idx]

        img = np.load(file_path).astype(np.float32)  # (H, W, 5)

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)  # (5, H, W)

        Y = torch.from_numpy(label_vec)  # (num_labels,), float32

        c_i = cls_name  # nome da espécie, ex: "03_brizantha_Agua_Boa_03"
        n_i = file_path.stem  # nome do arquivo (sem extensão), ex: "IMG_0003"

        return img, Y, c_i, n_i

#======================================================================

# results = test_results.copy()

def multioutput_classification_metrics_dataframe(results):

    n_concepts = results["true"].shape[1]

    metrics = pd.DataFrame()
    
    for i in range(n_concepts): # i = 0

        y_i_real = results["true"][:, i]
        y_i_pred = results["preds"][:, i]

        ith_metrics = classification_metrics_dataframe(
            y_real=y_i_real,
            y_pred=y_i_pred
        )

        metrics = pd.concat([metrics, ith_metrics], axis=0).reset_index(drop=True)

    return metrics

#======================================================================
#======================================================================
# Bloco convolucional auxiliar

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


# ======================================================================
# Modelo

class MultioutputSmallCNN(nn.Module):
    def __init__(self, in_channels=5, num_labels=8, base_channels=32):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(in_channels, base_channels),       # 5   -> 32
            conv_block(base_channels, base_channels * 2),  # 32  -> 64
            conv_block(base_channels * 2, base_channels * 4),  # 64  -> 128
            conv_block(base_channels * 4, base_channels * 8),  # 128 -> 256
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channels * 8, num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)  # sem sigmoid -> BCEWithLogitsLoss
        return logits

    # ------------------------------------------------------------------
    # Treino

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        device="cuda",
        threshold=0.5,
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=True,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        print("\n" + "="*60 + f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        history = {
            "train_loss": [], "val_loss": [],
            "val_f1_macro": [], "val_f1_micro": [], "val_exact_match": [],
        }

        best_val_f1_macro = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # ---- treino ----
            self.train()
            running_loss = 0.0
            n_samples = 0

            for imgs, Ys, c_is, n_is in train_loader:
                imgs, Ys = imgs.to(device), Ys.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, Ys)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                n_samples += imgs.size(0)

            train_loss = running_loss / n_samples

            # ---- validação ----
            val_loss, val_f1_macro, val_f1_micro, val_exact_match = self._evaluate(
                val_loader, criterion, device, threshold
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_micro"].append(val_f1_micro)
            history["val_exact_match"].append(val_exact_match)

            if verbose:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_f1_macro={val_f1_macro:.4f} | val_f1_micro={val_f1_micro:.4f} | "
                    f"val_exact_match={val_exact_match:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: F1 macro na validação) ----
            if val_f1_macro > best_val_f1_macro:
                best_val_f1_macro = val_f1_macro
                best_state = deepcopy(self.state_dict())
                torch.save(best_state, checkpoint_path)
                epochs_no_improve = 0
                if verbose:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_f1_macro={val_f1_macro:.4f})")
            else:
                epochs_no_improve += 1

            # ---- early stopping opcional ----
            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        # recarrega os melhores pesos encontrados durante o treino
        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device, threshold):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, Ys, c_is, n_is in loader:
            imgs, Ys = imgs.to(device), Ys.to(device)

            logits = self(imgs)
            loss = criterion(logits, Ys)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu())
            all_true.append(Ys.cpu())

        val_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        f1_macro = f1_score(true, preds, average="macro", zero_division=0)
        f1_micro = f1_score(true, preds, average="micro", zero_division=0)
        exact_match = float(np.mean(np.all(preds == true, axis=1)))

        return val_loss, f1_macro, f1_micro, exact_match

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda", threshold=0.5):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, Ys, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(Ys)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }






























