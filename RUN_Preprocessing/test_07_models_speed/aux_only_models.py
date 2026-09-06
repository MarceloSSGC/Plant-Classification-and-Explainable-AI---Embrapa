import numpy as np
import random
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

print(f"\n\033[100;40m\t     --- Auxiliar Only Models ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# SmallCNN

class MulticlassSmallCNN(nn.Module):
    MODEL_NAME = "SmallCNN"

    def __init__(self, in_channels=5, num_classes=31, base_channels=32, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_channels": base_channels,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        self.features = nn.Sequential(
            self._conv_block(in_channels, base_channels),            # 5   -> 32
            self._conv_block(base_channels, base_channels * 2),       # 32  -> 64
            self._conv_block(base_channels * 2, base_channels * 4),   # 64  -> 128
            self._conv_block(base_channels * 4, base_channels * 8),   # 128 -> 256
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        self.classifier = nn.Linear(base_channels * 8, num_classes)

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Bloco convolucional (agora interno à classe)

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)  # sem softmax -> CrossEntropyLoss
        return logits

    # ------------------------------------------------------------------
    # Helper para reportar os devices em uso

    def _print_device_report(self, device):
        print("\n\t Device report:")
        print(f"\t  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"\t  - GPU(s) visível(is): {torch.cuda.device_count()}")
            print(f"\t  - GPU atual: {torch.cuda.current_device()} "
                  f"({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"\t  - device solicitado para treino: {device}")
        print(f"\t  - device dos parâmetros do modelo: {next(self.parameters()).device}\n")

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=1,
    ):
        # ---- verbose: 0 = silencioso | 1 = resumo por época (default) | 2 = também progresso por batch ----
        self.to(device)

        # ---- reporta os devices disponíveis/utilizados antes de começar o treino ----
        self._print_device_report(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        if verbose >= 1:
            print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            # ---- passo de otimização (treino) ----
            self.train()
            for batch_idx, (imgs, y_is, c_is, n_is) in enumerate(train_loader, start=1):
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

                # ---- progresso por batch (só aparece com verbose=2) ----
                if verbose >= 2:
                    print(
                        f"\r    [Epoch {epoch:03d}/{epochs}] "
                        f"batch {batch_idx:04d}/{n_batches} - loss={loss.item():.4f}",
                        end="", flush=True,
                    )

            if verbose >= 2:
                print()  # quebra de linha após a barra de progresso da última batch

            # ---- avaliação em treino e validação (mesma métrica, mesmo critério) ----
            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose >= 1:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: loss na validação, não acurácia) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose >= 1:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # ---- early stopping opcional ----
            if patience is not None and epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        # recarrega os melhores pesos encontrados durante o treino
        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model



#======================================================================
#======================================================================
# Modelo - MobileNetV3 Small - Multiclass (single-label)

from torchvision.models import mobilenet_v3_small


class MulticlassMobileNetV3Small(nn.Module):
    MODEL_NAME = "MobileNetV3Small"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # ---- adapta a primeira camada para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.features[0][0]  # Conv2d(3, 16, kernel=3, stride=2, padding=1, bias=False)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.features[0][0] = new_conv

        # ---- adapta a cabeça de classificação para num_classes ----
        in_features = backbone.classifier[3].in_features  # 1024
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        if dropout is not None:
            backbone.classifier[2] = nn.Dropout(p=dropout, inplace=True)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

    # ------------------------------------------------------------------
    # Helper para reportar os devices em uso

    def _print_device_report(self, device):
        print("\n\t Device report:")
        print(f"\t  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"\t  - GPU(s) visível(is): {torch.cuda.device_count()}")
            print(f"\t  - GPU atual: {torch.cuda.current_device()} "
                  f"({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"\t  - device solicitado para treino: {device}")
        print(f"\t  - device dos parâmetros do modelo: {next(self.parameters()).device}\n")

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=1,
    ):
        # ---- verbose: 0 = silencioso | 1 = resumo por época (default) | 2 = também progresso por batch ----
        self.to(device)

        # ---- reporta os devices disponíveis/utilizados antes de começar o treino ----
        self._print_device_report(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        if verbose >= 1:
            print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            # ---- passo de otimização (treino) ----
            self.train()
            for batch_idx, (imgs, y_is, c_is, n_is) in enumerate(train_loader, start=1):
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

                # ---- progresso por batch (só aparece com verbose=2) ----
                if verbose >= 2:
                    print(
                        f"\r    [Epoch {epoch:03d}/{epochs}] "
                        f"batch {batch_idx:04d}/{n_batches} - loss={loss.item():.4f}",
                        end="", flush=True,
                    )

            if verbose >= 2:
                print()  # quebra de linha após a barra de progresso da última batch

            # ---- avaliação em treino e validação (mesma métrica, mesmo critério) ----
            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose >= 1:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: loss na validação, não acurácia) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose >= 1:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # ---- early stopping opcional ----
            if patience is not None and epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        # recarrega os melhores pesos encontrados durante o treino
        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model






# class MulticlassMobileNetV3Small(nn.Module):
#     MODEL_NAME = "MobileNetV3Small"

#     def __init__(
#         self,
#         in_channels=5,
#         num_classes=31,
#         pretrained=False,
#         dropout=0.2,
#         seed_model=None
#     ):
#         super().__init__()

#         self.seed_model = seed_model
#         self.set_seed(seed_model)

#         self.config = {
#             "in_channels": in_channels,
#             "num_classes": num_classes,
#             "pretrained": pretrained,
#             "dropout": dropout,
#             "seed_model": seed_model,
#         }

#         self.num_classes = num_classes

#         # ------------------------------------------------------------
#         # Backbone
#         # ------------------------------------------------------------

#         weights = "IMAGENET1K_V1" if pretrained else None
#         backbone = mobilenet_v3_small(weights=weights)

#         # ------------------------------------------------------------
#         # Adapta primeira convolução para N bandas
#         # ------------------------------------------------------------

#         old_conv = backbone.features[0][0]

#         new_conv = nn.Conv2d(
#             in_channels,
#             old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             padding=old_conv.padding,
#             dilation=old_conv.dilation,
#             groups=old_conv.groups,
#             bias=(old_conv.bias is not None),
#             padding_mode=old_conv.padding_mode,
#         )

#         if pretrained:
#             with torch.no_grad():

#                 # RGB
#                 n_copy = min(3, in_channels)

#                 new_conv.weight[:, :n_copy, :, :] = \
#                     old_conv.weight[:, :n_copy, :, :]

#                 # Bandas adicionais
#                 if in_channels > 3:
#                     mean_w = old_conv.weight.mean(
#                         dim=1,
#                         keepdim=True
#                     )

#                     new_conv.weight[:, 3:, :, :] = mean_w.repeat(
#                         1,
#                         in_channels - 3,
#                         1,
#                         1
#                     )

#         backbone.features[0][0] = new_conv

#         # ------------------------------------------------------------
#         # Classificador
#         # ------------------------------------------------------------

#         in_features = backbone.classifier[3].in_features

#         backbone.classifier[3] = nn.Linear(
#             in_features,
#             num_classes
#         )

#         if dropout is not None:
#             backbone.classifier[2] = nn.Dropout(
#                 p=dropout,
#                 inplace=True
#             )

#         self.backbone = backbone

#     # ================================================================
#     # Seed
#     # ================================================================

#     @staticmethod
#     def set_seed(seed=None):

#         if seed is None:
#             return

#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     # ================================================================
#     # Forward
#     # ================================================================

#     def forward(self, x):
#         return self.backbone(x)

#     # ================================================================
#     # Device report
#     # ================================================================

#     def _print_device_report(self, device):

#         print("\n\t Device report:")
#         print(
#             f"\t  - torch.cuda.is_available(): "
#             f"{torch.cuda.is_available()}"
#         )

#         if torch.cuda.is_available():

#             print(
#                 f"\t  - GPU(s) visível(is): "
#                 f"{torch.cuda.device_count()}"
#             )

#             print(
#                 f"\t  - GPU atual: "
#                 f"{torch.cuda.current_device()} "
#                 f"({torch.cuda.get_device_name(torch.cuda.current_device())})"
#             )

#         print(
#             f"\t  - device solicitado para treino: {device}"
#         )

#         print(
#             f"\t  - device dos parâmetros do modelo: "
#             f"{next(self.parameters()).device}\n"
#         )

#     # ================================================================
#     # Helpers de diagnóstico
#     # ================================================================

#     @staticmethod
#     def _tensor_stats(x):

#         x_detached = x.detach()

#         nan_count = torch.isnan(x_detached).sum().item()
#         inf_count = torch.isinf(x_detached).sum().item()

#         finite_mask = torch.isfinite(x_detached)

#         if finite_mask.any().item():

#             finite_values = x_detached[finite_mask]

#             min_value = finite_values.min().item()
#             max_value = finite_values.max().item()
#             mean_value = finite_values.float().mean().item()

#         else:

#             min_value = float("nan")
#             max_value = float("nan")
#             mean_value = float("nan")

#         return {
#             "nan": nan_count,
#             "inf": inf_count,
#             "min": min_value,
#             "max": max_value,
#             "mean": mean_value,
#         }

#     def _print_bad_batch(
#         self,
#         stage,
#         epoch,
#         batch_idx,
#         imgs,
#         y_is,
#         logits=None,
#         loss=None,
#         names=None,
#     ):

#         print("\n")
#         print("=" * 80)
#         print("🚨 DIAGNÓSTICO NUMÉRICO")
#         print("=" * 80)

#         print(f"Stage : {stage}")
#         print(f"Epoch : {epoch}")
#         print(f"Batch : {batch_idx}")

#         # ------------------------------------------------------------
#         # Input
#         # ------------------------------------------------------------

#         img_stats = self._tensor_stats(imgs)

#         print("\nINPUT")
#         print(f"  shape : {tuple(imgs.shape)}")
#         print(f"  dtype : {imgs.dtype}")
#         print(f"  device: {imgs.device}")

#         print(
#             f"  min/max/mean: "
#             f"{img_stats['min']:.6g} / "
#             f"{img_stats['max']:.6g} / "
#             f"{img_stats['mean']:.6g}"
#         )

#         print(
#             f"  NaN={img_stats['nan']} | "
#             f"Inf={img_stats['inf']}"
#         )

#         # ------------------------------------------------------------
#         # Labels
#         # ------------------------------------------------------------

#         print("\nLABELS")

#         print(
#             f"  min={y_is.min().item()} | "
#             f"max={y_is.max().item()}"
#         )

#         print(
#             f"  values={y_is.detach().cpu().tolist()}"
#         )

#         # ------------------------------------------------------------
#         # Arquivos
#         # ------------------------------------------------------------

#         if names is not None:

#             print("\nFILES")

#             for name in names:
#                 print(f"  {name}")

#         # ------------------------------------------------------------
#         # Logits
#         # ------------------------------------------------------------

#         if logits is not None:

#             logit_stats = self._tensor_stats(logits)

#             print("\nLOGITS")

#             print(
#                 f"  shape : {tuple(logits.shape)}"
#             )

#             print(
#                 f"  min/max/mean: "
#                 f"{logit_stats['min']:.6g} / "
#                 f"{logit_stats['max']:.6g} / "
#                 f"{logit_stats['mean']:.6g}"
#             )

#             print(
#                 f"  NaN={logit_stats['nan']} | "
#                 f"Inf={logit_stats['inf']}"
#             )

#         # ------------------------------------------------------------
#         # Loss
#         # ------------------------------------------------------------

#         if loss is not None:

#             print("\nLOSS")

#             try:
#                 print(f"  value={loss.detach().item()}")
#             except Exception:
#                 print("  value=<não disponível>")

#         print("=" * 80)
#         print()

#     # ================================================================
#     # Verifica gradientes
#     # ================================================================

#     def _check_gradients(
#         self,
#         epoch,
#         batch_idx,
#         max_abs_grad=1e6,
#     ):

#         largest_grad = 0.0
#         largest_name = None

#         for name, p in self.named_parameters():

#             if p.grad is None:
#                 continue

#             grad = p.grad.detach()

#             # NaN
#             if torch.isnan(grad).any().item():

#                 print("\n" + "=" * 80)
#                 print("🚨 NaN NO GRADIENTE")
#                 print(f"Epoch: {epoch}")
#                 print(f"Batch: {batch_idx}")
#                 print(f"Parameter: {name}")
#                 print("=" * 80)

#                 return False

#             # Inf
#             if torch.isinf(grad).any().item():

#                 print("\n" + "=" * 80)
#                 print("🚨 Inf NO GRADIENTE")
#                 print(f"Epoch: {epoch}")
#                 print(f"Batch: {batch_idx}")
#                 print(f"Parameter: {name}")
#                 print("=" * 80)

#                 return False

#             max_grad = grad.abs().max().item()

#             if max_grad > largest_grad:
#                 largest_grad = max_grad
#                 largest_name = name

#             # Gradiente absurdamente grande
#             if max_grad > max_abs_grad:

#                 print("\n" + "=" * 80)
#                 print("🚨 GRADIENTE EXPLOSIVO")
#                 print(f"Epoch: {epoch}")
#                 print(f"Batch: {batch_idx}")
#                 print(f"Parameter: {name}")
#                 print(f"max |grad|: {max_grad:.6e}")
#                 print("=" * 80)

#                 return False

#         return True

#     # ================================================================
#     # Verifica parâmetros depois do optimizer
#     # ================================================================

#     def _check_parameters(
#         self,
#         epoch,
#         batch_idx,
#     ):

#         for name, p in self.named_parameters():

#             data = p.detach()

#             if torch.isnan(data).any().item():

#                 print("\n" + "=" * 80)
#                 print("🚨 NaN NOS PARÂMETROS")
#                 print(f"Epoch: {epoch}")
#                 print(f"Batch: {batch_idx}")
#                 print(f"Parameter: {name}")
#                 print("=" * 80)

#                 return False

#             if torch.isinf(data).any().item():

#                 print("\n" + "=" * 80)
#                 print("🚨 Inf NOS PARÂMETROS")
#                 print(f"Epoch: {epoch}")
#                 print(f"Batch: {batch_idx}")
#                 print(f"Parameter: {name}")
#                 print("=" * 80)

#                 return False

#         return True

#     # ================================================================
#     # Fit
#     # ================================================================

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
#         verbose=1,

#         # ------------------------------------------------------------
#         # Diagnóstico
#         # ------------------------------------------------------------

#         debug_numerics=True,
#         loss_warning_threshold=20.0,
#         loss_abort_threshold=1e20,
#         max_abs_grad=1000.0,
#     ):

#         self.to(device)

#         self._print_device_report(device)

#         optimizer = optim.Adam(
#             self.parameters(),
#             lr=lr,
#             weight_decay=weight_decay
#         )

#         criterion = nn.CrossEntropyLoss()

#         if verbose >= 1:
#             print(
#                 f"\n\t Trainning... epochs: "
#                 f"\033[96;96m{epochs}\033[0m\n"
#             )

#         if debug_numerics:

#             print("\t Numerical debugging: ON")
#             print(
#                 f"\t  - loss warning threshold: "
#                 f"{loss_warning_threshold}"
#             )
#             print(
#                 f"\t  - loss abort threshold: "
#                 f"{loss_abort_threshold}"
#             )
#             print(
#                 f"\t  - max |gradient|: "
#                 f"{max_abs_grad:.2e}\n"
#             )

#         metric_names = [
#             "loss",
#             "acc",
#             "balanced_acc",
#             "f1_macro",
#             "f1_micro",
#             "precision_macro",
#             "recall_macro",
#             "kappa",
#         ]

#         history = {}

#         for m in metric_names:
#             history[f"train_{m}"] = []
#             history[f"val_{m}"] = []

#         best_val_loss = float("inf")
#         best_state = None
#         epochs_no_improve = 0

#         n_batches = len(train_loader)

#         # ============================================================
#         # Epochs
#         # ============================================================

#         for epoch in range(1, epochs + 1):

#             self.train()

#             # Para monitorar a própria fase de otimização
#             epoch_running_loss = 0.0
#             epoch_samples = 0

#             max_batch_loss = -float("inf")
#             max_batch_idx = None

#             for batch_idx, (
#                 imgs,
#                 y_is,
#                 c_is,
#                 n_is
#             ) in enumerate(train_loader, start=1):

#                 # ----------------------------------------------------
#                 # Antes da GPU
#                 # ----------------------------------------------------

#                 if debug_numerics:

#                     if torch.isnan(imgs).any().item() or \
#                        torch.isinf(imgs).any().item():

#                         self._print_bad_batch(
#                             "INPUT CPU",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "NaN/Inf encontrado no input CPU."
#                         )

#                     # labels também precisam ser válidos
#                     if y_is.min().item() < 0 or \
#                        y_is.max().item() >= self.num_classes:

#                         self._print_bad_batch(
#                             "INVALID LABEL",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "Label fora do intervalo válido."
#                         )

#                 # ----------------------------------------------------
#                 # CPU -> GPU
#                 # ----------------------------------------------------

#                 imgs = imgs.to(device)
#                 y_is = y_is.to(device)

#                 # ----------------------------------------------------
#                 # Input na GPU
#                 #
#                 # IMPORTANTE:
#                 # usamos isnan/isinf separadamente, pois durante
#                 # os testes torch.isfinite().all() apresentou
#                 # comportamento inconsistente em uma sessão.
#                 # ----------------------------------------------------

#                 if debug_numerics:

#                     if torch.isnan(imgs).any().item() or \
#                        torch.isinf(imgs).any().item():

#                         self._print_bad_batch(
#                             "INPUT GPU",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "NaN/Inf encontrado no input GPU."
#                         )

#                 # ----------------------------------------------------
#                 # Forward
#                 # ----------------------------------------------------

#                 optimizer.zero_grad(set_to_none=True)

#                 logits = self(imgs)

#                 if debug_numerics:

#                     if torch.isnan(logits).any().item() or \
#                        torch.isinf(logits).any().item():

#                         self._print_bad_batch(
#                             "FORWARD / LOGITS",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             logits=logits,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "NaN/Inf encontrado nos logits."
#                         )

#                 # ----------------------------------------------------
#                 # Loss
#                 # ----------------------------------------------------

#                 loss = criterion(logits, y_is)
#                 loss_value = loss.detach().item()

#                 if debug_numerics:

#                     if not np.isfinite(loss_value):

#                         self._print_bad_batch(
#                             "LOSS NaN/Inf",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             logits=logits,
#                             loss=loss,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "Loss NaN/Inf detectada."
#                         )

#                     if loss_value >= loss_abort_threshold:

#                         self._print_bad_batch(
#                             "LOSS EXPLOSIVA",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             logits=logits,
#                             loss=loss,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             f"Loss >= {loss_abort_threshold}."
#                         )

#                     if loss_value >= loss_warning_threshold:

#                         print(
#                             f"\n⚠️  Loss elevada | "
#                             f"epoch={epoch} | "
#                             f"batch={batch_idx}/{n_batches} | "
#                             f"loss={loss_value:.6f}"
#                         )

#                 # ----------------------------------------------------
#                 # Estatísticas da época
#                 # ----------------------------------------------------

#                 epoch_running_loss += (
#                     loss_value * imgs.size(0)
#                 )

#                 epoch_samples += imgs.size(0)

#                 if loss_value > max_batch_loss:
#                     max_batch_loss = loss_value
#                     max_batch_idx = batch_idx

#                 # ----------------------------------------------------
#                 # Backward
#                 # ----------------------------------------------------

#                 loss.backward()

#                 if debug_numerics:

#                     gradients_ok = self._check_gradients(
#                         epoch=epoch,
#                         batch_idx=batch_idx,
#                         max_abs_grad=max_abs_grad,
#                     )

#                     if not gradients_ok:

#                         self._print_bad_batch(
#                             "BACKWARD",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             logits=logits,
#                             loss=loss,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "Gradiente inválido/explosivo detectado."
#                         )

#                 # ----------------------------------------------------
#                 # Optimizer
#                 # ----------------------------------------------------

#                 optimizer.step()

#                 if debug_numerics:

#                     parameters_ok = self._check_parameters(
#                         epoch=epoch,
#                         batch_idx=batch_idx,
#                     )

#                     if not parameters_ok:

#                         self._print_bad_batch(
#                             "OPTIMIZER STEP",
#                             epoch,
#                             batch_idx,
#                             imgs,
#                             y_is,
#                             logits=logits,
#                             loss=loss,
#                             names=n_is,
#                         )

#                         raise RuntimeError(
#                             "Parâmetro NaN/Inf após optimizer.step()."
#                         )

#                 # ----------------------------------------------------
#                 # Print por batch
#                 # ----------------------------------------------------

#                 if verbose >= 2:

#                     train_loss_so_far = (
#                         epoch_running_loss / epoch_samples
#                     )

#                     print(
#                         f"\r"
#                         f"    [Epoch {epoch:03d}/{epochs}] "
#                         f"batch {batch_idx:04d}/{n_batches} | "
#                         f"loss={loss_value:.4f} | "
#                         f"avg={train_loss_so_far:.4f} | "
#                         f"max={max_batch_loss:.4f}",
#                         end="",
#                         flush=True,
#                     )

#             if verbose >= 2:
#                 print()

#             # --------------------------------------------------------
#             # Informação adicional da fase de otimização
#             # --------------------------------------------------------

#             optimization_loss = (
#                 epoch_running_loss / epoch_samples
#             )

#             if verbose >= 1:

#                 print(
#                     f"\n  Optimization epoch {epoch}: "
#                     f"avg_loss={optimization_loss:.6f} | "
#                     f"max_batch_loss={max_batch_loss:.6f} "
#                     f"(batch {max_batch_idx})"
#                 )

#             # ========================================================
#             # Avaliação
#             # ========================================================

#             train_metrics = self._evaluate(
#                 train_loader,
#                 criterion,
#                 device,
#                 split_name="TRAIN-EVAL",
#                 epoch=epoch,
#                 debug_numerics=debug_numerics,
#                 loss_abort_threshold=loss_abort_threshold,
#             )

#             val_metrics = self._evaluate(
#                 val_loader,
#                 criterion,
#                 device,
#                 split_name="VAL",
#                 epoch=epoch,
#                 debug_numerics=debug_numerics,
#                 loss_abort_threshold=loss_abort_threshold,
#             )

#             for m in metric_names:

#                 history[f"train_{m}"].append(
#                     train_metrics[m]
#                 )

#                 history[f"val_{m}"].append(
#                     val_metrics[m]
#                 )

#             if verbose >= 1:

#                 print(
#                     f"\n[Epoch {epoch:03d}/{epochs}] "
#                     f"train_loss={train_metrics['loss']:.4f} | "
#                     f"val_loss={val_metrics['loss']:.4f} | "
#                     f"train_acc={train_metrics['acc']:.4f} | "
#                     f"val_acc={val_metrics['acc']:.4f} | "
#                     f"train_f1_macro="
#                     f"{train_metrics['f1_macro']:.4f} | "
#                     f"val_f1_macro="
#                     f"{val_metrics['f1_macro']:.4f}"
#                 )

#             # ========================================================
#             # Checkpoint
#             # ========================================================

#             val_loss = val_metrics["loss"]

#             # nunca salva checkpoint numericamente inválido
#             if not np.isfinite(val_loss):

#                 raise RuntimeError(
#                     f"val_loss inválida na época {epoch}: "
#                     f"{val_loss}"
#                 )

#             if val_loss < best_val_loss:

#                 best_val_loss = val_loss
#                 best_state = deepcopy(
#                     self.state_dict()
#                 )

#                 torch.save(
#                     {
#                         "model_class": self.MODEL_NAME,
#                         "config": self.config,
#                         "state_dict": best_state,
#                     },
#                     checkpoint_path,
#                 )

#                 epochs_no_improve = 0

#                 if verbose >= 1:

#                     print(
#                         f"  -> novo melhor modelo salvo em "
#                         f"'{checkpoint_path}' "
#                         f"(val_loss={best_val_loss:.4f})"
#                     )

#             else:

#                 epochs_no_improve += 1

#             # ========================================================
#             # Early stopping
#             # ========================================================

#             if (
#                 patience is not None
#                 and epochs_no_improve >= patience
#             ):

#                 if verbose >= 1:

#                     print(
#                         f"  -> early stopping na época {epoch} "
#                         f"(sem melhora por {patience} épocas)"
#                     )

#                 break

#         # ============================================================
#         # Recupera melhor modelo
#         # ============================================================

#         if best_state is not None:
#             self.load_state_dict(best_state)

#         return history

#     # ================================================================
#     # Evaluate
#     # ================================================================

#     @torch.no_grad()
#     def _evaluate(
#         self,
#         loader,
#         criterion,
#         device,
#         split_name="EVAL",
#         epoch=None,
#         debug_numerics=True,
#         loss_abort_threshold=100.0,
#     ):

#         self.eval()

#         running_loss = 0.0
#         n_samples = 0

#         all_preds = []
#         all_true = []

#         max_batch_loss = -float("inf")
#         max_batch_idx = None

#         for batch_idx, (
#             imgs,
#             y_is,
#             c_is,
#             n_is
#         ) in enumerate(loader, start=1):

#             # --------------------------------------------------------
#             # CPU
#             # --------------------------------------------------------

#             if debug_numerics:

#                 if torch.isnan(imgs).any().item() or \
#                    torch.isinf(imgs).any().item():

#                     self._print_bad_batch(
#                         f"{split_name} INPUT CPU",
#                         epoch,
#                         batch_idx,
#                         imgs,
#                         y_is,
#                         names=n_is,
#                     )

#                     raise RuntimeError(
#                         f"NaN/Inf no input de {split_name}."
#                     )

#             imgs = imgs.to(device)
#             y_is = y_is.to(device)

#             # --------------------------------------------------------
#             # GPU input
#             # --------------------------------------------------------

#             if debug_numerics:

#                 if torch.isnan(imgs).any().item() or \
#                    torch.isinf(imgs).any().item():

#                     self._print_bad_batch(
#                         f"{split_name} INPUT GPU",
#                         epoch,
#                         batch_idx,
#                         imgs,
#                         y_is,
#                         names=n_is,
#                     )

#                     raise RuntimeError(
#                         f"NaN/Inf no input GPU de {split_name}."
#                     )

#             # --------------------------------------------------------
#             # Forward
#             # --------------------------------------------------------

#             logits = self(imgs)

#             if debug_numerics:

#                 if torch.isnan(logits).any().item() or \
#                    torch.isinf(logits).any().item():

#                     self._print_bad_batch(
#                         f"{split_name} LOGITS",
#                         epoch,
#                         batch_idx,
#                         imgs,
#                         y_is,
#                         logits=logits,
#                         names=n_is,
#                     )

#                     raise RuntimeError(
#                         f"NaN/Inf nos logits de {split_name}."
#                     )

#             # --------------------------------------------------------
#             # Loss
#             # --------------------------------------------------------

#             loss = criterion(logits, y_is)
#             loss_value = loss.item()

#             if debug_numerics:

#                 if not np.isfinite(loss_value):

#                     self._print_bad_batch(
#                         f"{split_name} LOSS NaN/Inf",
#                         epoch,
#                         batch_idx,
#                         imgs,
#                         y_is,
#                         logits=logits,
#                         loss=loss,
#                         names=n_is,
#                     )

#                     raise RuntimeError(
#                         f"Loss NaN/Inf em {split_name}."
#                     )

#                 if loss_value >= loss_abort_threshold:

#                     self._print_bad_batch(
#                         f"{split_name} LOSS EXPLOSIVA",
#                         epoch,
#                         batch_idx,
#                         imgs,
#                         y_is,
#                         logits=logits,
#                         loss=loss,
#                         names=n_is,
#                     )

#                     raise RuntimeError(
#                         f"Loss >= {loss_abort_threshold} "
#                         f"durante {split_name}."
#                     )

#             # --------------------------------------------------------
#             # Estatísticas
#             # --------------------------------------------------------

#             running_loss += (
#                 loss_value * imgs.size(0)
#             )

#             n_samples += imgs.size(0)

#             if loss_value > max_batch_loss:
#                 max_batch_loss = loss_value
#                 max_batch_idx = batch_idx

#             preds = torch.argmax(
#                 logits,
#                 dim=1
#             )

#             all_preds.append(
#                 preds.cpu()
#             )

#             all_true.append(
#                 y_is.cpu()
#             )

#         # ------------------------------------------------------------
#         # Proteção
#         # ------------------------------------------------------------

#         if n_samples == 0:
#             raise RuntimeError(
#                 f"Loader vazio em {split_name}."
#             )

#         avg_loss = running_loss / n_samples

#         if debug_numerics:

#             print(
#                 f"  [{split_name}] "
#                 f"avg_loss={avg_loss:.6f} | "
#                 f"max_batch_loss={max_batch_loss:.6f} "
#                 f"(batch {max_batch_idx})"
#             )

#         # ------------------------------------------------------------
#         # Métricas
#         # ------------------------------------------------------------

#         preds = torch.cat(
#             all_preds
#         ).numpy()

#         true = torch.cat(
#             all_true
#         ).numpy()

#         metrics = {
#             "loss": avg_loss,

#             "acc": float(
#                 np.mean(preds == true)
#             ),

#             "balanced_acc": balanced_accuracy_score(
#                 true,
#                 preds
#             ),

#             "f1_macro": f1_score(
#                 true,
#                 preds,
#                 average="macro",
#                 zero_division=0
#             ),

#             "f1_micro": f1_score(
#                 true,
#                 preds,
#                 average="micro",
#                 zero_division=0
#             ),

#             "precision_macro": precision_score(
#                 true,
#                 preds,
#                 average="macro",
#                 zero_division=0
#             ),

#             "recall_macro": recall_score(
#                 true,
#                 preds,
#                 average="macro",
#                 zero_division=0
#             ),

#             "kappa": cohen_kappa_score(
#                 true,
#                 preds
#             ),
#         }

#         return metrics

#     # ================================================================
#     # Predict
#     # ================================================================

#     @torch.no_grad()
#     def predict(
#         self,
#         loader,
#         device="cuda"
#     ):

#         self.to(device)
#         self.eval()

#         all_probs = []
#         all_preds = []
#         all_true = []

#         all_species = []
#         all_names = []

#         for batch_idx, (
#             imgs,
#             y_is,
#             c_is,
#             n_is
#         ) in enumerate(loader, start=1):

#             # valida CPU
#             if torch.isnan(imgs).any().item() or \
#                torch.isinf(imgs).any().item():

#                 raise RuntimeError(
#                     f"NaN/Inf no input de predict, "
#                     f"batch {batch_idx}."
#                 )

#             imgs = imgs.to(device)

#             logits = self(imgs)

#             if torch.isnan(logits).any().item() or \
#                torch.isinf(logits).any().item():

#                 raise RuntimeError(
#                     f"NaN/Inf nos logits de predict, "
#                     f"batch {batch_idx}."
#                 )

#             probs = torch.softmax(
#                 logits,
#                 dim=1
#             )

#             preds = torch.argmax(
#                 probs,
#                 dim=1
#             )

#             all_probs.append(
#                 probs.cpu()
#             )

#             all_preds.append(
#                 preds.cpu()
#             )

#             all_true.append(
#                 y_is
#             )

#             all_species.extend(
#                 c_is
#             )

#             all_names.extend(
#                 n_is
#             )

#         return {
#             "probs": torch.cat(
#                 all_probs
#             ).numpy(),

#             "preds": torch.cat(
#                 all_preds
#             ).numpy(),

#             "true": torch.cat(
#                 all_true
#             ).numpy(),

#             "species": all_species,

#             "filenames": all_names,
#         }

#     # ================================================================
#     # Load
#     # ================================================================

#     @classmethod
#     def load(
#         cls,
#         checkpoint_path,
#         device="cuda"
#     ):

#         checkpoint = torch.load(
#             checkpoint_path,
#             map_location=device
#         )

#         model = cls(
#             **checkpoint["config"]
#         )

#         model.load_state_dict(
#             checkpoint["state_dict"]
#         )

#         model.to(device)
#         model.eval()

#         return model

#======================================================================
#======================================================================
# Modelo - MobileNetV3Large - Multiclass (single-label)

from torchvision.models import mobilenet_v3_large


class MulticlassMobileNetV3Large(nn.Module):
    MODEL_NAME = "MobileNetV3Large"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        weights = "IMAGENET1K_V2" if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        # ---- adapta a primeira camada para aceitar 5 bandas em vez de 3 (RGB) ----
        # a primeira conv fica dentro de um bloco Conv2dNormActivation (features[0][0])
        old_conv = backbone.features[0][0]  # Conv2d(3, 16, kernel=3, stride=2, padding=1, bias=False)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.features[0][0] = new_conv

        # ---- adapta a cabeça de classificação para num_classes (com dropout customizado) ----
        # cabeça original: Linear(960,1280) -> Hardswish -> Dropout(0.2) -> Linear(1280,1000)
        in_features_last = backbone.classifier[0].in_features   # 960
        hidden_features = backbone.classifier[0].out_features   # 1280
        drop_p = dropout if dropout is not None else backbone.classifier[2].p

        backbone.classifier = nn.Sequential(
            nn.Linear(in_features_last, hidden_features),
            nn.Hardswish(),
            nn.Dropout(p=drop_p),
            nn.Linear(hidden_features, num_classes),
        )

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=True,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.train()
            for imgs, y_is, c_is, n_is in train_loader:
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: acurácia na validação) ----
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_acc={best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
    

#======================================================================
#======================================================================
# Modelo - EfficientNet-B0 - Multiclass (single-label)
# requer: pip install timm

import timm


class MulticlassEfficientNetB0(nn.Module):
    MODEL_NAME = "EfficientNetB0"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        # ---- cria o backbone EfficientNet-B0 já com a cabeça ajustada para num_classes ----
        backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # ---- adapta o stem (primeira conv) para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.conv_stem  # Conv2d(3, 32, kernel=3, stride=2, padding=1, bias=False)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.conv_stem = new_conv

        # ---- adapta a cabeça de classificação para num_classes (com dropout opcional) ----
        in_features = backbone.classifier.in_features  # 1280
        if dropout is not None:
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.classifier = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=True,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.train()
            for imgs, y_is, c_is, n_is in train_loader:
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: acurácia na validação) ----
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_acc={best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
        
#======================================================================
#======================================================================
# Modelo - ResNet18 - Multiclass (single-label)

from torchvision.models import resnet18


class MulticlassResNet18(nn.Module):
    MODEL_NAME = "ResNet18"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = resnet18(weights=weights)

        # ---- adapta a primeira camada para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.conv1  # Conv2d(3, 64, kernel=7, stride=2, padding=3, bias=False)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.conv1 = new_conv

        # ---- adapta a cabeça de classificação para num_classes (com dropout opcional) ----
        in_features = backbone.fc.in_features  # 512
        if dropout is not None:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

    # ------------------------------------------------------------------
    # Helper para reportar os devices em uso

    def _print_device_report(self, device):
        print("\n\t Device report:")
        print(f"\t  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"\t  - GPU(s) visível(is): {torch.cuda.device_count()}")
            print(f"\t  - GPU atual: {torch.cuda.current_device()} "
                  f"({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"\t  - device solicitado para treino: {device}")
        print(f"\t  - device dos parâmetros do modelo: {next(self.parameters()).device}\n")

    # ------------------------------------------------------------------
    # Treino

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=30,
        lr=1e-4,
        weight_decay=1e-4,
        device="cuda",
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=1,
    ):
        # ---- verbose: 0 = silencioso | 1 = resumo por época (default) | 2 = também progresso por batch ----
        self.to(device)

        # ---- reporta os devices disponíveis/utilizados antes de começar o treino ----
        self._print_device_report(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        if verbose >= 1:
            print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            self.train()
            for batch_idx, (imgs, y_is, c_is, n_is) in enumerate(train_loader, start=1):
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

                # ---- progresso por batch (só aparece com verbose=2) ----
                if verbose >= 2:
                    print(
                        f"\r    [Epoch {epoch:03d}/{epochs}] "
                        f"batch {batch_idx:04d}/{n_batches} - loss={loss.item():.4f}",
                        end="", flush=True,
                    )

            if verbose >= 2:
                print()  # quebra de linha após a barra de progresso da última batch

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose >= 1:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: loss na validação, não acurácia) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose >= 1:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model


#======================================================================
#======================================================================
# Modelo - ResNet50 - Multiclass (single-label)

from torchvision.models import resnet50


class MulticlassResNet50(nn.Module):
    MODEL_NAME = "ResNet50"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        weights = "IMAGENET1K_V2" if pretrained else None
        backbone = resnet50(weights=weights)

        # ---- adapta a primeira camada para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.conv1  # Conv2d(3, 64, kernel=7, stride=2, padding=3, bias=False)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.conv1 = new_conv

        # ---- adapta a cabeça de classificação para num_classes (com dropout opcional) ----
        in_features = backbone.fc.in_features  # 2048 (bottleneck do ResNet50)
        if dropout is not None:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

    # ------------------------------------------------------------------
    # Helper para reportar os devices em uso

    def _print_device_report(self, device):
        print("\n\t Device report:")
        print(f"\t  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"\t  - GPU(s) visível(is): {torch.cuda.device_count()}")
            print(f"\t  - GPU atual: {torch.cuda.current_device()} "
                  f"({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"\t  - device solicitado para treino: {device}")
        print(f"\t  - device dos parâmetros do modelo: {next(self.parameters()).device}\n")

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=1,
    ):
        # ---- verbose: 0 = silencioso | 1 = resumo por época (default) | 2 = também progresso por batch ----
        self.to(device)

        # ---- reporta os devices disponíveis/utilizados antes de começar o treino ----
        self._print_device_report(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        if verbose >= 1:
            print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            self.train()
            for batch_idx, (imgs, y_is, c_is, n_is) in enumerate(train_loader, start=1):
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

                # ---- progresso por batch (só aparece com verbose=2) ----
                if verbose >= 2:
                    print(
                        f"\r    [Epoch {epoch:03d}/{epochs}] "
                        f"batch {batch_idx:04d}/{n_batches} - loss={loss.item():.4f}",
                        end="", flush=True,
                    )

            if verbose >= 2:
                print()  # quebra de linha após a barra de progresso da última batch

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose >= 1:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: loss na validação, não acurácia) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose >= 1:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model



#======================================================================
#======================================================================
#======================================================================
# Modelo - ConvNeXt-Tiny - Multiclass (single-label)

from torchvision.models import convnext_tiny


class MulticlassConvNeXtTiny(nn.Module):
    MODEL_NAME = "ConvNeXtTiny"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
        }

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = convnext_tiny(weights=weights)

        # ---- adapta a primeira camada para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.features[0][0]  # Conv2d(3, 96, kernel=4, stride=4)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.features[0][0] = new_conv

        # ---- adapta a cabeça de classificação para num_classes (com dropout opcional) ----
        in_features = backbone.classifier[2].in_features  # 768
        if dropout is not None:
            backbone.classifier[2] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.classifier[2] = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

    # ------------------------------------------------------------------
    # Helper para reportar os devices em uso

    def _print_device_report(self, device):
        print("\n\t Device report:")
        print(f"\t  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"\t  - GPU(s) visível(is): {torch.cuda.device_count()}")
            print(f"\t  - GPU atual: {torch.cuda.current_device()} "
                  f"({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"\t  - device solicitado para treino: {device}")
        print(f"\t  - device dos parâmetros do modelo: {next(self.parameters()).device}\n")

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=1,
    ):
        # ---- verbose: 0 = silencioso | 1 = resumo por época (default) | 2 = também progresso por batch ----
        self.to(device)

        # ---- reporta os devices disponíveis/utilizados antes de começar o treino ----
        self._print_device_report(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        if verbose >= 1:
            print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            self.train()
            for batch_idx, (imgs, y_is, c_is, n_is) in enumerate(train_loader, start=1):
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

                # ---- progresso por batch (só aparece com verbose=2) ----
                if verbose >= 2:
                    print(
                        f"\r    [Epoch {epoch:03d}/{epochs}] "
                        f"batch {batch_idx:04d}/{n_batches} - loss={loss.item():.4f}",
                        end="", flush=True,
                    )

            if verbose >= 2:
                print()  # quebra de linha após a barra de progresso da última batch

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose >= 1:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: loss na validação, não acurácia) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose >= 1:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
    



#======================================================================
#======================================================================
# Modelo - ViT-Tiny - Multiclass (single-label)
# requer: pip install timm

import timm


class MulticlassViTTiny(nn.Module):
    MODEL_NAME = "ViTTiny"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None, img_size=224):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
            "img_size": img_size,
        }
        self.img_size = img_size

        # ---- cria o backbone ViT-Tiny (patch 16) já com a cabeça ajustada para num_classes ----
        backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

        # ---- adapta o patch embedding para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.patch_embed.proj  # Conv2d(3, 192, kernel=16, stride=16)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.patch_embed.proj = new_conv

        # ---- dropout opcional antes da cabeça de classificação ----
        if dropout is not None and hasattr(backbone, "head_drop"):
            backbone.head_drop = nn.Dropout(p=dropout)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        # ---- garante entrada no tamanho fixo esperado pelo ViT (abordagem padrão na literatura) ----
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

    # ------------------------------------------------------------------
    # Helper para reportar os devices em uso

    def _print_device_report(self, device):
        print("\n\t Device report:")
        print(f"\t  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"\t  - GPU(s) visível(is): {torch.cuda.device_count()}")
            print(f"\t  - GPU atual: {torch.cuda.current_device()} "
                  f"({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"\t  - device solicitado para treino: {device}")
        print(f"\t  - device dos parâmetros do modelo: {next(self.parameters()).device}\n")

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=1,
    ):
        # ---- verbose: 0 = silencioso | 1 = resumo por época (default) | 2 = também progresso por batch ----
        self.to(device)

        # ---- reporta os devices disponíveis/utilizados antes de começar o treino ----
        self._print_device_report(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        if verbose >= 1:
            print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            self.train()
            for batch_idx, (imgs, y_is, c_is, n_is) in enumerate(train_loader, start=1):
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

                # ---- progresso por batch (só aparece com verbose=2) ----
                if verbose >= 2:
                    print(
                        f"\r    [Epoch {epoch:03d}/{epochs}] "
                        f"batch {batch_idx:04d}/{n_batches} - loss={loss.item():.4f}",
                        end="", flush=True,
                    )

            if verbose >= 2:
                print()  # quebra de linha após a barra de progresso da última batch

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose >= 1:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: loss na validação, não acurácia) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose >= 1:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

#======================================================================
#======================================================================
# Modelo - ViT-Small - Multiclass (single-label)
# requer: pip install timm

import timm


class MulticlassViTSmall(nn.Module):
    MODEL_NAME = "ViTSmall"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None, img_size=224):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
            "img_size": img_size,
        }
        self.img_size = img_size

        # ---- cria o backbone ViT-Small (patch 16) já com a cabeça ajustada para num_classes ----
        backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

        # ---- adapta o patch embedding para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.patch_embed.proj  # Conv2d(3, 384, kernel=16, stride=16)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.patch_embed.proj = new_conv

        # ---- dropout opcional antes da cabeça de classificação ----
        if dropout is not None and hasattr(backbone, "head_drop"):
            backbone.head_drop = nn.Dropout(p=dropout)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        # ---- garante entrada no tamanho fixo esperado pelo ViT (abordagem padrão na literatura) ----
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=True,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.train()
            for imgs, y_is, c_is, n_is in train_loader:
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: acurácia na validação) ----
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_acc={best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

#======================================================================
#======================================================================
# Modelo - ViT-Base - Multiclass (single-label)
# requer: pip install timm

import timm


class MulticlassViTBase(nn.Module):
    MODEL_NAME = "ViTBase"

    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2, seed_model=None, img_size=224):
        super().__init__()

        # ---- reprodutibilidade: fixa a seed antes de instanciar as camadas ----
        self.seed_model = seed_model
        self.set_seed(seed_model)

        # ---- guarda os argumentos de construção para permitir carregamento genérico ----
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "seed_model": seed_model,
            "img_size": img_size,
        }
        self.img_size = img_size

        # ---- cria o backbone ViT-Base (patch 16) já com a cabeça ajustada para num_classes ----
        backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

        # ---- adapta o patch embedding para aceitar 5 bandas em vez de 3 (RGB) ----
        old_conv = backbone.patch_embed.proj  # Conv2d(3, 768, kernel=16, stride=16)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # copia os pesos RGB originais para os 3 primeiros canais (Blue, Green, Red)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
                # canais extras (NIR, Red Edge) recebem a média dos pesos RGB como inicialização
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1)

        backbone.patch_embed.proj = new_conv

        # ---- dropout opcional antes da cabeça de classificação ----
        if dropout is not None and hasattr(backbone, "head_drop"):
            backbone.head_drop = nn.Dropout(p=dropout)

        self.backbone = backbone

    # ------------------------------------------------------------------
    # Reprodutibilidade

    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        # ---- garante entrada no tamanho fixo esperado pelo ViT (abordagem padrão na literatura) ----
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.backbone(x)  # logits, sem softmax -> CrossEntropyLoss

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
        checkpoint_path="best_model.pt",
        patience=None,
        verbose=True,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f'\n\t Trainning...   epochs: \033[96;96m{epochs}\033[0m \n')

        metric_names = [
            "loss", "acc", "balanced_acc",
            "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "kappa",
        ]
        history = {}
        for m in metric_names:
            history[f"train_{m}"] = []
            history[f"val_{m}"] = []

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.train()
            for imgs, y_is, c_is, n_is in train_loader:
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

            train_metrics = self._evaluate(train_loader, criterion, device)
            val_metrics = self._evaluate(val_loader, criterion, device)

            for m in metric_names:
                history[f"train_{m}"].append(train_metrics[m])
                history[f"val_{m}"].append(val_metrics[m])

            if verbose:
                print(
                    f"\n[Epoch {epoch:03d}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                    f"train_f1_macro={train_metrics['f1_macro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

            # ---- checkpoint do melhor modelo (critério: acurácia na validação) ----
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = deepcopy(self.state_dict())

                # ---- salva estado + config, para permitir carregamento genérico ----
                torch.save(
                    {
                        "model_class": self.MODEL_NAME,
                        "config": self.config,
                        "state_dict": best_state,
                    },
                    checkpoint_path,
                )

                epochs_no_improve = 0
                if verbose:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_acc={best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  -> early stopping na época {epoch} (sem melhora por {patience} épocas)")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Avaliação interna (usada no fit, tanto para train quanto para val)

    @torch.no_grad()
    def _evaluate(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_true = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs, y_is = imgs.to(device), y_is.to(device)

            logits = self(imgs)
            loss = criterion(logits, y_is)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_is.cpu())

        avg_loss = running_loss / n_samples

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

        metrics = {
            "loss": avg_loss,
            "acc": float(np.mean(preds == true)),
            "balanced_acc": balanced_accuracy_score(true, preds),
            "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(true, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(true, preds, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(true, preds),
        }

        return metrics

    # ------------------------------------------------------------------
    # Predição

    @torch.no_grad()
    def predict(self, loader, device="cuda"):
        self.to(device)
        self.eval()

        all_probs, all_preds, all_true = [], [], []
        all_species, all_names = [], []

        for imgs, y_is, c_is, n_is in loader:
            imgs = imgs.to(device)
            logits = self(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_true.append(y_is)
            all_species.extend(c_is)
            all_names.extend(n_is)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "true": torch.cat(all_true).numpy(),
            "species": all_species,
            "filenames": all_names,
        }

    # ------------------------------------------------------------------
    # Carregamento genérico (o checkpoint carrega sua própria config)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

#======================================================================
#======================================================================

def model_class_function(MODEL_NAME: str):

    if MODEL_NAME == "SmallCNN":
        return MulticlassSmallCNN
    elif MODEL_NAME == "MobileNetV3Small":
        return MulticlassMobileNetV3Small
    elif MODEL_NAME == "MobileNetV3Large":
        return MulticlassMobileNetV3Large
    elif MODEL_NAME == "EfficientNetB0":
        return MulticlassEfficientNetB0
    elif MODEL_NAME == "ResNet18":
        return MulticlassResNet18
    elif MODEL_NAME == "ResNet50":
        return MulticlassResNet50
    elif MODEL_NAME == "ConvNeXtTiny":
        return MulticlassConvNeXtTiny
    elif MODEL_NAME == "ViTTiny":
        return MulticlassViTTiny
    elif MODEL_NAME == "ViTSmall":
        return MulticlassViTSmall
    elif MODEL_NAME == "ViTBase":
        return MulticlassViTBase
    else:
        raise ValueError(
            "MODEL_NAME not in list [SmallCNN, MobileNetV3Small, MobileNetV3Large, "
            "EfficientNetB0, ResNet18, ResNet50, ConvNeXtTiny, ViTTiny, ViTSmall, ViTBase]"
        )
    
#======================================================================
#======================================================================

def load_model_generic(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_class = model_class_function(checkpoint["model_class"])
    model = model_class(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()
    return model

#======================================================================
#======================================================================
