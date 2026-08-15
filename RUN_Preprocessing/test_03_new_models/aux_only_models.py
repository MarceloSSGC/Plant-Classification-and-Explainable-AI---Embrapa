import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

#======================================================================
# Modelo - MobileNetV3 Small - Multiclass (single-label)

from torchvision.models import mobilenet_v3_small


class MulticlassMobileNetV3Small(nn.Module):
    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2):
        super().__init__()

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
            # ---- passo de otimização (treino) ----
            self.train()
            for imgs, y_is, c_is, n_is in train_loader:
                imgs, y_is = imgs.to(device), y_is.to(device)

                optimizer.zero_grad()
                logits = self(imgs)
                loss = criterion(logits, y_is)
                loss.backward()
                optimizer.step()

            # ---- avaliação em treino e validação (mesma métrica, mesmo critério) ----
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
                torch.save(best_state, checkpoint_path)
                epochs_no_improve = 0
                if verbose:
                    print(f"  -> novo melhor modelo salvo em '{checkpoint_path}' (val_acc={best_val_acc:.4f})")
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

#---------------------------------------------------------------------
# model = MulticlassMobileNetV3Small(
#     in_channels=5,
#     num_classes=num_classes,
#     pretrained=True,   # ou True, se quiser testar a inicialização adaptada dos pesos ImageNet
#     dropout=0.2,
# )

# history = model.fit(
# train_loader=train_loader,
# val_loader=val_loader,
# epochs=epochs,
# lr=1e-3,
# weight_decay=1e-4,
# device="cuda",
# checkpoint_path="best_mobilenetv3_small.pt",
# patience=30,      # opcional, early stopping
# verbose=True,
#     )
        
#======================================================================
#======================================================================
# Modelo - ResNet18 - Multiclass (single-label)

from torchvision.models import resnet18


class MulticlassResNet18(nn.Module):
    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2):
        super().__init__()

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
                torch.save(best_state, checkpoint_path)
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

#---------------------------------------------------------------------
# model = MulticlassResNet18(
#     in_channels=5,
#     num_classes=num_classes,
#     pretrained=False,   # ou True para carregar pesos ImageNet adaptados
#     dropout=0.2,
# )

#======================================================================
#======================================================================

#======================================================================
# Modelo - ConvNeXt-Tiny - Multiclass (single-label)

from torchvision.models import convnext_tiny


class MulticlassConvNeXtTiny(nn.Module):
    def __init__(self, in_channels=5, num_classes=31, pretrained=False, dropout=0.2):
        super().__init__()

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
                torch.save(best_state, checkpoint_path)
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
    
#---------------------------------------------------------------------


# model = MulticlassConvNeXtTiny(
#     in_channels=5,
#     num_classes=num_classes,
#     pretrained=False,   # ou True para carregar pesos ImageNet adaptados
#     dropout=0.2,
# )
#---------------------------------------------------------------------
#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================
