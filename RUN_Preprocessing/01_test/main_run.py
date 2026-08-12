import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import json
from time import sleep, time
from sklearn.metrics import classification_report


# os.chdir("RUN_Preprocessing/01_test")

from aux_model import *

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#======================================================================
#======================================================================

print(f"\n\033[100;01m\t     --- Start Model ---     \t\033[0m\n")

#======================================================================
#======================================================================

def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
    print(data.shape)


#======================================================================
#======================================================================
# Params

PC = "Nitro"
INTERACTIVE = True
SEED = 10

print(f"\t  Interactive: \033[96;95m{INTERACTIVE} \033[0m\n")
print(f"\t  SEED: \033[96;95m{SEED} \033[0m\n")

#======================================================================
# Reproducibility

# Random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch - CPU
torch.manual_seed(SEED)

# PyTorch - CUDA/GPU
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Algoritmos determinísticos
torch.use_deterministic_algorithms(True)

# # cuDNN
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

#======================================================================
# Dataset & Directories

DATASET_NAME = "align_bands_ecc_affine_with_retry--best_band_otsu_green__SEED_10_AUG"
DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/run_preprocessing"

DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME)

if not os.path.isdir(DATA_DIR):
    raise ValueError("DATA_DIR doesnt exist")

#======================================================================
# Experiment Name and Directory

experiment_name = f"TEST_seg_&_aug__SEED_{SEED}_AUG"
experiment_type = "segmentation_augmentation"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Results/{experiment_type}/{experiment_name}"

if not os.path.isdir(DIR_EXP):
    os.makedirs(DIR_EXP)

#======================================================================
# Infos

DIR_INFOS = DIR_EXP + "/infos.json"

if os.path.isfile(DIR_INFOS):
    with open(DIR_INFOS, "r", encoding="utf-8") as file:
        infos = json.load(file)
else:
    if "infos.json" in os.listdir(DATA_DIR):
        old_info_dir = os.path.join(DATA_DIR, "infos.json")
        with open(old_info_dir, "r", encoding="utf-8") as file:
            infos = json.load(file)
    elif "aug_infos.json" in os.listdir(DATA_DIR):
        old_info_dir = os.path.join(DATA_DIR, "aug_infos.json")
        with open(old_info_dir, "r", encoding="utf-8") as file:
            infos = json.load(file)
    else:
        raise ValueError("infos.json was not found")

    infos["RUN"] = {
        "PC": PC,
        "INTERACTIVE": INTERACTIVE,
        "SEED": SEED,
        "DATA_DIR": DATA_DIR,
        "DIR_EXP": DIR_EXP,
    }
    with open(DIR_INFOS, "w", encoding="utf-8") as file:
        json.dump(infos, file)


#======================================================================
#======================================================================
# 1. Dataset PyTorch

DATA_DIR = Path(DATA_DIR)

TRAIN_DIR = DATA_DIR / "Train_Norm"
VAL_DIR   = DATA_DIR / "Val_Norm"
TEST_DIR  = DATA_DIR / "Test_Norm"


#======================================================================
# 2. DataLoaders

train_dataset = MultispectralWeedDataset(TRAIN_DIR)
# train_dataset.__dict__.keys()
# 'root_dir', 'transform', 'classes', 'class_to_idx', 'samples'
val_dataset = MultispectralWeedDataset(VAL_DIR)
test_dataset = MultispectralWeedDataset(TEST_DIR)

print("\nTrain:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset), "\n")

infos["RUN"]["n_samples"] = {
    "Train": len(train_dataset),
    "Val": len(val_dataset),
    "Test": len(test_dataset),
    }


batch_size = 8

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
# Cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_cuda_is_available = torch.cuda.is_available()
torch_cuda_get_device_name = torch.cuda.get_device_name(0)

infos["RUN"]["cuda"] = {
"device": str(device),
"torch_cuda_is_available": torch_cuda_is_available,
"torch_cuda_get_device_name": torch_cuda_get_device_name}

print(f"Device: \033[96;92m{device}\033[0m")
print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")

sleep(5)

#======================================================================
# Treinamento

num_classes = len(train_dataset.classes)  # deve ser 31
epochs = 30

best_model_dir = os.path.join(DIR_EXP, "best_model.pt")
loss_dir = os.path.join(DIR_EXP, "df_liss.csv")

if "model_trained" not in infos["RUN"].keys() or not infos["RUN"]["model_trained"]:

    print(f"\n\t \033[96;01m  Initialize Training  \033[0m")
    
    #-----------------------------------------------------------------

    # Instanciação do modelo

    model = MulticlassSmallCNN(
        in_channels=5,
        num_classes=num_classes,
        base_channels=32,
    )

    # ---- treino ----
    t_0 = time()
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        checkpoint_path=best_model_dir,
        patience=5,   # None se não quiser early stopping
        verbose=True,
    )
    t_1 = time()
    time_train = round(t_1-t_0)

    infos["RUN"]["model_trained"] = True

    df_loss = pd.DataFrame(history)
    df_loss.to_csv(loss_dir, index=False)

    best_epoch = int(np.argmax(history["val_f1_macro"])) 
    train_loss = history['train_loss'][best_epoch]
    val_loss = history['val_loss'][best_epoch]
    val_f1_macro = history['val_f1_macro'][best_epoch]
    val_f1_micro = history['val_f1_micro'][best_epoch]
    val_exact_match = history['val_exact_match'][best_epoch]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
    
    print(f"time_train: {time_train}")
    print(f"best_epoch: {best_epoch}")
    print(f"train_loss: {train_loss:.4f}")
    print(f"val_loss: {val_loss:.4f}")
    print(f"val_f1_macro: {val_f1_macro:.4f}")
    print(f"val_f1_micro: {val_f1_micro:.4f}")
    print(f"val_exact_match: {val_exact_match:.4f}")

    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")

    infos["RUN"]["model"] = {
        "device": str(device),
        "time_train": time_train,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1_macro": val_f1_macro,
        "val_f1_micro": val_f1_micro,
        "val_exact_match": val_exact_match,
    }

    with open(DIR_INFOS, "w", encoding="utf-8") as file:
        json.dump(infos, file, indent=4)

else:

    # Load Model

    print(f"\n\t --- \033[96;01m  Loading Model --- \033[0m")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Recriar a arquitetura (mesmos parâmetros do treino original)
    model = MulticlassSmallCNN(
        in_channels=5,
        num_classes=num_classes,   # precisa ser o mesmo valor usado no treino (31)
        base_channels=32,
    )

    # 2. Carregar os pesos salvos
    state_dict = torch.load(best_model_dir, map_location=device)
    model.load_state_dict(state_dict)

    # 3. Mover para o device e colocar em modo de avaliação
    model.to(device)
    model.eval()


#======================================================================
# Metrics

df_metric_train_dir = os.path.join(DIR_EXP, "df_metric_train.csv")
df_metric_val_dir = os.path.join(DIR_EXP, "df_metric_val.csv")
df_metric_test_dir = os.path.join(DIR_EXP, "df_metric_test.csv")

cl_report_train_dir = os.path.join(DIR_EXP, "cl_report_train.txt")
cl_report_val_dir = os.path.join(DIR_EXP, "cl_report_val.txt")
cl_report_test_dir = os.path.join(DIR_EXP, "cl_report_test.txt")

especies = sorted(os.listdir(TRAIN_DIR))

if not os.path.isfile(df_metric_val_dir) or not os.path.isfile(df_metric_test_dir):

    # Predicting

    # # Train
    # train_results = model.predict(
    #     loader=train_loader,
    #     device=device,
    #     threshold=0.5,
    # )
    # df_metric_train.to_csv(df_metric_train_dir, index=False)

    #------------------------------------------------------------------
    # Validation
    val_results = model.predict(val_loader, device=device)

    y_val_real = val_results["true"]
    y_val_pred = val_results["preds"]

    df_metrics_val = classification_metrics_dataframe(y_val_real, y_val_pred, especies)
    df_metrics_val.to_csv(df_metric_val_dir, index=False)

    labels_val_real = [especies[i] for i in y_val_real]
    labels_val_pred = [especies[i] for i in y_val_pred]
    cl_report_val = classification_report(labels_val_real, labels_val_pred)
    print(cl_report_val)
    with open(cl_report_val_dir, "w") as file:
        file.write(cl_report_val)

    #------------------------------------------------------------------
    # Test
    test_results  = model.predict(test_loader, device=device)

    y_test_real = test_results["true"]
    y_test_pred = test_results["preds"]

    df_metrics_test = classification_metrics_dataframe(y_test_real, y_test_pred, especies)
    df_metrics_test.to_csv(df_metric_test_dir, index=False)

    labels_test_real = [especies[i] for i in y_test_real]
    labels_test_pred = [especies[i] for i in y_test_pred]
    cl_report_test = classification_report(labels_test_real, labels_test_pred)
    print(cl_report_test)

    with open(cl_report_test_dir, "w") as file:
        file.write(cl_report_test)





#======================================================================
#======================================================================
#======================================================================

import shap

# ======================================================================
# 1. Preparar dados de background e de explicação
# ======================================================================

model.eval()
model.to(device)

# Loaders auxiliares só para o SHAP (num_workers=0 evita crash de subprocessos)
shap_train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

shap_test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

# Background: pequena amostra de imagens de treino (SHAP usa como "referência" de baseline)
background_imgs = []
for imgs, y_is, c_is, n_is in shap_train_loader:
    background_imgs.append(imgs)
    if len(background_imgs) * imgs.size(0) >= 4:   # 32 -> 4
        break
background_imgs = torch.cat(background_imgs)[:4].to(device)  # 32 -> 4

# Amostras que você quer explicar (ex. um batch do conjunto de teste)
explain_imgs, explain_y_is, explain_c_is, explain_n_is = next(iter(shap_test_loader))
explain_imgs = explain_imgs.to(device)

# ======================================================================
# 2. Criar o explainer
# ======================================================================

import gc

torch.cuda.empty_cache()
gc.collect()

explainer = shap.GradientExplainer(model, background_imgs)

# shap_values: lista com um array por classe (num_classes outputs -> lista de num_classes arrays)
# cada array tem shape (N, 5, H, W) -> mesma shape da imagem de entrada
shap_values = explainer.shap_values(explain_imgs)

# ======================================================================
# 3a. Agregar em importância por banda (TODAS as classes, como antes)
#     -> útil só como visão geral, mas mistura explicações de classes irrelevantes
# ======================================================================

shap_stack = np.stack(shap_values, axis=0)          # (num_classes, N, C, H, W)
shap_abs = np.abs(shap_stack)
band_importance_all_classes = shap_abs.mean(axis=(0, 1, 3, 4))  # (5,)

# ======================================================================
# 3b. Agregar em importância por banda SOMENTE da classe predita (mais correto p/ multiclasse)
# ======================================================================

with torch.no_grad():
    logits = model(explain_imgs)
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # classe predita por amostra

# Para cada amostra i, pega o shap_values da classe predita[i]
# shap_values[k] tem shape (N, C, H, W) -> um array por classe k
N = explain_imgs.shape[0]
shap_per_sample = np.stack(
    [shap_values[preds[i]][i] for i in range(N)],  # (C, H, W) por amostra
    axis=0
)  # (N, C, H, W)

shap_abs_pred = np.abs(shap_per_sample)
band_importance_pred_class = shap_abs_pred.mean(axis=(0, 2, 3))  # (5,)

# ======================================================================
# 4. Resultado
# ======================================================================

band_names = ["Band_1", "Band_2", "Band_3", "Band_4", "Band_5"]  # ajuste para seus nomes reais

print("=== Importância por banda (todas as classes, agregadas) ===")
for name, val in zip(band_names, band_importance_all_classes):
    print(f"{name}: {val:.6f}")

pct_all = 100 * band_importance_all_classes / band_importance_all_classes.sum()
for name, val in zip(band_names, pct_all):
    print(f"{name}: {val:.2f}%")

print("\n=== Importância por banda (apenas classe predita por amostra) ===")
for name, val in zip(band_names, band_importance_pred_class):
    print(f"{name}: {val:.6f}")

pct_pred = 100 * band_importance_pred_class / band_importance_pred_class.sum()
for name, val in zip(band_names, pct_pred):
    print(f"{name}: {val:.2f}%")

#======================================================================























