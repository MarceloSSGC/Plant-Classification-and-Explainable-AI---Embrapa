import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import argparse
from time import sleep, time

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Concepts/First_Model/No_View__Shape_01")

from auxiliar_concept import *

#======================================================================
#======================================================================

print(f"\n\033[100;01m\t     --- Start Model ---     \t\033[0m\n")

#======================================================================
# Params

interactive = True
print(f"\n\t  Interactive: \033[96;95m{interactive} \033[0m\n")

#======================================================================
# Interactive or Argparse

if interactive: 
    SEED = 10
    ALIGN = True
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, required=True)
    parser.add_argument("--ALIGN", type=int, required=True)
    args = parser.parse_args()

    SEED = args.SEED
    ALIGN = False if args.ALIGN == 0 else True 


print("\n" + "- "*40)
print(f"\n SEED: \033[96;91m{SEED}\033[0m")
print(f" ALIGN: \033[96;91m{ALIGN}\033[0m\n")
print("- "*40 + "\n")
sleep(3)

# Control
# x = 1/0
#======================================================================
# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#======================================================================
# Dataset & Directories

DATASET_NAME = "PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5"
DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/"

if not ALIGN:
    DATA_DIR = os.path.join(DATA_DIR, "PlantaDaninha_BoaVista")
else:
    DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME)

#======================================================================
# Experiment Name and Directory

experiment_name = f"concepts_no_view__shape_{SEED}_ALIGN_{ALIGN}"
experiment_type = "concepts_no_view"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/{experiment_type}/{experiment_name}"

#======================================================================
# Infos

DIR_INFOS = DIR_EXP + "/infos.json"

with open(DIR_INFOS, "r", encoding="utf-8") as file:
    infos = json.load(file)


#======================================================================
# 1. Dataset PyTorch

DIR_EXP = Path(DIR_EXP)

TRAIN_DIR = DIR_EXP / "Train_Norm"
VAL_DIR   = DIR_EXP / "Val_Norm"
TEST_DIR  = DIR_EXP / "Test_Norm"

df_shape = pd.read_csv(f"{DIR_EXP}/df_shape.csv")

#======================================================================
# 2. DataLoaders

train_dataset = MultispectralWeedMultilabelDataset(TRAIN_DIR, df_shape)
# 'root_dir', 'classes', 'label_cols', 'species_to_label', 'samples'

val_dataset = MultispectralWeedMultilabelDataset(VAL_DIR, df_shape)

test_dataset = MultispectralWeedMultilabelDataset(TEST_DIR, df_shape)


print("\nTrain:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset), "\n")

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

infos["cuda"] = {
"device": str(device),
"torch_cuda_is_available": torch_cuda_is_available,
"torch_cuda_get_device_name": torch_cuda_get_device_name}

print(f"Device: \033[96;92m{device}\033[0m")
print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")

#======================================================================
# Treinamento

num_labels = len(train_dataset.label_cols)  # ex.: 5 (c_11, c_15, c_16, c_17, c_18)
in_channels = 5  # número de bandas


if "model_trained" not in infos.keys() or not infos["model_trained"]:

    print(f"\n\t \033[96;01m  Initialize Training  \033[0m")
    
    # ======================================================================
    # Instanciação do modelo

    model = MultioutputSmallCNN(
        in_channels=in_channels,
        num_labels=num_labels,
        base_channels=32,
    )

    # ======================================================================
    # Treino (fit)
        
    epochs = 2

    best_model_dir = str(DIR_EXP / "best_model.pt")
    history_dir = str(DIR_EXP / "history.pkl")
    loss_dir = str(DIR_EXP / "df_loss.csv")

    t_0 = time()
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        threshold=0.5,
        checkpoint_path=best_model_dir,
        patience=7,        # early stopping opcional; remova/None se não quiser
        verbose=True,
    )
    t_1 = time()
    time_train = round(t_1-t_0)

    infos["model_trained"] = True

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

    infos["model"] = {
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

    with open(DIR_INFOS, "w") as file:
        json.dump(infos, file, indent=4)

else:

    # Load Model

    print(f"\n\t --- \033[96;01m  Loading Model --- \033[0m")


    # 1. Recriar a arquitetura com os MESMOS hiperparâmetros do treino original
    model = MultioutputSmallCNN(
        in_channels=5,
        num_labels=num_labels,      # precisa ser o mesmo valor usado no treino
        base_channels=32,
    )

    # 2. Carregar os pesos salvos
    state_dict = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

#======================================================================
# Metrics

df_metric_train_dir = str(DIR_EXP / "df_metric_train.csv")
df_metric_val_dir = str(DIR_EXP / "df_metric_val.csv")
df_metric_test_dir = str(DIR_EXP / "df_metric_test.csv")

if not os.path.isfile(df_metric_val_dir) or not os.path.isfile(df_metric_test_dir):

    # Predicting

    # # Train
    # train_results = model.predict(
    #     loader=train_loader,
    #     device=device,
    #     threshold=0.5,
    # )

    # Validation
    val_results = model.predict(
        loader=val_loader,
        device=device,
        threshold=0.5,
    )

    # Test
    test_results = model.predict(
        loader=test_loader,
        device=device,
        threshold=0.5,
    )
    # 'probs', 'preds', 'true', 'species', 'filenames'
                

    # df_metric_train = multioutput_classification_metrics_dataframe(train_results)
    df_metric_val = multioutput_classification_metrics_dataframe(val_results)
    df_metric_test = multioutput_classification_metrics_dataframe(test_results)

    # df_metric_train.to_csv(df_metric_train_dir, index=False)
    df_metric_val.to_csv(df_metric_val_dir, index=False)
    df_metric_test.to_csv(df_metric_test_dir, index=False)


#======================================================================
#======================================================================

import shap
import numpy as np
import torch

# ======================================================================
# 1. Preparar dados de background e de explicação
# ======================================================================

model.eval()
model.to(device)

# Background: pequena amostra de imagens de treino (SHAP usa como "referência" de baseline)
# Poucas amostras já bastam (ex. 20-50) — GradientExplainer não precisa de muitas

background_imgs = []
for imgs, Ys, c_is, n_is in train_loader:
    background_imgs.append(imgs)
    if len(background_imgs) * imgs.size(0) >= 32:
        break
background_imgs = torch.cat(background_imgs)[:32].to(device)

# Amostras que você quer explicar (ex. um batch do conjunto de teste)
explain_imgs, explain_Ys, explain_c_is, explain_n_is = next(iter(test_loader))
explain_imgs = explain_imgs.to(device)

# ======================================================================
# 2. Criar o explainer (modelo como um todo -> soma/média sobre as 5 saídas)
# ======================================================================

explainer = shap.GradientExplainer(model, background_imgs)

# shap_values: lista com um array por output (5 outputs -> lista de 5 arrays)
# cada array tem shape (N, 5, H, W) -> mesma shape da imagem de entrada
shap_values = explainer.shap_values(explain_imgs)

# ======================================================================
# 3. Agregar em importância por banda (modelo como um todo)
# ======================================================================

# shap_values é uma lista (uma entrada por output) de arrays (N, C=5, H, W)
# Passo 1: empilhar os outputs -> (num_outputs, N, C, H, W)
shap_stack = np.stack(shap_values, axis=0)

# Passo 2: valor absoluto (não importa a direção do efeito, só a magnitude)
shap_abs = np.abs(shap_stack)

# Passo 3: média sobre outputs, amostras (N) e pixels (H, W) -> sobra só por banda (C=5)
band_importance = shap_abs.mean(axis=(0, 1, 3, 4))  # shape: (5,)

# ======================================================================
# 4. Resultado
# ======================================================================

band_names = ["Band_1", "Band_2", "Band_3", "Band_4", "Band_5"]  # ajuste para seus nomes reais

for name, val in zip(band_names, band_importance):
    print(f"{name}: {val:.6f}")

# Importância relativa (%) -- soma 100%
band_importance_pct = 100 * band_importance / band_importance.sum()
for name, val in zip(band_names, band_importance_pct):
    print(f"{name}: {val:.2f}%")

#======================================================================

















































