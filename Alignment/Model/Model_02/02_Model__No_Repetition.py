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
from time import sleep

# import sys
# MODEL_DIR = Path.cwd() / "Model_02"
# print("Diretório:", MODEL_DIR)
# print("Existe:", MODEL_DIR.exists())
# print("auxiliar.py existe:", (MODEL_DIR / "auxiliar.py").is_file())
# sys.path.insert(0, str(MODEL_DIR))
# print("Path.cwd():", Path.cwd())

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment/Model/Model_02")

from auxiliar import *

#======================================================================
#======================================================================

print(f"\n\033[100;01m\t     --- Start Model ---     \t\033[0m\n")

#======================================================================
# Params

interactive = False
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

if not ALIGN:
    DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista"
else:
    DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine"

# Experiment Name and Directory

experiment_name = f"first_test__15-07__SEED_{SEED}_ALIGN_{ALIGN}"
experiment_type = "alignment_compare"
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

#------------------------------------------------------------------------
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

print("\nTrain:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset), "\n")

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
print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m")

#======================================================================
# Treinamento

if "model_trained" not in infos.keys() or not infos["model_trained"]:
    print(f"\n\t \033[100;01m  Initialize Training  \033[0m")
    
    model = SmallMultispectralCNN(
        num_classes=len(class_to_idx),
        input_channels=5
    )

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=35,
        learning_rate=1e-3,
        save_path=DIR_EXP / "best_small_cnn_multispectral.pth"
    )

    infos["model_trained"] = True

    best_epoch = int(model.best_epoch)-1
    total_params = sum(p.numel() for p in model.parameters())
    infos["model"] = {
        "total_params": total_params,
        "best_epoch": model.best_epoch,
        "best_train_loss": model.history["train_loss"][best_epoch],
        "best_train_acc": model.history["train_acc"][best_epoch],
        "best_val_loss": model.history["val_loss"][best_epoch],
        "best_val_acc": model.history["val_acc"][best_epoch],
        "best_model_dir": str(DIR_EXP / "best_small_cnn_multispectral.pth")
    }

    with open(DIR_INFOS, "w") as file:
        json.dump(infos, file, indent=4) 
else:
    print(f"\n\t \033[100;01m  Loading Model...  \033[0m")
    model = SmallMultispectralCNN(
        num_classes=len(class_to_idx),
        input_channels=5
    )

    model.load_state_dict(
        torch.load(
            DIR_EXP / "best_small_cnn_multispectral.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

#======================================================================
# Loss

loss_path = DIR_EXP / "df_loss.csv"

if not os.path.isfile(loss_path):

    df_loss = pd.DataFrame()
    df_loss["train_loss"] = model.history["train_loss"]
    df_loss["val_loss"] = model.history["val_loss"]
    df_loss["train_acc"] = model.history["train_acc"]
    df_loss["val_acc"] = model.history["val_acc"]

    df_loss.to_csv(loss_path, index=False)
    print("df_loss:\033[96;92m Saved \033[0m")

# df_loss[["train_loss","val_loss"]].plot()
# df_loss[["train_acc","val_acc"]].plot()

#======================================================================
# Prediction

# y_train = get_labels(train_loader)
# y_val   = get_labels(val_loader)
# y_test  = get_labels(test_loader)

# y_train, train_pred, train_pred_prob = model.predict_with_labels(train_loader)
y_val, y_val_pred, y_val_pred_prob = model.predict_with_labels(val_loader)
y_test, y_test_pred, y_test_pred_prob = model.predict_with_labels(test_loader)

#======================================================================
# 5. Avaliação  Validation e Test

# Validation
df_metrics_val = classification_metrics_dataframe(
    y_real=y_val,
    y_pred=y_val_pred
)
df_metrics_val.columns = [x + "_val" for x in df_metrics_val.columns]
h(df_metrics_val)

df_metrics_test = classification_metrics_dataframe(
    y_real=y_test,
    y_pred=y_test_pred
)
df_metrics_test.columns = [x + "_tst" for x in df_metrics_test.columns]
h(df_metrics_test)

# df_metrics_test = classification_metrics_dataframe(
#     y_real=y_test,
#     y_pred=y_test_pred,
#     class_names=idx_to_class
# )

df_metrics = pd.concat([df_metrics_val, df_metrics_test], axis=1)

df_metrics.to_csv(DIR_EXP / "df_metrics.csv", index=False)

#======================================================================
# 6. Relatório de classificação

target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\nClassification Report - \033[96;92mVAL\033[0m")
val_report = classification_report(
    y_val,
    y_val_pred,
    target_names=target_names,
    zero_division=0
)
print(val_report)

print("\nClassification Report - \033[96;92mTEST\033[0m")
test_report = classification_report(
    y_test,
    y_test_pred,
    target_names=target_names,
    zero_division=0
)

print(test_report)

with open(DIR_EXP / "report_VAL.txt", "w", encoding="utf-8") as f:
    f.write(val_report)

with open(DIR_EXP / "report_TEST.txt", "w", encoding="utf-8") as f:
    f.write(test_report)

#======================================================================
# 7. Matriz de confusão

print("\nConfusion Matrix - VAL")
val_conf_mat = confusion_matrix(y_val, y_val_pred)
val_conf_mat = pd.DataFrame(val_conf_mat, index=list(idx_to_class.values()), columns=list(idx_to_class.values()))
val_conf_mat.to_csv(DIR_EXP / "conf_mat_VAL.csv")
# print(val_conf_mat)

print("\nConfusion Matrix - TEST")
test_conf_mat = confusion_matrix(y_test, y_test_pred)
test_conf_mat = pd.DataFrame(test_conf_mat, index=list(idx_to_class.values()), columns=list(idx_to_class.values()))
test_conf_mat.to_csv(DIR_EXP / "conf_mat_TEST.csv")

# print(test_conf_mat.values)

#======================================================================


