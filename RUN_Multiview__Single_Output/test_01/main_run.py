import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import json
from time import sleep, time
from sklearn.metrics import classification_report

# Auxiliar
try:
    from .aux_model import *
    from .aux_only_models import *
except ImportError:
    from aux_model import *
    from aux_only_models import *

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


#======================================================================
#======================================================================

# import yaml

# def load_config(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)
    
# yaml_test_name = "Multiview_01.yaml"
# # path = f"/run/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_test_name}"
# path = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_test_name}"
# config = load_config(path)

# for x in config:
#     print(f"{x}: \033[96;96m{config[x]}\033[0m")

#======================================================================
#======================================================================

def run_training(config):

    #======================================================================
    #======================================================================

    print(f"\n\033[100;01m\t     --- Start Model ---     \t\033[0m\n")

    #======================================================================
    #======================================================================
    # Params

    PC = config["PC"]
    INTERACTIVE = config["INTERACTIVE"]
    SEED_MODEL = config["MODEL"]["SEED_MODEL"]

    print(f"\t  PC: \033[96;95m{PC} \033[0m\n")
    print(f"\t  Interactive: \033[96;95m{INTERACTIVE} \033[0m\n")
    print(f"\t  SEED_MODEL: \033[96;95m{SEED_MODEL} \033[0m\n")

    #======================================================================
    # Reproducibility

    # Random
    random.seed(SEED_MODEL)

    # NumPy
    np.random.seed(SEED_MODEL)

    # PyTorch - CPU
    torch.manual_seed(SEED_MODEL)

    # PyTorch - CUDA/GPU
    torch.cuda.manual_seed(SEED_MODEL)
    torch.cuda.manual_seed_all(SEED_MODEL)

    # Algoritmos determinísticos
    torch.use_deterministic_algorithms(True)

    # # cuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    #======================================================================
    # Dataset & Directories

    SPLIT_DATA_NAME = config["SPLIT_DATA_NAME"]
    SPLIT_NAME_TYPE = config["SPLIT_NAME_TYPE"]

    if PC == "NITRO":
        DATA_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{SPLIT_NAME_TYPE}/{SPLIT_DATA_NAME}"
    else:
        DATA_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{SPLIT_NAME_TYPE}/{SPLIT_DATA_NAME}"

    if not os.path.isdir(DATA_DIR):
        raise ValueError("DATA_DIR doesnt exist")

    #======================================================================
    # Experiment Name and Directory

    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
    experiment_type = config["EXPERIMENT_TYPE"]

    if PC == "NITRO":
       DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Results/{experiment_type}/{EXPERIMENT_NAME}"
    else:
       DIR_EXP = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Results/{experiment_type}/{EXPERIMENT_NAME}"


    if not os.path.isdir(DIR_EXP):
        os.makedirs(DIR_EXP)

    #======================================================================
    # Infos

    model_info_dir = DIR_EXP + "/mld_info.json"

    if os.path.isfile(model_info_dir):
        with open(model_info_dir, "r", encoding="utf-8") as file:
            mdl_info = json.load(file)
    else:
        if "split_info.json" in os.listdir(DATA_DIR):
            old_info_dir = os.path.join(DATA_DIR, "split_info.json")
            with open(old_info_dir, "r", encoding="utf-8") as file:
                mdl_info = json.load(file)

        else:
            raise ValueError("split_info.json was not found")

        mdl_info["RUN"] = {
            "SEED_MODEL": SEED_MODEL,
            "DATA_DIR": DATA_DIR,
            # "DIR_EXP": DIR_EXP,

            "N_BANDS": mdl_info['MULTIVIEW']['N_BANDS']
        }

        with open(model_info_dir, "w", encoding="utf-8") as file:
            json.dump(mdl_info, file)


    #======================================================================
    #======================================================================
    # 1. Dataset PyTorch

    DATA_DIR = Path(DATA_DIR)

    TRAIN_DIR = DATA_DIR / "Train_Norm"
    VAL_DIR   = DATA_DIR / "Val_Norm"
    TEST_DIR  = DATA_DIR / "Test_Norm"

    #======================================================================
    # 2. DataLoaders

    # train_dataset.__dict__.keys()
    # 'root_dir', 'transform', 'classes', 'class_to_idx', 'samples'

    train_dataset = MultispectralWeedDataset(TRAIN_DIR)
    val_dataset = MultispectralWeedDataset(VAL_DIR)
    test_dataset = MultispectralWeedDataset(TEST_DIR)

    print(f"\nTrain: \033[96;92m{len(train_dataset)}\033[0m")
    print(f"Val: \033[96;92m{len(val_dataset)}\033[0m")
    print(f"Test: \033[96;92m{len(test_dataset)}\033[0m \n")

    mdl_info["RUN"]["n_samples"] = {
        "Train": len(train_dataset),
        "Val": len(val_dataset),
        "Test": len(test_dataset),
        }

    N_BANDS = mdl_info["RUN"]['N_BANDS']
    config["MODEL"]["N_BANDS"] = N_BANDS
    batch_size = config["MODEL"]["BATCH_SIZE"]
    num_workers = config["MODEL"]["NUM_WORKERS"]
    pin_memory = config["MODEL"]["PIN_MEMORY"]
    persistent_workers = config["MODEL"]["PERSISTENT_WORKERS"]

    train_loader = DataLoader(
        train_dataset,
        # test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    #======================================================================
    # Cuda

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_cuda_is_available = torch.cuda.is_available()
    torch_cuda_get_device_name = torch.cuda.get_device_name(0)

    mdl_info["RUN"]["cuda"] = {
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
    epochs = config["MODEL"]["EPOCHS"]

    best_model_dir = os.path.join(DIR_EXP, "best_model.pt")
    loss_dir = os.path.join(DIR_EXP, "df_loss.csv")

    MODEL_NAME = config["MODEL"]["MODEL_NAME"]

    if "model_trained" not in mdl_info["RUN"].keys() or not mdl_info["RUN"]["model_trained"]:

        #-----------------------------------------------------------------

        print("-"*80 + f"\n\t \033[96;01m  Initialize Training  \033[0m\n")
        for x in config["MODEL"]:
            print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")
        print("\n" + "-"*80)

        # SmallCNN  MobileNetV3Small   ResNet18
        #-----------------------------------------------------------------
        # Instanciação do modelo

        PRETRAINED = config["MODEL"]["PRETRAINED"]
        DROPOUT = config["MODEL"]["DROPOUT"]
        
        model_class = model_class_function(MODEL_NAME)

        if MODEL_NAME == "SmallCNN":
            model = model_class(
                in_channels=N_BANDS,
                num_classes=num_classes,
                base_channels=32,
                dropout=DROPOUT,
            )
        else:
            model = model_class(
                in_channels=N_BANDS,
                num_classes=num_classes,
                pretrained=PRETRAINED,   # ou True para carregar pesos ImageNet adaptados
                dropout=DROPOUT,
            )

        # ---- Treino ----
 
        t_0 = time()
        history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        device=str(device),
        checkpoint_path=best_model_dir,
        patience=30,                        # opcional, early stopping
        verbose=True,
        )
        t_1 = time()
        # ---- Fim de Treino ----

        elapsed = round(t_1 - t_0)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        time_train = f"{hours} hour, {minutes} min, {seconds} sec"
        print(f'\n time_train: {time_train}')

        mdl_info["RUN"]["model_trained"] = True

        df_loss = pd.DataFrame(history)
        df_loss.to_csv(loss_dir, index=False)

        best_epoch = int(np.argmax(history["val_f1_macro"])) 

        # Loss
        train_loss = history['train_loss'][best_epoch]
        val_loss = history['val_loss'][best_epoch]

        # ACC
        train_acc = history['train_acc'][best_epoch]
        val_acc = history['val_acc'][best_epoch]

        # Balanced ACC
        train_balanced_acc = history['train_balanced_acc'][best_epoch]
        val_balanced_acc = history['val_balanced_acc'][best_epoch]

        # f1 Macro
        train_f1_macro = history['train_f1_macro'][best_epoch]
        val_f1_macro = history['val_f1_macro'][best_epoch]

        # f1 Micro
        train_f1_micro = history['train_f1_micro'][best_epoch]
        val_f1_micro = history['val_f1_micro'][best_epoch]

        # Precision Macro
        train_precision_macro = history['train_precision_macro'][best_epoch]
        val_precision_macro = history['val_precision_macro'][best_epoch]

        # Recall Macro
        train_recall_macro = history['train_recall_macro'][best_epoch]
        val_recall_macro = history['val_recall_macro'][best_epoch]

        # Kappa
        train_kappa = history['train_kappa'][best_epoch]
        val_kappa = history['val_kappa'][best_epoch]

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
        
        print(f"time_train: \033[96;92m{time_train}\033[0m")
        print(f"best_epoch: \033[96;92m{best_epoch}\033[0m\n")
        print(f"train_acc: \033[96;92m{train_acc:.4f}\033[0m")
        print(f"val_acc: \033[96;92m{val_acc:.4f}\033[0m\n")
        print(f"train_loss: \033[96;92m{train_loss:.4f}\033[0m")
        print(f"val_acc: \033[96;92m{val_acc:.4f}\033[0m\n")

        print(f"Total de parâmetros: \033[96;92m{total_params:,}\033[0m")
        print(f"Parâmetros treináveis: \033[96;92m{trainable_params:,}\033[0m\n")

        mdl_info["RUN"]["model"] = {
            "device": str(device),
            "time_train": time_train,
            "epochs": epochs,
            "best_epoch": best_epoch,

            "total_params": total_params,
            "trainable_params": trainable_params,

            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_balanced_acc": train_balanced_acc,
            "val_balanced_acc": val_balanced_acc,
            "train_f1_macro": train_f1_macro,
            "val_f1_macro": val_f1_macro,
            "train_f1_micro": train_f1_micro,
            "val_f1_micro": val_f1_micro,
            "train_precision_macro": train_precision_macro,
            "val_precision_macro": val_precision_macro,
            "train_recall_macro": train_recall_macro,
            "val_recall_macro": val_recall_macro,
            "train_kappa": train_kappa,
            "val_kappa": val_kappa,

        }
        with open(model_info_dir, "w", encoding="utf-8") as file:
            json.dump(mdl_info, file, indent=4)

    else:

        # Load Model

        print("-"*80 + f"\n\t --- \033[96;01m  Loading Model --- \033[0m \n")
        for x in config["MODEL"]:
            print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")
        print("\n" + "-"*80)

        model = load_model_generic(best_model_dir, device=str(device))

        print(f"\n\t --- \033[96;01m  Loaded --- \033[0m \n")
        print(f"model_class: \033[96;92m{model.MODEL_NAME if hasattr(model, 'MODEL_NAME') else model.__class__.__name__}\033[0m")
        print(f"in_channels (bandas): \033[96;92m{model.config['in_channels']}\033[0m")
        print(f"num_classes: \033[96;92m{model.config['num_classes']}\033[0m")

    #======================================================================
    #
    # param = {
    #     "SEED_MODEL": SEED_MODEL,
    #     "N_BANDS": N_BANDS,
    #     "N_BANDS": N_BANDS,
    #     "time_train": mdl_info["RUN"]['model']['time_train'],
    #     "epochs": mdl_info["RUN"]['model']['epochs'],
    #     "best_epoch": mdl_info["RUN"]['model']['best_epoch'],
    #     "trainable_params": mdl_info["RUN"]['model']['trainable_params'],
    # }

    df_model = pd.DataFrame([config["MODEL"]])
    df_model["SEED_MODEL"] = SEED_MODEL
    df_model["N_BANDS"] = N_BANDS
    df_model["TIME_TRAIN"] = mdl_info["RUN"]['model']['time_train']
    df_model["EPOCHS"] = mdl_info["RUN"]['model']['epochs']
    df_model["BEST_EPOCH"] = mdl_info["RUN"]['model']['best_epoch']
    df_model["N_PARAMS"] = mdl_info["RUN"]['model']['trainable_params']

    #======================================================================
    # Metrics

    # df_metric_train_dir = os.path.join(DIR_EXP, "df_metric_train.csv")
    df_metric_val_dir = os.path.join(DIR_EXP, "df_metric_val.csv")
    df_metric_test_dir = os.path.join(DIR_EXP, "df_metric_test.csv")

    # cl_report_train_dir = os.path.join(DIR_EXP, "cl_report_train.txt")
    cl_report_val_dir = os.path.join(DIR_EXP, "cl_report_val.txt")
    cl_report_test_dir = os.path.join(DIR_EXP, "cl_report_test.txt")

    especies = sorted(os.listdir(TRAIN_DIR))

    if not os.path.isfile(df_metric_val_dir) or not os.path.isfile(df_metric_test_dir):

        # Predicting

        print(f"\n\t --- \033[96;01m  Model Predicting --- \033[0m\n")

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
        df_metrics_val = pd.concat([df_model, df_metrics_val], axis=1)
        df_metrics_val_col = list(df_metrics_val.columns)
        df_metrics_val["period"] = "val"
        df_metrics_val_col.insert(0, "period")
        df_metrics_val = df_metrics_val[df_metrics_val_col]
        df_metrics_val.to_csv(df_metric_val_dir, index=False)

        labels_val_real = [especies[i] for i in y_val_real]
        labels_val_pred = [especies[i] for i in y_val_pred]
        cl_report_val = classification_report(labels_val_real, labels_val_pred)
        print("\n\t --- Validation --- \n")
        print(cl_report_val)

        with open(cl_report_val_dir, "w") as file:
            file.write(cl_report_val)

        #------------------------------------------------------------------
        # Test
        test_results  = model.predict(test_loader, device=device)

        y_test_real = test_results["true"]
        y_test_pred = test_results["preds"]

        df_metrics_test = classification_metrics_dataframe(y_test_real, y_test_pred, especies)
        df_metrics_test = pd.concat([df_model, df_metrics_test], axis=1)
        df_metrics_test_col = list(df_metrics_test.columns)
        df_metrics_test["period"] = "test"
        df_metrics_test_col.insert(0, "period")
        df_metrics_test = df_metrics_test[df_metrics_test_col]
        df_metrics_test.to_csv(df_metric_test_dir, index=False)

        labels_test_real = [especies[i] for i in y_test_real]
        labels_test_pred = [especies[i] for i in y_test_pred]
        cl_report_test = classification_report(labels_test_real, labels_test_pred)
        print("\n\t --- Test --- \n")
        print(cl_report_test)

        with open(cl_report_test_dir, "w") as file:
            file.write(cl_report_test)


    #======================================================================
    #======================================================================
    #======================================================================
    # SHAP

    APPLY_SHAP = config["APPLY_SHAP"]

    if APPLY_SHAP:

        #======================================================================
        #======================================================================

        print(f"\n\033[100;01m\t     --- Start SHAAP ---     \t\033[0m\n")

        #======================================================================
        #======================================================================

        sleep(5)

        import gc
        import shap

        # ======================================================================
        # 1. Preparar dados de background e de explicação
        # ======================================================================

        model.eval()
        model.to(device)

        BACKGROUND_SIZE = 4
        EXPLAIN_BATCH = 4
        NSAMPLES = 20   # <-- controla o buffer interno (nsamples, num_bands, H, W); default do SHAP é 200

        shap_train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
        )

        shap_test_loader = DataLoader(
            test_dataset,
            batch_size=EXPLAIN_BATCH,
            shuffle=False,
            num_workers=0,
        )

        background_imgs = []
        for imgs, y_is, c_is, n_is in shap_train_loader:
            background_imgs.append(imgs)
            if len(background_imgs) * imgs.size(0) >= BACKGROUND_SIZE:
                break
        background_imgs = torch.cat(background_imgs)[:BACKGROUND_SIZE].to(device)

        explain_imgs, explain_y_is, explain_c_is, explain_n_is = next(iter(shap_test_loader))
        explain_imgs = explain_imgs.to(device)

        num_bands = explain_imgs.shape[1]
        band_names = [f"Band_{i+1}" for i in range(num_bands)]

        # ======================================================================
        # 2. Criar o explainer
        # ======================================================================

        torch.cuda.empty_cache()
        gc.collect()

        explainer = shap.GradientExplainer(model, background_imgs)

        shap_values, indexes = explainer.shap_values(
            explain_imgs,
            nsamples=NSAMPLES,
            ranked_outputs=1,
        )

        shap_pred_class = shap_values[0]
        pred_class_idx = indexes[:, 0]

        del explainer, shap_values
        torch.cuda.empty_cache()
        gc.collect()

        # ======================================================================
        # 3. Agregar em importância por banda (classe predita por amostra)
        # ======================================================================

        shap_abs_pred = np.abs(shap_pred_class)
        band_importance_pred_class = shap_abs_pred.mean(axis=(0, 2, 3))

        # ======================================================================
        # 4. Resultado
        # ======================================================================

        print("=== Importância por banda (classe predita por amostra) ===")
        for name, val in zip(band_names, band_importance_pred_class):
            print(f"{name}: {val:.6f}")

        pct_pred = 100 * band_importance_pred_class / band_importance_pred_class.sum()
        for name, val in zip(band_names, pct_pred):
            print(f"{name}: {val:.2f}%")



