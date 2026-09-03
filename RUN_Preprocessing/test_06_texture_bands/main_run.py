import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import json
from time import sleep, time
from sklearn.metrics import classification_report

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

import torch

# Auxiliar
try:
    from aux_model import *
    from aux_only_models import *
except ImportError:
    from RUN_Preprocessing.test_06_texture_bands.aux_model import *
    from RUN_Preprocessing.test_06_texture_bands.aux_only_models import *

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


#======================================================================
#======================================================================

# import yaml

# def load_config(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)
    
# yaml_test_name = "MTV_model_01.yaml"

# # Dante
# path = f"/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa/config/{yaml_test_name}"


# # path = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_test_name}"
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
    MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]

    print(f"\t  PC: \033[96;95m{PC} \033[0m")
    print(f"\t  Interactive: \033[96;95m{INTERACTIVE} \033[0m")
    print(f"\t  SEED_MODEL: \033[96;95m{SEED_MODEL} \033[0m\n")
    print(f"\t  MULTIVIEW_DATA_NICKNAME: \033[96;95m{MULTIVIEW_DATA_NICKNAME} \033[0m\n")

    # print(os.getcwd())

    if PC not in ["NITRO", "HELIOS", "DANTE"]:
        raise ValueError(f"PC: {PC} not correct")

    #======================================================================
    # PC Directory

    if PC == "NITRO":
        PC_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
    elif PC == "HELIOS":
        PC_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
    else:
        PC_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos"

    #======================================================================
    # GPU

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

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

    AUGMENTATION = config["AUGMENTATION"]

    SPLIT_DATA_NAME = config["SPLIT_DATA_NAME"]
    SPLIT_DATE_TYPE = config["SPLIT_DATE_TYPE"]

    AUG_DATE_NAME = config["AUG_DATE_NAME"]
    AUG_DATE_TYPE = config["AUG_DATE_TYPE"]

    if AUGMENTATION:
        TRAIN_DATA_DIR = f"Augmentation/{AUG_DATE_TYPE}/{AUG_DATE_NAME}"
    else:
        TRAIN_DATA_DIR = f"Split/{SPLIT_DATE_TYPE}/{SPLIT_DATA_NAME}"

    DATA_DIR = f"{PC_DIR}/Datasets/{TRAIN_DATA_DIR}"

    if not os.path.isdir(DATA_DIR):
        raise ValueError(f"DATA_DIR doesnt exist - {DATA_DIR[-60:]}")

    #======================================================================
    # Experiment Name and Directory

    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
    EXPERIMENT_TYPE = AUG_DATE_TYPE if AUGMENTATION else SPLIT_DATE_TYPE

    config["EXPERIMENT_TYPE"] = EXPERIMENT_TYPE

    DIR_EXP = f"{PC_DIR}/Results/{EXPERIMENT_TYPE}/{MULTIVIEW_DATA_NICKNAME}/{EXPERIMENT_NAME}"

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
        elif "aug_infos.json" in os.listdir(DATA_DIR):
            old_info_dir = os.path.join(DATA_DIR, "aug_infos.json")
            with open(old_info_dir, "r", encoding="utf-8") as file:
                mdl_info = json.load(file)
        else:
            raise ValueError("split_info.json or aug_infos.json was not found")

        mdl_info["RUN"] = {
            "EXPERIMENT_NAME": EXPERIMENT_NAME,
            "EXPERIMENT_TYPE": EXPERIMENT_TYPE,
            "AUGMENTATION": AUGMENTATION,
            "SEED_MODEL": SEED_MODEL,
            "DATA_DIR": DATA_DIR,
            "DIR_EXP": DIR_EXP,

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
    # 2. Datasets

    # train_dataset.__dict__.keys()
    # 'root_dir', 'transform', 'classes', 'class_to_idx', 'samples'

    train_dataset = MultispectralWeedDataset(TRAIN_DIR)
    val_dataset = MultispectralWeedDataset(VAL_DIR)
    test_dataset = MultispectralWeedDataset(TEST_DIR)

    print(f"\nTrain: \033[96;92m{len(train_dataset)}\033[0m")
    print(f"Val: \033[96;92m{len(val_dataset)}\033[0m")
    print(f"Test: \033[96;92m{len(test_dataset)}\033[0m \n")

    #======================================================================
    # Model Config

    mdl_info["RUN"]["n_samples"] = {
        "Train": len(train_dataset),
        "Val": len(val_dataset),
        "Test": len(test_dataset),
        }

    N_BANDS = mdl_info["RUN"]['N_BANDS']
    config["MODEL"]["N_BANDS"] = N_BANDS

    epochs = config["MODEL"]["EPOCHS"]
    batch_size = config["MODEL"]["BATCH_SIZE"]
    lr = config["MODEL"]["LR"]
    num_workers = config["MODEL"]["NUM_WORKERS"]
    # num_workers = 0
    pin_memory = config["MODEL"]["PIN_MEMORY"]
    persistent_workers = config["MODEL"]["PERSISTENT_WORKERS"]
    # persistent_workers = False

    # Model Config

    mdl_info["RUN"]["MODEL_CONFIG"] = {
        "MODEL_NAME": config["MODEL"]["MODEL_NAME"],
        "SEED_MODEL": config["MODEL"]["SEED_MODEL"],
        "AUGMENTATION": AUGMENTATION,

        "PRETRAINED": config["MODEL"]["PRETRAINED"],
        "BATCH_SIZE": config["MODEL"]["BATCH_SIZE"],
        "LR": config["MODEL"]["LR"],
        "EPOCHS": config["MODEL"]["EPOCHS"],
        "DROPOUT": config["MODEL"]["DROPOUT"],

        "N_BANDS": N_BANDS,
        "NUM_WORKERS": config["MODEL"]["NUM_WORKERS"],
        "PIN_MEMORY": config["MODEL"]["PIN_MEMORY"],
        "PERSISTENT_WORKERS": config["MODEL"]["PERSISTENT_WORKERS"],

    }

    #======================================================================
    # 2.1 DataLoaders

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
    torch_cuda_get_device_name = torch.cuda.get_device_name(0) if torch_cuda_is_available else None

    mdl_info["RUN"]["cuda"] = {
    "device": str(device),
    "torch_cuda_is_available": torch_cuda_is_available,
    "torch_cuda_get_device_name": torch_cuda_get_device_name}

    print(f"Device: \033[96;92m{device}\033[0m")
    print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
    print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")

    sleep(3)

    #======================================================================
    # Treinamento

    num_classes = len(train_dataset.classes)  # deve ser 31

    best_model_dir = os.path.join(DIR_EXP, "best_model.pt")
    loss_dir = os.path.join(DIR_EXP, "df_loss.csv")

    MODEL_NAME = config["MODEL"]["MODEL_NAME"]

    if "model_trained" not in mdl_info["RUN"].keys() or not mdl_info["RUN"]["model_trained"]:

        #-----------------------------------------------------------------

        print("-"*80 + f"\n\t \033[96;01m  Initialize Training  \033[0m\n")
        print(f" EXPERIMENT_NAME: \033[96;93m{EXPERIMENT_NAME}\033[0m\n")

        print(f"\n AUGMENTATION: \033[96;95m{AUGMENTATION}\033[0m\n")

        for x in config["MODEL"]:
            print(f" {x}: \033[96;96m{config['MODEL'][x]}\033[0m")

        print(f"\n MULTIVIEW_DATA_NICKNAME: \033[96;95m {config['MULTIVIEW_DATA_NICKNAME']} \033[0m")

        print(f"\n VIEWS:")
        for X in config['VIEWS']:
            for y in config['VIEWS'][X]:
                print(f"{X}: \033[96;93m{config['VIEWS'][X][y]}\033[0m")

        print("\n" + "-"*80)

        # SmallCNN MobileNetV3Small ResNet18 ConvNeXtTiny ViTTiny

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
                seed_model=SEED_MODEL
            )
        else:
            model = model_class(
                in_channels=N_BANDS,
                num_classes=num_classes,
                pretrained=PRETRAINED,   # ou True para carregar pesos ImageNet adaptados
                dropout=DROPOUT,
                seed_model=SEED_MODEL
            )

        #-----------------------------------------------------------------
        # Train
 
        t_0 = time()
        history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=1e-4,
        device=str(device),
        checkpoint_path=best_model_dir,
        patience=30,                        # opcional, early stopping
        verbose=True,
        )
        t_1 = time()

        #-----------------------------------------------------------------
        # Time

        elapsed = round(t_1 - t_0)

        # Tempo total
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        time_train = f"{hours} hour, {minutes} min, {seconds} sec"

        # Tempo médio por época
        elapsed_epoch = round(elapsed / epochs)

        hours_epoch = elapsed_epoch // 3600
        minutes_epoch = (elapsed_epoch % 3600) // 60
        seconds_epoch = elapsed_epoch % 60

        time_epoch = f"{hours_epoch} hour, {minutes_epoch} min, {seconds_epoch} sec"

        print(f"\ntime_train: {time_train}")
        print(f"time_epoch: {time_epoch}")

        #-----------------------------------------------------------------

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

        mdl_info["RUN"]["training"] = {
            "device": str(device),
            "time_train": time_train,
            "time_epoch": time_epoch,
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
        print(f"num_classes: \033[96;92m{model.config['num_classes']}\033[0m\n")

    #======================================================================
    # Model Params

    df_model = pd.DataFrame([config["MODEL"]])
    df_model["SEED_MODEL"] = SEED_MODEL
    df_model["NICKNAME"] = config['MULTIVIEW_DATA_NICKNAME']
    df_model["AUGMENTATION"] = AUGMENTATION
    df_model["N_BANDS"] = N_BANDS
    df_model["TIME_TRAIN"] = mdl_info["RUN"]['training']['time_train']
    df_model["TIME_EPOCH"] = mdl_info["RUN"]['training']['time_epoch']
    df_model["EPOCHS"] = mdl_info["RUN"]['training']['epochs']
    df_model["BEST_EPOCH"] = mdl_info["RUN"]['training']['best_epoch']
    df_model["N_PARAMS"] = mdl_info["RUN"]['training']['trainable_params']
    df_model["DEVICE"] = str(device)
    df_model["CUDA_IS_AVAILABLE"] = torch_cuda_is_available
    df_model["CUDA_DEVICE_NAME"] = torch_cuda_get_device_name

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

        print(f"\n\033[100;01m\t     --- Start SHAP ---     \t\033[0m\n")

        #======================================================================
        # Files and Diretories

        df_shap_por_classe_dir = os.path.join(DIR_EXP, "df_shap_por_classe.csv")
        df_diagnostico_dir = os.path.join(DIR_EXP, "df_diagnostico.csv")

        if not os.path.isfile(df_shap_por_classe_dir) or not os.path.isfile(df_diagnostico_dir):

            #======================================================================

            import gc
            import shap

            # ======================================================================
            # CUSTO COMPUTACIONAL — ajuste estas variáveis para controlar tempo/memória
            # ======================================================================
            NSAMPLES = 20         # nº de interpolações por imagem explicada
            BACKGROUND_SIZE = 4   # nº de imagens de referência (background), vindas de train
            # Com 31 classes (1 imagem cada), o custo total é ~31x o de uma única imagem.
            # Se ficar muito lento/pesado, reduza NSAMPLES primeiro.
            # ======================================================================

            model.eval()
            model.to(device)

            # ======================================================================
            # 1. Selecionar 1 imagem de teste por classe verdadeira
            # ======================================================================

            # test_dataset.samples é uma lista de (file_path, cls_idx, cls_name)
            # vinda da MultispectralWeedDataset -> usamos para pegar 1 amostra por classe

            one_per_class = {}  # cls_idx -> índice na lista test_dataset.samples
            for sample_idx, (file_path, cls_idx, cls_name) in enumerate(test_dataset.samples):
                if cls_idx not in one_per_class:
                    one_per_class[cls_idx] = sample_idx

            missing_classes = [c for c in test_dataset.class_to_idx.values() if c not in one_per_class]
            if missing_classes:
                missing_names = [test_dataset.classes[c] for c in missing_classes]
                print(f"[aviso] classes sem nenhuma imagem no conjunto de teste: {missing_names}")

            # ordena por cls_idx para o dataframe final sair na ordem das classes (0..30)
            selected = sorted(one_per_class.items())  # lista de (cls_idx, sample_idx)

            # ======================================================================
            # 2. Preparar background (referência), vindo de train
            # ======================================================================

            shap_train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
            )

            background_imgs = []
            for imgs, y_is, c_is, n_is in shap_train_loader:
                background_imgs.append(imgs)
                if len(background_imgs) * imgs.size(0) >= BACKGROUND_SIZE:
                    break
            background_imgs = torch.cat(background_imgs)[:BACKGROUND_SIZE].to(device)

            num_bands = background_imgs.shape[1]
            band_names = [f"banda_{i+1}" for i in range(num_bands)]

            print("background_imgs.shape:", tuple(background_imgs.shape))
            print(f"Classes selecionadas: {len(selected)} de {len(test_dataset.classes)}")

            # ======================================================================
            # 3. Criar o explainer
            # ======================================================================

            torch.cuda.empty_cache()
            gc.collect()

            explainer = shap.GradientExplainer(model, background_imgs)

            # ======================================================================
            # 4. Loop: 1 imagem por classe -> SHAP -> importância por banda (%)
            # ======================================================================

            rows = []          # uma linha por classe: percentuais por banda
            row_labels = []     # nome da classe (índice do dataframe final)
            diagnostics = []    # info extra por classe (arquivo, classe predita, bateu com verdadeira?)

            for cls_idx, sample_idx in selected:
                file_path, true_cls_idx, true_cls_name = test_dataset.samples[sample_idx]

                img, y_i, c_i, n_i = test_dataset[sample_idx]
                single_img = img.unsqueeze(0).to(device)  # (1, num_bands, H, W)

                sv, idx = explainer.shap_values(
                    single_img,
                    nsamples=NSAMPLES,
                    ranked_outputs=1,
                )

                sv_arr = sv[0]
                if torch.is_tensor(sv_arr):
                    sv_arr = sv_arr.detach().cpu().numpy()
                else:
                    sv_arr = np.asarray(sv_arr)
                sv_arr = np.squeeze(sv_arr)  # -> (num_bands, H, W)

                if sv_arr.shape != (num_bands,) + tuple(single_img.shape[2:]):
                    raise RuntimeError(
                        f"Classe '{true_cls_name}': shape inesperado após squeeze: {sv_arr.shape}. "
                        f"Esperado: {(num_bands,) + tuple(single_img.shape[2:])}."
                    )

                pred_class = idx[0, 0]
                pred_class = int(pred_class.detach().cpu().item()) if torch.is_tensor(pred_class) else int(pred_class)

                # importância por banda para essa imagem (média espacial)
                band_importance = np.abs(sv_arr).mean(axis=(1, 2))  # (num_bands,)
                pct = 100 * band_importance / band_importance.sum()

                rows.append(pct)
                row_labels.append(true_cls_name)
                diagnostics.append({
                    "classe_idx": true_cls_idx,
                    "classe_nome": true_cls_name,
                    "arquivo": file_path.stem,
                    "classe_predita_idx": pred_class,
                    "classe_predita_nome": test_dataset.classes[pred_class],
                    "acertou": pred_class == true_cls_idx,
                })

                torch.cuda.empty_cache()

            del explainer
            gc.collect()
            torch.cuda.empty_cache()

            # ======================================================================
            # 5. Montar o DataFrame final (31, 10)
            # ======================================================================

            df_shap_por_classe = pd.DataFrame(rows, index=row_labels, columns=band_names)
            df_shap_por_classe.to_csv(df_shap_por_classe_dir, index=True)

            df_diagnostico = pd.DataFrame(diagnostics)
            df_diagnostico.to_csv(df_diagnostico_dir, index=True)

            print("\n=== DataFrame de importância por banda, por classe (%) ===")
            print(df_shap_por_classe.round(2))

            print(f"\nAcurácia nas imagens explicadas: {df_diagnostico['acertou'].mean():.2%}")
            print("\n=== Diagnóstico por classe ===")
            print(df_diagnostico)


        #==================================================================================================
        #==================================================================================================
        print("\n\n" + "="*30 + f"\n\n\t\033[100;01m --- FIM --- \033[100;0m\n\n"  + "="*30)
        #==================================================================================================
        #==================================================================================================






#         #======================================================================

#         shap_dir = os.path.join(DIR_EXP, "df_shap.csv")

#         if not os.path.isfile(shap_dir):
                
#             #======================================================================

#             sleep(5)

#             import gc
#             import shap

#             # ======================================================================
#             # CUSTO COMPUTACIONAL — ajuste estas variáveis para controlar tempo/memória
#             # ======================================================================
#             NSAMPLES = 20         # nº de interpolações por imagem explicada (maior impacto no custo)
#             BACKGROUND_SIZE = 4   # nº de imagens de referência (background)
#             EXPLAIN_BATCH = 4     # nº de imagens a explicar (processadas uma a uma, ver nota abaixo)
#             # H, W da imagem também pesam bastante no custo, mas não são ajustados aqui
#             # (dependem do pré-processamento anterior).
#             # ======================================================================

#             model.eval()
#             model.to(device)

#             # ======================================================================
#             # 1. Preparar dados de background e de explicação
#             # ======================================================================

#             shap_train_loader = DataLoader(
#                 train_dataset,
#                 batch_size=4,
#                 shuffle=True,
#                 num_workers=0,
#             )

#             shap_test_loader = DataLoader(
#                 test_dataset,
#                 batch_size=EXPLAIN_BATCH,
#                 shuffle=False,
#                 num_workers=0,
#             )

#             background_imgs = []
#             for imgs, y_is, c_is, n_is in shap_train_loader:
#                 background_imgs.append(imgs)
#                 if len(background_imgs) * imgs.size(0) >= BACKGROUND_SIZE:
#                     break
#             background_imgs = torch.cat(background_imgs)[:BACKGROUND_SIZE].to(device)

#             explain_imgs, explain_y_is, explain_c_is, explain_n_is = next(iter(shap_test_loader))
#             explain_imgs = explain_imgs.to(device)

#             num_bands = explain_imgs.shape[1]
#             band_names = [f"Band_{i+1}" for i in range(num_bands)]

#             print("explain_imgs.shape:", tuple(explain_imgs.shape))
#             print("background_imgs.shape:", tuple(background_imgs.shape))

#             # ======================================================================
#             # 2. Criar o explainer
#             # ======================================================================

#             torch.cuda.empty_cache()
#             gc.collect()

#             explainer = shap.GradientExplainer(model, background_imgs)

#             # ======================================================================
#             # 3. Explicar UMA IMAGEM POR VEZ
#             #    (evita depender do comportamento de batching interno do shap com
#             #     ranked_outputs, que nessa versão está descartando a dimensão do lote)
#             # ======================================================================

#             per_image_shap = []      # vai acumular um array (num_bands, H, W) por imagem
#             per_image_pred_class = []

#             for i in range(explain_imgs.shape[0]):
#                 single_img = explain_imgs[i:i+1]  # mantém a dim de batch, shape (1, num_bands, H, W)

#                 sv, idx = explainer.shap_values(
#                     single_img,
#                     nsamples=NSAMPLES,
#                     ranked_outputs=1,
#                 )

#                 # ---- sv[0]: garante que está em CPU/numpy antes de manipular ----
#                 sv_arr = sv[0]
#                 if torch.is_tensor(sv_arr):
#                     sv_arr = sv_arr.detach().cpu().numpy()
#                 else:
#                     sv_arr = np.asarray(sv_arr)

#                 sv_arr = np.squeeze(sv_arr)  # remove eixos extras de tamanho 1

#                 if sv_arr.shape != (num_bands,) + tuple(single_img.shape[2:]):
#                     raise RuntimeError(
#                         f"Imagem {i}: shape inesperado após squeeze: {sv_arr.shape}. "
#                         f"Esperado: {(num_bands,) + tuple(single_img.shape[2:])}. "
#                         f"Investigue a versão do shap antes de prosseguir."
#                     )

#                 per_image_shap.append(sv_arr)

#                 # ---- idx: garante que está em CPU antes de converter para int ----
#                 pred_class = idx[0, 0]
#                 if torch.is_tensor(pred_class):
#                     pred_class = pred_class.detach().cpu().item()
#                 else:
#                     pred_class = int(np.asarray(pred_class))
#                 per_image_pred_class.append(int(pred_class))

#                 torch.cuda.empty_cache()

#             del explainer
#             gc.collect()
#             torch.cuda.empty_cache()

#             # ======================================================================
#             # 4. Empilhar em (EXPLAIN_BATCH, num_bands, H, W) e agregar por banda
#             # ======================================================================

#             shap_pred_class = np.stack(per_image_shap, axis=0)  # (EXPLAIN_BATCH, num_bands, H, W)
#             print("shap_pred_class.shape (empilhado):", shap_pred_class.shape)

#             assert shap_pred_class.shape[0] == explain_imgs.shape[0], (
#                 "Número de imagens no resultado empilhado não bate com EXPLAIN_BATCH."
#             )
#             assert shap_pred_class.shape[1] == num_bands, (
#                 "Eixo de bandas não bate com num_bands após empilhamento."
#             )

#             shap_abs_pred = np.abs(shap_pred_class)
#             band_importance_pred_class = shap_abs_pred.mean(axis=(0, 2, 3))  # média sobre imagens + espaço -> (num_bands,)

#             assert band_importance_pred_class.shape == (num_bands,)

#             # ======================================================================
#             # 5. Resultado
#             # ======================================================================

#             print(f"\nClasses preditas por imagem: {per_image_pred_class}")

#             print("\n=== Importância por banda (média sobre as imagens explicadas) ===")
#             for name, val in zip(band_names, band_importance_pred_class):
#                 print(f"{name}: {val:.6f}")

#             pct_pred = 100 * band_importance_pred_class / band_importance_pred_class.sum()
#             for name, val in zip(band_names, pct_pred):
#                 print(f"{name}: {val:.2f}%")

#             print(f"\nSoma das porcentagens (deve ser ~100%): {pct_pred.sum():.2f}%")



#             df_shap = pd.DataFrame(
#                 [band_importance_pred_class, pct_pred],
#                 index=["importancia_media", "percentual"],
#                 columns=[f"banda_{i+1}" for i in range(num_bands)],
#             )

#             print(df_shap)

#             df_shap.to_csv(shap_dir, index=True)
            

            # #==================================================================================================
            # #==================================================================================================
            # print("\n\n" + "="*30 + f"\n\n\t\033[100;01m --- FIM --- \033[100;0m\n\n"  + "="*30)
            # #==================================================================================================
            # #==================================================================================================


        # import gc
        # import shap

        # # ======================================================================
        # # 1. Preparar dados de background e de explicação
        # # ======================================================================

        # model.eval()
        # model.to(device)

        # BACKGROUND_SIZE = 4
        # EXPLAIN_BATCH = 4
        # NSAMPLES = 20

        # shap_train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=4,
        #     shuffle=True,
        #     num_workers=0,
        # )

        # shap_test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=EXPLAIN_BATCH,
        #     shuffle=False,
        #     num_workers=0,
        # )

        # background_imgs = []
        # for imgs, y_is, c_is, n_is in shap_train_loader:
        #     background_imgs.append(imgs)
        #     if len(background_imgs) * imgs.size(0) >= BACKGROUND_SIZE:
        #         break
        # background_imgs = torch.cat(background_imgs)[:BACKGROUND_SIZE].to(device)

        # explain_imgs, explain_y_is, explain_c_is, explain_n_is = next(iter(shap_test_loader))
        # explain_imgs = explain_imgs.to(device)

        # num_bands = explain_imgs.shape[1]
        # band_names = [f"Band_{i+1}" for i in range(num_bands)]

        # # ======================================================================
        # # 2. Criar o explainer
        # # ======================================================================

        # torch.cuda.empty_cache()
        # gc.collect()

        # explainer = shap.GradientExplainer(model, background_imgs)

        # shap_values, indexes = explainer.shap_values(
        #     explain_imgs,
        #     nsamples=NSAMPLES,
        #     ranked_outputs=1,
        # )

        # shap_pred_class = np.asarray(shap_values[0])
        # pred_class_idx = indexes[:, 0]

        # del explainer, shap_values
        # torch.cuda.empty_cache()
        # gc.collect()

        # # ======================================================================
        # # 3. Agregar em importância por banda (classe predita por amostra)
        # # ======================================================================

        # print("explain_imgs.shape:", tuple(explain_imgs.shape))
        # print("shap_pred_class.shape (bruto):", shap_pred_class.shape)

        # # ranked_outputs pode adicionar um eixo extra de tamanho 1 no final -> remove
        # if shap_pred_class.ndim == explain_imgs.ndim + 1 and shap_pred_class.shape[-1] == 1:
        #     shap_pred_class = shap_pred_class.squeeze(-1)
        #     print("shap_pred_class.shape (após squeeze):", shap_pred_class.shape)

        # # localiza o eixo das bandas em vez de assumir axis=1 -----------------
        # if shap_pred_class.shape == tuple(explain_imgs.shape):
        #     channel_axis = 1  # caso esperado: (N, C, H, W), igual ao input
        # else:
        #     candidates = [ax for ax, size in enumerate(shap_pred_class.shape) if size == num_bands]
        #     if len(candidates) != 1:
        #         raise RuntimeError(
        #             f"Não foi possível identificar com segurança o eixo das bandas. "
        #             f"shap_pred_class.shape={shap_pred_class.shape}, num_bands={num_bands}, "
        #             f"candidatos={candidates}. Investigue manualmente antes de prosseguir."
        #         )
        #     channel_axis = candidates[0]
        #     print(f"[aviso] shape diferente do esperado; eixo de bandas inferido = {channel_axis}")

        # reduce_axes = tuple(ax for ax in range(shap_pred_class.ndim) if ax != channel_axis)

        # shap_abs_pred = np.abs(shap_pred_class)
        # band_importance_pred_class = shap_abs_pred.mean(axis=reduce_axes)  # deve virar (num_bands,)

        # assert band_importance_pred_class.shape == (num_bands,), (
        #     f"Esperado shape ({num_bands},), obtive {band_importance_pred_class.shape}. "
        #     f"channel_axis usado: {channel_axis}"
        # )

        # # ======================================================================
        # # 4. Resultado
        # # ======================================================================

        # print("\n=== Importância por banda (classe predita por amostra) ===")
        # for name, val in zip(band_names, band_importance_pred_class):
        #     print(f"{name}: {val:.6f}")

        # pct_pred = 100 * band_importance_pred_class / band_importance_pred_class.sum()
        # for name, val in zip(band_names, pct_pred):
        #     print(f"{name}: {val:.2f}%")

        # print(f"\nSoma das porcentagens (deve ser ~100%): {pct_pred.sum():.2f}%")




