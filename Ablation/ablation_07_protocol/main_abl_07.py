import os
import pandas as pd
import json

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

# Auxiliar
from aux_plot import *
from aux_model import *
from aux_only_models import *
from aux_transformations import *

#======================================================================
#======================================================================

def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
    print(data.shape)

def p(data):
    print(pd.DataFrame(data).to_string())
    print(data.shape)

#======================================================================
#======================================================================
# PC Directory

PC = "EUROPA"

if PC == "NITRO":
    PC_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
elif PC == "HELIOS":
    PC_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
elif PC == "DANTE":
    PC_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos"
elif PC == "EUROPA":
    PC_DIR = f"/home/u1469618/Documentos/Datasets/Embrapa_Experimentos"
else:
    raise ValueError(f"PC: {PC} is not correct")

#======================================================================
# Directories

# DANTE

VAL_DATA_DIR = f"{PC_DIR}/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Val_Norm"


# Dante
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__MobileNetV3Small__DROPOUT_0.2_BATCH_SIZE_8_lr_0.0001_EPOCHS_30_AUG_True_SEED_10"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__ResNet18__DROPOUT_0.2_BATCH_SIZE_8_lr_0.0001_EPOCHS_30_AUG_True"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__ViTTiny__DROPOUT_0.2_BATCH_SIZE_8_lr_0.0001_EPOCHS_30_AUG_True_SEED_20"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__SmallCNN__DROPOUT_0.2_BATCH_SIZE_8_lr_0.0001_EPOCHS_30_AUG_True"

# EUROPA
EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__MobileNetV3Small__DROPOUT_0.2_BATCH_SIZE_12_lr_0.0001_EPOCHS_30_AUG_True_SEED_7"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__ResNet18__DROPOUT_0.2_BATCH_SIZE_12_lr_0.0001_EPOCHS_30_AUG_True_SEED_7"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__SmallCNN__DROPOUT_0.2_BATCH_SIZE_12_lr_0.0001_EPOCHS_30_AUG_True_SEED_7"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__ConvNeXtTiny__DROPOUT_0.2_BATCH_SIZE_12_lr_0.0001_EPOCHS_30_AUG_True_SEED_7"
# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__ViTTiny__DROPOUT_0.2_BATCH_SIZE_12_lr_0.0001_EPOCHS_30_AUG_True_SEED_7"




BAND_TYPE = "Multiview_Texture__AUG"
EXP_TYPE = "RGB_NIR_RE"
# EXP_TYPE = "RGB_entropy"
EXP_DIR = f"{PC_DIR}/Results/{BAND_TYPE}/{EXP_TYPE}/{EXP_NAME}"


ABL_NAME = "Ablation_07_protocol"
ABL_DIR = f"{PC_DIR}/Ablation/{ABL_NAME}/{BAND_TYPE}/{EXP_TYPE}/{EXP_NAME}"

#-----------------------------------------------------------------------
# NITRO

# VAL_DATA_DIR = f"---{PC_DIR}/Datasets/Multiview_5_BANDS/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20/Val_Norm/"

# EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__MobileNetV3Small__DROPOUT_0.2_BATCH_SIZE_8_lr_0.0001_EPOCHS_30_AUG_True"
# BAND_TYPE = "Multiview_Texture__AUG"
# EXP_TYPE = "RGB_NIR_RE"
# EXP_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Results_Dante_2026-09-10/{BAND_TYPE}/{EXP_TYPE}/{EXP_NAME}"


# ABL_NAME = "Ablation_06_seven_trans"
# ABL_DIR = f"{PC_DIR}/Ablation/{ABL_NAME}/{BAND_TYPE}/{EXP_TYPE}/{EXP_NAME}"

#======================================================================
# See image

especies = sorted(os.listdir(VAL_DATA_DIR))
especie = "03_brizantha_Agua_Boa_03"
files = sorted(os.listdir(os.path.join(VAL_DATA_DIR, especie)))
file_name = files[0]
file_fir = os.path.join(VAL_DATA_DIR, especie, file_name)

img = np.load(file_fir).astype("float32")

plot_rgb(img)

count_connected_components(img)
img = keep_bigger_components(img, 20)

img_trans = suppress_texture(img)
img_trans = suppress_texture_mask_aware(img)
img_trans = suppress_local_shape_contour(img, sigma=2)
img_trans = suppress_shape_elastic_deformation(img, intensity=30)
img_trans = elastic_deformation_local(img, intensity=20, size=50)
plot_rgb(img_trans)

mdl_info_dir = os.path.join(EXP_DIR, 'mld_info.json')

with open(mdl_info_dir, "r") as f:
    mdl_info = json.load(f)


#======================================================================
# All Models


TRANS_DICT = {
    suppress_patch_shuffle: [0, 2, 4, 6, 8, 12, 16],
    suppress_patch_rotation: [0, 2, 4, 6, 8, 12, 16],

    suppress_gaussian_blur_mask_aware: [0, 1, 2, 3, 4, 5, 6],
    suppress_bilateral_filter_mask_aware: [0, 1, 2, 3, 4, 5, 6],

    suppress_rgb_color: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_rgb_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],

    suppress_nir_re_to_mean: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
}


transformation = suppress_patch_shuffle
transformation_params = [0, 2, 4, 6, 8, 12, 16]

#======================================================================

def do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
        do_again=False
):

    #------------------------------------------------------------------
    # DIRS

    output_dir = ABL_DIR
    os.makedirs(output_dir, exist_ok=True)

    df_all_dir = f"{output_dir}/df_all_{transformation.__name__}.csv"
    # fig_dir = f"{output_dir}/ablation_study.png"

    if not os.path.isfile(df_all_dir) or do_again:

        #------------------------------------------------------------------
        # INFO

        mdl_info_dir = os.path.join(EXP_DIR, 'mld_info.json')

        with open(mdl_info_dir, "r") as f:
            mdl_info = json.load(f)

        #------------------------------------------------------------------

        model_name = mdl_info["RUN"]["MODEL_CONFIG"]["MODEL_NAME"]
        if mdl_info["RUN"]["MODEL_CONFIG"]["AUGMENTATION"]:
            model_name += "_AUG"

        print(f"model_name: \033[100;40m   {model_name}    \033[0m")


        #------------------------------------------------------------------
        # Val Metrics

        val_metric_dir = os.path.join(EXP_DIR, 'df_metric_val.csv')
        df_metric_val = pd.read_csv(val_metric_dir)

        #------------------------------------------------------------------
        # 1. Dataset PyTorch

        DATA_DIR = Path(VAL_DATA_DIR)

        VAL_DIR  = DATA_DIR 

        #----------------------------------------------------------------------
        # Cuda

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_cuda_is_available = torch.cuda.is_available()
        torch_cuda_get_device_name = torch.cuda.get_device_name(0) if torch_cuda_is_available else None

        print(f"Device: \033[96;92m{device}\033[0m")
        print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
        print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")

        #----------------------------------------------------------------------
        # Load Model

        model_dir = os.path.join(EXP_DIR, 'best_model.pt')
        model = load_model_generic(model_dir, device=str(device))

        #----------------------------------------------------------------------
        # especies

        especies = sorted(os.listdir(VAL_DIR))

        #----------------------------------------------------------------------
        # Cases

        def define_trans(param):

            if param == 0 or param is None:
                def trans_function(img):
                    return img
            else:
                def trans_function(img):
                    return transformation(img, param)

            return trans_function

        #-------------------------

        transformations_list = [define_trans(k) for k in transformation_params]

        name_cases_list = [f"{transformation.__name__}_{k}" for k in transformation_params]

        #----------------------------------------------------------------------
        # For all Bands

        df_all = df_metric_val.copy()

        df_all_col = list(df_all.columns)
        df_all["EXP"] = "Original"
        df_all_col.insert(21, "EXP")

        df_all = df_all[df_all_col]

        df_temp = df_all.iloc[:, :22].copy()

        for i, transform in enumerate(transformations_list):   # i = 0

            print("\n"+ "="*80 + f"\n i: {i} -- {transformation_params[i]} - {name_cases_list[i]}")
            # transform = transformations_list[0]

            val_dataset = WeedDataset_Transform(VAL_DIR, transform=transform)

            print(f"mean: \n{val_dataset[2][0].mean(axis=(1, 2))}")
            print(f"std: \n{val_dataset[2][0].std(axis=(1, 2))}")


            N_BANDS = int(mdl_info["RUN"]['N_BANDS'])
            batch_size = int(df_metric_val.loc[0, "BATCH_SIZE"])
            num_workers = int(df_metric_val.loc[0, "NUM_WORKERS"])
            pin_memory = int(df_metric_val.loc[0, "PIN_MEMORY"])
            persistent_workers = int(df_metric_val.loc[0, "PERSISTENT_WORKERS"])

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )

            new_val_results = model.predict(val_loader, device=device)

            y_val_real = new_val_results["true"]
            y_val_pred = new_val_results["preds"]

            df_metric_val_abl = classification_metrics_dataframe(y_val_real, y_val_pred, especies)

            # df_temp["EXP"] = "Only_" + "_".join([str(x) for x in sorted(set(range(5)) - set(bands))])
            df_temp["EXP"] = name_cases_list[i]
            df_metric_val_abl = pd.concat([df_temp, df_metric_val_abl], axis=1)

            df_all = pd.concat([df_all, df_metric_val_abl], axis=0).reset_index(drop=True)

        p(df_all)

        df_all.to_csv(df_all_dir, index=False)

    else:
        df_all = pd.read_csv(df_all_dir)

    return df_all

        # #------------------------------------------------------------------
        # # PLOT

        # fig_acc, ax_acc = plot_ablation_4_metric(df_all, title=f"Spectral Band Ablation Study - {model_name}", figsize=(9, 6))
        # # fig_prc_micro, ax_prc_micro = plot_ablation_metric(df_all, metric="precision_micro", title=f"Spectral Band Ablation Study - {model_name}")

        # #------------------------------------------------------------------
        # # Save

        # #-------------------------
        # # Metrics

        # df_all.to_csv(df_all_dir, index=False)

        # #-------------------------
        # # Image ACC

        # fig_acc.savefig(
        #     fig_dir,
        #     dpi=300,
        #     bbox_inches="tight"
        # )


#======================================================================
#======================================================================

TRANS_DICT = {
    suppress_patch_shuffle: [0, 2, 4, 6, 8, 12, 16],
    suppress_patch_rotation: [0, 2, 4, 6, 8, 12, 16],

    suppress_gaussian_blur_mask_aware: [0, 1, 2, 3, 4, 5, 6],
    suppress_bilateral_filter_mask_aware: [0, 1, 2, 3, 4, 5, 6],

    suppress_rgb_color: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_rgb_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],

    suppress_nir_re_to_mean: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
}

#======================================================================
# Shape

transformation = suppress_patch_shuffle
transformation_params = [0, 2, 4, 6, 8, 12, 16]

df_patch_shuffle = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
        # True
)

title = transformation.__name__
df_patch_shuffle_ = df_patch_shuffle.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_patch_shuffle_["acuracia"])], norm=True, title=title, figsize=(10, 6))

#----------------------------------------------------------------------
transformation = suppress_patch_rotation
transformation_params = [0, 2, 4, 6, 8, 12, 16]

df_patch_rotation = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
        # True
)

title = transformation.__name__
df_patch_rotation_ = df_patch_rotation.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_patch_rotation_["acuracia"])], norm=True, title=title, figsize=(10, 6))

#======================================================================
# Texture

transformation = suppress_gaussian_blur_mask_aware
transformation_params = [0, 1, 2, 3, 4, 5, 6]

df_gaussian_blur = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

title = transformation.__name__
df_gaussian_blur_ = df_gaussian_blur.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_gaussian_blur_["acuracia"])], norm=True, title=title, figsize=(10, 6))

#----------------------------------------------------------------------

transformation = suppress_bilateral_filter_mask_aware
transformation_params = [0, 1, 2, 3, 4, 5, 6]

df_bilateral_filter = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
        # True     
)

title = transformation.__name__
df_bilateral_filter_ = df_bilateral_filter.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_bilateral_filter_["acuracia"])], norm=True, title=title, figsize=(10, 6))


#======================================================================
# Color

transformation = suppress_rgb_color
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]

df_rgb_color = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

title = transformation.__name__
df_rgb_color_ = df_rgb_color.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_rgb_color_["acuracia"])], norm=True, title=title, figsize=(10, 6))

#----------------------------------------------------------------------
transformation = suppress_rgb_channel_shuffle
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]

df_rgb_channel_shuffle = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

title = transformation.__name__
df_rgb_channel_shuffle_ = df_rgb_channel_shuffle.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_rgb_channel_shuffle["acuracia"])], norm=True, title=title, figsize=(10, 6))

#======================================================================
# Spectrum

transformation = suppress_nir_re_to_mean
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]

df_nir_re_to_mean = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

title = transformation.__name__
df_nir_re_to_mean_ = df_nir_re_to_mean.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_nir_re_to_mean_["acuracia"])], norm=True, title=title, figsize=(10, 6))

#----------------------------------------------------------------------

transformation = suppress_nir_re_to_green
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]

df_nir_re_to_green = do_ablation_07(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

title = transformation.__name__
df_nir_re_to_green_ = df_nir_re_to_green.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_nir_re_to_green_["acuracia"])], norm=True, title=title, figsize=(10, 6))


#======================================================================
#======================================================================

title = "MobileNetV3Small"

plotar_multiplas_linhas([df_patch_shuffle_["acuracia"], 
                         df_patch_rotation_["acuracia"],
                         df_gaussian_blur_["acuracia"], 
                         df_bilateral_filter_["acuracia"], 
                         df_rgb_color_["acuracia"], 
                         df_rgb_channel_shuffle_["acuracia"], 
                         df_nir_re_to_mean_["acuracia"], 
                         df_nir_re_to_green_["acuracia"]], 
                         nomes=["patch_shuffle", "patch_rotation", 
                                "gaussian_blur", "bilateral_filter", 
                                "rgb_color", "rgb_channel_shuffle", 
                                "nir_re_to_mean", "nir_re_to_green"], norm=True, title=title, figsize=(15, 9))



plotar_multiplas_linhas([df_patch_shuffle_["acuracia"], 
                         df_gaussian_blur_["acuracia"],
                         df_rgb_color_["acuracia"],
                         df_nir_re_to_green_["acuracia"]], 
                         nomes=["patch_shuffle", "gaussian_blur", "rgb_color", "nir_re_to_green"], norm=True, title=title, figsize=(15, 9))


#======================================================================
#======================================================================

import numpy as np


def degradation_metrics(accuracies):
    """
    Calcula métricas de degradação para uma transformação.

    Parameters
    ----------
    accuracies : list ou array
        Acurácias ordenadas da imagem original (nível 0)
        até a maior intensidade de transformação.

        Exemplo:
        [0.92, 0.90, 0.91, 0.89, 0.82, 0.67, 0.50]

    Returns
    -------
    tuple
        (variacao_bruta, variacao_percentual, auc)
    """

    acc = np.asarray(accuracies, dtype=float)

    if acc.ndim != 1 or len(acc) < 2:
        raise ValueError("accuracies deve ser uma lista 1D com pelo menos 2 valores.")

    baseline = acc[0]

    if baseline == 0:
        raise ValueError("A acurácia baseline não pode ser zero.")

    # 1. Variação bruta
    variacao_bruta = baseline - acc[-1]

    # 2. Variação percentual
    variacao_percentual = variacao_bruta / baseline

    # 3. AUC da curva relativa, com intensidade normalizada em [0, 1]
    x = np.linspace(0, 1, len(acc))
    acc_relativa = acc / baseline

    auc = np.trapz(acc_relativa, x)

    return variacao_bruta, variacao_percentual, auc


import numpy as np
import pandas as pd


def degradation_metrics(accuracies):
    """
    Retorna:
        (gross_variation, percentage_variation, auc)
    """

    acc = np.asarray(accuracies, dtype=float)

    if acc.ndim != 1 or len(acc) < 2:
        raise ValueError("accuracies deve conter pelo menos 2 valores.")

    baseline = acc[0]

    if baseline == 0:
        raise ValueError("A acurácia baseline não pode ser zero.")

    # Variação bruta
    gross_variation = baseline - acc[-1]

    # Variação percentual
    percentage_variation = (
        (baseline - acc[-1]) / baseline
    ) * 100

    # AUC relativa ao baseline
    x = np.linspace(0, 1, len(acc))
    relative_acc = acc / baseline

    auc = np.trapz(relative_acc, x)

    return gross_variation, percentage_variation, auc


def transformations_metrics(trans_acc_dict):
    """
    Calcula as métricas de degradação para cada transformação.

    Parameters
    ----------
    trans_acc_dict : dict
        Dicionário:
            chave  -> nome da transformação
            valor  -> Series/lista de acurácias ordenadas por intensidade

    Returns
    -------
    pd.DataFrame
    """

    rows = []

    for transformation, accuracies in trans_acc_dict.items():

        gross_var, percentage_var, auc = degradation_metrics(
            accuracies
        )

        rows.append({
            "Transformations": transformation,
            "Gross Variation": gross_var,
            "Percentage Variation": percentage_var,
            "AUC": auc
        })

    return pd.DataFrame(rows)

#======================================================================


plotar_multiplas_linhas([df_patch_shuffle_["acuracia"], 
                         df_gaussian_blur_["acuracia"],
                         df_rgb_color_["acuracia"],
                         df_nir_re_to_green_["acuracia"]], 
                         nomes=["patch_shuffle", "gaussian_blur", "rgb_color", "nir_re_to_green"], norm=True, title=title, figsize=(15, 9))

degradation_metrics(list(df_patch_shuffle_["acuracia"]))

trans_acc_dict = {
    "patch_shuffle": df_patch_shuffle_["acuracia"], 
    "gaussian_blur": df_gaussian_blur_["acuracia"], 
    "rgb_color": df_rgb_color_["acuracia"], 
    "nir_re_to_green": df_nir_re_to_green_["acuracia"], 
}

transformations_metrics(trans_acc_dict).sort_values("AUC", ascending=True)

#----------------------------------------------------------------------

plotar_multiplas_linhas([df_patch_shuffle_["acuracia"], 
                         df_patch_rotation_["acuracia"],
                         df_gaussian_blur_["acuracia"], 
                         df_bilateral_filter_["acuracia"], 
                         df_rgb_color_["acuracia"], 
                         df_rgb_channel_shuffle_["acuracia"], 
                         df_nir_re_to_mean_["acuracia"], 
                         df_nir_re_to_green_["acuracia"]], 
                         nomes=["patch_shuffle", "patch_rotation", 
                                "gaussian_blur", "bilateral_filter", 
                                "rgb_color", "rgb_channel_shuffle", 
                                "nir_re_to_mean", "nir_re_to_green"], norm=True, title=title, figsize=(15, 9))


degradation_metrics(list(df_patch_shuffle_["acuracia"]))

trans_acc_dict = {
    "patch_shuffle": df_patch_shuffle_["acuracia"], 
    "patch_rotation": df_patch_rotation_["acuracia"], 
    "gaussian_blur": df_gaussian_blur_["acuracia"], 
    "bilateral_filter": df_bilateral_filter_["acuracia"], 
    "rgb_color": df_rgb_color_["acuracia"], 
    "rgb_channel_shuffle": df_rgb_channel_shuffle_["acuracia"], 
    "nir_re_to_mean": df_nir_re_to_mean_["acuracia"], 
    "nir_re_to_green": df_nir_re_to_green_["acuracia"], 
}

transformations_metrics(trans_acc_dict).sort_values("AUC", ascending=True)


#======================================================================
#======================================================================
