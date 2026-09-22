import os
import numpy as np
import pandas as pd

# Auxiliar
from aux_plot import *
from aux_transformations import *
from aux_feature_metrics import *

#======================================================================
#======================================================================
# Dataset

PC = "NITRO"

if PC == "NITRO":
    PC_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
elif PC == "DANTE":
    PC_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos"
else:
    PC_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"

#======================================================================
# DATA_DIR

DATA_DIR = f"{PC_DIR}/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Val_Norm"


#======================================================================


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_curves_plot(df: pd.DataFrame, df_dir: str, title="", xlabel="", ylabel="") -> None:
    """
    Plota cada coluna numérica do DataFrame como uma curva
    e salva a figura em df_dir.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cujas colunas numéricas representam curvas.

    df_dir : str
        Caminho completo para salvar a figura, incluindo nome
        e extensão.

        Exemplo:
        "/home/user/resultados/curvas.png"
    """

    output_path = Path(df_dir)

    # Cria o diretório caso ele ainda não exista
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Seleciona apenas colunas numéricas
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("O DataFrame não possui colunas numéricas.")

    plt.figure(figsize=(10, 6))

    for column in numeric_df.columns:
        plt.plot(
            numeric_df.index,
            numeric_df[column],
            label=str(column)
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


#======================================================================
# Dir 

results_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Trans_Measure"

# TRANS_DICT = {
#     suppress_patch_shuffle: [0, 2, 4, 6, 8, 12, 16],
#     suppress_patch_rotation: [0, 2, 4, 6, 8, 12, 16],
#     suppress_gaussian_blur_mask_aware: [0, 1, 2, 3, 4, 5, 6],
#     suppress_bilateral_filter_mask_aware: [0, 1, 2, 3, 4, 5, 6],
#     suppress_rgb_color: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     # suppress_colors: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     # suppress_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_rgb_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_nir_re_to_mean: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
# }

TRANS_DICT = {
    suppress_patch_shuffle: [0, 2, 4, 6, 8, 12, 16],
    suppress_gaussian_blur_mask_aware: [0, 1, 2, 3, 4, 5, 6],
    suppress_rgb_color: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
}


TRANS_DICT = {
    suppress_patch_shuffle: [0, 2, 4, 5, 6, 7, 8],
    suppress_gaussian_blur_mask_aware: [0, 1, 2, 2.5, 3, 3.5, 4],
    suppress_rgb_color: [0, 0.13, 0.27, 0.41, 0.55, 0.69 ,0.83],
    suppress_nir_re_to_green: [0, 0.05, 0.11, 0.17, 0.23, 0.29, 0.35],
}

#======================================================================
# Measures

MEASURE_LIST = [
    long_range_spatial_organization,
    local_variance_LV,
    mean_chroma_new__mean_chroma,
    spectral_rgb_residual
]

#======================================================================

measure = long_range_spatial_organization
measure = local_variance_LV
measure = mean_chroma_new__mean_chroma
measure = spectral_rgb_residual

def scale_trans_measure(
        DATA_DIR,
        TRANS_DICT,
        measure,
        n_especie=1,
        comparative=False
):

    # control: same number of params

    first_number = True
    for x in TRANS_DICT:
        n_p = len(TRANS_DICT[x])
        if first_number:
            f_p = n_p
            first_number = False
        elif n_p != f_p:
            raise ValueError("n_params not the same")

    #----------------------------------------------

    trans_list = list(TRANS_DICT.keys())
    trans_names = [met.__name__ for met in TRANS_DICT]

    all_values = {x: [] for x in trans_names}

    especies = sorted(os.listdir(DATA_DIR))

    if n_especie == -1:
        n = len(especies)
    else:
        n = n_especie

    for especie in especies[:n]:    # especie = especies[0]

        especie_dir = os.path.join(DATA_DIR, especie)
        files = sorted(set([x for x in os.listdir(especie_dir)]))

        for k, file_name in enumerate(files):     # k, file_name = 0, files[0]

            print(f"{especie} - file: {k} of {len(files)}")

            file_dir = os.path.join(especie_dir, file_name)

            img_5b = np.load(file_dir).astype("float32")

            for i, trans in enumerate(trans_list):    # i, trans = 0, trans_list[0]

                trans_params = TRANS_DICT[trans]

                ith_values = []

                for j, param in enumerate(trans_params): # j, param = 0, trans_params[0]

                    # print(f"(k, i, j): {k, i, j} - {trans_names[i]}")

                    jth_img_trans = trans(img_5b, param)

                    if comparative:
                        met_value = measure(img_5b, jth_img_trans)
                    else:
                        met_value = measure(jth_img_trans)

                    if not (isinstance(met_value, int) or isinstance(met_value, float) or isinstance(met_value, np.ndarray)):
                        raise ValueError(f"met_value: {met_value}")
                    
                    ith_values.append(met_value)

                ith_np = np.array([x/(ith_values[0]+1e-16) for x in ith_values])

                all_values[trans_names[i]].append(ith_np)

    result = pd.DataFrame({x: np.r_[all_values[x]].mean(axis=0) for x in all_values.keys()})

    result.plot(title=measure.__name__, grid=True, figsize=(12, 9))

    return result







