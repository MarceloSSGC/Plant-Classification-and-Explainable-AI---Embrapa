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
#     suppress_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_nir_re_to_mean: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
# }

TRANS_DICT = {
    suppress_patch_shuffle: [0, 2, 4, 6, 8, 12, 16],
    suppress_patch_rotation: [0, 2, 4, 6, 8, 12, 16],
    suppress_gaussian_blur_mask_aware: [0, 1, 2, 3, 4, 5, 6],
    suppress_bilateral_filter_mask_aware: [0, 1, 2, 3, 4, 5, 6],
    suppress_rgb_color: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    # suppress_colors: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    # suppress_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_rgb_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_nir_re_to_mean: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
    suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
}

#======================================================================

# SHAPE
# shape_metric_list = [
#     (edge_ssim_ESSIM, True),
#     (gradient_correlation_GC, True),
#     (long_range_spatial_organization, False),
#     (shape_descriptors__area, False),
#     (shape_descriptors__perimeter, False),
#     (shape_descriptors__compactness, False),
#     (shape_descriptors__solidity, False),
#     (shape_descriptors__hu_1, False),
#     (shape_descriptors__hu_2, False),
#     (coarse_ssim_mask_aware, True),
# ]

shape_metric_list = [
    (edge_ssim_ESSIM, True),
    (gradient_correlation_GC, True),
    (long_range_spatial_organization, False),
    # (shape_descriptors__area, False),
    # (shape_descriptors__perimeter, False),
    # (shape_descriptors__compactness, False),
    # (shape_descriptors__solidity, False),
    # (shape_descriptors__hu_1, False),
    # (shape_descriptors__hu_2, False),
    # (coarse_ssim_mask_aware, True),
]

#----------------------------------------------------------------------
# TEXTURE

# texture_metric_list = [
#     (local_variance_LV, False),
#     (high_frequency_energy_HFE, False),

#     (laplacian_variance_mask_aware, False),
#     (glcm_contrast_energy__New__energy, False),
#     (glcm_contrast_energy__New__contrast, False),

#     (high_low_freq_energy_ratio_new__ratio, False),
#     (high_low_freq_energy_ratio_new__energy_high, False),
#     (high_low_freq_energy_ratio_new__energy_low, False),

# ]

texture_metric_list = [
    (local_variance_LV, False),
    (high_frequency_energy_HFE, False),

    (laplacian_variance_mask_aware, False),
    # (glcm_contrast_energy__New__energy, False),
    # (glcm_contrast_energy__New__contrast, False),

    (high_low_freq_energy_ratio_new__ratio, False),
    (high_low_freq_energy_ratio_new__energy_high, False),
    (high_low_freq_energy_ratio_new__energy_low, False),
]

#----------------------------------------------------------------------
# COLOR

color_metric_list = [
    (mean_chroma_new__mean_chroma, False),
    (mean_chroma_new__std_chroma, False),
]

#----------------------------------------------------------------------
# ESPECTRAL

# espectral_metric_list = [
#     (spectral_intraband_variance__mean_variance, False),
#     (spectral_correlation_with_green__mean_correlation, False),
#     (spectral_wasserstein_distance__mean, True),
#     (spectral_angle_distance, True),
#     (spectral_angle_similarity, True),
# ]

espectral_metric_list = [
    # (spectral_intraband_variance__mean_variance, False),
    # (spectral_correlation_with_green__mean_correlation, False),
    # (spectral_wasserstein_distance__mean, True),
    # (spectral_angle_distance, True),
    (spectral_angle_similarity, True),
    (spectral_rgb_residual, False),
]


measure_name = "Measures_02"


best_col = ['suppress_patch_shuffle', 'suppress_gaussian_blur_mask_aware', 'suppress_rgb_color', 'suppress_nir_re_to_green']

for n in [1, 5, -1]:    # n=5

    for metric, comparative  in shape_metric_list:  # metric, comparative = shape_metric_list[6]

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Shape")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")

        result = pd.read_csv(df_dir).drop("Unnamed: 0", axis=1)

        # result.plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)
        # result[best_col].plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)

        plot_dir = os.path.join(result_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)

        full_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_FULL.png")
        save_curves_plot(result, full_dir, title=metric.__name__, xlabel="intensity")

        some_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_SOME.png")
        save_curves_plot(result[best_col], some_dir, title=metric.__name__, xlabel="intensity")

    for metric, comparative  in texture_metric_list: # metric, comparative = texture_metric_list[2]

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Texture")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")

        result = pd.read_csv(df_dir).drop("Unnamed: 0", axis=1)

        # result.plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)
        # result[best_col].plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)

        plot_dir = os.path.join(result_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)

        full_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_FULL.png")
        save_curves_plot(result, full_dir, title=metric.__name__, xlabel="intensity")

        some_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_SOME.png")
        save_curves_plot(result[best_col], some_dir, title=metric.__name__, xlabel="intensity")

    for metric, comparative  in color_metric_list:  # metric, comparative = color_metric_list[0]

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Color")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")

        result = pd.read_csv(df_dir).drop("Unnamed: 0", axis=1)

        # result.plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)
        # result[best_col].plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)

        plot_dir = os.path.join(result_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)

        full_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_FULL.png")
        save_curves_plot(result, full_dir, title=metric.__name__, xlabel="intensity")

        some_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_SOME.png")
        save_curves_plot(result[best_col], some_dir, title=metric.__name__, xlabel="intensity")


    for metric, comparative  in espectral_metric_list:  # metric, comparative = color_metric_list[0]

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Espectral")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")

        result = pd.read_csv(df_dir).drop("Unnamed: 0", axis=1)

        # result.plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)
        # result[best_col].plot(title=f"{metric.__name__}__n_{n}", figsize=(12, 9), grid=True)

        plot_dir = os.path.join(result_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)

        full_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_FULL.png")
        save_curves_plot(result, full_dir, title=metric.__name__, xlabel="intensity")

        some_dir = os.path.join(plot_dir, f"{metric.__name__}__n_{n}_SOME.png")
        save_curves_plot(result[best_col], some_dir, title=metric.__name__, xlabel="intensity")

#======================================================================
#======================================================================
#======================================================================
#======================================================================

from pathlib import Path
import pandas as pd
import numpy as np


def build_delta_g_table(root_dir: str) -> pd.DataFrame:
    """
    Constrói a tabela Delta g_k para as métricas de Shape, Texture,
    Color e Espectral.

    Estrutura esperada
    ------------------
    root_dir/
        Shape/
            Plots/              # ignorada
            df_metric_1.csv
            df_metric_2.csv
            ...
        Texture/
            ...
        Color/
            ...
        Espectral/
            ...

    Cada CSV deve possuir:
        - colunas: transformações;
        - linhas: níveis de intensidade;
        - primeira linha: intensidade 0 (normalizada para 1);
        - última linha: intensidade máxima.

    Para cada métrica g_i e transformação t_j, calcula:

        Delta_(g_i, t_j) =
            | g_i(lambda_max) - g_i(lambda_0) |

    Como lambda_0 já está normalizado para 1:

        Delta_(g_i, t_j) =
            | g_i(lambda_max) - 1 |

    Returns
    -------
    pd.DataFrame
        Linhas:
            (feature, metric)

        Colunas:
            transformações

        Entrada (i, j):
            Delta da métrica i causado pela transformação j
            na intensidade máxima.
    """

    root_dir = Path(root_dir)

    features = [
        "Shape",
        "Texture",
        "Color",
        "Espectral"
    ]

    rows = []

    for feature in features:

        feature_dir = root_dir / feature

        if not feature_dir.exists():
            print(f"[AVISO] Pasta não encontrada: {feature_dir}")
            continue

        # glob("*.csv") considera somente os CSVs diretamente
        # dentro da pasta, portanto Plots/ já é naturalmente ignorada.
        csv_files = sorted(feature_dir.glob("*.csv"))

        for csv_path in csv_files:

            df = pd.read_csv(csv_path)

            # Remove coluna criada automaticamente quando um DataFrame
            # foi salvo com index=True.
            unnamed_cols = [
                col for col in df.columns
                if str(col).startswith("Unnamed:")
            ]

            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)

            if df.empty:
                print(f"[AVISO] CSV vazio: {csv_path}")
                continue

            # Garantir que os valores das transformações sejam numéricos
            df = df.apply(pd.to_numeric, errors="coerce")

            # Primeira linha = imagem original
            original = df.iloc[0]

            # Última linha = maior intensidade
            maximum = df.iloc[-1]

            # Delta g
            delta = np.abs(maximum - original)

            # Nome da medida
            metric_name = csv_path.stem

            # Remove "df_" apenas para deixar a tabela mais limpa
            if metric_name.startswith("df_"):
                metric_name = metric_name[3:]

            row = {
                "feature": feature,
                "metric": metric_name,
                **delta.to_dict()
            }

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    delta_table = pd.DataFrame(rows)

    # Feature + métrica identificam cada linha
    delta_table = delta_table.set_index(
        ["feature", "metric"]
    )

    return delta_table


delta_1 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Trans_Measure/Measures_02/Especies_1"

df_delta_1 = build_delta_g_table(delta_1)

table_dir_1 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Trans_Measure/Tables/df_table_1.csv"
df_delta_1.to_csv(table_dir_1, index=True)


rename_metrics = {
    "edge_ssim_ESSIM__n_1": "ESSIM",
    "gradient_correlation_GC__n_1": "GC",
    "long_range_spatial_organization__n_1": "LRSO",
    "high_frequency_energy_HFE__n_1": "HFE",
    "laplacian_variance_mask_aware__n_1": "Laplacian Variance",
    "mean_chroma_new__mean_chroma__n_1": "Mean Chroma",
    "mean_chroma_new__std_chroma__n_1": "Std Chroma",
    "spectral_rgb_residual__n_1": "Spectral RGB Residual",
}

df_delta_1 = df_delta_1.rename(index=rename_metrics, level="metric")


"""
Interpretação

Cada valor representa quanto aquela métrica mudou, em porcentagem, entre a imagem original e a 
intensidade máxima da transformação.

Como você multiplicou por 100, por exemplo:
- ESSIM x Patch Shuffle = 97.60 → a métrica ESSIM mudou 97,6% em relação ao valor original.
- ESSIM x Gaussian Blur = 14.85 → mudou apenas 14,85%.

Um detalhe importante: como você usa diferença absoluta, valores acima de \(100\%\), como GC = 100.11, 
são possíveis. Isso significa que a alteração foi ligeiramente maior que o próprio valor original.
"""

#======================================================================

delta_5 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Trans_Measure/Measures_02/Especies_5"

df_delta_5 = build_delta_g_table(delta_5)

table_dir_5 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Trans_Measure/Tables/df_table_5.csv"
df_delta_5.to_csv(table_dir_5, index=True)


rename_metrics = {
    "edge_ssim_ESSIM__n_5": "ESSIM",
    "gradient_correlation_GC__n_5": "GC",
    "long_range_spatial_organization__n_5": "LRSO",
    "high_frequency_energy_HFE__n_5": "HFE",
    "laplacian_variance_mask_aware__n_5": "Laplacian Variance",
    "mean_chroma_new__mean_chroma__n_5": "Mean Chroma",
    "mean_chroma_new__std_chroma__n_5": "Std Chroma",
    "spectral_rgb_residual__n_5": "Spectral RGB Residual",
}

df_delta_5 = df_delta_5.rename(index=rename_metrics, level="metric")



#======================================================================


# import os
# import numpy as np
# import pandas as pd

# from joblib import Parallel, delayed
# from tqdm.auto import tqdm


# def eval_feature_metrics(
#         DATA_DIR,
#         TRANS_DICT,
#         metric,
#         n_jobs=0
# ):

#     trans_list = [list(met.keys())[0] for met in TRANS_DICT]
#     trans_names = [list(met.keys())[0].__name__ for met in TRANS_DICT]

#     # ------------------------------------------------------------
#     # Processa uma única imagem
#     # ------------------------------------------------------------
#     def process_file(especie, file_name):

#         file_dir = os.path.join(DATA_DIR, especie, file_name)

#         img_5b = np.load(file_dir).astype("float32")

#         file_values = {}

#         for i, trans in enumerate(trans_list):

#             trans_params = TRANS_DICT[i][trans]

#             ith_values = []

#             for param in trans_params:

#                 img_trans = trans(img_5b, param)

#                 ith_values.append(metric(img_trans))

#             ith_np = np.array([
#                 x / (ith_values[0] + 1e-16)
#                 for x in ith_values
#             ])

#             file_values[trans_names[i]] = ith_np

#         return file_values


#     # ------------------------------------------------------------
#     # Lista todas as imagens
#     # ------------------------------------------------------------
#     especies = sorted(os.listdir(DATA_DIR))

#     tasks = []

#     for especie in especies:

#         especie_dir = os.path.join(DATA_DIR, especie)
#         files = sorted(os.listdir(especie_dir))

#         for file_name in files:
#             tasks.append((especie, file_name))


#     # ------------------------------------------------------------
#     # Execução sequencial
#     # ------------------------------------------------------------
#     if n_jobs == 0:

#         results = []

#         for especie, file_name in tqdm(
#             tasks,
#             total=len(tasks),
#             desc="Processando imagens",
#             unit="img"
#         ):

#             results.append(
#                 process_file(especie, file_name)
#             )


#     # ------------------------------------------------------------
#     # Execução paralela
#     # ------------------------------------------------------------
#     else:

#         result_generator = Parallel(
#             n_jobs=n_jobs,
#             return_as="generator_unordered"
#         )(
#             delayed(process_file)(
#                 especie,
#                 file_name
#             )
#             for especie, file_name in tasks
#         )

#         results = list(
#             tqdm(
#                 result_generator,
#                 total=len(tasks),
#                 desc=f"Processando ({n_jobs} jobs)",
#                 unit="img"
#             )
#         )


#     # ------------------------------------------------------------
#     # Agrupa resultados
#     # ------------------------------------------------------------
#     all_values = {
#         name: []
#         for name in trans_names
#     }

#     for result in results:

#         for name in trans_names:
#             all_values[name].append(result[name])


#     # ------------------------------------------------------------
#     # Média final
#     # ------------------------------------------------------------
#     return pd.DataFrame({
#         name: np.stack(all_values[name]).mean(axis=0)
#         for name in trans_names
#     })


# df_trans_met = eval_feature_metrics(VAL_DIR, TRANS_DICT, metric, 30)

# df_trans_met.plot()
# pd.DataFrame({x: np.r_[all_values[x]].mean(axis=0) for x in all_values.keys()}).plot()


# #======================================================================
# #======================================================================

# # Dante
# feat_met_dir = f"{PC_DIR}/Feature_Metrics/eval_01"

# os.makedirs(feat_met_dir, exist_ok=True)


# metric_list = [
#     long_range_spatial_organization,
#     shape_descriptors__area,
#     shape_descriptors__perimeter,
#     shape_descriptors__hu_1,
#     shape_descriptors__hu_2,

#     laplacian_variance,
#     glcm_contrast_energy__energy,
#     glcm_contrast_energy__energy_per_distance,
#     high_low_frequency_energy_ratio_GPT,
#     high_low_freq_energy_ratio__ratio,
#     high_low_freq_energy_ratio__energy_high,
#     high_low_freq_energy_ratio__energy_low,

#     # mean_lab_chroma_GPT,
#     mean_chroma_zscore,
#     mean_chroma__mean_chroma,
#     mean_chroma__std_chroma,
#     rgb_channel_divergence_GPT__rgb_divergence,
#     rgb_channel_divergence_TEST,
#     hue_distribution_metrics_GPT__hue_entropy,
#     hue_distribution_metrics_GPT__hue_circular_variance,
#     hue_distribution_metrics_GPT__valid_hue_fraction

# ]


# for metric in metric_list:  # metric = metric_list[1]

#     print(f"metric: {metric.__name__}")

#     df_metric_dir = os.path.join(feat_met_dir, f"df_{metric.__name__}.csv")

#     if not os.path.isfile(df_metric_dir):
#         df_trans_met = eval_feature_metrics(VAL_DIR, TRANS_DICT, metric, 20)
#         df_trans_met.to_csv(df_metric_dir, index=False)
#     else:
#         df_trans_met = pd.read_csv(df_metric_dir)

#     # print(df_trans_met)



# for metric in metric_list:

#     # metric = metric_list[4]

#     print(f"metric: {metric.__name__}")

#     df_metric_dir = os.path.join(feat_met_dir, f"df_{metric.__name__}.csv")

#     df_trans_met = pd.read_csv(df_metric_dir)

#     df_trans_met.plot(figsize=(10, 6), title=metric.__name__)


# #======================================================================
# #======================================================================
# #======================================================================
# #======================================================================

# from aux_instance_opt import *

# metric = long_range_spatial_organization


# especies = sorted(os.listdir(VAL_DIR))
# especie = "03_brizantha_Agua_Boa_03"
# files = sorted(os.listdir(os.path.join(VAL_DIR, especie)))
# file_name = files[0]
# file_fir = os.path.join(VAL_DIR, especie, file_name)

# img = np.load(file_fir).astype("float32")

# plot_rgb(img)

# mean_chroma_zscore(img+30)
# laplacian_variance_luminance_GPT(img)

# for metric in metric_list:
#     print(f"metric: {metric.__name__}")
#     print(f"-> {metric(img)} \n")



# g_1 = long_range_spatial_organization
# g_2 = laplacian_variance_luminance_GPT
# g_3 = mean_chroma_zscore




# optimizer = DirectInstanceSuppression(
#     img_5b=img,
#     g_1=g_1,
#     g_2=g_2,
#     g_3=g_3,
#     lambda_g2=10.0,
#     lambda_g3=10.0,
# )

# img_5b_suppress = optimizer.fit(
#     epochs=200,
#     lr=1e-3,
#     perturbation_size=1e-3,
# )

# plot_rgb(img_5b_suppress)
