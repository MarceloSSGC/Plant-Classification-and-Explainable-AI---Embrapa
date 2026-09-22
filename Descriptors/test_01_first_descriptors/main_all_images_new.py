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


# TRANS_DICT = {
#     suppress_patch_shuffle: [0, 2, 4, 6, 8, 12, 16],
#     # suppress_patch_rotation: [0, 2, 4, 6, 8, 12, 16],
#     suppress_gaussian_blur_mask_aware: [0, 1, 2, 3, 4, 5, 6],
#     # suppress_bilateral_filter_mask_aware: [0, 1, 2, 3, 4, 5, 6],
#     # suppress_colors: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_rgb_color: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     # suppress_channel_shuffle: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     # suppress_nir_re_to_mean: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
#     suppress_nir_re_to_green: [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1],
# }

# metric = local_variance_LV

#======================================================================


def eval_feature_metrics(
        DATA_DIR,
        TRANS_DICT,
        metric,
        comparative=False,
        n_especie=1
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
                        met_value = metric(img_5b, jth_img_trans)
                    else:
                        met_value = metric(jth_img_trans)

                    if not (isinstance(met_value, int) or isinstance(met_value, float) or isinstance(met_value, np.ndarray)):
                        raise ValueError(f"met_value: {met_value}")
                    
                    ith_values.append(met_value)

                ith_np = np.array([x/(ith_values[0]+1e-16) for x in ith_values])

                all_values[trans_names[i]].append(ith_np)

    result = pd.DataFrame({x: np.r_[all_values[x]].mean(axis=0) for x in all_values.keys()})

    # result.plot()

    return result




#======================================================================

# best_col = ['suppress_patch_shuffle', 'suppress_gaussian_blur_mask_aware', 'suppress_rgb_color', 'suppress_nir_re_to_green']

# metric = spectral_rgb_residual

# result = eval_feature_metrics(
#                     DATA_DIR,
#                     TRANS_DICT,
#                     metric,
#                     False,
#                     1)


# result
# result.plot(title=metric.__name__, figsize=(12, 9))
# result[best_col].plot(title=metric.__name__, figsize=(12, 9))


#======================================================================
# Dir 

results_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Trans_Measure"

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
shape_metric_list = [
    (edge_ssim_ESSIM, True),
    (gradient_correlation_GC, True),
    (long_range_spatial_organization, False),
    (shape_descriptors__area, False),
    (shape_descriptors__perimeter, False),
    (shape_descriptors__compactness, False),
    (shape_descriptors__solidity, False),
    (shape_descriptors__hu_1, False),
    (shape_descriptors__hu_2, False),
    (coarse_ssim_mask_aware, True),
]

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
texture_metric_list = [
    (local_variance_LV, False),
    (high_frequency_energy_HFE, False),

    (laplacian_variance_mask_aware, False),
    (glcm_contrast_energy__New__energy, False),
    (glcm_contrast_energy__New__contrast, False),

    (high_low_freq_energy_ratio_new__ratio, False),
    (high_low_freq_energy_ratio_new__energy_high, False),
    (high_low_freq_energy_ratio_new__energy_low, False),
]

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

espectral_metric_list = [
    (spectral_intraband_variance__mean_variance, False),
    (spectral_correlation_with_green__mean_correlation, False),
    (spectral_wasserstein_distance__mean, True),
    (spectral_angle_distance, True),
    (spectral_angle_similarity, True),
    (spectral_rgb_residual, True),
]

espectral_metric_list = [
    # (spectral_intraband_variance__mean_variance, False),
    # (spectral_correlation_with_green__mean_correlation, False),
    # (spectral_wasserstein_distance__mean, True),
    # (spectral_angle_distance, True),
    (spectral_angle_similarity, True),
    (spectral_rgb_residual, False),
]


#----------------------------------------------------------------------

measure_name = "Measures_02"

for n in [1, 5, -1]:

    for metric, comparative  in shape_metric_list:

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Shape")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")
        if not os.path.isfile(df_dir):
            df_result = eval_feature_metrics(
                                DATA_DIR,
                                TRANS_DICT,
                                metric,
                                comparative,
                                n)
            df_result.to_csv(df_dir)

    for metric, comparative  in texture_metric_list:

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Texture")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")
        if not os.path.isfile(df_dir):
            df_result = eval_feature_metrics(
                                DATA_DIR,
                                TRANS_DICT,
                                metric,
                                comparative,
                                n)
            df_result.to_csv(df_dir)

    for metric, comparative  in color_metric_list:

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Color")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")
        if not os.path.isfile(df_dir):
            df_result = eval_feature_metrics(
                                DATA_DIR,
                                TRANS_DICT,
                                metric,
                                comparative,
                                n)
            df_result.to_csv(df_dir)


    for metric, comparative  in espectral_metric_list:

        print(f"\n\t\033[100;40m -- \033[100;40m{metric.__name__} - {n}  \033[100;0m")
        result_dir = os.path.join(results_dir, measure_name, f"Especies_{n}/Espectral")
        os.makedirs(result_dir, exist_ok=True)
        df_dir = os.path.join(result_dir, f"df_{metric.__name__}__n_{n}.csv")
        if not os.path.isfile(df_dir):
            df_result = eval_feature_metrics(
                                DATA_DIR,
                                TRANS_DICT,
                                metric,
                                comparative,
                                n)
            df_result.to_csv(df_dir)

# #======================================================================



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
