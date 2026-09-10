import os
import numpy as np
import pandas as pd

# Auxiliar
from aux_plot import *
from aux_transformations import *
from aux_feature_metrics import *

#======================================================================
#======================================================================
# Directories

DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Val_Norm"
LOCAL_DATA = "/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa/Feature_Metrics/test_01_val_data/local_data"

especies = sorted(os.listdir(DATA_DIR))

#======================================================================

# ["color", "texture", "shape"]

#======================================================================
# Metric Color

metric = mean_chroma__mean_chroma

color_array, texture_array, shape_array = features_from_transformations(DATA_DIR, metric)

color_array_norm = color_array / color_array[:, 0][:, None]
texture_array_norm = texture_array / texture_array[:, 0][:, None]
shape_array_norm = shape_array / shape_array[:, 0][:, None]

color_mean = color_array_norm.mean(axis=0)
texture_mean = texture_array_norm.mean(axis=0)
shape_mean = shape_array_norm.mean(axis=0)

plotar_barras([(i, x) for i, x in enumerate(color_mean)])
plotar_barras([(i, x) for i, x in enumerate(texture_mean)])
plotar_barras([(i, x) for i, x in enumerate(shape_mean)])

plotar_multiplas_linhas([color_mean, texture_mean, shape_mean], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "entropy")

#======================================================================
# Metric Texture

metric = laplacian_variance_luminance_GPT

color_array, texture_array, shape_array = features_from_transformations(DATA_DIR, metric)

color_array_norm = color_array / color_array[:, 0][:, None]
texture_array_norm = texture_array / texture_array[:, 0][:, None]
shape_array_norm = shape_array / shape_array[:, 0][:, None]

color_mean = color_array_norm.mean(axis=0)
texture_mean = texture_array_norm.mean(axis=0)
shape_mean = shape_array_norm.mean(axis=0)

plotar_barras([(i, x) for i, x in enumerate(color_mean)])
plotar_barras([(i, x) for i, x in enumerate(texture_mean)])
plotar_barras([(i, x) for i, x in enumerate(shape_mean)])


plotar_multiplas_linhas([color_mean, texture_mean, shape_mean], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "entropy")

#======================================================================
# Metric Shape

metric = coarse_ssim_gpt

color_array, texture_array, shape_array = features_from_transformations(DATA_DIR, metric, comparative=True)

color_array_norm = color_array / color_array[:, 0][:, None]
texture_array_norm = texture_array / texture_array[:, 0][:, None]
shape_array_norm = shape_array / shape_array[:, 0][:, None]

color_mean = color_array_norm.mean(axis=0)
texture_mean = texture_array_norm.mean(axis=0)
shape_mean = shape_array_norm.mean(axis=0)

plotar_barras([(i, x) for i, x in enumerate(color_mean)])
plotar_barras([(i, x) for i, x in enumerate(texture_mean)])
plotar_barras([(i, x) for i, x in enumerate(shape_mean)])


plotar_multiplas_linhas([color_mean, texture_mean, shape_mean], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "entropy")













