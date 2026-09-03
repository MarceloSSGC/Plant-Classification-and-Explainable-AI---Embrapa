import os
import numpy as np

# Auxiliar
from aux_plot import *
from aux_transformations import *
from aux_feature_metrics import *

#======================================================================
#======================================================================
#======================================================================
# Dataset

PC = "DANTE"

if PC == "DANTE":
    PC_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/"
else:
    PC_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/"

#======================================================================
# DATA_DIR

# DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu"
DATA_DIR = f"{PC_DIR}/Datasets/Aligned/align_bands_ecc_affine_with_retry"

especies = sorted(os.listdir(DATA_DIR))

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(DATA_DIR, especie)
files = sorted(set([x[:-6]for x in os.listdir(especie_dir)]))

file_name = files[0]
# file_name = "IMG_0008"

#======================================================================
# Load

img_5b = load_5b_from_dir(especie_dir, file_name)

plot_rgb(img_5b)

#----------------------------------------------------------------------
# # Load from Augmentation

# DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20_AUG/Train_Norm"
# especies = sorted(os.listdir(DATA_DIR))
# especie = "36_Unha_de_gato_Serra_da_Prata_06"
# especie_dir = os.path.join(DATA_DIR, especie)
# files = sorted(os.listdir(especie_dir))

# file_name = files[0]
# file_dir = os.path.join(especie_dir, file_name)

# img_5b = np.load(file_dir)

# print(f"img_5b.shape: {img_5b.shape}")
# print(f"img_5b.mean: {img_5b.mean(axis=(0, 1))}")
# print(f"img_5b.std: {img_5b.std(axis=(0, 1))}")

#======================================================================

"""
M_1 = Long-range spatial organization
M_2 = High-frequency texture
M_3 = Chromaticity
"""

#======================================================================
#======================================================================
# Shape

img_5b_tr = suppress_shape(img_5b)

plot_rgb(img_5b_tr)

long_range_spatial_organization(img_5b)
long_range_spatial_organization(img_5b_tr)


img_5b_tr = suppress_shape(img_5b, patch_size = 256)
plot_rgb(img_5b_tr)
long_range_spatial_organization(img_5b_tr)


values_list = []
for ps in [0, 1028, 512, 256, 128, 64, 32]:  # ps = 64

    print("\n"+"="*40+f"\n patch_size: {ps}")
    if ps == 0:
        img_5b_tr = img_5b.copy()
    else:
        img_5b_tr = suppress_shape(img_5b, patch_size = ps)
    # plot_rgb(img_5b_tr)
    lrso = long_range_spatial_organization(img_5b_tr)

    print(f"ps: {ps} - long_range_spatial_organization: {lrso}")

    values_list.append((ps, lrso))

plotar_barras(values_list)



