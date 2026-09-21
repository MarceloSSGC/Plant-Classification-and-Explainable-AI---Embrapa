import os
import numpy as np
import pandas as pd

# Auxiliar
from aux_plot import *
from aux_transformations import *
from aux_feature_metrics import *

#======================================================================
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

especies = sorted(os.listdir(DATA_DIR))

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(DATA_DIR, especie)
files = sorted(os.listdir(especie_dir))

file_name = files[-1]
file_dir = os.path.join(especie_dir, file_name)

#======================================================================
# Load

img_5b = np.load(file_dir)

plot_rgb(img_5b)

#======================================================================
# Transformations
#======================================================================
# SHAPE

# suppress_patch_shuffle

img_5b_trans = suppress_patch_shuffle(img_5b, 16)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))

#----------------------------------------------------------------------
# suppress_patch_rotation

img_5b_trans = suppress_patch_rotation(img_5b, 8)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))

#======================================================================
#======================================================================
# Texture

# Gaussian_blur_mask_aware

img_5b_trans = suppress_gaussian_blur_mask_aware(img_5b, 10)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))


#----------------------------------------------------------------------
# Bilateral_filter

img_5b_trans = suppress_bilateral_filter_mask_aware(img_5b, 0)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))

plot_rgb(img_5b)

#======================================================================
#======================================================================
# Color

# Gaussian_blur_mask_aware

img_5b_trans = suppress_colors(img_5b, 1)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))


#----------------------------------------------------------------------
# Channel Shuffle

img_5b_trans = suppress_channel_shuffle(img_5b, 1)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))

plot_rgb(img_5b)

#======================================================================
#======================================================================
# Espectral

# suppress_nir_re

img_5b_trans = suppress_nir_re_to_mean(img_5b, 0.5)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))
plot_rgb(img_5b, bands_ch=(4, 3, 2))


#----------------------------------------------------------------------
# suppress_nir_re_to_green

img_5b_trans = suppress_nir_re_to_green(img_5b, 1)

plot_rgb(img_5b_trans)
plot_rgb(img_5b_trans, bands_ch=(4, 3, 2))
plot_rgb(img_5b, bands_ch=(4, 3, 2))

plot_rgb(img_5b)

#======================================================================
#======================================================================
#======================================================================
# Measures

# SHAPE

edge_ssim_ESSIM(img_5b, img_5b)
edge_ssim_ESSIM(img_5b, img_5b_trans)

gradient_correlation_GC(
    img_5b,
    img_5b_trans,
    sobel_ksize=11,
    channel="luminance"
)


long_range_spatial_organization(img_5b)

shape_descriptors__area(img_5b)
shape_descriptors__perimeter(img_5b)
shape_descriptors__compactness(img_5b)
shape_descriptors__solidity(img_5b)
shape_descriptors__hu_1(img_5b)
shape_descriptors__hu_2(img_5b)

coarse_ssim_mask_aware(img_5b, img_5b_trans)

#----------------------------------------------------------------------
# Texture

local_variance_LV(img_5b)
high_frequency_energy_HFE(img_5b)


laplacian_variance_mask_aware(img_5b)
glcm_contrast_energy__New__energy(img_5b)
glcm_contrast_energy__New__contrast(img_5b)

high_low_freq_energy_ratio_new__ratio(img_5b)
high_low_freq_energy_ratio_new__energy_high(img_5b)
high_low_freq_energy_ratio_new__energy_low(img_5b)

#----------------------------------------------------------------------
# Color

mean_chroma_new__mean_chroma(img_5b)
mean_chroma_new__std_chroma(img_5b)

#----------------------------------------------------------------------
# Espectral

spectral_intraband_variance__mean_variance(img_5b)
spectral_correlation_with_green__mean_correlation(img_5b)

spectral_wasserstein_distance(img_5b, img_5b_trans)


spectral_wasserstein_distance(img_5b, suppress_nir_re_to_green(img_5b, 1))
spectral_angle_distance(img_5b, suppress_nir_re_to_green(img_5b, 1))













#======================================================================

"""
M_1 = Long-range spatial organization
M_2 = High-frequency texture
M_3 = Chromaticity

"""

#======================================================================
#======================================================================
# Shape

img_5b_tr = suppress_shape(img_5b, patch_size=1024)

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
    # lrso = long_range_spatial_organization(img_5b_tr)
    lrso = long_range_spatial_autocorr(img_5b_tr)

    print(f"ps: {ps} - long_range_spatial_organization: {lrso}")

    values_list.append((ps, lrso))

plotar_barras(values_list)
plotar_linhas(values_list, norm=True)


#======================================================================
# Texture

sh_info = shape_descriptors(img_5b)
sh_info = shape_descriptors(img_5b_tr)

values_list = []
for ps in [0, 1028, 512, 256, 128, 64, 32]:  # ps = 64

    print("\n"+"="*40+f"\n patch_size: {ps}")
    if ps == 0:
        img_5b_tr = img_5b.copy()
    else:
        img_5b_tr = suppress_shape(img_5b, patch_size = ps)
    # plot_rgb(img_5b_tr)
    # lrso = long_range_spatial_organization(img_5b_tr)
    sh_info = shape_descriptors(img_5b_tr)

    print(f"ps: {ps} - shape_descriptors: {sh_info}")

    values_list.append((ps, sh_info))

# ['area', 'perimeter', 'compactness', 'solidity', 'hu_1', 'hu_2',
#  'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7']


y = "area"
for y in list(values_list[0][1].keys()):
    print(f" y: {y}")
    plotar_barras([(x[0], x[1][y]) for x in values_list])



sh_info = shape_descriptors_cl(img_5b)
sh_info = shape_descriptors_cl(img_5b_tr)

plot_rgb(0.5*(img_5b_tr + img_5b))

#----------------------------------------------------------------------

#======================================================================
# Color

plot_rgb(suppress_rgb_color(img_5b, descolorir=1))



#======================================================================
#======================================================================
#======================================================================

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = long_range_spatial_autocorr

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]
# lrso_shape = [abs(metric(x)) for x in imgs_shape]

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__)


lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]

plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)))
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)))
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)))


#----------------------------------------------------------------------
# shape_descriptors
# shape_descriptors_cl
# coarse_ssim
# coarse_ssim_gpt

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = shape_descriptors_cl_TEST

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title="shape_descriptors_cl_area")



plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"])


lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)))
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)))
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)))


#----------------------------------------------------------------------
# Shape

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = coarse_ssim_gpt

lrso_color = [metric(x, y) for x, y in zip(imgs_color, [img_5b for i in range(7)])]
lrso_texture = [metric(x, y) for x, y in zip(imgs_texture, [img_5b for i in range(7)])]
lrso_shape = [metric(x, y) for x, y in zip(imgs_shape, [img_5b for i in range(7)])]


plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__)


lrso_color_norm = [x/max(lrso_color) for x in lrso_color]
lrso_texture_norm = [x/max(lrso_texture) for x in lrso_texture]
lrso_shape_norm = [x/max(lrso_shape) for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)))
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)))
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)))

#======================================================================
#======================================================================
# **M₂ (Gaussian Blur):**
# 1. Variância do Laplaciano (luminância)

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = laplacian_variance_luminance_GPT

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]
# lrso_shape = [abs(metric(x)) for x in imgs_shape]

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__)




lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")

#----------------------------------------------------------------------
# **M₂ (Gaussian Blur):**
# 2. GLCM contrast/energy, offset pequeno
 
imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = glcm_contrast_energy_TEST

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]


plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title="glcm_contrast_energy__contrast_per_distance")




lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")

#----------------------------------------------------------------------
# **M₂ (Gaussian Blur):**
# 2. GLCM contrast/energy, offset pequeno
 
imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = glcm_contrast_energy_test

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]
# lrso_shape = [abs(metric(x)) for x in imgs_shape]

lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")

#----------------------------------------------------------------------
# **M₂ (Gaussian Blur):**
# 3. Razão energia alta/baixa frequência (FFT/wavelet)

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = high_low_freq_energy_ratio_TEST

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "energy_low")



lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")

#======================================================================
#======================================================================
#======================================================================
# **M₃ (Grayscale/Dessaturação):**
# 1. Chroma média (Lab ou HSV-S)

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = mean_chroma_test

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]

# plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
#                         title=metric.__name__)

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "std_chroma")


lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")


#======================================================================
# 2. Divergência entre canais RGB (|R-G|, |G-B|, |R-B|)

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = rgb_channel_divergence_TEST

lrso_color = [metric(x) for x in imgs_color]
lrso_texture = [metric(x) for x in imgs_texture]
lrso_shape = [metric(x) for x in imgs_shape]

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "max_divergence")


lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")

#======================================================================
# 3. Entropia/variância circular do histograma de Hue

imgs_color = step_suppress_rgb_color(img_5b)
imgs_texture = step_suppress_texture(img_5b)
imgs_shape = step_suppress_shape(img_5b)

metric = hue_histogram_stats_TEST

lrso_color = [metric(x)+1e-16 for x in imgs_color]
lrso_texture = [metric(x)+1e-16 for x in imgs_texture]
lrso_shape = [metric(x)+1e-16 for x in imgs_shape]

plotar_multiplas_linhas([lrso_color, lrso_texture, lrso_shape], ["color", "texture", "shape"],
                        title=metric.__name__ + "_" + "entropy")


lrso_color_norm = [x/lrso_color[0] for x in lrso_color]
lrso_texture_norm = [x/lrso_texture[0] for x in lrso_texture]
lrso_shape_norm = [x/lrso_shape[0] for x in lrso_shape]


plotar_barras(list(zip(step_suppress_rgb_color(return_list=True), lrso_color_norm)), title="Color Trans")
plotar_barras(list(zip(step_suppress_texture(return_list=True), lrso_texture_norm)), title="Texture Trans")
plotar_barras(list(zip(step_suppress_shape(return_list=True), lrso_shape_norm)), title="Shape Trans")


#======================================================================







#======================================================================
#======================================================================
#======================================================================


#======================================================================




