import os
import rasterio
import matplotlib.pyplot as plt

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Segmentation")
from auxiliar import *

#======================================================================
#======================================================================
# Dataset

# DATA_DIR = "/home/marcelo/Documents/Datasets/PlantaDaninha_BoaVista"
DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine"

#======================================================================
#======================================================================
# Plot Band

species = sorted(os.listdir(DATA_DIR))

specie = "36_Unha_de_gato_Serra_da_Prata_06"
specie_dir = DATA_DIR + f"/{specie}"

file_names = sorted(os.listdir(specie_dir))

file_name = 'IMG_0069_2.tif'


# Plotagem
# float: valores entre 0 e 1;
# uint8: valores entre 0 e 255.

plot_tif_dir(specie_dir, file_name)


#======================================================================
#======================================================================
# 1 Band

# Threshold Manual

file_path = os.path.join(specie_dir, file_name)

with rasterio.open(file_path) as src:
    band = src.read(1)  # Lê a primeira (ou única) banda do arquivo

print(f"max pixel: {band.max()}")
print(f"min pixel: {band.min()}")

"""
0        -> preto
65535    -> branco
32768    -> cinza médio
"""

# Pixels acima do threshold
threshold = 10_000

band_threshold = band.copy()
band_threshold[band_threshold < threshold] = 0

plot_tif(band_threshold)
plot_tif(band>19552)
plot_tif(band)

plot_both(band, band_threshold)


plot_hist_band(band, bins=50)
plot_hist_band(band_threshold, bins=50)

#======================================================================
# Otsu

from skimage.filters import threshold_otsu


# Calcula automaticamente o threshold
th = threshold_otsu(band)

print(f"Threshold de Otsu = {th}")

# Cria máscara binária
mask = band > th
mask = band < th

band_masked = band.copy()
band_masked[mask == 0] = 0
plot_both(band, band_masked)


#======================================================================
# Recomendação:
# ✔ CLAHE → Gaussian → Otsu → Abertura/Fechamento morfológico

mask = segment_band_otsu(band)
print(f"mask - max pixel: {mask.max()}")
print(f"mask - min pixel: {mask.min()}")

band_masked = band.copy()
band_masked[mask == 0] = 0
plot_both(band, band_masked)

mask = segment_band_otsu(band, invert=True)

band_masked = band.copy()
band_masked[mask == 0] = 0
plot_both(band, band_masked)


#======================================================================
#======================================================================
#======================================================================
# 3 Bandas

# Plot Band

species = sorted(os.listdir(DATA_DIR))

specie = "32_Jurubebinha_Serra_da_Prata_03"
specie_dir = DATA_DIR + f"/{specie}"

file_names = [x[:-6] for x in sorted(os.listdir(specie_dir))]

file_name = file_names[0]

plot_rgb_tif_dir(specie_dir, file_name, (3, 2, 1))

plot_rgb_tif_dir(specie_dir, file_name, (4, 5, 1))


rgb_img = load_rgb_from_dir(specie_dir, file_name, (3, 2, 1))
print(f"rgb_img.shape : {rgb_img.shape}")

# min max - Band
print(f"rgb_img min max: {rgb_img[:, :, 0].min(), rgb_img[:, :, 0].max()}")
print(f"rgb_img min max: {rgb_img[:, :, 1].min(), rgb_img[:, :, 1].max()}")
print(f"rgb_img min  max {rgb_img[:, :, 2].min(), rgb_img[:, :, 2].max()}")

plot_rgb(rgb_img)


#======================================================================
# ExG (Excess Green) + Otsu 

rgb_img_masked = exg_otsu_segmentation(rgb_img)

print(f"rgb_img_masked.shape : {rgb_img_masked.shape}")

print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 0].min(), rgb_img_masked[:, :, 0].max()}")
print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 1].min(), rgb_img_masked[:, :, 1].max()}")
print(f"rgb_img_masked min  max {rgb_img_masked[:, :, 2].min(), rgb_img_masked[:, :, 2].max()}")

plot_rgb(rgb_img_masked)

plot_segmentation(rgb_img, rgb_img_masked)

#======================================================================
# SLIC + Merge

rgb_img_masked = slic_merge_segmentation(rgb_img)

print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 0].min(), rgb_img_masked[:, :, 0].max()}")
print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 1].min(), rgb_img_masked[:, :, 1].max()}")
print(f"rgb_img_masked min  max {rgb_img_masked[:, :, 2].min(), rgb_img_masked[:, :, 2].max()}")

plot_rgb(rgb_img_masked)

plot_segmentation(rgb_img, rgb_img_masked)

#======================================================================
#======================================================================
#======================================================================
# Contour

rgb_img_contour = contour_exg_otsu_segmentation(
    rgb_img_masked,
    contour_thickness=3,
    color="red",
)

plot_rgb(rgb_img_contour)

plot_segmentation(rgb_img, rgb_img_contour)
plot_segmentation(rgb_img, mask)


#======================================================================
#======================================================================
#======================================================================
#======================================================================

