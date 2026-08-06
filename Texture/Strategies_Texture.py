import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Texture")
from auxiliar_texture import *

#======================================================================
#======================================================================
# Dataset

WHC_DATA = "PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5"
DATA_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{WHC_DATA}"

#======================================================================
#======================================================================
# Plot Band

species = sorted(os.listdir(DATA_DIR))

specie = "01_malva_branca_Agua_Boa_01"

specie_dir = DATA_DIR + f"/{specie}"

file_names = sorted(os.listdir(specie_dir))

file_name = 'IMG_0017_2.tif'


# Plotagem
# float: valores entre 0 e 1;
# uint8: valores entre 0 e 255.

plot_tif_dir(specie_dir, file_name)


#======================================================================
# Load band

file_path = os.path.join(specie_dir, file_name)

with rasterio.open(file_path) as src:
    band = src.read(1)  # Lê banda do arquivo

print(f"max pixel: {band.max()}")
print(f"min pixel: {band.min()}")

plot_tif(band)

#======================================================================
# 1 Band

# Variância Local

var_band = local_var_band(band, window_size=10)
plot_tif(var_band)

plot_both(band, var_band)

# segmentada
mask = segment_band_otsu(band)
band_masked = band.copy()
band_masked[mask == 0] = 0
plot_tif(band_masked)

var_band_masked = local_var_band(band_masked, window_size=10)
plot_tif(var_band_masked)

plot_both(band_masked, var_band_masked)


#----------------------------------------------------------------------
# Entropia Local

ent_band = local_entropy_band(band, window_size=5)

plot_tif(ent_band)
plot_both(band, ent_band)


# segmentada
ent_band_masked = local_entropy_band(band_masked, window_size=5)
plot_tif(ent_band_masked)
plot_both(band_masked, ent_band_masked)

#----------------------------------------------------------------------
# Sobel

sobel_band = sobel_band_filter(band)
plot_tif(sobel_band)
plot_both(band, sobel_band)

sobel_band_masked = sobel_band_filter(band_masked)
plot_tif(sobel_band_masked)
plot_both(band_masked, sobel_band_masked)

#----------------------------------------------------------------------

# # testar
# from skimage.morphology import footprint_rectangle

# entropy_band = entropy(band_norm, footprint_rectangle((window_size, window_size)))

#======================================================================
