import os

print(f'os.getcwd(): {os.getcwd()}')

from aux_index import *

#======================================================================
#======================================================================
#======================================================================
# Dataset

DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu"

especies = sorted(os.listdir(DATA_DIR))

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(DATA_DIR, especie)
files = sorted(set([x[:-6]for x in os.listdir(especie_dir)]))


file_name = files[0]

img_5b = load_5b_from_dir(especie_dir, file_name)

plot_rgb(img_5b)

plot_band(img_5b[:, :, 2])

plot_band(img_5b[:, :, 3], title="NIR")

plot_5bands(img_5b)

#======================================================================
# NDVI

NDVI = apply_ndvi(img_5b)

plot_band(NDVI, title="NDVI")

print(NDVI.shape)
# (960, 1280)
#======================================================================
# NDRE

NDRE = apply_ndre(img_5b)

plot_band(NDRE, title="NDRE")

print(NDRE.shape)
# (960, 1280)

#======================================================================
# GNDVI

GNDVI = apply_gndvi(img_5b)

plot_band(GNDVI, title="GNDVI")

print(GNDVI.shape)
# (960, 1280)


#======================================================================
# EVI

EVI = apply_evi(img_5b)

plot_band(EVI, title="EVI")

print(EVI.shape)
# (960, 1280)

#======================================================================

plot_rgb(img_5b)
plot_band(NDVI, title="NDVI")
plot_band(NDRE, title="NDRE")
plot_band(GNDVI, title="GNDVI")
plot_band(EVI, title="EVI")



#======================================================================

#======================================================================



