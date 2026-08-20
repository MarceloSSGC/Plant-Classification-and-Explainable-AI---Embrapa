import os

print(f'os.getcwd(): {os.getcwd()}')

from aux_color import *

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
#======================================================================
# 

#                          ┌── V ─────────────── intensidade espectral
#                          │
#  [B,G,R,NIR,RE] ── T₅ ───┼── S ─────────────── contraste/saturação espectral
#                          │
#                          ├── H₁ ──┐
#                          ├── H₂ ──┼──────────── direção/assinatura espectral
#                          └── H₃ ──┘

#======================================================================
#======================================================================


img_hsv_5b = convert_5b_to_hsv_nir_re(img_5b)

print(img_hsv_5b.shape)
# (960, 1280, 5)

plot_rgb(img_5b)
plot_rgb(img_hsv_5b)

plot_5bands(img_hsv_5b)
plot_5bands(img_5b)



