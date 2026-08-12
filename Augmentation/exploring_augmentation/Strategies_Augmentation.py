import os
import numpy as np
import pandas as pd

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Augmentation/exploring_augmentation/")

from auxiliar_augmentation import *

#======================================================================
#======================================================================
# Dataset & Directories

# DATASET_NAME = "Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu"
DATASET_NAME = "PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5"
DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/"

DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME)

#======================================================================
# Experiment Name and Directory

experiment_name = f"augmentation_TEST"
experiment_type = "data_augmentation_TESTING"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/{experiment_type}/{experiment_name}"

if not os.path.isdir(DIR_EXP):
    os.makedirs(DIR_EXP)

#======================================================================

especies = sorted(os.listdir(DATA_DIR))

dist_especies = dict()

for especie in especies: # especie = especies[0]

    especie_dir = os.path.join(DATA_DIR, especie)

    n_smaple = len(set([x[:-6] for x in os.listdir(especie_dir)]))

    dist_especies.update({especie: n_smaple})

#======================================================================

# Rotação aleatória: 0–360°.
# Flip horizontal: probabilidade de ~50%.
# Flip vertical: probabilidade de ~50%.
# Translação pequena: até ~5–10% da largura/altura.
# Scale/Zoom leve: algo como 0,9–1,1×.

# | Nº de imagens originais | Augmentation sugerido | Fator final aprox. | Exemplo     |
# | ----------------------: | --------------------: | -----------------: | ----------- |
# |                **< 15** |  +3 a +4 por original |               4–5× | 7 → 28–35   |
# |               **15–24** |  +2 a +3 por original |               3–4× | 20 → 60–80  |
# |               **25–39** |  +1 a +2 por original |               2–3× | 30 → 60–90  |
# |               **40–59** |       +1 por original |                 2× | 45 → 90     |
# |               **60–89** |  +0 a +1 por original |               1–2× | 63 → 63–126 |
# |                **≥ 90** |                nenhum |                 1× | 112 → 112   |


# Random Crop — recorte de regiões da imagem.
# Shear — inclinação/deformação geométrica.
# Affine Transform — combinação de rotação, escala, translação e shear.
# Perspective Transform — alteração de perspectiva.
# Elastic Deformation — deformações locais da imagem.
# Gaussian Noise — adição de ruído.
# Gaussian Blur — suavização/desfoque.
# Brightness/Gain — alteração da intensidade.
# Contrast — alteração do contraste.
# Gamma correction — transformação não linear das intensidades.
# Cutout / Random Erasing — remoção de pequenas regiões.
# MixUp — combinação de duas imagens e seus rótulos.
# CutMix — insere uma região de uma imagem em outra.
# Augmentation espectral — pequenas perturbações nas bandas espectrais.


#======================================================================
#======================================================================
# Choose IMAGE

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(DATA_DIR, especie)

file_names = sorted(set([x[:-6] for x in os.listdir(especie_dir)]))

file_name = "IMG_0069"

img_5b = load_5b_from_dir(especie_dir, file_name)
print(f'img_5b: {img_5b.shape}')

plot_rgb(img_5b)

#------------------------------------------------------------------------
# Rotation: 0–360°.

# Radom
img_5b_rot = random_rotation(img_5b)
plot_rgb(img_5b_rot)


img_5b_rot = random_rotation(img_5b, 90)
plot_rgb(img_5b_rot)
plot_segmentation(img_5b, img_5b_rot)

#------------------------------------------------------------------------
# Translação pequena: até ~5–10%

img_5b_trans = random_translation(img_5b, max_fraction=0.1)
plot_rgb(img_5b_trans)

plot_segmentation(img_5b, img_5b_trans)


img_5b_trans = deterministic_translation(img_5b, max_fraction=0.05)
plot_rgb(img_5b_trans)
plot_segmentation(img_5b, img_5b_trans)

img_5b_trans = deterministic_translation_manual(img_5b)
plot_rgb(img_5b)
plot_rgb(img_5b_trans)
plot_segmentation(img_5b, img_5b_trans)

img_5b_trans = random_translate(img_5b)
plot_rgb(img_5b)
plot_rgb(img_5b_trans)
plot_segmentation(img_5b, img_5b_trans)

#------------------------------------------------------------------------
# Scale/Zoom

img_5b_zoom = scale_image(img_5b, scale=0.9)
plot_rgb(img_5b)
plot_rgb(img_5b_zoom)
plot_segmentation(img_5b, img_5b_zoom)


#------------------------------------------------------------------------
# Gaussian Noise — adição de ruído

img_5b_gauss = gaussian_noise(img_5b, noise_fraction=0.1)
plot_rgb(img_5b)
plot_rgb(img_5b_gauss)
plot_segmentation(img_5b, img_5b_gauss)

#------------------------------------------------------------------------
# Cutout / Random Erasing — remoção de pequenas regiões

img_5b_cut = random_cutout(img_5b, size_fraction=0.1)
plot_rgb(img_5b)
plot_rgb(img_5b_cut)
plot_segmentation(img_5b, img_5b_cut)


#------------------------------------------------------------------------
#------------------------------------------------------------------------
# mixup_same_class

especie = "20_Guanxuma_Paludo_02"

especie_dir = os.path.join(DATA_DIR, especie)

file_names = sorted(set([x[:-6] for x in os.listdir(especie_dir)]))


file_names = sorted(set([x[:-6] for x in os.listdir(especie_dir)]))

file_name_1 = file_names[2]
file_name_2 = file_names[1]

img_5b_1 = load_5b_from_dir(especie_dir, file_name_1)
img_5b_2 = load_5b_from_dir(especie_dir, file_name_2)

img_mix = mixup_same_class(img_5b_1, img_5b_2, lam=0.5) # lam combinção 
plot_rgb(img_5b_1)
plot_rgb(img_5b_2)

plot_rgb(img_mix)

plot_segmentation(img_5b_1, img_mix)
plot_segmentation(img_5b_2, img_mix)

















#------------------------------------------------------------------------
# Combo
img_5b_x = scale_image(random_rotation(img_5b, 30), 1.2)
plot_rgb(img_5b_x)
plot_segmentation(img_5b, img_5b_x)


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


# | Nº de imagens originais | Augmentation sugerido | Fator final aprox. | Exemplo     |
# | ----------------------: | --------------------: | -----------------: | ----------- |
# |                **< 15** |  +3 a +4 por original |               4–5× | 7 → 28–35   |
# |               **15–24** |  +2 a +3 por original |               3–4× | 20 → 60–80  |
# |               **25–39** |  +1 a +2 por original |               2–3× | 30 → 60–90  |
# |               **40–59** |       +1 por original |                 2× | 45 → 90     |
# |               **60–89** |  +0 a +1 por original |               1–2× | 63 → 63–126 |
# |                **≥ 90** |                nenhum |                 1× | 112 → 112   |


pd.Series(dist_especies).plot(kind="bar")

dist_especies_new = {x: int(dist_especies[x] * 0.7) for x in dist_especies}
pd.Series(dist_especies_new).plot(kind="bar")


dist_especies_aug = dict()

for especie in especies:

    n_smaple = dist_especies_new[especie]
    print(f"especie: {especie} \t {n_smaple}")

    if n_smaple < 15:
        factor = n_smaple * 4
    elif n_smaple < 24:
        factor = n_smaple * 3
    elif n_smaple < 40:
        factor = n_smaple * 2
    else:
        factor = n_smaple
    print(f"resample: \033[96;94m{factor}\033[0m\n")

    dist_especies_aug[especie] = factor

pd.Series(dist_especies_aug).plot(kind="bar")













