import os
import rasterio
import matplotlib.pyplot as plt

from auxiliar_segmentation import *
from aux_transformations import *

#======================================================================
# best_band_otsu_green

# HD_Externo 8T
DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Aligned/align_bands_ecc_affine_with_retry"


especies = sorted(os.listdir(DATA_DIR))
especie = "03_brizantha_Agua_Boa_03"
files = sorted(os.listdir(os.path.join(DATA_DIR, especie)))
file_name = files[0]
base_name = file_name[:-6]
file_fir = os.path.join(DATA_DIR, especie, file_name)

img = load_5b_from_dir(os.path.join(DATA_DIR, especie), base_name).astype("float32")

img_seg, mask, best_band = segment_best_band_otsu_green(img)

plot_rgb(img_seg)

mask_5b = np.stack([mask, mask, mask, mask, mask], axis=2)
plot_rgb(mask_5b)

plot_rgb(img)


img_trans = suppress_texture_mask_aware(img)
plot_rgb(img_trans)

img_seg_trans = suppress_texture_mask_aware(img_seg)
plot_rgb(img_seg)
plot_rgb_no_norm(img_seg)
plot_rgb(img_seg_trans)


mask_seg = mask_from_segmented(img_seg)
mask_seg_trans = mask_from_segmented(img_seg_trans)

plot_rgb(mask_seg)
plot_rgb(mask_seg_trans)




plot_rgb(img)

count_connected_components(img)
img = keep_bigger_components(img, 20)

img_trans = suppress_texture(img)
img_trans = suppress_texture_mask_aware(img, sigma=2)
img_trans = suppress_colors(img)
plot_rgb(img_trans)








