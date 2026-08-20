import os

print(f'os.getcwd(): {os.getcwd()}')

from aux_shape import *

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
# Canny / Edge map

edges = apply_canny_1b(img_5b[:, :, 3])
plot_band(edges, title="Canny - NIR")

all_edges = apply_canny_5b(img_5b)
plot_5bands(all_edges)

#======================================================================
# Sobel / Gradient map

sobel = apply_sobel_1b(img_5b[:, :, 3])
plot_band(sobel, title="Sobel - NIR")

all_sobel = apply_sobel_5b(img_5b)
plot_5bands(all_sobel)

#======================================================================
# Silhouette / Segmentation mask

silhouette = apply_silhouette_1b(img_5b[:, :, 3])
plot_band(silhouette, title="silhouette - NIR")

all_silhouette = apply_silhouette_5b(img_5b)
plot_5bands(all_silhouette)

#======================================================================

plot_5bands(img_5b)
plot_5bands(all_edges)
plot_5bands(all_sobel)
plot_5bands(all_silhouette)


#======================================================================
#======================================================================
#======================================================================































