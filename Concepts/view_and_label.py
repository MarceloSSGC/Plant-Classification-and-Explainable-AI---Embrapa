import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista" 
os.chdir(f"{base_dir}/Concepts")

from auxiliar import *


#======================================================================
# Dataset

DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine"

especies = sorted(os.listdir(DATA_DIR))

#======================================================================

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(DATA_DIR, especie)

files = sorted([x[:-6] for x in os.listdir(especie_dir)])
files = sorted(set(files))

print(f'n_files: {len(files)}')
# base_name = files[0]

# plot_rgb_tif_dir(especie_dir, base_name, (3, 2, 1))


for i in range(min(10, len(files))):
    plot_rgb_tif_dir(especie_dir, files[i], (3, 2, 1))

files

#======================================================================
# Save PNG


base_name = "IMG_0076"

png_image_dir = f"{base_dir}/IMAGES_PNG/{especie}"


if not os.path.isdir(png_image_dir):
    os.makedirs(png_image_dir)

from_tiff_to_png(especie_dir, base_name, (3, 2, 1), png_image_dir)




#======================================================================

print(f'{len(os.listdir(f"{base_dir}/IMAGES_PNG/"))}')

for x in os.listdir(f"{base_dir}/IMAGES_PNG/"):
    print(f'\n {x}: {len(os.listdir(f"{base_dir}/IMAGES_PNG/{x}"))}')

#======================================================================








