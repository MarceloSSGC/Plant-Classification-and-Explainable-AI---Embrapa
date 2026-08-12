import os
import numpy as np

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Segmentation")
from Segmentation.auxiliar_segmentation import *

#======================================================================

print(f"\n\033[100;01m\t     --- Start Segmentation ---     \t\033[0m\n")

#======================================================================
# Dataset & Directories

DATASET_NAME = "PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5"
BASE_DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/"

DATA_DIR = os.path.join(BASE_DATA_DIR, DATASET_NAME)

#======================================================================
# New Dataset

NEW_DATASET_NAME = "Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu"


#======================================================================

especies = sorted(os.listdir(DATA_DIR))

for especie in especies[:31]:    # especie = especies[0]

    print("\n"+"="*60 + f'\n\t especie: {especie}')

    new_especie_dir = os.path.join(BASE_DATA_DIR, NEW_DATASET_NAME, especie)

    if not os.path.isdir(new_especie_dir):
        os.makedirs(new_especie_dir)

    old_especie_dir = os.path.join(BASE_DATA_DIR, DATASET_NAME, especie)
    file_names = sorted(set([x[:-6] for x in os.listdir(old_especie_dir)]))

    for ith, ith_file in enumerate(file_names):     # ith, ith_file = 0, file_names[0]

        print(f'ith: {ith} - {len(file_names)} \t file: {ith_file} \t {especie}')

        final_dir = os.path.join(new_especie_dir, ith_file)

        if not os.path.isfile(final_dir + "_1.tif") or \
            not os.path.isfile(final_dir + "_2.tif") or \
            not os.path.isfile(final_dir + "_3.tif") or \
            not os.path.isfile(final_dir + "_4.tif") or \
            not os.path.isfile(final_dir + "_5.tif"):

            print(f'segmentation...')
            img_5b = load_5b_from_dir(old_especie_dir, ith_file)
            print(f"img_5b.shape : {img_5b.shape} \n")

            # img_5b_seg, mask = segment_best_band_otsu(img_5b)
            img_5b_seg, mask, best_band = segment_best_band_otsu_green(img_5b)

            # plot_rgb(img_5b_seg)

            plot_segmentation(img_5b, img_5b_seg)

            save_segmented_bands(img_5b_seg, new_especie_dir, ith_file)




        
        # orgn_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5/01_malva_branca_Agua_Boa_01/IMG_0015_1.tif"
        # seg_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu/01_malva_branca_Agua_Boa_01/IMG_0015_1.tif"

        # import tifffile

        # img = tifffile.imread(orgn_dir)
        # print(f'orgn_dir:')
        # print(img.shape)
        # print(img.dtype)
        # print(img.min(), img.max())

        # img = tifffile.imread(seg_dir)
        # print(f'\nseg_dir:')
        # print(img.shape)
        # print(img.dtype)
        # print(img.min(), img.max())

#======================================================================
#======================================================================
#======================================================================
