import os
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Segmentation")
from Segmentation.auxiliar_segmentation import *

print(os.listdir())

#======================================================================
#======================================================================

def img_to_png(rgb_img: np.ndarray, save_dir: str, base_name: str):
    """
    Salva uma imagem RGB (numpy) no formato PNG.

    Parameters
    ----------
    rgb_img : np.ndarray
        Imagem RGB no formato (H, W, 3).
    save_dir : str
        Diretório onde a imagem será salva.
    base_name : str
        Nome base do arquivo (sem extensão).
    """

    os.makedirs(save_dir, exist_ok=True)

    # Normaliza para [0, 255] caso não esteja em uint8
    if rgb_img.dtype != np.uint8:
        rgb = rgb_img.astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb_img

    save_path = os.path.join(save_dir, f"{base_name}.png")
    Image.fromarray(rgb).save(save_path)


#======================================================================
#======================================================================
# Dataset

WHC_DATA = "PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5"
DATA_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{WHC_DATA}"

#----------------------------------------------------------------------
# To Save RGB

rgb_dir = "/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/IMAGES/RGB_images/RGB_images"

# 
# rgb_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/IMAGES/RGB_images"

#----------------------------------------------------------------------
# To Save Segmented

# seg_rgb_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/IMAGES/SEG_RGB_images"

#======================================================================
# Save Segmented


species = sorted(os.listdir(DATA_DIR))

for specie in species:  # specie = species[0]

    print(f"specie: {specie}")
    
    specie_dir = os.path.join(DATA_DIR, specie)

    files = sorted(set([x[:-6] for x in os.listdir(specie_dir)]))

    for k, base_name in enumerate(files):     # k, base_name = 0, files[0]

        rgb_img = load_rgb_from_dir(specie_dir, base_name, (3, 2, 1))
        print(f"k : {k} of {len(files)} -- {base_name}")

        # min max - Band
        # print(f"rgb_img min max: {rgb_img[:, :, 0].min(), rgb_img[:, :, 0].max()}")
        # print(f"rgb_img min max: {rgb_img[:, :, 1].min(), rgb_img[:, :, 1].max()}")
        # print(f"rgb_img min  max {rgb_img[:, :, 2].min(), rgb_img[:, :, 2].max()}")

        # plot_rgb(rgb_img)

        #------------------------------------------------------------------
        # ExG (Excess Green) + Otsu 

        # rgb_img_masked = exg_otsu_segmentation(rgb_img)
        # print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 0].min(), rgb_img_masked[:, :, 0].max()}")
        # print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 1].min(), rgb_img_masked[:, :, 1].max()}")
        # print(f"rgb_img_masked min  max {rgb_img_masked[:, :, 2].min(), rgb_img_masked[:, :, 2].max()}")

        # #------------------------------------------------------------------

        # # plot_rgb(rgb_img)
        # plot_rgb(rgb_img_masked)
        # plot_segmentation(rgb_img, rgb_img_masked)

        #------------------------------------------------------------------

        new_especie_dir = os.path.join(rgb_dir, specie)

        if not os.path.isdir(new_especie_dir):
            os.makedirs(new_especie_dir)

        file_dir = os.path.join(new_especie_dir, base_name + ".png")

        if not os.path.isfile(file_dir):
            img_to_png(rgb_img, new_especie_dir, base_name)


