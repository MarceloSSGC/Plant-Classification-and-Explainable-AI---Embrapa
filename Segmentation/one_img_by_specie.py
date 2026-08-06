import os
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Segmentation")
from auxiliar import *

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

WHC_DATA = "PlantaDaninha_BoaVista_Aligned_ecc_affine"
DATA_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{WHC_DATA}"

#======================================================================
# selected images

selected_imgs_dir = "/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/IMAGES/IMAGES_PNG"

selection = dict()
for x in sorted(os.listdir(selected_imgs_dir)): # x = "31_Cipo_Serra_da_Prata_02"
    y = os.listdir(os.path.join(selected_imgs_dir, x))[0][:-4]
    selection.update({x: y})

#======================================================================
# Save Segmented

save_dir = "/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/IMAGES/segmented"

species = sorted(os.listdir(DATA_DIR))

for specie in species:  # specie = species[1]
    
    base_name = selection[specie]
    specie_dir = os.path.join(DATA_DIR, specie)

    rgb_img = load_rgb_from_dir(specie_dir, base_name, (3, 2, 1))
    print(f"rgb_img.shape : {rgb_img.shape}")

    # min max - Band
    print(f"rgb_img min max: {rgb_img[:, :, 0].min(), rgb_img[:, :, 0].max()}")
    print(f"rgb_img min max: {rgb_img[:, :, 1].min(), rgb_img[:, :, 1].max()}")
    print(f"rgb_img min  max {rgb_img[:, :, 2].min(), rgb_img[:, :, 2].max()}")

    # plot_rgb(rgb_img)

    #------------------------------------------------------------------
    # ExG (Excess Green) + Otsu 

    rgb_img_masked = exg_otsu_segmentation(rgb_img)
    print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 0].min(), rgb_img_masked[:, :, 0].max()}")
    print(f"rgb_img_masked min max: {rgb_img_masked[:, :, 1].min(), rgb_img_masked[:, :, 1].max()}")
    print(f"rgb_img_masked min  max {rgb_img_masked[:, :, 2].min(), rgb_img_masked[:, :, 2].max()}")

    #------------------------------------------------------------------

    # plot_rgb(rgb_img)
    plot_rgb(rgb_img_masked)
    plot_segmentation(rgb_img, rgb_img_masked)

    #------------------------------------------------------------------

    new_especie_dir = os.path.join(save_dir, specie)

    if not os.path.isdir(new_especie_dir):
        os.makedirs(new_especie_dir)

    file_dir = os.path.join(new_especie_dir, base_name + ".png")

    if not os.path.isfile(file_dir):
        img_to_png(rgb_img_masked, new_especie_dir, base_name)



