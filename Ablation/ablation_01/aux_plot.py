import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar PLOT ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# 5 bands

def load_5b_from_dir(image_dir: str, base_name: str):

    channels = []

    for band in range(1, 6):
        band_file = f"{base_name}_{band}.tif"
        band_path = os.path.join(image_dir, band_file)

        with rasterio.open(band_path) as src:
            img = src.read(1).astype(np.float32)

        channels.append(img)

    img = np.dstack(channels)

    return img

#======================================================================

def plot_rgb(rgb_image, bands_ch=(3, 2, 1)):

    channels = []

    for band in range(3):
        
        img = rgb_image[:, :, band].astype(np.float32)

        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)

    rgb = np.dstack(channels)

    plt.figure(figsize=(15, 9))
    plt.imshow(rgb)
    plt.title(f"RGB - {bands_ch}")
    plt.axis("off")
    plt.show()


#======================================================================

def plot_two_imgs(rgb_img, rgb_img_masked):
    """
    Plota a imagem original e a imagem segmentada lado a lado.

    Parameters
    ----------
    rgb_img : np.ndarray
        Imagem RGB original (H, W, 3).

    rgb_img_masked : np.ndarray
        Imagem RGB após segmentação (H, W, 3).
    """

    channels = []
    for band in range(3):
        img = rgb_img[:, :, band].astype(np.float32)
        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)
    rgb = np.dstack(channels)

    channels = []
    for band in range(3):
        img = rgb_img_masked[:, :, band].astype(np.float32)
        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)
    rgb_masked = np.dstack(channels)

    fig, ax = plt.subplots(1, 2, figsize=(15, 9))

    ax[0].imshow(rgb)
    ax[0].set_title("Imagem original")
    ax[0].axis("off")

    ax[1].imshow(rgb_masked)
    ax[1].set_title("Imagem segmentada")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

#======================================================================
#======================================================================
#======================================================================
#======================================================================
