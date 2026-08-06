import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar ---     \t\033[0m\n")

#======================================================================
# 3 bands

def plot_rgb_tif_dir(image_dir: str, base_name: str, bands: tuple):

    if len(bands) != 3:
        raise ValueError("'bands' deve conter exatamente três valores.")

    if any(b < 1 or b > 5 for b in bands):
        raise ValueError("As bandas devem estar entre 1 e 5.")

    # Remove o número da banda e a extensão
    # base_name = file_name.rsplit("_", 1)[0]

    channels = []

    for band in bands:
        band_file = f"{base_name}_{band}.tif"
        band_path = os.path.join(image_dir, band_file)

        with rasterio.open(band_path) as src:
            img = src.read(1).astype(np.float32)

        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)

    rgb = np.dstack(channels)

    plt.figure(figsize=(15, 9))
    plt.imshow(rgb)
    plt.title(f"{base_name}  RGB={bands}")
    plt.axis("off")
    plt.show()

#---------------------------------------------------------------------

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

#---------------------------------------------------------------------

def load_rgb_from_dir(image_dir: str, base_name: str, bands: tuple):

    if len(bands) != 3:
        raise ValueError("'bands' deve conter exatamente três valores.")

    if any(b < 1 or b > 5 for b in bands):
        raise ValueError("As bandas devem estar entre 1 e 5.")

    channels = []

    for band in bands:
        band_file = f"{base_name}_{band}.tif"
        band_path = os.path.join(image_dir, band_file)

        with rasterio.open(band_path) as src:
            img = src.read(1).astype(np.float32)

        channels.append(img)

    rgb = np.dstack(channels)

    return rgb

#---------------------------------------------------------------------

def from_tiff_to_png(image_dir: str, base_name: str, bands: tuple, png_dir: str):
    if len(bands) != 3:
        raise ValueError("'bands' deve conter exatamente três valores.")

    if any(b < 1 or b > 5 for b in bands):
        raise ValueError("As bandas devem estar entre 1 e 5.")

    os.makedirs(png_dir, exist_ok=True)

    channels = []

    for band in bands:
        band_file = f"{base_name}_{band}.tif"
        band_path = os.path.join(image_dir, band_file)

        with rasterio.open(band_path) as src:
            img = src.read(1).astype(np.float32)

        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)

    rgb = np.dstack(channels)

    # Converte para uint8
    rgb = (rgb * 255).astype(np.uint8)

    png_path = os.path.join(png_dir, f"{base_name}.png")
    Image.fromarray(rgb).save(png_path)