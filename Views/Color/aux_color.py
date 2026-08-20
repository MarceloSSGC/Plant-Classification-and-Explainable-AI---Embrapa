import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu
import tifffile
import cv2

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar INDEX ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# PLOT

# 1 band

def plot_band(img, title="Band"):
    """
    Plota uma imagem de uma única banda.

    Parameters
    ----------
    img : np.ndarray
        Array 2D de dimensão (H, W).
    title : str
        Título da imagem.
    """
    if img.ndim != 2:
        raise ValueError(f"Esperado array 2D (H, W), recebido {img.shape}")

    plt.figure(figsize=(10, 7))
    plt.imshow(img, cmap="gray")
    plt.colorbar(label="Pixel value")
    plt.title(title)
    plt.axis("off")
    plt.show()

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
# plot_5bands


def plot_5bands(img_5b, fontsize=18):
    """
    Plota lado a lado as 5 bandas de uma imagem multiespectral.

    Parameters
    ----------
    img_5b : np.ndarray
        Array de dimensão (H, W, 5).
    fontsize : int or float, optional
        Tamanho da fonte dos títulos das bandas. Padrão: 14.
    """
    band_mapping = {
        0: "Blue",
        1: "Green",
        2: "Red",
        3: "NIR",
        4: "Red Edge",
    }

    if img_5b.ndim != 3 or img_5b.shape[2] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for i, ax in enumerate(axes):
        ax.imshow(img_5b[:, :, i], cmap="gray")
        ax.set_title(band_mapping[i], fontsize=fontsize)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

#======================================================================
#======================================================================
#======================================================================
# [B, G, R, NIR, RedEdge] --> [H, S, V, NIR, RedEdge]

import cv2

def convert_5b_to_hsv_nir_re(img_5b):
    """
    Converte uma imagem multiespectral:

        [B, G, R, NIR, RedEdge]

    para:

        [H, S, V, NIR, RedEdge]

    Parameters
    ----------
    img_5b : np.ndarray
        Array (H, W, 5), dtype float32.

        Ordem:
        0 = Blue
        1 = Green
        2 = Red
        3 = NIR
        4 = Red Edge

    Returns
    -------
    result : np.ndarray
        Array (H, W, 5), dtype float32.

        Ordem:
        0 = Hue
        1 = Saturation
        2 = Value
        3 = NIR
        4 = Red Edge
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float32)

    # ---------------------------------------------------------
    # Bandas
    # ---------------------------------------------------------
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    nir = img[:, :, 3]
    red_edge = img[:, :, 4]

    # ---------------------------------------------------------
    # Máscara da planta
    # ---------------------------------------------------------
    mask = np.any(img != 0, axis=-1)

    # ---------------------------------------------------------
    # Cria RGB na ordem esperada pelo OpenCV
    # ---------------------------------------------------------
    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    # ---------------------------------------------------------
    # Normaliza RGB para [0, 1]
    # ---------------------------------------------------------
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)

    rgb_values = rgb[mask]

    if rgb_values.size > 0:
        vmin = rgb_values.min()
        vmax = rgb_values.max()

        if vmax > vmin:
            rgb_norm[mask] = (
                (rgb[mask] - vmin) /
                (vmax - vmin)
            )

    # ---------------------------------------------------------
    # RGB -> HSV
    #
    # Para float32 no OpenCV:
    # H: [0, 360)
    # S: [0, 1]
    # V: [0, 1]
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(
        rgb_norm,
        cv2.COLOR_RGB2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Fundo permanece zero
    H[~mask] = 0
    S[~mask] = 0
    V[~mask] = 0

    # ---------------------------------------------------------
    # Empilha:
    # [H, S, V, NIR, RedEdge]
    # ---------------------------------------------------------
    result = np.stack(
        [H, S, V, nir, red_edge],
        axis=-1
    ).astype(np.float32)

    return result





















