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
# NDVI


def apply_ndvi(img_5b, eps=1e-8):
    """
    Calcula o NDVI a partir de uma imagem multiespectral de 5 bandas.

    Ordem das bandas:
        0 = Blue
        1 = Green
        2 = Red
        3 = NIR
        4 = Red Edge

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5).

    eps : float
        Valor pequeno para evitar divisão por zero.

    Returns
    -------
    ndvi : np.ndarray
        Mapa NDVI (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # Bandas
    red = img_5b[:, :, 2].astype(np.float32)
    nir = img_5b[:, :, 3].astype(np.float32)

    # Máscara da planta
    mask = (red != 0) | (nir != 0)

    # NDVI
    denominator = nir + red

    ndvi = np.zeros_like(red, dtype=np.float32)

    valid = mask & (np.abs(denominator) > eps)

    ndvi[valid] = (
        (nir[valid] - red[valid]) /
        denominator[valid]
    )

    return ndvi

#======================================================================
# NDRE


def apply_ndre(img_5b, eps=1e-8):
    """
    Calcula o NDRE a partir de uma imagem multiespectral de 5 bandas.

    Ordem das bandas:
        0 = Blue
        1 = Green
        2 = Red
        3 = NIR
        4 = Red Edge

    Fórmula:
        NDRE = (NIR - RedEdge) / (NIR + RedEdge)

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5).

    eps : float
        Valor pequeno para evitar divisão por zero.

    Returns
    -------
    ndre : np.ndarray
        Mapa NDRE (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # Bandas
    nir = img_5b[:, :, 3].astype(np.float32)
    red_edge = img_5b[:, :, 4].astype(np.float32)

    # Máscara da planta
    mask = (nir != 0) | (red_edge != 0)

    # Denominador
    denominator = nir + red_edge

    # Inicializa resultado
    ndre = np.zeros_like(nir, dtype=np.float32)

    # Pixels válidos
    valid = mask & (np.abs(denominator) > eps)

    # NDRE
    ndre[valid] = (
        (nir[valid] - red_edge[valid]) /
        denominator[valid]
    )

    return ndre

#======================================================================
# GNDVI


def apply_gndvi(img_5b, eps=1e-8):
    """
    Calcula o GNDVI a partir de uma imagem multiespectral de 5 bandas.

    Ordem das bandas:
        0 = Blue
        1 = Green
        2 = Red
        3 = NIR
        4 = Red Edge

    Fórmula:
        GNDVI = (NIR - Green) / (NIR + Green)

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5).

    eps : float
        Valor pequeno para evitar divisão por zero.

    Returns
    -------
    gndvi : np.ndarray
        Mapa GNDVI (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # Bandas
    green = img_5b[:, :, 1].astype(np.float32)
    nir = img_5b[:, :, 3].astype(np.float32)

    # Máscara da planta
    mask = (green != 0) | (nir != 0)

    # Denominador
    denominator = nir + green

    # Inicializa resultado
    gndvi = np.zeros_like(nir, dtype=np.float32)

    # Pixels válidos
    valid = mask & (np.abs(denominator) > eps)

    # GNDVI
    gndvi[valid] = (
        (nir[valid] - green[valid]) /
        denominator[valid]
    )

    return gndvi


#======================================================================
# EVI


def apply_evi(img_5b, eps=1e-8):
    """
    Calcula o EVI a partir de uma imagem multiespectral de 5 bandas.

    Ordem das bandas:
        0 = Blue
        1 = Green
        2 = Red
        3 = NIR
        4 = Red Edge

    Fórmula:
        EVI = 2.5 * (NIR - Red) /
              (NIR + 6*Red - 7.5*Blue + 1)

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5).

    eps : float
        Valor pequeno para evitar divisão por zero.

    Returns
    -------
    evi : np.ndarray
        Mapa EVI (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # Bandas
    blue = img_5b[:, :, 0].astype(np.float32)
    red  = img_5b[:, :, 2].astype(np.float32)
    nir  = img_5b[:, :, 3].astype(np.float32)

    # Máscara da planta
    mask = (blue != 0) | (red != 0) | (nir != 0)
    # mask = np.ones_like(nir, dtype=np.float32)

    # Denominador
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0

    # Inicializa resultado
    evi = np.zeros_like(nir, dtype=np.float32)

    # Pixels válidos
    valid = mask & (np.abs(denominator) > eps)

    # EVI
    evi[valid] = (
        2.5 * (nir[valid] - red[valid])
        / denominator[valid]
    )

    return evi

#======================================================================
#======================================================================
#======================================================================




