import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Texture---     \t\033[0m\n")

#======================================================================
# 1 Band

def plot_tif_dir(image_dir: str, file_name: str):
    
    file_path = os.path.join(image_dir, file_name)

    with rasterio.open(file_path) as src:
        image = src.read(1)  # Lê a primeira (ou única) banda do arquivo

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title(file_name)
    plt.axis("off")
    plt.show()

#---------------------------------------------------------------------

def plot_tif(band):

    plt.figure(figsize=(6, 6))
    plt.imshow(band, cmap="gray")
    plt.title("Band")
    plt.axis("off")
    plt.show()

#---------------------------------------------------------------------

def plot_both(band, mask):
    plt.figure(figsize=(15, 9))
    # plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(band, cmap="gray")
    plt.title("Band")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Mask")

    plt.show()

#---------------------------------------------------------------------

def plot_hist_band(band, bins=256):
    plt.hist(band.ravel(), bins=bins)
    plt.xlabel("Valor do pixel")
    plt.ylabel("Frequência")
    plt.show()

#---------------------------------------------------------------------


#======================================================================
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

def plot_segmentation(rgb_img, rgb_img_masked):
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
# Segmentation

# 1 Band

from typing import Tuple
import cv2
import numpy as np


def segment_band_otsu(
    band: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    gaussian_kernel: Tuple[int, int] = (5, 5),
    opening_kernel_size: int = 3,
    closing_kernel_size: int = 5,
    invert: bool = False,
) -> np.ndarray:
    """
    Segmenta uma banda usando:
    CLAHE -> Gaussian Blur -> Otsu -> abertura -> fechamento.

    Parâmetros
    ----------
    band:
        Imagem bidimensional, por exemplo com shape (960, 1280).
        Pode ser uint8 ou uint16.

    clip_limit:
        Limite de contraste do CLAHE.

    tile_grid_size:
        Tamanho da grade utilizada pelo CLAHE.

    gaussian_kernel:
        Kernel do filtro gaussiano. Os valores devem ser positivos e ímpares.

    opening_kernel_size:
        Tamanho do kernel da abertura morfológica.

    closing_kernel_size:
        Tamanho do kernel do fechamento morfológico.

    invert:
        False: pixels acima do threshold tornam-se brancos.
        True: pixels abaixo do threshold tornam-se brancos.

    Retorno
    -------
    np.ndarray:
        Máscara binária uint8 com valores 0 e 255.
    """

    if not isinstance(band, np.ndarray):
        raise TypeError("'band' deve ser um array NumPy.")

    if band.ndim != 2:
        raise ValueError(
            f"'band' deve ter duas dimensões. Shape recebido: {band.shape}"
        )

    if band.size == 0:
        raise ValueError("'band' não pode estar vazio.")

    if np.isnan(band).any() or np.isinf(band).any():
        raise ValueError("'band' contém valores NaN ou infinitos.")

    if any(k <= 0 or k % 2 == 0 for k in gaussian_kernel):
        raise ValueError(
            "Os valores de 'gaussian_kernel' devem ser positivos e ímpares."
        )

    if opening_kernel_size <= 0 or closing_kernel_size <= 0:
        raise ValueError("Os kernels morfológicos devem ser maiores que zero.")

    # O CLAHE do OpenCV aceita imagens uint8 ou uint16.
    if band.dtype not in (np.uint8, np.uint16):
        band_min = band.min()
        band_max = band.max()

        if band_max == band_min:
            raise ValueError("A imagem possui intensidade constante.")

        band_work = cv2.normalize(
            band,
            None,
            alpha=0,
            beta=65535,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint16)
    else:
        band_work = band.copy()

    # 1. Aumenta o contraste local
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced = clahe.apply(band_work)

    # 2. Reduz ruído e pequenas variações
    blurred = cv2.GaussianBlur(
        enhanced,
        gaussian_kernel,
        sigmaX=0,
    )

    # 3. Segmentação automática por Otsu
    threshold_type = (
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    )

    otsu_threshold, mask = cv2.threshold(
        blurred,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU,
    )

    # Garante uma máscara uint8 com valores 0 e 255,
    # inclusive quando a entrada é uint16.
    mask = (mask > 0).astype(np.uint8) * 255

    # Elemento estruturante elíptico
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (opening_kernel_size, opening_kernel_size),
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size),
    )

    # 4. Remove pequenas regiões brancas isoladas
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        opening_kernel,
    )

    # 5. Preenche pequenos buracos e descontinuidades
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
    )

    return mask


#======================================================================
#======================================================================
# Texture

# 1 Band

# variância local

import numpy as np
from scipy.ndimage import uniform_filter

def local_var_band(band, window_size=5):
    """
    Calcula a variância local de uma banda.

    Parameters
    ----------
    band : ndarray
        Imagem de entrada (2D).
    window_size : int, optional
        Tamanho da janela quadrada (default=5).

    Returns
    -------
    var_band : ndarray
        Mapa de variância local.
    """
    band = band.astype(np.float32)

    mean = uniform_filter(band, size=window_size)
    mean_sq = uniform_filter(band**2, size=window_size)

    var_band = mean_sq - mean**2

    return var_band

#----------------------------------------------------------------------
# Entropia Local

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import square

def local_entropy_band(band, window_size=5):
    """
    Calcula a entropia local de uma banda.

    Parameters
    ----------
    band : ndarray
        Imagem de entrada (2D).
    window_size : int, optional
        Tamanho da janela quadrada (default=5).

    Returns
    -------
    entropy_band : ndarray
        Mapa de entropia local.
    """
    band = band.astype(np.float32)

    # Normaliza para 8 bits (0-255)
    band_norm = ((band - band.min()) /
                 (band.max() - band.min()) * 255).astype(np.uint8)

    entropy_band = entropy(band_norm, square(window_size))

    return entropy_band

#----------------------------------------------------------------------
# Sobel

import numpy as np
from scipy.ndimage import sobel

def sobel_band_filter(band):
    """
    Aplica o filtro Sobel em uma banda.

    Parameters
    ----------
    band : ndarray
        Imagem de entrada (2D).

    Returns
    -------
    sobel_mag : ndarray
        Magnitude do gradiente calculada pelo filtro Sobel.
    """
    band = band.astype(np.float32)

    # Gradientes horizontal e vertical
    gx = sobel(band, axis=1, mode='reflect')
    gy = sobel(band, axis=0, mode='reflect')

    # Magnitude do gradiente
    sobel_mag = np.hypot(gx, gy)

    return sobel_mag

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

#======================================================================









