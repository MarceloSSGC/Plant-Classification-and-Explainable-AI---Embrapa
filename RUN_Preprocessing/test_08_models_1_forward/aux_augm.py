import os
import numpy as np
import matplotlib.pyplot as plt

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Augmentation RUN ---     \t\t\033[0m\n")

#======================================================================

def plot_rgb(rgb_image, bands_ch=(3, 2, 1)):

    channels = []

    idx_bands_ch = [x-1 for x in bands_ch]

    for band in idx_bands_ch:
        
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
# Rotation: 0–360°.

import numpy as np
from scipy.ndimage import rotate


def random_rotation(img_multiband, angle="random"):
    """
    Aplica uma rotação a uma imagem multibanda.

    Parameters
    ----------
    img_multiband : np.ndarray
        Array com shape (H, W, N_BANDS).

    angle : float or str, optional
        Ângulo de rotação em graus.
        Se "random", utiliza um ângulo aleatório entre 0 e 360.

    Returns
    -------
    img_rotated : np.ndarray
        Imagem rotacionada com o mesmo shape e dtype da entrada.

        Exemplos:
        (900, 1100, 37) -> (900, 1100, 37)
        (900, 1100, 32) -> (900, 1100, 32)
    """
    if angle == "random":
        angle = np.random.uniform(0, 360)

    img_rotated = rotate(
        img_multiband,
        angle=angle,
        axes=(0, 1),       # rotaciona H e W, preservando N_BANDS
        reshape=False,     # mantém (H, W, N_BANDS)
        order=1,           # interpolação bilinear
        mode="constant",
        cval=0
    )

    return img_rotated.astype(img_multiband.dtype)

#======================================================================
# Translação pequena: até ~5–10%
import numpy as np
from scipy.ndimage import shift


def deterministic_translation(img_multiband, max_fraction=0.05):
    """
    Aplica uma translação determinística/aleatória (-1/1)
    na altura e largura de uma imagem multibanda.

    Parameters
    ----------
    img_multiband : np.ndarray
        Array com shape (H, W, N_BANDS).

    max_fraction : float
        Fração máxima de deslocamento.
        0.05 = 5%, 0.10 = 10%.

    Returns
    -------
    img_translated : np.ndarray
        Imagem transladada com o mesmo shape e dtype da entrada.

        Exemplos:
        (900, 1100, 37) -> (900, 1100, 37)
        (900, 1100, 32) -> (900, 1100, 32)
    """

    H, W, _ = img_multiband.shape

    dy = np.random.choice([-1, 1]) * max_fraction * H
    dx = np.random.choice([-1, 1]) * max_fraction * W

    img_translated = shift(
        img_multiband,
        shift=(dy, dx, 0),  # desloca H e W, preservando N_BANDS
        order=0,
        mode="constant",
        cval=0,
        prefilter=False
    )

    return img_translated.astype(img_multiband.dtype)


#======================================================================
# Scale/Zoom
import numpy as np
from scipy.ndimage import zoom


def scale_image(img_multiband, scale=1.1):
    """
    Aplica scale/zoom em uma imagem multibanda mantendo o shape original.

    Parameters
    ----------
    img_multiband : np.ndarray
        Array com shape (H, W, N_BANDS).

    scale : float
        Fator de escala.
        0.9 = reduz 10%
        1.0 = original
        1.1 = aumenta 10%

    Returns
    -------
    img_scaled : np.ndarray
        Imagem escalada com o mesmo shape e dtype da entrada.

        Exemplos:
        (900, 1100, 37) -> (900, 1100, 37)
        (900, 1100, 32) -> (900, 1100, 32)
    """

    H, W, _ = img_multiband.shape

    # Aplica scale somente nas dimensões espaciais.
    # Todas as N_BANDS permanecem juntas.
    scaled = zoom(
        img_multiband,
        zoom=(scale, scale, 1),
        order=1
    )

    img_scaled = np.zeros_like(img_multiband)

    h, w, _ = scaled.shape

    if scale >= 1:
        # Crop central para recuperar o shape original
        y0 = (h - H) // 2
        x0 = (w - W) // 2

        img_scaled = scaled[y0:y0 + H, x0:x0 + W, :]

    else:
        # Centraliza a imagem reduzida e preenche as bordas com 0
        y0 = (H - h) // 2
        x0 = (W - w) // 2

        img_scaled[y0:y0 + h, x0:x0 + w, :] = scaled

    return img_scaled.astype(img_multiband.dtype)

#======================================================================
# Cutout / Random Erasing — remoção de pequenas regiões
import numpy as np


def random_cutout(img_multiband, seed):
    """
    Remove uma região retangular aleatória da imagem,
    preenchendo-a com zero em todas as bandas.

    Parameters
    ----------
    img_multiband : np.ndarray
        Array com shape (H, W, N_BANDS).

    seed : int or None
        Seed para tornar o resultado reproduzível.

    Returns
    -------
    img_cutout : np.ndarray
        Imagem com Cutout, mantendo o mesmo shape e dtype da entrada.

        Exemplos:
        (900, 1100, 37) -> (900, 1100, 37)
        (900, 1100, 32) -> (900, 1100, 32)
    """

    size_fraction = 0.10

    rng = np.random.default_rng(seed)

    H, W, _ = img_multiband.shape

    cut_h = int(H * size_fraction)
    cut_w = int(W * size_fraction)

    # Posição aleatória
    y0 = rng.integers(0, H - cut_h + 1)
    x0 = rng.integers(0, W - cut_w + 1)

    img_cutout = img_multiband.copy()

    # Remove a mesma região em todas as N_BANDS
    img_cutout[
        y0:y0 + cut_h,
        x0:x0 + cut_w,
        :
    ] = 0

    return img_cutout


#======================================================================
#======================================================================
#======================================================================
# Augmentation Compilation

# aug_params = [
#     [("rotation", 180)],
#     [("rotation", 90), ("translation", 0.1)],
#     [("rotation", 270), ("scale", 1.1)], 
#     [("translation", 0.1), ("scale", 0.9)], 
#     [("cutout", 39), ("cutout", 139), ("cutout", 10)]
# ]

def augmentation_compilation(img_5b, method, param):

    if method == "rotation":
        new_img_5b = random_rotation(img_5b, param)
    elif method == "translation":
        new_img_5b = deterministic_translation(img_5b, param)
    elif method == "scale":
        new_img_5b = scale_image(img_5b, param)
    elif method == "cutout":    # 39, 239, 10
        new_img_5b = random_cutout(img_5b, param)
    else:
        raise ValueError("method not defined")

    return new_img_5b


