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

def random_rotation(img_5b, angle="random"):
    """
    Aplica uma rotação aleatória entre 0 e 360 graus
    a uma imagem multibanda.

    Parameters
    ----------
    img_5b : np.ndarray
        Array (H, W, C), por exemplo (960, 1280, 5).

    Returns
    -------
    img_rot : np.ndarray
        Imagem rotacionada com o mesmo shape da entrada.
    """
    if angle == "random":
        angle = np.random.uniform(0, 360)
    
    img_rot = rotate(
        img_5b,
        angle=angle,
        axes=(0, 1),       # rotaciona somente H e W
        reshape=False,     # mantém (H, W, 5)
        order=1,           # interpolação bilinear
        mode="constant",
        cval=0
    )

    return img_rot.astype(img_5b.dtype)

#======================================================================
# Translação pequena: até ~5–10%

from scipy.ndimage import shift

def deterministic_translation(img_5b, max_fraction=0.05):
    """
    Aplica translação deterministica/aleatória(-1/1) da altura e largura.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multibanda (H, W, C).

    max_fraction : float
        Fração máxima de deslocamento.
        0.05 = 5%, 0.10 = 10%.

    Returns
    -------
    img_trans : np.ndarray
        Imagem transladada, mantendo o shape original.
    """

    H, W, _ = img_5b.shape

    dy = np.random.choice([-1, 1]) * max_fraction * H
    dx = np.random.choice([-1, 1]) * max_fraction * W

    img_trans = shift(
        img_5b,
        shift=(dy, dx, 0),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False
    )

    return img_trans.astype(img_5b.dtype)


#======================================================================
# Scale/Zoom

from scipy.ndimage import zoom


def scale_image(img_5b, scale=1.1):
    """
    Aplica scale/zoom em imagem multibanda mantendo o shape original.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem (H, W, C).

    scale : float
        Fator de escala.
        0.9 = reduz 10%
        1.0 = original
        1.1 = aumenta 10%

    Returns
    -------
    out : np.ndarray
        Imagem (H, W, C), com mesmo dtype da original.
    """

    H, W, C = img_5b.shape

    # Scale somente nas dimensões espaciais.
    # As 5 bandas permanecem juntas.
    scaled = zoom(
        img_5b,
        zoom=(scale, scale, 1),
        order=1
    )

    out = np.zeros_like(img_5b)

    h, w, _ = scaled.shape

    if scale >= 1:
        # Crop central
        y0 = (h - H) // 2
        x0 = (w - W) // 2

        out = scaled[y0:y0 + H, x0:x0 + W, :]

    else:
        # Centraliza a imagem reduzida e deixa bordas = 0
        y0 = (H - h) // 2
        x0 = (W - w) // 2

        out[y0:y0 + h, x0:x0 + w, :] = scaled

    return out.astype(img_5b.dtype)


#======================================================================
# Cutout / Random Erasing — remoção de pequenas regiões

def random_cutout(img_5b, seed):
    """
    Remove uma região retangular aleatória da imagem,
    preenchendo-a com zero em todas as bandas.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multibanda (H, W, 5).

    size_fraction : float
        Tamanho aproximado do recorte em relação às
        dimensões da imagem.
        Ex.: 0.10 = 10% da altura e largura.

    seed : int ou None
        Seed para tornar o resultado reproduzível.

    Returns
    -------
    img_cutout : np.ndarray
        Imagem com Cutout, mantendo shape e dtype.
    """

    size_fraction=0.10

    rng = np.random.default_rng(seed)

    H, W, _ = img_5b.shape

    cut_h = int(H * size_fraction)
    cut_w = int(W * size_fraction)

    # Posição aleatória
    y0 = rng.integers(0, H - cut_h + 1)
    x0 = rng.integers(0, W - cut_w + 1)

    img_cutout = img_5b.copy()

    # Mesma região removida nas 5 bandas
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


