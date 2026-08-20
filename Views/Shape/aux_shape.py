import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu
import tifffile
import cv2

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar SHAPE ---     \t\t\033[0m\n")

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
# Canny / Edge map

def apply_canny_1b(img_1b, threshold1=100, threshold2=200):
    """
    Aplica Canny Edge Detection em uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), podendo ser float32.
    threshold1 : int
        Limiar inferior do Canny.
    threshold2 : int
        Limiar superior do Canny.

    Returns
    -------
    edges : np.ndarray
        Edge map (H, W), dtype uint8, com valores 0 ou 255.
    """
    
    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    # Normaliza para [0, 255]
    img_norm = cv2.normalize(
        img_1b,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Canny
    edges = cv2.Canny(
        img_norm,
        threshold1,
        threshold2
    )

    return edges

#---------------------------------------------------------------------

def apply_canny_5b(img_5b, threshold1=100, threshold2=200):

    canny_bands = []
    for i in range(img_5b.shape[-1]):
        canny_bands.append(apply_canny_1b(img_5b[:, :, i], threshold1=threshold1, threshold2=threshold2))

    return np.stack(canny_bands, axis=-1)


#======================================================================
# Sobel / Gradient map

def apply_sobel_1b(img_1b, ksize=3):
    """
    Aplica Sobel em uma única banda e retorna
    o módulo do gradiente.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W).
    ksize : int
        Tamanho do kernel Sobel (3, 5, 7...).

    Returns
    -------
    gradient : np.ndarray
        Mapa de magnitude do gradiente (H, W), float32.
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    img = img_1b.astype(np.float32)

    # Gradiente horizontal
    grad_x = cv2.Sobel(
        img, cv2.CV_32F,
        dx=1, dy=0,
        ksize=ksize
    )

    # Gradiente vertical
    grad_y = cv2.Sobel(
        img, cv2.CV_32F,
        dx=0, dy=1,
        ksize=ksize
    )

    # Magnitude do gradiente
    gradient = cv2.magnitude(grad_x, grad_y)

    return gradient

#---------------------------------------------------------------------

def apply_sobel_5b(img_5b, ksize=3):
    sobel_bands = []
    for i in range(img_5b.shape[-1]):
        sobel_bands.append(apply_sobel_1b(img_5b[:, :, i], ksize=ksize))

    return np.stack(sobel_bands, axis=-1)

#======================================================================
# Silhouette / Segmentation mask


def apply_silhouette_1b(img_1b):
    """
    Gera uma máscara de silhueta a partir de uma banda já segmentada.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), onde pixels com valor 0 representam fundo.

    Returns
    -------
    silhouette : np.ndarray
        Máscara binária (H, W), dtype uint8:
        0 = fundo
        1 = planta
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    silhouette = (img_1b != 0).astype(np.uint8)

    return silhouette

#---------------------------------------------------------------------

def apply_silhouette_5b(img_5b):
    silhouette_bands = []
    for i in range(img_5b.shape[-1]):
        silhouette_bands.append(apply_silhouette_1b(img_5b[:, :, i]))

    return np.stack(silhouette_bands, axis=-1)

#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================




















