import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Segmentation---     \t\033[0m\n")

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
#---------------------------------------------------------------------
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
# 3 Bands

# ExG (Excess Green) + Otsu
def exg_otsu_segmentation(rgb_img):
    """
    Segmentação utilizando Excess Green (ExG) + Otsu.

    Parameters
    ----------
    rgb_img : np.ndarray
        Imagem RGB de dimensão (H, W, 3).

    Returns
    -------
    rgb_img_masked : np.ndarray
        Imagem RGB segmentada, com o fundo zerado.
    """

    # Canais
    R = rgb_img[:, :, 0]
    G = rgb_img[:, :, 1]
    B = rgb_img[:, :, 2]

    # Excess Green
    exg = 2 * G - R - B

    # Limiar de Otsu
    threshold = threshold_otsu(exg)

    # Máscara binária
    mask = exg > threshold

    # Aplicar máscara na imagem RGB
    rgb_img_masked = rgb_img.copy()
    rgb_img_masked[~mask] = 0

    return rgb_img_masked

#======================================================================
# SLIC + Merge

from skimage.segmentation import slic, relabel_sequential
from skimage import graph


def slic_merge_segmentation(rgb_img):
    """
    Segmenta uma imagem de três bandas usando:

    1. SLIC para gerar superpixels;
    2. fusão de regiões vizinhas por similaridade espectral;
    3. ExG + Otsu sobre as regiões fusionadas;
    4. aplicação da máscara à imagem original.

    Parameters
    ----------
    rgb_img : np.ndarray
        Imagem com formato (H, W, 3).

    Returns
    -------
    rgb_img_masked : np.ndarray
        Imagem segmentada, com os pixels de fundo zerados.
    """

    rgb_img = np.asarray(rgb_img)

    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("rgb_img deve possuir formato (H, W, 3).")

    if not np.isfinite(rgb_img).all():
        raise ValueError("rgb_img contém valores NaN ou infinitos.")

    # Normalização por canal somente para o processamento.
    rgb_normalized = np.zeros_like(rgb_img, dtype=np.float32)

    for channel in range(3):
        band = rgb_img[:, :, channel].astype(np.float32)

        band_min = band.min()
        band_max = band.max()

        if band_max > band_min:
            rgb_normalized[:, :, channel] = (
                (band - band_min) / (band_max - band_min)
            )

    # 1. Geração dos superpixels.
    slic_labels = slic(
        rgb_normalized,
        n_segments=300,
        compactness=10,
        sigma=1,
        start_label=0,
        channel_axis=-1
    )

    # 2. Construção do grafo de adjacência.
    rag = graph.rag_mean_color(
        rgb_normalized,
        slic_labels,
        mode="distance"
    )

    # Fusão de superpixels espectralmente semelhantes.
    merged_labels = graph.cut_threshold(
        slic_labels,
        rag,
        thresh=0.10
    )

    merged_labels, _, _ = relabel_sequential(merged_labels)

    # 3. Excess Green.
    red = rgb_normalized[:, :, 0]
    green = rgb_normalized[:, :, 1]
    blue = rgb_normalized[:, :, 2]

    exg = 2.0 * green - red - blue

    # Calcula o ExG médio de cada região fusionada.
    region_ids = np.unique(merged_labels)
    region_exg = np.zeros(region_ids.size, dtype=np.float32)

    for index, region_id in enumerate(region_ids):
        region_exg[index] = exg[merged_labels == region_id].mean()

    # Otsu aplicado aos valores médios das regiões.
    if np.unique(region_exg).size > 1:
        threshold = threshold_otsu(region_exg)
    else:
        threshold = region_exg[0]

    vegetation_regions = region_ids[region_exg > threshold]

    mask = np.isin(merged_labels, vegetation_regions)

    # 4. Aplicação da máscara na imagem original.
    rgb_img_masked = np.zeros_like(rgb_img)
    rgb_img_masked[mask] = rgb_img[mask]

    return rgb_img_masked

#======================================================================
#======================================================================
#======================================================================
# Contour

import cv2
import numpy as np


def contour_exg_otsu_segmentation(
    rgb_img_masked,
    contour_thickness=2,
    color="red",
):
    """
    Destaca a borda interna e externa de uma imagem RGB já segmentada.

    Parameters
    ----------
    rgb_img_masked : np.ndarray
        Imagem RGB segmentada com shape (H, W, 3).
        O fundo deve possuir valor zero nos três canais.

    contour_thickness : int, optional
        Espessura aproximada da borda para cada lado da fronteira.

    color : str, optional
        Cor da borda:
        red, green, blue, yellow, cyan, magenta ou white.

    Returns
    -------
    img_with_border : np.ndarray
        Imagem RGB com a borda interna e externa destacada.
    """

    if not isinstance(rgb_img_masked, np.ndarray):
        raise TypeError("'rgb_img_masked' deve ser um array NumPy.")

    if rgb_img_masked.ndim != 3 or rgb_img_masked.shape[2] != 3:
        raise ValueError(
            "A imagem deve possuir shape (H, W, 3). "
            f"Shape recebido: {rgb_img_masked.shape}"
        )

    if contour_thickness < 1:
        raise ValueError("'contour_thickness' deve ser maior ou igual a 1.")

    # Determina o valor máximo de acordo com o dtype da imagem.
    if np.issubdtype(rgb_img_masked.dtype, np.integer):
        max_value = np.iinfo(rgb_img_masked.dtype).max
    elif np.issubdtype(rgb_img_masked.dtype, np.floating):
        max_value = 1.0
    else:
        raise TypeError(
            f"Tipo de dado não suportado: {rgb_img_masked.dtype}"
        )

    # As cores estão definidas na ordem RGB.
    colors = {
        "red":     (max_value, 0, 0),
        "green":   (0, max_value, 0),
        "blue":    (0, 0, max_value),
        "yellow":  (max_value, max_value, 0),
        "cyan":    (0, max_value, max_value),
        "magenta": (max_value, 0, max_value),
        "white":   (max_value, max_value, max_value),
    }

    color = color.lower()

    if color not in colors:
        raise ValueError(
            f"Cor '{color}' inválida. "
            f"Escolha entre: {', '.join(colors.keys())}."
        )

    # Região segmentada: qualquer canal diferente de zero.
    mask = np.any(rgb_img_masked != 0, axis=2).astype(np.uint8) * 255

    # Kernel ímpar. Quanto maior, mais larga será a faixa da borda.
    kernel_size = 2 * contour_thickness + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    # Expande a máscara para o lado externo.
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Reduz a máscara para formar o lado interno.
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Faixa contendo a borda interna e externa.
    border_mask = cv2.subtract(dilated_mask, eroded_mask) > 0

    # Não modifica a imagem original.
    img_with_border = rgb_img_masked.copy()

    # Desenha a faixa da borda.
    img_with_border[border_mask] = colors[color]

    return img_with_border




#======================================================================






















