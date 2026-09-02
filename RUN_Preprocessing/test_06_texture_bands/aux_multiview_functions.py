import os
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

try:
    from temp_transformation import *
except ImportError:
    from RUN_Preprocessing.test_06_texture_bands.temp_transformation import *


#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Multiview Functions ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
#======================================================================
# PLOT

# 1 band

def plot_band(img, title="Band", figsize=(12, 9)):
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

    plt.figure(figsize=figsize)
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

def plot_rgb(rgb_image, bands_ch=(2, 1, 0)):

    channels = []

    for band in bands_ch:
        
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

def plot_rgb_no_norm(rgb_image, bands_ch=(2, 1, 0)):

    channels = []

    for band in bands_ch:
        
        img = rgb_image[:, :, band].astype(np.float32)

        # Normalização individual para [0,1]
        # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
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
# SHAPE

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------

#======================================================================
#======================================================================
# COLOR

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

def convert_5b_to_hsv(img_5b):
    return convert_5b_to_hsv_nir_re(img_5b)[:, :, [0, 1, 2]]

#----------------------------------------------------------------------

#======================================================================
#======================================================================
# TEXTURE

#----------------------------------------------------------------------
# tophat

def clip_percentile_normalize(
    array: np.ndarray,
    percentile: float = 0.95
) -> np.ndarray:
    """
    Limita os valores acima de um percentil e normaliza para [0, 1].

    Parameters
    ----------
    array : np.ndarray
        Array de entrada, esperado como float32.
    percentile : float
        Percentil entre 0 e 1. Ex.: 0.9 = percentil 90.

    Returns
    -------
    np.ndarray
        Array float32 normalizado entre 0 e 1.
    """
    array = np.asarray(array, dtype=np.float32)

    # Calcula o valor correspondente ao percentil
    upper = np.quantile(array, percentile)

    # Limita valores acima do percentil
    clipped = np.minimum(array, upper)

    # Normalização min-max para [0, 1]
    min_val = clipped.min()
    max_val = clipped.max()

    if max_val == min_val:
        return np.zeros_like(clipped, dtype=np.float32)

    normalized = (clipped - min_val) / (max_val - min_val)

    return normalized.astype(np.float32)


import numpy as np
from skimage.morphology import white_tophat, black_tophat, disk
from scipy.ndimage import distance_transform_edt


def _prepare_filled_image(img_1b, mask, bg_value):
    """Normaliza e preenche o fundo com o valor de folha mais próximo,
    evitando que a borda folha/fundo contamine o filtro."""
    img = img_1b.astype(np.float64)

    if mask is None:
        mask = img != bg_value
    mask = mask.astype(bool)

    if not mask.any():
        raise ValueError("Máscara vazia — nenhum pixel de folha detectado.")

    img_min, img_max = np.nanmin(img[mask]), np.nanmax(img[mask])
    img_norm = (img - img_min) / (img_max - img_min)

    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    img_filled = img_norm[iy, ix]
    img_filled[mask] = img_norm[mask]

    return img_filled, mask


def apply_tophat(img_1b, mask=None, bg_value=0.0, radius=5, mode='white'):
    """
    Aplica Top-hat (ou Black-hat) morfológico em uma banda única,
    realçando estruturas finas mais claras (ou mais escuras) que a
    vizinhança imediata — bom para nervuras e pequenas texturas.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se None, inferida via img_1b != bg_value.
    bg_value : float
        Valor considerado "fundo" quando mask=None.
    radius : int
        Raio do elemento estruturante (disco). Define o tamanho máximo
        de estrutura realçada — deve ser um pouco MAIOR que a espessura
        da nervura em pixels (estruturas maiores que o footprint não
        são realçadas, ficam "removidas" pela abertura/fechamento).
        Comece pequeno (3-5) para nervuras finas, aumente se as
        nervuras principais forem grossas.
    mode : str
        'white' = white_tophat, realça estruturas mais CLARAS que o
                  entorno (imagem menos sua abertura morfológica).
        'black' = black_tophat, realça estruturas mais ESCURAS que o
                  entorno (fechamento morfológico menos a imagem).
        Depende de como a nervura aparece na sua banda — teste os dois.

    Retorna
    -------
    np.ndarray
        Mapa top-hat/black-hat, mesma dimensão de img_1b, zerado fora
        da folha.
    """
    if mode not in ('white', 'black'):
        raise ValueError("mode deve ser 'white' ou 'black'.")

    img_filled, mask = _prepare_filled_image(img_1b, mask, bg_value)
    footprint = disk(radius)

    if mode == 'white':
        result = white_tophat(img_filled, footprint=footprint)
    else:
        result = black_tophat(img_filled, footprint=footprint)

    result = result * mask.astype(np.float32)

    return result.astype(np.float32)

def apply_tophat_pecentil(img_1b, mask=None, bg_value=0.0, radius=5, mode='white',percentile=0.99):
    img_temp = apply_tophat(img_1b, mask=mask, bg_value=bg_value, radius=radius, mode=mode)
    return clip_percentile_normalize(img_temp, percentile=percentile)


#----------------------------------------------------------------------
# Structure tensor (tensor de estrutura local)

import numpy as np
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from scipy.ndimage import distance_transform_edt


def _prepare_filled_image(img_1b, mask, bg_value):
    """Normaliza e preenche o fundo com o valor de folha mais próximo,
    evitando que a borda folha/fundo contamine o filtro."""
    img = img_1b.astype(np.float64)

    if mask is None:
        mask = img != bg_value
    mask = mask.astype(bool)

    if not mask.any():
        raise ValueError("Máscara vazia — nenhum pixel de folha detectado.")

    img_min, img_max = np.nanmin(img[mask]), np.nanmax(img[mask])
    img_norm = (img - img_min) / (img_max - img_min)

    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    img_filled = img_norm[iy, ix]
    img_filled[mask] = img_norm[mask]

    return img_filled, mask


def apply_structure_tensor(img_1b, mask=None, bg_value=0.0, sigma=2):
    """
    Calcula o tensor de estrutura local de uma banda única, retornando
    mapas de orientação dominante e coerência.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se None, inferida via img_1b != bg_value.
    bg_value : float
        Valor considerado "fundo" quando mask=None.
    sigma : float
        Desvio-padrão do kernel Gaussiano usado para agregar informação
        local (janela efetiva de análise). Maior = orientação mais
        "suave"/macro; menor = mais sensível a detalhe fino (nervuras
        finas), porém mais ruidoso.

    Retorna
    -------
    orientation : np.ndarray
        Ângulo (em radianos, entre -pi/2 e pi/2) da direção dominante
        da estrutura local em cada pixel. Sem significado fora da folha
        (zerado pela máscara).
    coherence : np.ndarray
        Grau de "quão bem definida" é essa direção, entre 0 e 1.
        Próximo de 1 = estrutura fortemente direcional (nervura clara,
        borda nítida); próximo de 0 = região isotrópica/plana (sem
        direção dominante — tecido foliar uniforme, por exemplo).
    """
    img_filled, mask = _prepare_filled_image(img_1b, mask, bg_value)

    # Componentes do tensor de estrutura
    Arr, Arc, Acc = structure_tensor(img_filled, sigma=sigma, order='rc')

    # Autovalores (l1 >= l2)
    l1, l2 = structure_tensor_eigenvalues((Arr, Arc, Acc))

    # Orientação dominante: ângulo do autovetor associado ao maior autovalor
    orientation = 0.5 * np.arctan2(2 * Arc, Acc - Arr)

    # Coerência: quão anisotrópica é a estrutura local
    denom = l1 + l2
    coherence = np.where(denom > 1e-10, (l1 - l2) / denom, 0.0)

    orientation = orientation * mask.astype(np.float32)
    coherence = coherence * mask.astype(np.float32)

    return orientation.astype(np.float32), coherence.astype(np.float32)


#----------------------------------------------------------------------

#======================================================================
#======================================================================
# INDEX

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

#----------------------------------------------------------------------
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


#----------------------------------------------------------------------
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


#----------------------------------------------------------------------

#======================================================================
#======================================================================
#======================================================================


def one_view_function(img_5b, type_view, funct, param):

    band_mapping = {
        "All": [0, 1, 2, 3, 4],
        "Blue": 0,
        "Green": 1,
        "Red": 2,
        "NIR": 3,
        "Red Edge": 4,
    }

    # print(f'\n\ntype_view: {type_view}')
    # print(f'funct: {funct}')
    # print(f'param: {param}\n\n')

    if "which_band" not in param.keys() or len(param["which_band"]) == 0:
        raise ValueError("param/which_band has a problem")

    if isinstance(param["which_band"], str):
        band_chose = band_mapping[param["which_band"]]
    elif isinstance(param["which_band"], list):
        band_chose = [band_mapping[y] for y in param["which_band"]]
    else:
        raise ValueError("param['which_band'] has a problem")

    #--------------------------------------------------------------

    if type_view == "BANDS":
        img_xb = img_5b[:, :, band_chose]
        return img_xb

    #--------------------------------------------------------------
    elif type_view == "SHAPE":

        if funct == "Canny":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(apply_canny_1b(img_1b, threshold1=100, threshold2=200), axis=-1)

        elif funct == "Silhouette":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(apply_silhouette_1b(img_1b), axis=-1)
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")

    #--------------------------------------------------------------
    elif type_view == "COLOR":

        if funct == "HSV":
            return convert_5b_to_hsv(img_5b)
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")

    #--------------------------------------------------------------
    elif type_view == "TEXTURE":

        if funct == "Tophat":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(clip_percentile_normalize(apply_tophat(img_1b)), axis=-1)
        elif funct == "Coherence":
            img_1b = img_5b[:, :, band_chose]
            orientation_map, coherence_map = apply_structure_tensor(img_1b, bg_value=0.0, sigma=2)
            return np.expand_dims(coherence_map, axis=-1)  
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")

    #--------------------------------------------------------------
    elif type_view == "INDEX":
        if funct == "NDVI":
            return np.expand_dims(apply_ndvi(img_5b), axis=-1)
        elif funct == "NDRE":
            return np.expand_dims(apply_ndre(img_5b), axis=-1)
        elif funct == "GNDVI":
            return np.expand_dims(apply_gndvi(img_5b), axis=-1)  
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")
     
    #--------------------------------------------------------------
    else:
        raise ValueError(f"type_view: {type_view} not defined")

#======================================================================

# type_view = ith_trans['type_view']
# funct = ith_trans['funct']
# param = ith_trans['param']


def one_view_function_v_texture(img_5b, type_view, funct, param):

    band_mapping = {
        "All": [0, 1, 2, 3, 4],
        "RGB": [0, 1, 2],
        "Blue": 0,
        "Green": 1,
        "Red": 2,
        "NIR": 3,
        "Red Edge": 4,
    }

    # print(f'\n\ntype_view: {type_view}')
    # print(f'funct: {funct}')
    # print(f'param: {param}\n\n')

    if "which_band" not in param.keys() or len(param["which_band"]) == 0:
        raise ValueError("param/which_band has a problem")

    if isinstance(param["which_band"], str):
        band_chose = band_mapping[param["which_band"]]
    elif isinstance(param["which_band"], list):
        band_chose = [band_mapping[y] for y in param["which_band"]]
    else:
        raise ValueError("param['which_band'] has a problem")

    #--------------------------------------------------------------

    if type_view == "BANDS":
        img_xb = img_5b[:, :, band_chose]
        return img_xb

    #--------------------------------------------------------------
    elif type_view == "SHAPE":

        if funct == "Canny":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(apply_canny_1b(img_1b, threshold1=100, threshold2=200), axis=-1)

        elif funct == "Silhouette":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(apply_silhouette_1b(img_1b), axis=-1)
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")

    #--------------------------------------------------------------
    elif type_view == "COLOR":

        if funct == "HSV":
            return convert_5b_to_hsv(img_5b)
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")

    #--------------------------------------------------------------
    elif type_view == "TEXTURE":

        param_copy = deepcopy(param)
        del param_copy['which_band']

        if funct == "Tophat":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(clip_percentile_normalize(apply_tophat(img_1b, **param_copy)), axis=-1)
        elif funct == "tensor":
            img_1b = img_5b[:, :, band_chose]
            orientation_map, coherence_map = apply_structure_tensor(img_1b, **param_copy)
            return np.expand_dims(coherence_map, axis=-1)
        elif funct == "LBP":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(apply_lbp_1b(img_1b, **param_copy), axis=-1)
        elif funct == "entropy":
            img_1b = img_5b[:, :, band_chose]
            return np.expand_dims(apply_local_entropy(img_1b, **param_copy), axis=-1)
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")

    #--------------------------------------------------------------
    elif type_view == "INDEX":
        if funct == "NDVI":
            return np.expand_dims(apply_ndvi(img_5b), axis=-1)
        elif funct == "NDRE":
            return np.expand_dims(apply_ndre(img_5b), axis=-1)
        elif funct == "GNDVI":
            return np.expand_dims(apply_gndvi(img_5b), axis=-1)  
        else:
            raise ValueError(f"funct: {funct} not in {type_view}")
     
    #--------------------------------------------------------------
    else:
        raise ValueError(f"type_view: {type_view} not defined")



#======================================================================
# Multiview Function


# "VIEWS": {
#     "BANDS": {
#         0: {
#             "funct": "Iden",
#             "param": {
#                 "which_band": "RGB"
#             }
#         }
#     },
#     "TEXTURE": {
#         0: {
#             "funct": "Tophat",
#             "param": {
#                 "which_band": "Green"
#             }
#         },
#         1: {
#             "funct": "Coherence",
#             "param": {
#                 "which_band": "Green"
#             }
#         }
#     }
# }


# trans_list = [
#     {
#         "type_view": "SHAPE",
#         "funct": "Canny",
#         "param": {"which_band": "NIR"}
#     },
#     {
#         "type_view": "SHAPE",
#         "funct": "Silhouette",
#         "param": {"which_band": "NIR"}
#     },
#     {
#         "type_view": "COLOR",
#         "funct": "HSV",
#         "param": {"which_band": "All"}
#     },
#     {
#         "type_view": "TEXTURE",
#         "funct": "Tophat",
#         "param": {"which_band": "NIR"}
#     },
#     {
#         "type_view": "TEXTURE",
#         "funct": "Coherence",
#         "param": {"which_band": "NIR"}
#     },
#     {
#         "type_view": "INDEX",
#         "funct": "NDVI",
#         "param": {"which_band": "All"}
#     },
#     {
#         "type_view": "INDEX",
#         "funct": "NDRE",
#         "param": {"which_band": "All"}
#     },
#     {
#         "type_view": "INDEX",
#         "funct": "GNDVI",
#         "param": {"which_band": "All"}
#     },
# ]

def mulview_one_img_function(img_5b, trans_list, plot_img=False):

    if trans_list == None or trans_list == "None":
        return img_5b

    #--------------------------------------------------------------

    array_view = []

    for ith_trans in trans_list: # ith_trans = trans_list[k]  k=0  k+=1

        ith_view = one_view_function_v_texture(img_5b, **ith_trans)

        if plot_img:
            if ith_view.shape[-1] == 1:
                plot_band(ith_view[:, :, 0])
            else:
                plot_rgb(ith_view)

        array_view.append(ith_view)

    return np.concatenate(array_view, axis=-1)

#======================================================================
# Test

# DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu"
# especies = sorted(os.listdir(DATA_DIR))
# especie = "36_Unha_de_gato_Serra_da_Prata_06"
# especie_dir = os.path.join(DATA_DIR, especie)
# files = sorted(set([x[:-6]for x in os.listdir(especie_dir)]))
# file_name = files[0]
# img_5b = load_5b_from_dir(especie_dir, file_name)

# type_view = "Shape"
# funct = "Canny"
# param = {"which_band": "NIR"}

#======================================================================
#======================================================================
#======================================================================

def multiview_main_function(trans_list, SEG_DATA_DIR, MTV_DATA_DIR):

    especies = sorted(os.listdir(SEG_DATA_DIR))
    if "seg_info.json" in especies:
        especies.remove("seg_info.json")

    take_n_bands = True

    for i, especie in enumerate(especies):    # i, especie = 0, especies[0]

        print(f'i: {i} - especie: {especie}')

        new_files_dir = os.path.join(MTV_DATA_DIR, especie)
        os.makedirs(new_files_dir, exist_ok=True)


        old_files_dir = os.path.join(SEG_DATA_DIR, especie)
        files = sorted(set([x[:-6] for x in os.listdir(old_files_dir)]))

        for file_name in files:              # file_name = files[0]

            file_dir = os.path.join(new_files_dir, f"{file_name}.npy")

            if not os.path.isfile(file_dir):

                img_5b = load_5b_from_dir(old_files_dir, file_name)

                img_multiview = mulview_one_img_function(img_5b, trans_list)

                if take_n_bands:
                    n_bands = img_multiview.shape[-1]
                    print(f"\n\t\033[100;40m --- n_bands: {n_bands} ---   \033[100;0m")
                    take_n_bands = False

                np.save(file_dir, img_multiview)

    return n_bands

#======================================================================
#======================================================================


def views_dict_to_string(views_dict):
    """
    Converte um dicionário de configuração em uma string identificadora.

    Exemplo:
        {
            'BANDS': {
                0: {
                    'funct': 'Iden',
                    'param': {'which_band': 'All'}
                }
            },
            ...
        }

    Retorna algo como:
        BANDS_Iden_which_band_All__SHAPE_Canny_which_band_NIR_...
    """

    parts = []

    for category, functions in views_dict.items():

        category_parts = [str(category)]

        for _, config in functions.items():

            # Nome da função
            funct = config.get("funct")

            if funct is not None:
                category_parts.append(str(funct))

            # Parâmetros da função
            params = config.get("param", {})

            for param_name, param_value in params.items():
                category_parts.append(str(param_name))
                category_parts.append(str(param_value))

        # Junta os elementos dentro de uma categoria com "_"
        parts.append("_".join(category_parts))

    # Separa categorias com "__"
    return "__".join(parts)


#======================================================================
#======================================================================
