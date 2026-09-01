import os
import numpy as np

#======================================================================
#======================================================================
#======================================================================
# LBP (Local Binary Pattern)

from skimage.feature import local_binary_pattern

def apply_lbp_1b(img_1b, radius=1, n_points=8):
    """
    Aplica Local Binary Pattern (LBP) em uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), com fundo igual a zero.
    radius : int
        Raio da vizinhança do LBP.
    n_points : int
        Número de pontos da vizinhança.

    Returns
    -------
    lbp : np.ndarray
        Mapa LBP (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    # Máscara da planta
    mask = img_1b != 0

    # Normalização usando apenas pixels da planta
    img_uint8 = np.zeros_like(img_1b, dtype=np.uint8)

    values = img_1b[mask]

    if values.size > 0:
        vmin = values.min()
        vmax = values.max()

        if vmax > vmin:
            img_uint8[mask] = (
                255 * (values - vmin) / (vmax - vmin)
            ).astype(np.uint8)

    # LBP
    lbp = local_binary_pattern(
        img_uint8,
        P=n_points,
        R=radius,
        method="uniform"
    ).astype(np.float32)

    # Mantém fundo = 0
    lbp[~mask] = 0

    return lbp

#---------------------------------------------------------------------

def apply_lbp_5b(img_5b, radius=1, n_points=8):

    canny_bands = []
    for i in range(img_5b.shape[-1]):
        canny_bands.append(apply_lbp_1b(img_5b[:, :, i], radius=radius, n_points=n_points))

    return np.stack(canny_bands, axis=-1)

#---------------------------------------------------------------------
# Parameters

# My
# radius=1, n_points=8
# radius=2, n_points=1
# radius=4, n_points=4
# radius=6, n_points=6

# | Teste     | `radius` | `n_points` | Característica              |
# | --------- | -------: | ---------: | --------------------------- |
# | **LBP-1** |        1 |          8 | Textura fina/local          |
# | **LBP-2** |        2 |         16 | Textura local-intermediária |
# | **LBP-3** |        3 |         24 | Textura intermediária       |
# | **LBP-4** |        4 |         32 | Textura mais ampla          |

#======================================================================
#======================================================================
# Entropia local (skimage.filters.rank.entropy)

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


def apply_local_entropy(img_1b, mask=None, bg_value=0.0, radius=5, levels=32):
    """
    Calcula a entropia local de uma banda única — mede a desordem/
    complexidade da distribuição de tons dentro de cada janela,
    ignorando pixels de fundo (não deixa a borda folha/fundo
    contaminar o resultado).

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se None, inferida via img_1b != bg_value.
    bg_value : float
        Valor considerado "fundo" quando mask=None.
    radius : int
        Raio do elemento estruturante (disco) — define o tamanho da
        janela de análise. Menor = textura mais fina/local (melhor
        para nervuras finas); maior = mais suave/estrutural.
    levels : int
        Número de níveis de cinza para quantização. rank.entropy exige
        imagem inteira (uint8/uint16), então a banda float32 precisa
        ser quantizada antes — mesma lógica usada no GLCM.

    Retorna
    -------
    np.ndarray
        Mapa de entropia local, mesma dimensão de img_1b, zerado fora
        da folha.
    """
    img = img_1b.astype(np.float64)

    if mask is None:
        mask = img != bg_value
    mask = mask.astype(bool)

    if not mask.any():
        raise ValueError("Máscara vazia — nenhum pixel de folha detectado.")

    # Quantiza para uint8 (rank.entropy exige imagem inteira)
    img_min, img_max = np.nanmin(img[mask]), np.nanmax(img[mask])
    img_norm = (img - img_min) / (img_max - img_min)
    img_norm = np.clip(img_norm, 0, 1)
    img_quant = (img_norm * (levels - 1)).astype(np.uint8)

    footprint = disk(radius)

    result = entropy(img_quant, footprint=footprint, mask=mask.astype(np.uint8))

    result = result * mask.astype(np.float32)

    return result.astype(np.float32)

#---------------------------------------------------------------------
# Parameters

# My
# radius=5, levels=32
# radius=3, levels=32
# radius=1, levels=16
# radius=1, levels=32


#======================================================================
#======================================================================
# tophat

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


#---------------------------------------------------------------------

# bg_value=0.0, radius=15, mode='black'
# bg_value=0.0, radius=15, mode='white'

#======================================================================
#======================================================================




#======================================================================
#======================================================================


#======================================================================
#======================================================================
#======================================================================
# Transformations

# Transformation - Only RGB

def trans__only_RGB(img_5b):
    return img_5b[:, :, [0, 1, 2]]

#======================================================================
# Transformation - LBP (Local Binary Pattern)

def trans__lbp(img_5b):

    img_rgb = trans__only_RGB(img_5b)

    img_G = img_rgb[:, :, 1]

    img_1b_p1 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=5, levels=32)
    img_1b_p2 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=1, levels=8)
    img_1b_p3 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=2, levels=1)
    img_1b_p4 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=4, levels=4)
    img_1b_p5 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=6, levels=6)

    img_full = np.concatenate([img_rgb, img_1b_p1[:, :, np.newaxis], 
                               img_1b_p2[:, :, np.newaxis], img_1b_p3[:, :, np.newaxis],
                               img_1b_p4[:, :, np.newaxis], img_1b_p5][:, :, np.newaxis],
                               axis=2)
    return img_full

# radius=1, n_points=8
# radius=2, n_points=1
# radius=4, n_points=4
# radius=6, n_points=6

#======================================================================
# Transformation - Entropia local (skimage.filters.rank.entropy)

# def apply_local_entropy(img_1b, mask=None, bg_value=0.0, radius=5, levels=32):

def trans__entropy(img_5b):

    img_rgb = trans__only_RGB(img_5b)

    img_G = img_rgb[:, :, 1]

    img_1b_p1 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=5, levels=32)
    img_1b_p2 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=3, levels=32)
    img_1b_p3 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=1, levels=16)
    img_1b_p4 = apply_local_entropy(img_G, mask=None, bg_value=0.0, radius=1, levels=32)

    img_full = np.concatenate([img_rgb, img_1b_p1[:, :, np.newaxis], 
                               img_1b_p2[:, :, np.newaxis], img_1b_p3[:, :, np.newaxis], 
                               img_1b_p4[:, :, np.newaxis]], axis=2)

    return img_full

# radius=5, levels=32
# radius=3, levels=32
# radius=1, levels=16
# radius=1, levels=32

#======================================================================
# Transformation - Tophat

# apply_tophat(img_1b, mask=None, bg_value=0.0, radius=5, mode='white')
def trans__tophat(img_5b):

    img_rgb = trans__only_RGB(img_5b)

    img_B = img_rgb[:, :, 0]
    img_G = img_rgb[:, :, 1]

    img_1b_p1 = apply_tophat(img_B, mask=None, bg_value=0.0, radius=5, mode='white')
    img_1b_p2 = apply_tophat(img_B, mask=None, bg_value=0.0, radius=5, mode='black')

    img_1b_p3 = apply_tophat(img_G, mask=None, bg_value=0.0, radius=5, mode='white')
    img_1b_p4 = apply_tophat(img_G, mask=None, bg_value=0.0, radius=5, mode='black')

    img_full = np.concatenate([img_rgb, img_1b_p1, img_1b_p2, img_1b_p3, img_1b_p4], axis=2)

    return img_full

# bg_value=0.0, radius=15, mode='black'
# bg_value=0.0, radius=15, mode='white'

#======================================================================

#======================================================================
#======================================================================
#======================================================================


def trans_function_bands_number(transformation_name):

    n_bands = None
    if transformation_name == "5_bands":
        n_bands = 5

    elif transformation_name == "only_RGB":
        n_bands = 3

    elif transformation_name == "LBP":
        n_bands = 8

    elif transformation_name == "Entropy":
        n_bands = 7

    elif transformation_name == "Tophat":
        n_bands = 7

    else:
        raise ValueError("transformation_name not in list")

    return n_bands


def trans_function(transformation_name):

    if transformation_name == "5_bands":
        transformation = None

    elif transformation_name == "only_RGB":
        transformation = trans__only_RGB

    elif transformation_name == "Entropy":
        transformation = trans__entropy

    elif transformation_name == "LBP":
        transformation = trans__lbp

    elif transformation_name == "Tophat":
        transformation = trans__tophat

    else:
        raise ValueError("transformation_name not in list")

    return transformation































