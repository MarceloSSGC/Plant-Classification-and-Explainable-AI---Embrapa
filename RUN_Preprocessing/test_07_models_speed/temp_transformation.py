import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
# # from skimage.filters import threshold_otsu
# import tifffile
# import cv2

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar TEMP Transformation ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# PLOT

# 1 band

def plot_band(img, title="Band", figsize=(15, 9)):
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
# Params

# My
# radius=1, n_points=8
# radius=3, n_points=3
# radius=4, n_points=4

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
# Params

# My
# radius=1, levels=32
# radius=1, levels=16
# radius=2, levels=16


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
# Params

# My
# radius=20, mode='white'
# radius=10, mode='black'


#======================================================================
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


#---------------------------------------------------------------------
# Params

# My
# bg_value=0.0, sigma=1
# bg_value=0.0, sigma=2


#======================================================================
#======================================================================
#======================================================================

from scipy import ndimage


def keep_bigger_components(
    img: np.ndarray,
    n_components: int = 1
) -> np.ndarray:
    """
    Mantém apenas as n maiores componentes conexas de uma imagem segmentada.

    Parameters
    ----------
    img : np.ndarray
        Imagem segmentada nos formatos:
            - (X, Y): imagem de uma banda
            - (X, Y, N): imagem com N bandas

        Um pixel é considerado background quando todas as suas bandas
        possuem valor 0.

    n_components : int, default=1
        Número de maiores componentes conexas a serem mantidas.
        Deve ser >= 1.

    Returns
    -------
    img_bigger_comp : np.ndarray
        Imagem contendo apenas as n maiores componentes conexas,
        com o mesmo shape e dtype da imagem de entrada.
    """

    # Validação
    if not isinstance(img, np.ndarray):
        raise TypeError("img deve ser um numpy.ndarray.")

    if img.ndim not in (2, 3):
        raise ValueError(
            "img deve possuir dimensão (X, Y) ou (X, Y, N). "
            f"Recebido: {img.shape}"
        )

    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components deve ser um inteiro >= 1.")

    # Guarda a dimensionalidade original
    original_ndim = img.ndim

    # Converte temporariamente (X, Y) -> (X, Y, 1)
    if original_ndim == 2:
        img_work = img[..., np.newaxis]
    else:
        img_work = img

    # Máscara 2D:
    # foreground se pelo menos uma banda for diferente de zero
    mask = np.any(img_work != 0, axis=-1)

    # Conectividade 8
    structure = np.ones((3, 3), dtype=np.uint8)

    # Identifica componentes conexas
    labeled, num_components = ndimage.label(
        mask,
        structure=structure
    )

    # Se não houver componentes
    if num_components == 0:
        return np.zeros_like(img)

    # Área de cada componente
    areas = np.bincount(labeled.ravel())

    # Label 0 corresponde ao background
    areas[0] = 0

    # Número de componentes que serão mantidas
    n_keep = min(n_components, num_components)

    # Labels das n maiores componentes
    biggest_labels = np.argpartition(
        areas,
        -n_keep
    )[-n_keep:]

    # Máscara final
    bigger_components_mask = np.isin(
        labeled,
        biggest_labels
    )

    # Aplica a máscara em todas as bandas
    img_bigger_comp = np.where(
        bigger_components_mask[..., np.newaxis],
        img_work,
        0
    )

    # Retorna com a mesma dimensionalidade da entrada
    if original_ndim == 2:
        img_bigger_comp = img_bigger_comp[..., 0]

    return img_bigger_comp


#======================================================================











