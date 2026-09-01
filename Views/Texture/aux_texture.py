import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu
import tifffile
import cv2

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Texture ---     \t\t\033[0m\n")

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


#======================================================================
# Gabor filter bank

from skimage.filters import gabor

def apply_gabor_1b(
    img_1b,
    frequencies=(0.1, 0.2),
    thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)
):
    """
    Aplica um banco de filtros Gabor em uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), com fundo igual a zero.

    frequencies : tuple
        Frequências espaciais utilizadas pelos filtros Gabor.

    thetas : tuple
        Orientações dos filtros em radianos.

    Returns
    -------
    gabor_maps : np.ndarray
        Array (H, W, K), onde K é o número total de filtros:
        K = len(frequencies) * len(thetas).

        O fundo permanece com valor zero.
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    # Máscara da planta
    mask = img_1b != 0

    # Normalização para [0, 1]
    img_norm = np.zeros_like(img_1b, dtype=np.float32)

    values = img_1b[mask]

    if values.size > 0:
        vmin = values.min()
        vmax = values.max()

        if vmax > vmin:
            img_norm[mask] = (
                (values - vmin) / (vmax - vmin)
            ).astype(np.float32)

    # ---------------------------------------------------------
    # Banco de filtros Gabor
    # ---------------------------------------------------------

    gabor_maps = []

    for frequency in frequencies:
        for theta in thetas:

            real, imag = gabor(
                img_norm,
                frequency=frequency,
                theta=theta
            )

            # Magnitude da resposta
            magnitude = np.sqrt(real**2 + imag**2)

            # Mantém fundo = 0
            magnitude[~mask] = 0

            gabor_maps.append(
                magnitude.astype(np.float32)
            )

    # (H, W, K)
    return np.stack(gabor_maps, axis=-1)


# ---------------------------------------------------------------------


def apply_gabor_5b(
    img_5b,
    frequencies=(0.1, 0.2),
    thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)
):
    """
    Aplica o banco de filtros Gabor às 5 bandas.

    Returns
    -------
    gabor_5b : np.ndarray
        Array (H, W, 5, K)
    """

    if img_5b.ndim != 3:
        raise ValueError(
            f"Esperado array (H, W, B), recebido {img_5b.shape}"
        )

    gabor_bands = []

    for i in range(img_5b.shape[-1]):

        gabor_band = apply_gabor_1b(
            img_5b[:, :, i],
            frequencies=frequencies,
            thetas=thetas
        )

        gabor_bands.append(gabor_band)

    # (H, W, 5, K)
    return np.stack(gabor_bands, axis=2)

#======================================================================
# Wavelet decomposition

import pywt

def apply_wavelet_1b(img_1b, wavelet="haar"):
    """
    Aplica decomposição Wavelet 2D de nível 1 em uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), com fundo igual a zero.
    wavelet : str
        Wavelet utilizada. Ex.: "haar", "db2", "sym2".

    Returns
    -------
    wavelet_maps : np.ndarray
        Array (H2, W2, 4), contendo:
        [LL, LH, HL, HH]
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    img = img_1b.astype(np.float32)

    # Decomposição Wavelet 2D
    LL, (LH, HL, HH) = pywt.dwt2(img, wavelet)

    wavelet_maps = np.stack(
        [LL, LH, HL, HH],
        axis=-1
    ).astype(np.float32)

    return wavelet_maps


# ---------------------------------------------------------------------


def apply_wavelet_5b(img_5b, wavelet="haar"):
    """
    Aplica decomposição Wavelet 2D em todas as bandas.

    Returns
    -------
    wavelet_5b : np.ndarray
        Array (H2, W2, 5, 4)

        Último eixo:
        0 = LL
        1 = LH
        2 = HL
        3 = HH
    """

    if img_5b.ndim != 3:
        raise ValueError(
            f"Esperado array (H, W, B), recebido {img_5b.shape}"
        )

    wavelet_bands = []

    for i in range(img_5b.shape[-1]):

        wavelet_band = apply_wavelet_1b(
            img_5b[:, :, i],
            wavelet=wavelet
        )

        wavelet_bands.append(wavelet_band)

    return np.stack(wavelet_bands, axis=2)

#======================================================================
# Local Variance

from scipy.ndimage import uniform_filter

def apply_local_variance_1b(img_1b, window_size=5):
    """
    Calcula a variância local de uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), com fundo igual a zero.
    window_size : int
        Tamanho da janela local.

    Returns
    -------
    local_var : np.ndarray
        Mapa de variância local (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    img = img_1b.astype(np.float32)

    # Máscara da planta
    mask = img != 0

    # Média local
    mean = uniform_filter(
        img,
        size=window_size,
        mode="reflect"
    )

    # Média local do quadrado
    mean_sq = uniform_filter(
        img ** 2,
        size=window_size,
        mode="reflect"
    )

    # Var(X) = E[X²] - E[X]²
    local_var = mean_sq - mean ** 2

    # Corrige pequenos valores negativos por erro numérico
    local_var = np.maximum(local_var, 0)

    # Mantém fundo = 0
    local_var[~mask] = 0

    return local_var.astype(np.float32)

#======================================================================
# Frangi / Hessian vesselness

from skimage.filters import frangi

def apply_frangi_1b(
    img_1b,
    sigmas=range(1, 5),
    black_ridges=False
):
    """
    Aplica Frangi / Hessian vesselness em uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), com fundo igual a zero.
    sigmas : iterable
        Escalas analisadas pelo filtro.
        Valores maiores detectam estruturas mais largas.
    black_ridges : bool
        False -> destaca estruturas claras sobre fundo escuro.
        True  -> destaca estruturas escuras sobre fundo claro.

    Returns
    -------
    frangi_map : np.ndarray
        Mapa Frangi (H, W), dtype float32.
        O fundo permanece com valor zero.
    """

    if img_1b.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W), recebido {img_1b.shape}"
        )

    # Máscara da planta
    mask = img_1b != 0

    # Normalização para [0, 1]
    img_norm = np.zeros_like(img_1b, dtype=np.float32)

    values = img_1b[mask]

    if values.size > 0:
        vmin = values.min()
        vmax = values.max()

        if vmax > vmin:
            img_norm[mask] = (
                (values - vmin) / (vmax - vmin)
            ).astype(np.float32)

    # Frangi / Hessian vesselness
    frangi_map = frangi(
        img_norm,
        sigmas=sigmas,
        black_ridges=black_ridges
    ).astype(np.float32)

    # Mantém fundo = 0
    frangi_map[~mask] = 0

    return frangi_map

#======================================================================
# DoG / LoG

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace


def texture_filter(img_1b, method="dog", sigma1=1.0, sigma2=2.0):
    """
    Aplica DoG ou LoG em uma imagem de uma única banda.

    Parameters
    ----------
    img_1b : np.ndarray
        Imagem 2D (H, W), preferencialmente float32.
    method : {"dog", "log"}
        Filtro a ser aplicado.
    sigma1 : float
        Escala principal.
    sigma2 : float
        Segunda escala usada apenas no DoG.

    Returns
    -------
    result : np.ndarray
        Imagem 2D filtrada, float32.
    """

    img = img_1b.astype(np.float32)

    if method.lower() == "dog":
        g1 = gaussian_filter(img, sigma=sigma1)
        g2 = gaussian_filter(img, sigma=sigma2)
        result = g1 - g2

    elif method.lower() == "log":
        result = gaussian_laplace(img, sigma=sigma1)

    else:
        raise ValueError("method deve ser 'dog' ou 'log'")

    return result.astype(np.float32)



#======================================================================
# normalização robusta → CLAHE → Frangi multiescala

import numpy as np
from skimage import exposure
from skimage.filters import frangi


def texture_frangi(img_1b,
                   sigmas=range(1, 6),
                   clip_limit=0.02):
    """
    Realça estruturas internas alongadas, como nervuras foliares.

    Pipeline:
        banda -> normalização -> CLAHE -> Frangi multiescala

    Parameters
    ----------
    img_1b : np.ndarray
        Banda 2D (H, W), por exemplo img_5b[:, :, 1].

    sigmas : iterable
        Escalas analisadas pelo Frangi.
        Valores maiores detectam estruturas mais largas.

    clip_limit : float
        Intensidade do CLAHE.

    Returns
    -------
    result : np.ndarray
        Imagem 2D float32, normalizada em [0, 1].
    """

    img = img_1b.astype(np.float32)

    # --------------------------------------------------
    # 1. Normalização robusta
    # Evita que poucos pixels extremos dominem o contraste
    # --------------------------------------------------
    p_low, p_high = np.percentile(img, (1, 99))

    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-8)

    # --------------------------------------------------
    # 2. CLAHE: aumenta contraste LOCAL
    # --------------------------------------------------
    img_clahe = exposure.equalize_adapthist(
        img,
        clip_limit=clip_limit
    )

    # --------------------------------------------------
    # 3. Frangi multiescala
    # Detecta estruturas lineares/ridges
    # --------------------------------------------------
    result = frangi(
        img_clahe,
        sigmas=sigmas,
        black_ridges=False
    )

    # Normalização final
    result = result.astype(np.float32)

    rmin = result.min()
    rmax = result.max()

    result = (result - rmin) / (rmax - rmin + 1e-8)

    return result


#======================================================================
# glcm

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from scipy.ndimage import zoom

def glcm_texture_map(img_1b, prop='contrast', levels=32, window_size=15,
                      distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                      stride=4):
    """
    Calcula um mapa de textura GLCM em janela deslizante, retornando um
    array 2D de mesma dimensão que img_1b.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda (ex: img_5b[:, :, i]).
    prop : str
        Propriedade de Haralick a calcular por janela:
        'contrast', 'homogeneity', 'energy', 'correlation',
        'dissimilarity', 'ASM'.
    levels : int
        Número de níveis de cinza para quantização.
    window_size : int
        Tamanho do lado da janela (deve ser ímpar, ex: 15, 21, 31).
        Janelas maiores = textura mais "suave"/macro;
        janelas menores = textura mais local/fina, porém mais ruidosa.
    distances : tuple
        Distâncias em pixels para a GLCM.
    angles : tuple
        Ângulos considerados (a feature final é a média entre eles).
    stride : int
        Passo do deslizamento da janela. stride=1 é pixel a pixel (lento);
        stride=4 ou 8 calcula esparsamente e depois reamostra pro
        tamanho original (muito mais rápido).

    Retorna
    -------
    np.ndarray
        Array 2D, mesma dimensão de img_1b, com o valor da feature
        de textura em cada posição.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar.")

    h, w = img_1b.shape
    half = window_size // 2

    # 1. Normaliza e quantiza a imagem inteira uma única vez
    img_min, img_max = np.nanmin(img_1b), np.nanmax(img_1b)
    if img_max - img_min == 0:
        raise ValueError("Banda com valor constante — GLCM não aplicável.")
    img_norm = (img_1b - img_min) / (img_max - img_min)
    img_quant = (img_norm * (levels - 1)).astype(np.uint8)

    # 2. Padding para lidar com bordas
    img_padded = np.pad(img_quant, half, mode='reflect')

    # 3. Posições esparsas (de acordo com o stride)
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    sparse_map = np.zeros((len(ys), len(xs)), dtype=np.float32)

    # 4. Calcula GLCM em cada posição esparsa
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            window = img_padded[y:y + window_size, x:x + window_size]

            glcm = graycomatrix(
                window,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True
            )
            sparse_map[i, j] = graycoprops(glcm, prop).mean()

    # 5. Reamostra o mapa esparso de volta ao tamanho original
    zoom_factors = (h / sparse_map.shape[0], w / sparse_map.shape[1])
    texture_map = zoom(sparse_map, zoom_factors, order=1)  # interpolação bilinear

    # Garante dimensão exata (zoom pode arredondar diferente por causa de float)
    if texture_map.shape != (h, w):
        texture_map = resize(texture_map, (h, w), preserve_range=True,
                              anti_aliasing=True)

    return texture_map.astype(np.float32)

import numpy as np
from skimage.feature import graycomatrix, graycoprops

def glcm_texture_map(img_1b, prop='homogeneity', levels=32, window_size=15,
                      distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    Calcula um mapa de textura GLCM pixel a pixel (sem stride, sem
    interpolação), retornando um array 2D de mesma dimensão que img_1b.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda (ex: img_5b[:, :, i]).
    prop : str
        Propriedade de Haralick: 'contrast', 'homogeneity', 'energy',
        'correlation', 'dissimilarity', 'ASM'.
    levels : int
        Número de níveis de cinza para quantização.
    window_size : int
        Tamanho do lado da janela (ímpar). Controla o nível de detalhe:
        menor = mais fino/local, maior = mais suave/estrutural.
    distances : tuple
        Distâncias em pixels para a GLCM.
    angles : tuple
        Ângulos considerados (a feature final é a média entre eles).

    Retorna
    -------
    np.ndarray
        Array 2D, mesma dimensão de img_1b, com o valor da feature
        de textura em cada posição — nítido, sem blur de interpolação.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar.")

    h, w = img_1b.shape
    half = window_size // 2

    # 1. Normaliza e quantiza a imagem inteira uma única vez
    img_min, img_max = np.nanmin(img_1b), np.nanmax(img_1b)
    if img_max - img_min == 0:
        raise ValueError("Banda com valor constante — GLCM não aplicável.")
    img_norm = (img_1b - img_min) / (img_max - img_min)
    img_quant = (img_norm * (levels - 1)).astype(np.uint8)

    # 2. Padding para lidar com bordas
    img_padded = np.pad(img_quant, half, mode='reflect')

    # 3. Calcula GLCM em CADA posição (sem pular nenhuma)
    texture_map = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            window = img_padded[y:y + window_size, x:x + window_size]

            glcm = graycomatrix(
                window,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True
            )
            texture_map[y, x] = graycoprops(glcm, prop).mean()

    return texture_map


from scipy.ndimage import uniform_filter

def local_roughness_map(img_1b, window_size=9):
    """
    Mapa de rugosidade via desvio-padrão local — rápido, vetorizado,
    pixel a pixel, sem necessidade de numba.
    """
    img = img_1b.astype(np.float32)
    mean = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img**2, size=window_size)
    variance = mean_sq - mean**2
    variance = np.clip(variance, 0, None)  # evita negativos por erro numérico
    return np.sqrt(variance)  # desvio-padrão local = "rugosidade"


import numpy as np
from scipy.ndimage import uniform_filter

def local_roughness_map_masked(img_1b, mask=None, window_size=9, bg_value=0.0):
    """
    Mapa de rugosidade (desvio-padrão local) que ignora pixels de fundo
    dentro da janela, evitando que a transição folha-fundo domine o
    resultado nas bordas.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray ou None
        Máscara booleana (True = folha/válido, False = fundo).
        Se None, é inferida automaticamente como img_1b != bg_value.
    window_size : int
        Tamanho da janela (ímpar). Reduza para detalhe mais fino
        (nervuras finas), ex: 5 ou 7.
    bg_value : float
        Valor considerado "fundo" quando mask=None (ex: 0.0 se a
        imagem segmentada tem fundo zerado).

    Retorna
    -------
    np.ndarray
        Mapa de rugosidade, mesma dimensão de img_1b. Pixels de fundo
        são retornados como 0.
    """
    img = img_1b.astype(np.float32)

    if mask is None:
        mask = img != bg_value
    mask = mask.astype(np.float32)

    # Estatísticas locais ponderadas pela máscara (equivalente a excluir
    # pixels de fundo da média/variância local)
    count = uniform_filter(mask, size=window_size)
    count_safe = np.where(count > 0, count, 1)  # evita divisão por zero

    sum_img = uniform_filter(img * mask, size=window_size)
    sum_sq = uniform_filter((img ** 2) * mask, size=window_size)

    mean = sum_img / count_safe
    mean_sq = sum_sq / count_safe
    variance = np.clip(mean_sq - mean ** 2, 0, None)

    roughness = np.sqrt(variance)
    roughness[mask == 0] = 0  # zera o fundo explicitamente

    return roughness
#======================================================================
import numpy as np
from skimage.filters import frangi
from scipy.ndimage import distance_transform_edt

def apply_frangi(img_1b, mask=None, bg_value=0.0, sigmas=range(1, 6, 1),
                  black_ridges=False, alpha=0.5, beta=0.5, gamma=0.05):
    """
    Frangi com tratamento de borda: preenche o fundo por extrapolação
    suave (nearest-neighbor da folha) antes de filtrar, evitando que a
    transição folha/fundo domine a normalização interna do filtro.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se None, é inferida automaticamente
        via img_1b != bg_value.
    bg_value : float
        Valor considerado "fundo" quando mask=None (ajuste conforme sua
        segmentação — parece ser 0 pelas imagens que você mostrou).
    gamma : float
        Fixo manualmente (em vez de automático) — controla sensibilidade
        a contraste. Comece baixo (0.01–0.1); o valor automático do
        skimage costuma ser dominado pela borda folha/fundo.

    Retorna
    -------
    np.ndarray
        Mapa de vesselness, mesma dimensão de img_1b, zerado fora da folha.
    """
    img = img_1b.astype(np.float64)

    if mask is None:
        mask = img != bg_value
    mask = mask.astype(bool)

    if not mask.any():
        raise ValueError("Máscara vazia — nenhum pixel de folha detectado.")

    img_min, img_max = np.nanmin(img[mask]), np.nanmax(img[mask])
    img_norm = (img - img_min) / (img_max - img_min)

    # Preenche o fundo com o valor do pixel de folha mais próximo,
    # eliminando o salto abrupto folha->fundo antes de filtrar
    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    img_filled = img_norm[iy, ix]
    img_filled[mask] = img_norm[mask]

    vesselness = frangi(
        img_filled, sigmas=sigmas, alpha=alpha, beta=beta,
        gamma=gamma, black_ridges=black_ridges
    )

    vesselness = vesselness * mask.astype(np.float32)
    return vesselness.astype(np.float32)

#======================================================================
# sato, meijering

import numpy as np
from skimage.filters import sato, meijering
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


def apply_sato(img_1b, mask=None, bg_value=0.0, sigmas=range(1, 6, 1),
                black_ridges=False):
    """
    Aplica o filtro de Sato (tubeness) em uma banda única.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se None, inferida via img_1b != bg_value.
    bg_value : float
        Valor considerado "fundo" quando mask=None.
    sigmas : iterable de floats
        Escalas de detecção — mesma lógica do Frangi (sigmas pequenos
        para nervuras finas, maiores para nervuras principais).
    black_ridges : bool
        True = detecta estruturas mais escuras; False = mais claras
        que o entorno. Teste os dois.

    Retorna
    -------
    np.ndarray
        Mapa de tubeness, mesma dimensão de img_1b, zerado fora da folha.
    """
    img_filled, mask = _prepare_filled_image(img_1b, mask, bg_value)

    result = sato(img_filled, sigmas=sigmas, black_ridges=black_ridges)
    result = result * mask.astype(np.float32)

    return result.astype(np.float32)


def apply_meijering(img_1b, mask=None, bg_value=0.0, sigmas=range(1, 6, 1),
                     alpha=None, black_ridges=False):
    """
    Aplica o filtro de Meijering (neuriteness) em uma banda única.

    Parâmetros
    ----------
    img_1b : np.ndarray
        Imagem 2D de uma única banda.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se None, inferida via img_1b != bg_value.
    bg_value : float
        Valor considerado "fundo" quando mask=None.
    sigmas : iterable de floats
        Escalas de detecção.
    alpha : float ou None
        Controla a forma do filtro (ajuste de plate-like vs line-like).
        None usa o padrão do skimage.
    black_ridges : bool
        True = detecta estruturas mais escuras; False = mais claras
        que o entorno. Teste os dois.

    Retorna
    -------
    np.ndarray
        Mapa de neuriteness, mesma dimensão de img_1b, zerado fora da folha.
    """
    img_filled, mask = _prepare_filled_image(img_1b, mask, bg_value)

    result = meijering(img_filled, sigmas=sigmas, alpha=alpha,
                        black_ridges=black_ridges)
    result = result * mask.astype(np.float32)

    return result.astype(np.float32)

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

#======================================================================
# skeleton

import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_otsu


def apply_skeletonize(enhanced_map, mask=None, threshold=None,
                       min_size=20, method='zhang'):
    """
    Binariza um mapa de estruturas já realçado (ex: saída de Frangi,
    Sato, Meijering ou top-hat) e extrai o esqueleto (linha de 1 pixel
    de espessura) das nervuras/estruturas detectadas.

    Parâmetros
    ----------
    enhanced_map : np.ndarray
        Mapa 2D já realçado (ex: vessel_map, sato_map, tophat_map das
        funções anteriores) — NÃO a banda bruta.
    mask : np.ndarray booleana ou None
        True = folha, False = fundo. Se fornecida, garante que nenhuma
        estrutura seja detectada fora da folha (redundante se o mapa
        de entrada já veio zerado no fundo, mas seguro manter).
    threshold : float ou None
        Limiar para binarização. Se None, é calculado automaticamente
        via método de Otsu — mas Otsu pode falhar se a imagem tiver
        muitos pixels de fundo zerados (bimodalidade artificial). Se o
        resultado vier ruim, defina manualmente (ex: percentil da
        distribuição de valores não-zero).
    min_size : int
        Remove componentes conectados menores que este número de
        pixels antes de esqueletizar — elimina ruído pontual que
        geraria "fiapos" isolados no esqueleto.
    method : str
        'zhang' (padrão, só 2D) ou 'lee'. Zhang costuma dar esqueletos
        mais finos e estáveis para imagens 2D.

    Retorna
    -------
    skeleton : np.ndarray (bool)
        Máscara binária do esqueleto, mesma dimensão de enhanced_map.
    binary : np.ndarray (bool)
        A máscara binarizada intermediária (antes de esqueletizar) —
        útil para debug/visualização, ver se o threshold está bom.
    """
    img = enhanced_map.astype(np.float64)

    if mask is not None:
        img = img * mask.astype(np.float64)

    # 1. Determina o limiar
    if threshold is None:
        nonzero = img[img > 0]
        if nonzero.size == 0:
            raise ValueError("Mapa vazio — nenhum valor positivo para binarizar.")
        threshold = threshold_otsu(nonzero)

    # 2. Binariza
    binary = img > threshold

    # 3. Remove ruído pontual pequeno antes de esqueletizar
    if min_size > 0:
        binary = remove_small_objects(binary, min_size=min_size)

    # 4. Esqueletiza
    skeleton = skeletonize(binary, method=method)

    return skeleton, binary


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

#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================


import numpy as np
from scipy.ndimage import distance_transform_edt


def stretch_border_to_background(img, size=10):
    """
    Estica os valores da borda da região segmentada para dentro
    do fundo, sem modificar nenhum pixel original da planta.

    Parameters
    ----------
    img : np.ndarray
        Imagem segmentada:
        (H, W) ou (H, W, C).
        Fundo deve ser zero.

    size : int
        Quantidade de pixels que a borda será expandida
        para dentro do fundo.

    Returns
    -------
    result : np.ndarray
        Imagem com a borda expandida.
    """

    # Máscara da planta
    if img.ndim == 2:
        mask = img != 0

    elif img.ndim == 3:
        mask = np.any(img != 0, axis=2)

    else:
        raise ValueError("img deve ter dimensão (H,W) ou (H,W,C)")

    # Distância dos pixels do fundo até a planta.
    # indices fornece a posição do pixel da planta mais próximo.
    distance, indices = distance_transform_edt(
        ~mask,
        return_indices=True
    )

    # Fundo que será preenchido
    expand = (~mask) & (distance <= size)

    result = img.copy()

    rows = indices[0]
    cols = indices[1]

    if img.ndim == 2:
        result[expand] = img[rows[expand], cols[expand]]

    else:
        result[expand, :] = img[
            rows[expand],
            cols[expand],
            :
        ]

    return result



import numpy as np


def shuffle_patches(img, grid=(4, 4), seed=None):
    """
    Divide a imagem em blocos e embaralha suas posições.

    Parameters
    ----------
    img : np.ndarray
        Imagem (H, W) ou (H, W, C).

    grid : tuple
        Número de divisões (linhas, colunas).
        Exemplo: (4, 4) gera 16 blocos.

    seed : int ou None
        Seed para tornar o embaralhamento reproduzível.

    Returns
    -------
    shuffled : np.ndarray
        Imagem com os blocos embaralhados.
    """

    rows, cols = grid
    H, W = img.shape[:2]

    # Limites dos blocos
    y_edges = np.linspace(0, H, rows + 1, dtype=int)
    x_edges = np.linspace(0, W, cols + 1, dtype=int)

    # Para permitir embaralhamento direto, os blocos
    # precisam ter o mesmo tamanho.
    if H % rows != 0 or W % cols != 0:
        raise ValueError(
            f"A dimensão {(H, W)} deve ser divisível por {grid}."
        )

    patches = []

    for i in range(rows):
        for j in range(cols):
            patch = img[
                y_edges[i]:y_edges[i + 1],
                x_edges[j]:x_edges[j + 1],
                ...
            ].copy()

            patches.append(patch)

    # Embaralhamento
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(patches))

    shuffled = np.empty_like(img)

    k = 0
    for i in range(rows):
        for j in range(cols):

            patch = patches[permutation[k]]

            shuffled[
                y_edges[i]:y_edges[i + 1],
                x_edges[j]:x_edges[j + 1],
                ...
            ] = patch

            k += 1

    return shuffled




