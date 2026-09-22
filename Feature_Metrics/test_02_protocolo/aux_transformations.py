from pathlib import Path
import numpy as np
import pandas as pd

from time import sleep
from copy import deepcopy

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Transformations ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# Shape

import numpy as np


def suppress_patch_shuffle(
    image: np.ndarray,
    grid_size: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Suprime informação de shape / organização espacial por Patch Shuffle.

    A imagem é dividida em uma grade grid_size x grid_size e os patches
    são permutados aleatoriamente.

    A MESMA permutação espacial é aplicada simultaneamente a todas
    as bandas, preservando o vetor espectral de cada pixel.

    Quanto maior o grid_size:
        - menores os patches;
        - maior a quebra da organização espacial;
        - maior, em geral, a intensidade da supressão de shape.

    Parameters
    ----------
    image : np.ndarray
        Imagem multibanda no formato (H, W, C).

    grid_size : int
        Número de patches em cada eixo.

        Exemplos:
            grid_size = 2  -> supressão fraca
            grid_size = 3
            grid_size = 4
            grid_size = 5
            grid_size = 6
            grid_size = 8  -> supressão forte

    seed : int
        Semente da permutação, garantindo reprodutibilidade.

    Returns
    -------
    np.ndarray
        Imagem com shape (H, W, C), com os patches embaralhados.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Esperado array (H, W, C), recebido {image.shape}"
        )


    if grid_size == 0:
        return image
    
    H, W, C = image.shape
    orig_dtype = image.dtype

    # Tamanho necessário para que H e W sejam divisíveis por grid_size
    patch_h = int(np.ceil(H / grid_size))
    patch_w = int(np.ceil(W / grid_size))

    Hp = patch_h * grid_size
    Wp = patch_w * grid_size

    pad_h = Hp - H
    pad_w = Wp - W

    # Padding apenas quando necessário
    padded = np.pad(
        image,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )

    # ---------------------------------------------------------
    # Divide em grid_size x grid_size patches
    # ---------------------------------------------------------

    patches = (
        padded
        .reshape(
            grid_size,
            patch_h,
            grid_size,
            patch_w,
            C
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(
            grid_size * grid_size,
            patch_h,
            patch_w,
            C
        )
    )

    # ---------------------------------------------------------
    # Embaralha os patches
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(
        grid_size * grid_size
    )

    patches = patches[permutation]

    # ---------------------------------------------------------
    # Reconstrói a imagem
    # ---------------------------------------------------------

    shuffled = (
        patches
        .reshape(
            grid_size,
            grid_size,
            patch_h,
            patch_w,
            C
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(Hp, Wp, C)
    )

    # Remove padding
    shuffled = shuffled[:H, :W, :]

    return shuffled.astype(orig_dtype)


# Tamanhos de referência para gerar a curva
# pequeno -> forma bastante destruída, textura local preservada
# médio
# grande -> mais estrutura global preservada
PATCH_SIZE_LEVELS = {
    "pequeno": 16,
    "medio": 64,
    "grande": 128,
}


def suppress_shape_all_levels(image: np.ndarray, seed: int = 42):
    """
    Aplica suppress_shape em todos os níveis pré-definidos de patch size.

    Note: usa a mesma seed para todos os níveis por padrão, mas como o
    número de patches muda com patch_size, as permutações resultantes
    são naturalmente diferentes entre níveis (não é um problema).

    Returns
    -------
    dict[str, np.ndarray]
        Chaves: "pequeno", "medio", "grande" -> arrays (H, W, 5).
    """
    return {
        level: suppress_patch_shuffle(image, patch_size=ps, seed=seed)
        for level, ps in PATCH_SIZE_LEVELS.items()
    }

#----------------------------------------------------------------------

def step_suppress_patch_shuffle(image: np.ndarray = None, patch_size_list: list = [0, 2, 3, 4, 6, 8, 12], return_list=False):

    if return_list:
        return patch_size_list

    imgs_list = []

    for patch_size in patch_size_list:
        if patch_size is None or patch_size == 0:
            imgs_list.append(image)
        else:
            imgs_list.append(suppress_patch_shuffle(image, patch_size))

    return imgs_list


#======================================================================
# Patch Rotation

def suppress_patch_rotation(
    image: np.ndarray,
    grid_size: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Suprime informação de shape / continuidade espacial via Patch Rotation.

    A imagem é dividida em uma grade grid_size x grid_size.
    Cada patch é rotacionado independentemente por um ângulo aleatório
    escolhido entre {0, 90, 180, 270} graus.

    A rotação é aplicada simultaneamente a todas as bandas, preservando
    o vetor espectral de cada pixel.

    Quanto maior o grid_size:
        - menores os patches;
        - mais local é a perturbação;
        - maior a quebra da continuidade de bordas e estruturas locais.

    Parameters
    ----------
    image : np.ndarray
        Imagem multibanda no formato (H, W, C).

    grid_size : int
        Número de células da grade em cada eixo.

    seed : int
        Semente para garantir reprodutibilidade.

    Returns
    -------
    np.ndarray
        Imagem transformada, com o mesmo shape e dtype da entrada.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Esperado array (H, W, C), recebido {image.shape}"
        )


    if grid_size == 0:
        return image

    H, W, C = image.shape
    orig_dtype = image.dtype

    # Para permitir rotações de 90° sem interpolação,
    # utilizamos patches quadrados.
    patch_size = int(
        np.ceil(max(H, W) / grid_size)
    )

    Hp = patch_size * grid_size
    Wp = patch_size * grid_size

    pad_h = Hp - H
    pad_w = Wp - W

    padded = np.pad(
        image,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )

    result = padded.copy()

    rng = np.random.default_rng(seed)

    # Cada patch recebe uma rotação independente
    for i in range(grid_size):
        for j in range(grid_size):

            y0 = i * patch_size
            y1 = (i + 1) * patch_size

            x0 = j * patch_size
            x1 = (j + 1) * patch_size

            patch = padded[y0:y1, x0:x1, :]

            # k = 0,1,2,3
            # corresponde a 0°, 90°, 180°, 270°
            k = rng.integers(0, 4)

            rotated_patch = np.rot90(
                patch,
                k=k,
                axes=(0, 1)
            )

            result[y0:y1, x0:x1, :] = rotated_patch

    # Remove padding
    result = result[:H, :W, :]

    return result.astype(orig_dtype)


#----------------------------------------------------------------------

def step_suppress_patch_rotation(image: np.ndarray = None, patch_size_list: list = [0, 2, 3, 4, 6, 8, 12], return_list=False):

    if return_list:
        return patch_size_list

    imgs_list = []

    for patch_size in patch_size_list:
        if patch_size is None or patch_size == 0:
            imgs_list.append(image)
        else:
            imgs_list.append(suppress_patch_rotation(image, patch_size))

    return imgs_list



#======================================================================
#======================================================================
#======================================================================
# Texture

import numpy as np
from scipy.ndimage import gaussian_filter

# Níveis de blur pré-definidos (sigma do filtro gaussiano, em pixels)
BLUR_LEVELS = {
    "leve": 1.0,
    "medio": 3.0,
    "forte": 6.0,
}


def suppress_texture(image: np.ndarray, sigma: float = 5) -> np.ndarray:
    """
    Aplica supressão de textura (item 5.1) via low-pass gaussiano
    em cada banda independentemente.

    Reduz componentes de alta frequência (detalhes/textura local),
    preservando aproximadamente a estrutura global e as tendências
    espectrais de baixa frequência de cada banda.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem [B, G, R, NIR, RE].
    sigma : float
        Desvio-padrão do filtro gaussiano (em pixels). Quanto maior,
        mais forte o blur. Use os valores de referência em BLUR_LEVELS
        ou um valor customizado.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), mesmo dtype de entrada, com cada
        banda borrada independentemente (sem misturar bandas).
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")
    if sigma <= 0:
        raise ValueError(f"sigma deve ser > 0, recebido {sigma}")

    orig_dtype = image.dtype
    image_f = image.astype(np.float64)

    blurred = np.empty_like(image_f)
    for band in range(image.shape[-1]):
        # sigma aplicado só nos eixos espaciais (H, W), nunca entre bandas
        blurred[..., band] = gaussian_filter(image_f[..., band], sigma=sigma)

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        blurred = np.clip(blurred, info.min, info.max)

    return blurred.astype(orig_dtype)


def suppress_texture_all_levels(image: np.ndarray):
    """
    Aplica suppress_texture em todos os níveis pré-definidos
    (leve, médio, forte), retornando um dicionário com os resultados.

    Útil para gerar a curva "sem blur -> leve -> médio -> forte"
    de queda de desempenho.

    Returns
    -------
    dict[str, np.ndarray]
        Chaves: "leve", "medio", "forte" -> arrays (H, W, 5).
    """
    return {
        level: suppress_texture(image, sigma=s)
        for level, s in BLUR_LEVELS.items()
    }

#----------------------------------------------------------------------

def step_suppress_texture(image: np.ndarray = None, sigma_list: list = [0, 1, 2, 3, 4, 5, 6], return_list=False):

    if return_list:
        return sigma_list

    imgs_list = []

    for sigma in sigma_list:
        if sigma == 0:
            imgs_list.append(image)
        else:
            imgs_list.append(suppress_texture(image, sigma))

    return imgs_list

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def suppress_gaussian_blur_mask_aware(
    image: np.ndarray,
    sigma: float = 5
) -> np.ndarray:
    """
    Aplica supressão de textura via low-pass gaussiano, ignorando o
    background da imagem.

    Funciona tanto para imagens segmentadas originais quanto para
    imagens segmentadas normalizadas por Z-score.

    O background é identificado como os pixels que possuem,
    simultaneamente, o valor mínimo de cada uma das 5 bandas.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem
        [B, G, R, NIR, RE].

    sigma : float
        Desvio-padrão do filtro gaussiano, em pixels.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), com o mesmo dtype da entrada.
        O background mantém exatamente seus valores originais.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if sigma == 0:
        return image



    orig_dtype = image.dtype
    image_f = image.astype(np.float64)

    # Valor correspondente ao background em cada banda.
    background_values = np.min(image_f, axis=(0, 1))

    # Background: pixel possui simultaneamente o mínimo das 5 bandas.
    # isclose evita problemas de precisão em ponto flutuante.
    background_mask = np.all(
        np.isclose(
            image_f,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    # Foreground é o complemento do background.
    mask = ~background_mask
    mask_f = mask.astype(np.float64)

    # Blur da máscara.
    blurred_mask = gaussian_filter(
        mask_f,
        sigma=sigma
    )

    blurred = np.zeros_like(image_f)

    for band in range(image.shape[-1]):

        # Somente foreground contribui para o blur.
        weighted_band = image_f[..., band] * mask_f

        blurred_values = gaussian_filter(
            weighted_band,
            sigma=sigma
        )

        # Média gaussiana considerando apenas foreground.
        np.divide(
            blurred_values,
            blurred_mask,
            out=blurred[..., band],
            where=blurred_mask > 0
        )

    # Restaura exatamente o background da imagem de entrada.
    blurred[background_mask] = image_f[background_mask]

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        blurred = np.clip(
            blurred,
            info.min,
            info.max
        )

    return blurred.astype(orig_dtype)

#----------------------------------------------------------------------

def step_suppress_gaussian_blur_mask_aware(image: np.ndarray = None, sigma_list: list = [0, 1, 2, 3, 4, 5, 6], return_list=False):

    if return_list:
        return sigma_list

    imgs_list = []

    for sigma in sigma_list:
        if sigma == 0:
            imgs_list.append(image)
        else:
            imgs_list.append(suppress_gaussian_blur_mask_aware(image, sigma))

    return imgs_list


#======================================================================
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def suppress_bilateral_filter_mask_aware(
    image: np.ndarray,
    intensity: float = 3.0
) -> np.ndarray:
    """
    Suprime textura usando Bilateral Filter apenas na região da planta.

    Compatível com:
        - imagens segmentadas originais: fundo = 0;
        - imagens segmentadas normalizadas por Z-score:
          fundo = mínimo simultâneo das 5 bandas.

    O parâmetro `intensity` controla conjuntamente:
        - tamanho da vizinhança;
        - sigma espacial;
        - sigma radiométrico.

    Quanto maior `intensity`, maior a suavização da textura.

    Parameters
    ----------
    image : np.ndarray
        Imagem (H, W, 5).

    intensity : float
        Intensidade da supressão. Deve ser > 0.

        Exemplos aproximados:
            0.5 -> muito fraco
            1.0 -> fraco
            2.0 -> moderado
            3.0 -> forte
            5.0 -> muito forte

        Não há limite superior fixo.

    Returns
    -------
    np.ndarray
        Imagem filtrada com mesmo shape e dtype.
        O fundo permanece exatamente igual ao original.
    """

    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if intensity == 0:
        return image

    orig_dtype = image.dtype
    img = image.astype(np.float32)

    H, W, C = img.shape

    # =========================================================
    # Máscara
    # =========================================================

    background_values = np.min(
        img,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    plant_mask = ~background_mask

    if not np.any(plant_mask):
        return image.copy()

    # =========================================================
    # Parâmetros derivados da intensidade
    # =========================================================

    # Diâmetro sempre ímpar.
    d = 2 * int(np.ceil(intensity)) + 1

    sigma_space = 2.0 * intensity

    # Como cada banda será padronizada pelo desvio-padrão
    # da planta, sigma_color torna-se aproximadamente
    # independente da escala radiométrica original.
    sigma_color = 0.35 * intensity

    # =========================================================
    # Preenchimento temporário do background
    # =========================================================
    #
    # Para impedir que o valor do fundo contamine a borda
    # da planta, cada pixel de fundo recebe temporariamente
    # o valor do pixel de planta mais próximo.
    #
    # distance_transform_edt retorna os índices do foreground
    # mais próximo.

    _, nearest_indices = distance_transform_edt(
        background_mask,
        return_indices=True
    )

    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    result = img.copy()

    # =========================================================
    # Bilateral por banda
    # =========================================================

    for band in range(C):

        band_img = img[..., band]

        plant_values = band_img[plant_mask]

        mean = np.mean(plant_values)
        std = np.std(plant_values)

        if std < 1e-12:
            continue

        # -----------------------------------------
        # Padronização apenas para tornar
        # sigma_color comparável entre bandas.
        # -----------------------------------------

        normalized = (
            band_img - mean
        ) / std

        # -----------------------------------------
        # Background temporariamente preenchido
        # pela planta mais próxima.
        # -----------------------------------------

        filled = normalized.copy()

        filled[background_mask] = normalized[
            nearest_y[background_mask],
            nearest_x[background_mask]
        ]

        # -----------------------------------------
        # Bilateral Filter
        # -----------------------------------------

        filtered = cv2.bilateralFilter(
            filled.astype(np.float32),
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            borderType=cv2.BORDER_REFLECT101
        )

        # Retorna à escala original
        filtered = filtered * std + mean

        # Só modifica a planta
        result[..., band][plant_mask] = (
            filtered[plant_mask]
        )

    # =========================================================
    # Restaura fundo exatamente
    # =========================================================

    result[background_mask] = img[background_mask]

    # =========================================================
    # Dtype
    # =========================================================

    if np.issubdtype(orig_dtype, np.integer):

        info = np.iinfo(orig_dtype)

        result = np.clip(
            np.rint(result),
            info.min,
            info.max
        )

    return result.astype(orig_dtype)

#----------------------------------------------------------------------

def step_suppress_bilateral_filter_mask_aware(image: np.ndarray = None, sigma_list: list = [0, 1, 2, 3, 4, 5, 6], return_list=False):

    if return_list:
        return sigma_list

    imgs_list = []

    for sigma in sigma_list:
        if sigma == 0:
            imgs_list.append(image)
        else:
            imgs_list.append(suppress_bilateral_filter_mask_aware(image, sigma))

    return imgs_list


#======================================================================
#======================================================================
#======================================================================
# Color

def suppress_rgb_color(image: np.ndarray, discolor: float = 1.0) -> np.ndarray:
    """
    Aplica a transformação de supressão de cor RGB (item 5.3).

    Recebe uma imagem multiespectral com 5 bandas na ordem:
    [Blue, Green, Red, NIR, RedEdge]

    Converte as bandas B, G, R em uma única banda grayscale e interpola
    entre a imagem original (discolor=0) e a versão totalmente
    dessaturada (discolor=1), preservando NIR e Red Edge inalterados
    em qualquer caso.

    Resultado: [B', G', R', NIR, RE], onde
        canal' = (1 - discolor) * canal_original + discolor * grayscale

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), dtype float ou uint, bandas na ordem
        [B, G, R, NIR, RE].
    discolor : float
        Intensidade da dessaturação, entre 0 e 1.
        0 = imagem original (RGB intacto).
        1 = imagem totalmente grayscale (equivalente à função original).
        Valores intermediários = interpolação linear entre as duas.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5) com mesmo dtype de entrada.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")

    if not (0.0 <= discolor <= 1.0):
        raise ValueError(f"discolor deve estar em [0, 1], recebido {discolor}")

    orig_dtype = image.dtype

    blue  = image[..., 0].astype(np.float64)
    green = image[..., 1].astype(np.float64)
    red   = image[..., 2].astype(np.float64)
    nir   = image[..., 3]
    red_edge = image[..., 4]

    # Pesos de luminosidade padrão (Rec. 601)
    # grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
    grayscale = (1/3) * red + (1/3) * green + (1/3) * blue

    # Interpolação linear entre canal original e grayscale
    blue_out  = (1 - discolor) * blue  + discolor * grayscale
    green_out = (1 - discolor) * green + discolor * grayscale
    red_out   = (1 - discolor) * red   + discolor * grayscale

    # Ajusta dtype de volta ao original (evita overflow/truncamento indevido)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        blue_out  = np.clip(blue_out, info.min, info.max)
        green_out = np.clip(green_out, info.min, info.max)
        red_out   = np.clip(red_out, info.min, info.max)

    blue_out  = blue_out.astype(orig_dtype)
    green_out = green_out.astype(orig_dtype)
    red_out   = red_out.astype(orig_dtype)

    result = np.stack(
        [blue_out, green_out, red_out, nir, red_edge],
        axis=-1
    )

    return result


def suppress_colors(
    image: np.ndarray,
    discolor: float = 1.0
) -> np.ndarray:
    """
    Aplica supressão de cor a uma imagem multibanda.

    Para cada pixel, calcula a média aritmética de todas as bandas e
    interpola cada banda entre seu valor original (discolor=0) e essa
    média (discolor=1).

    Para uma imagem com N bandas:

        mean = (band_1 + band_2 + ... + band_N) / N

        band_i' = (1 - discolor) * band_i + discolor * mean

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, N), onde N >= 1 representa o número
        de bandas da imagem.

    discolor : float, default=1.0
        Intensidade da supressão de cor, entre 0 e 1.

        0 = imagem original.
        1 = todas as bandas recebem a média aritmética das N bandas.
        Valores intermediários realizam uma interpolação linear entre
        cada banda original e a média.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, N), com o mesmo dtype da entrada.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Esperado array (H, W, N), recebido {image.shape}"
        )

    if image.shape[-1] < 1:
        raise ValueError(
            "A imagem deve possuir pelo menos uma banda."
        )

    if not (0.0 <= discolor <= 1.0):
        raise ValueError(
            f"discolor deve estar em [0, 1], recebido {discolor}"
        )

    orig_dtype = image.dtype

    # Converte para float para evitar overflow durante os cálculos
    image_float = image.astype(np.float64)

    # Média aritmética de todas as bandas para cada pixel.
    # keepdims=True mantém shape (H, W, 1), permitindo broadcasting
    # sobre as N bandas.
    mean = np.mean(
        image_float,
        axis=-1,
        keepdims=True
    )

    # Interpolação linear de cada banda em direção à média
    result = (
        (1.0 - discolor) * image_float
        + discolor * mean
    )

    # Garante os limites do dtype original
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(
            result,
            info.min,
            info.max
        )

    return result.astype(orig_dtype)

#----------------------------------------------------------------------

def step_suppress_colors(image: np.ndarray = None, intensity_list: list = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1], return_list=False):

    if return_list:
        return intensity_list

    imgs_list = []

    for intensity in intensity_list:
        imgs_list.append(suppress_colors(image, intensity))

    return imgs_list


#======================================================================
# Channel Shuffle

import numpy as np


def suppress_channel_shuffle(
    image: np.ndarray,
    intensity: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Suprime informação de cor/espectral por Channel Shuffle.

    Os canais são permutados de forma que NENHUM canal permaneça
    em sua posição original.

    A intensidade controla uma combinação linear entre a imagem
    original e a imagem com canais permutados:

        output = (1 - intensity) * original
                 + intensity * shuffled

    intensity = 0.0 -> imagem original
    intensity = 1.0 -> Channel Shuffle completo

    O background permanece exatamente inalterado.

    Compatível com:
        - imagem segmentada original:
              fundo = 0
        - imagem segmentada normalizada por Z-score:
              fundo = mínimo simultâneo das bandas

    Parameters
    ----------
    image : np.ndarray
        Imagem no formato (H, W, C).

    intensity : float
        Intensidade da transformação em [0, 1].

    seed : int
        Seed da permutação.

    Returns
    -------
    np.ndarray
        Imagem transformada com mesmo shape e dtype da entrada.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Esperado array (H, W, C), recebido {image.shape}"
        )

    if not 0.0 <= intensity <= 1.0:
        raise ValueError(
            f"intensity deve estar em [0, 1], recebido {intensity}"
        )

    H, W, C = image.shape

    if C < 2:
        raise ValueError(
            "Channel Shuffle requer pelo menos 2 canais."
        )

    orig_dtype = image.dtype
    img = image.astype(np.float64)

    # ---------------------------------------------------------
    # Identificação do background
    # ---------------------------------------------------------

    background_values = np.min(
        img,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    # ---------------------------------------------------------
    # Derangement dos canais:
    # nenhuma banda fica em sua posição original
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    original_order = np.arange(C)

    while True:
        permutation = rng.permutation(C)

        if np.all(permutation != original_order):
            break

    shuffled = img[..., permutation]

    # ---------------------------------------------------------
    # Combinação linear
    # ---------------------------------------------------------

    result = (
        (1.0 - intensity) * img
        + intensity * shuffled
    )

    # Fundo permanece exatamente igual
    result[background_mask] = img[background_mask]

    # ---------------------------------------------------------
    # Restaura dtype
    # ---------------------------------------------------------

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)

        result = np.clip(
            np.rint(result),
            info.min,
            info.max
        )

    return result.astype(orig_dtype)

#----------------------------------------------------------------------

def step_suppress_channel_shuffle(image: np.ndarray = None, intensity_list: list = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1], return_list=False):

    if return_list:
        return intensity_list

    imgs_list = []

    for intensity in intensity_list:
        imgs_list.append(suppress_channel_shuffle(image, intensity))

    return imgs_list

#======================================================================
import numpy as np


def suppress_rgb_channel_shuffle(
    image: np.ndarray,
    intensity: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Aplica Channel Shuffle somente às bandas RGB.

    Ordem esperada:
        [B, G, R, NIR, RE]

    As bandas B, G e R são permutadas de forma que NENHUMA
    permaneça em sua posição original.

    As bandas NIR e RE permanecem exatamente inalteradas.

    A intensidade controla:

        output_RGB =
            (1 - intensity) * RGB_original
            + intensity * RGB_shuffled

    intensity = 0.0
        -> imagem original.

    intensity = 1.0
        -> shuffle completo de B, G e R.

    O background permanece exatamente inalterado.

    Compatível com:
        - imagens segmentadas originais;
        - imagens segmentadas normalizadas por Z-score.

    Parameters
    ----------
    image : np.ndarray
        Imagem (H, W, 5) com bandas:
        [B, G, R, NIR, RE].

    intensity : float
        Intensidade em [0, 1].

    seed : int
        Seed para escolha da permutação RGB.

    Returns
    -------
    np.ndarray
        Imagem transformada com mesmo shape e dtype.
    """

    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if not 0.0 <= intensity <= 1.0:
        raise ValueError(
            f"intensity deve estar em [0, 1], recebido {intensity}"
        )

    orig_dtype = image.dtype
    img = image.astype(np.float64)

    # =========================================================
    # 1. Background
    # =========================================================

    background_values = np.min(
        img,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    # =========================================================
    # 2. Derangement somente de B, G e R
    # =========================================================
    #
    # As únicas duas permutações de 3 elementos nas quais
    # nenhum elemento permanece na posição original são:
    #
    # [1, 2, 0]
    # [2, 0, 1]
    #

    rng = np.random.default_rng(seed)

    permutations = np.array([
        [1, 2, 0],
        [2, 0, 1]
    ])

    permutation = permutations[
        rng.integers(0, 2)
    ]

    shuffled_rgb = img[..., permutation]

    # =========================================================
    # 3. Combinação linear somente no RGB
    # =========================================================

    result = img.copy()

    result[..., :3] = (
        (1.0 - intensity) * img[..., :3]
        + intensity * shuffled_rgb
    )

    # NIR e RE nunca são modificados:
    # result[..., 3] = img[..., 3]
    # result[..., 4] = img[..., 4]

    # Background exatamente igual
    result[background_mask] = img[background_mask]

    # =========================================================
    # 4. Restaura dtype
    # =========================================================

    if np.issubdtype(orig_dtype, np.integer):

        info = np.iinfo(orig_dtype)

        result = np.clip(
            np.rint(result),
            info.min,
            info.max
        )

    return result.astype(orig_dtype)


#======================================================================
#======================================================================
# Espectral

import numpy as np


def suppress_nir_re_to_mean(
    image: np.ndarray,
    intensity: float = 1.0
) -> np.ndarray:
    """
    Suprime progressivamente a informação espacial das bandas NIR e RE,
    aproximando cada banda da sua própria média na região da planta.

    Bandas esperadas:
        [B, G, R, NIR, RE]

    A transformação é:

        NIR_new = (1 - intensity) * NIR
                  + intensity * mean(NIR)

        RE_new  = (1 - intensity) * RE
                  + intensity * mean(RE)

    onde as médias são calculadas APENAS sobre os pixels da planta.

    As bandas B, G e R permanecem exatamente inalteradas.

    intensity = 0:
        imagem original.

    intensity = 1:
        todos os pixels da planta na banda NIR recebem mean(NIR),
        e todos os pixels da planta na banda RE recebem mean(RE).

    O background permanece exatamente inalterado.

    Compatível com:
        - imagens segmentadas originais:
              background = 0
        - imagens segmentadas normalizadas por Z-score:
              background = mínimo simultâneo das 5 bandas

    Parameters
    ----------
    image : np.ndarray
        Imagem multiespectral com shape (H, W, 5).

    intensity : float
        Intensidade da supressão em [0, 1].

    Returns
    -------
    np.ndarray
        Imagem transformada com mesmo shape e dtype da entrada.
    """

    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if not 0.0 <= intensity <= 1.0:
        raise ValueError(
            f"intensity deve estar em [0, 1], recebido {intensity}"
        )

    orig_dtype = image.dtype
    img = image.astype(np.float64)

    # ---------------------------------------------------------
    # Máscara da planta
    # ---------------------------------------------------------

    background_values = np.min(
        img,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    plant_mask = ~background_mask

    if not np.any(plant_mask):
        return image.copy()

    result = img.copy()

    # Índices:
    # 0 = B
    # 1 = G
    # 2 = R
    # 3 = NIR
    # 4 = RE

    for band in [3, 4]:

        band_values = img[..., band]

        # Média somente sobre a planta
        band_mean = np.mean(
            band_values[plant_mask]
        )

        # Combinação linear somente na planta
        result[..., band][plant_mask] = (
            (1.0 - intensity)
            * band_values[plant_mask]
            + intensity
            * band_mean
        )

    # Fundo exatamente igual ao original
    result[background_mask] = img[background_mask]

    # ---------------------------------------------------------
    # Restaurar dtype
    # ---------------------------------------------------------

    if np.issubdtype(orig_dtype, np.integer):

        info = np.iinfo(orig_dtype)

        result = np.clip(
            np.rint(result),
            info.min,
            info.max
        )

    return result.astype(orig_dtype)

#----------------------------------------------------------------------

def step_suppress_nir_re_to_mean(image: np.ndarray = None, intensity_list: list = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1], return_list=False):

    if return_list:
        return intensity_list

    imgs_list = []

    for intensity in intensity_list:
        imgs_list.append(suppress_nir_re_to_mean(image, intensity))

    return imgs_list

#======================================================================


def suppress_nir_re_to_green(
    image: np.ndarray,
    intensity: float = 1.0
) -> np.ndarray:
    """
    Suprime progressivamente a informação das bandas NIR e RE,
    aproximando ambas da banda Green.

    Bandas esperadas:
        [B, G, R, NIR, RE]

    Transformação:

        NIR_new = (1 - intensity) * NIR
                  + intensity * Green

        RE_new  = (1 - intensity) * RE
                  + intensity * Green

    As bandas B, G e R permanecem exatamente inalteradas.

    intensity = 0:
        imagem original.

    intensity = 1:
        NIR = Green
        RE  = Green

    O background permanece exatamente inalterado.

    Compatível com:
        - imagens segmentadas originais:
              background = 0
        - imagens segmentadas normalizadas por Z-score:
              background = mínimo simultâneo das 5 bandas

    Parameters
    ----------
    image : np.ndarray
        Imagem multiespectral com shape (H, W, 5).

    intensity : float
        Intensidade da supressão em [0, 1].

    Returns
    -------
    np.ndarray
        Imagem transformada com mesmo shape e dtype da entrada.
    """

    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if not 0.0 <= intensity <= 1.0:
        raise ValueError(
            f"intensity deve estar em [0, 1], recebido {intensity}"
        )

    orig_dtype = image.dtype
    img = image.astype(np.float64)

    # ---------------------------------------------------------
    # Máscara da planta
    # ---------------------------------------------------------

    background_values = np.min(
        img,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    plant_mask = ~background_mask

    if not np.any(plant_mask):
        return image.copy()

    result = img.copy()

    # Bandas:
    # 0 = B
    # 1 = G
    # 2 = R
    # 3 = NIR
    # 4 = RE

    green = img[..., 1]

    # NIR -> Green
    result[..., 3][plant_mask] = (
        (1.0 - intensity) * img[..., 3][plant_mask]
        + intensity * green[plant_mask]
    )

    # RE -> Green
    result[..., 4][plant_mask] = (
        (1.0 - intensity) * img[..., 4][plant_mask]
        + intensity * green[plant_mask]
    )

    # Fundo exatamente igual ao original
    result[background_mask] = img[background_mask]

    if np.issubdtype(orig_dtype, np.integer):

        info = np.iinfo(orig_dtype)

        result = np.clip(
            np.rint(result),
            info.min,
            info.max
        )

    return result.astype(orig_dtype)

#----------------------------------------------------------------------

def step_suppress_nir_re_to_green(image: np.ndarray = None, intensity_list: list = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1], return_list=False):

    if return_list:
        return intensity_list

    imgs_list = []

    for intensity in intensity_list:
        imgs_list.append(suppress_nir_re_to_green(image, intensity))

    return imgs_list

#======================================================================
#======================================================================
# Alignment

def suppress_band_alignment(
    image: np.ndarray,
    shifts: list,
    max_shift: int = 12,
    mode: str = "reflect",
    seed: int = 42,
) -> np.ndarray:
    """
    Aplica desalinhamento espacial entre bandas (item 5.4), deslocando
    cada banda por um vetor (dx, dy) diferente.

    Isso quebra a correspondência espacial pixel-a-pixel entre bandas
    (o vetor espectral [B, G, R, NIR, RE] deixa de corresponder à mesma
    região física da planta), mas preserva dentro de cada banda:
    valores originais, histograma, textura interna e forma.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem [B, G, R, NIR, RE].
    shifts : list de 5 tuplas (dx, dy), opcional
        Deslocamento explícito por banda, em pixels. dx > 0 desloca
        para a direita, dy > 0 desloca para baixo. Se None, os
        deslocamentos são sorteados aleatoriamente dentro de
        [-max_shift, max_shift].
    max_shift : int
        Amplitude máxima do deslocamento sorteado (usado só se
        `shifts` não for fornecido).
    mode : str
        Modo de preenchimento da borda exposta pelo deslocamento
        (repassado para np.pad): "reflect" (padrão), "edge", "wrap".
        Evita introduzir zeros artificiais nas bordas.
    seed : int, opcional
        Semente para reprodutibilidade dos deslocamentos sorteados.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), mesmo dtype de entrada, com cada
        banda deslocada independentemente.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")

    H, W, C = image.shape
    orig_dtype = image.dtype

    if shifts is None:
        rng = np.random.default_rng(seed)
        shifts = [
            (int(rng.integers(-max_shift, max_shift + 1)),
             int(rng.integers(-max_shift, max_shift + 1)))
            for _ in range(C)
        ]
    if len(shifts) != C:
        raise ValueError(f"shifts deve ter {C} tuplas (uma por banda), recebido {len(shifts)}")

    result = np.empty_like(image)

    for band, (dx, dy) in enumerate(shifts):
        band_data = image[..., band]

        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)

        padded = np.pad(
            band_data,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=mode,
        )

        # Recorta de volta ao tamanho original, já deslocado
        shifted = padded[
            pad_bottom: pad_bottom + H,
            pad_right: pad_right + W,
        ]

        result[..., band] = shifted

    return result.astype(orig_dtype)


# Configuração de referência sugerida no material de origem
DEFAULT_BAND_SHIFTS = [
    (10, 0),    # Blue    -> 10 px direita
    (0, -5),    # Green   -> 5 px cima
    (-8, 0),    # Red     -> 8 px esquerda
    (0, 12),    # NIR     -> 12 px baixo
    (5, 5),     # Red Edge -> deslocamento diagonal
]


def suppress_band_alignment_default(image: np.ndarray) -> np.ndarray:
    """
    Aplica suppress_band_alignment com os deslocamentos de referência
    sugeridos (equivalentes ao exemplo do material de origem).
    """
    return suppress_band_alignment(image, shifts=DEFAULT_BAND_SHIFTS)

#======================================================================
#======================================================================



def suppress_spatial_organization(image: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Aplica supressão completa da organização espacial (item 5.6) via
    pixel shuffle conjunto: a MESMA permutação aleatória de posições é
    aplicada às cinco bandas simultaneamente.

    Cada vetor espectral [B, G, R, NIR, RE] de um pixel permanece
    intacto (assinatura espectral preservada), mas é realocado para
    uma posição espacial aleatória da imagem.

    Destrói: forma global, contornos, textura, relações de vizinhança
    e toda a organização/localização espacial.

    Preserva: valores originais das cinco bandas, correspondência
    espectral por pixel, e a distribuição global das assinaturas
    espectrais presentes na imagem.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem [B, G, R, NIR, RE].
    seed : int, opcional
        Semente para reprodutibilidade da permutação.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), mesmo dtype de entrada, com os
        pixels (vetores de 5 bandas) reorganizados espacialmente.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")

    H, W, C = image.shape
    orig_dtype = image.dtype

    # Achata para (H*W, C): cada linha é o vetor espectral de um pixel
    flat = image.reshape(-1, C)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.shape[0])

    shuffled_flat = flat[perm]

    result = shuffled_flat.reshape(H, W, C)

    return result.astype(orig_dtype)

#======================================================================
#======================================================================
#======================================================================
