import os
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
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue

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

#----------------------------------------------------------------------

def step_suppress_rgb_color(image: np.ndarray = None, discolor_list: list = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1], return_list=False):

    if return_list:
        return discolor_list

    imgs_list = []

    for discolor in discolor_list:
        imgs_list.append(suppress_rgb_color(image, discolor))

    return imgs_list

#----------------------------------------------------------------------

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


def suppress_texture_all_levels(image: np.ndarray) -> dict[str, np.ndarray]:
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

def suppress_texture_mask_aware(
    image: np.ndarray,
    sigma: float = 5
) -> np.ndarray:
    """
    Aplica supressão de textura via low-pass gaussiano, ignorando o
    background da imagem.

    Um pixel é considerado background somente quando todas as 5 bandas
    possuem valor zero:

        [B, G, R, NIR, RE] == [0, 0, 0, 0, 0]

    O filtro gaussiano é aplicado independentemente em cada banda,
    considerando apenas pixels pertencentes à imagem (foreground).
    Pixels de background não contribuem para o cálculo do blur e
    permanecem com valor zero no resultado.

    Isso evita que valores zero do background reduzam artificialmente
    os valores dos pixels próximos às bordas do foreground.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem
        [B, G, R, NIR, RE].

    sigma : float
        Desvio-padrão do filtro gaussiano, em pixels.
        Quanto maior, mais forte a supressão de textura.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), com o mesmo dtype da entrada.
        O background permanece zero.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if sigma <= 0:
        raise ValueError(
            f"sigma deve ser > 0, recebido {sigma}"
        )

    orig_dtype = image.dtype
    image_f = image.astype(np.float64)

    # Foreground: pixel em que pelo menos uma das 5 bandas é diferente de zero.
    # Background: todas as 5 bandas são zero.
    mask = np.any(image != 0, axis=-1)

    # Máscara em float para utilização no filtro.
    mask_f = mask.astype(np.float64)

    # Gaussian blur da máscara.
    #
    # Isso representa, para cada posição, quanto da vizinhança
    # considerada pelo filtro pertence ao foreground.
    blurred_mask = gaussian_filter(
        mask_f,
        sigma=sigma
    )

    blurred = np.zeros_like(image_f)

    for band in range(image.shape[-1]):

        # Remove explicitamente o background antes do blur.
        # Isso é importante porque somente os pixels válidos devem
        # contribuir para o resultado.
        weighted_band = image_f[..., band] * mask_f

        # Blur dos valores da banda.
        blurred_values = gaussian_filter(
            weighted_band,
            sigma=sigma
        )

        # Normalização pela quantidade efetiva de foreground
        # considerada pelo filtro.
        #
        # Onde blurred_mask > 0:
        #
        #     resultado = blur(valores * máscara) / blur(máscara)
        #
        # Isso impede que o background zero seja contabilizado
        # como parte da média.
        np.divide(
            blurred_values,
            blurred_mask,
            out=blurred[..., band],
            where=blurred_mask > 0
        )

    # Garante que o background original continue exatamente zero.
    blurred[~mask] = 0

    # Retorna aos limites do dtype original.
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        blurred = np.clip(
            blurred,
            info.min,
            info.max
        )

    return blurred.astype(orig_dtype)


#======================================================================
#======================================================================
# Shape

def suppress_shape(image: np.ndarray, patch_size: int = 128, seed: int | None = 42) -> np.ndarray:
    """
    Aplica supressão de shape / organização espacial (item 5.2) via
    patch shuffle, usando a MESMA permutação de patches nas cinco bandas.

    Cada posição da planta mantém seu vetor espectral [B, G, R, NIR, RE]
    associado (pois a permutação é idêntica entre bandas), mas a
    organização espacial global é destruída ao embaralhar os patches.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem [B, G, R, NIR, RE].
    patch_size : int
        Tamanho do lado do patch quadrado (em pixels). Quanto menor,
        mais a forma global é destruída; quanto maior, mais estrutura
        global é preservada.
    seed : int, opcional
        Semente para reprodutibilidade da permutação.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), mesmo dtype de entrada, com patches
        reordenados (mesma ordem em todas as bandas).
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size deve ser > 0, recebido {patch_size}")

    H, W, C = image.shape
    orig_dtype = image.dtype

    # Padding para que H e W sejam múltiplos de patch_size.
    # Usamos reflect para não introduzir bordas artificiais (zeros)
    # que criariam um distribution shift extra e evitável.
    pad_h = (-H) % patch_size
    pad_w = (-W) % patch_size
    padded = np.pad(
        image,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )
    Hp, Wp, _ = padded.shape
    n_rows = Hp // patch_size
    n_cols = Wp // patch_size
    n_patches = n_rows * n_cols

    # Reorganiza em (n_patches, patch_size, patch_size, C)
    patches = (
        padded
        .reshape(n_rows, patch_size, n_cols, patch_size, C)
        .transpose(0, 2, 1, 3, 4)          # (n_rows, n_cols, ph, pw, C)
        .reshape(n_patches, patch_size, patch_size, C)
    )

    # Uma única permutação, aplicada igualmente a todas as bandas
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_patches)
    shuffled_patches = patches[perm]

    # Reconstrói a imagem a partir dos patches embaralhados
    shuffled = (
        shuffled_patches
        .reshape(n_rows, n_cols, patch_size, patch_size, C)
        .transpose(0, 2, 1, 3, 4)          # (n_rows, ph, n_cols, pw, C)
        .reshape(Hp, Wp, C)
    )

    # Remove o padding, voltando ao tamanho original
    result = shuffled[:H, :W, :]

    return result.astype(orig_dtype)


# Tamanhos de referência para gerar a curva
# pequeno -> forma bastante destruída, textura local preservada
# médio
# grande -> mais estrutura global preservada
PATCH_SIZE_LEVELS = {
    "pequeno": 16,
    "medio": 64,
    "grande": 128,
}


def suppress_shape_all_levels(image: np.ndarray, seed: int | None = None) -> dict[str, np.ndarray]:
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
        level: suppress_shape(image, patch_size=ps, seed=seed)
        for level, ps in PATCH_SIZE_LEVELS.items()
    }

#----------------------------------------------------------------------

def step_suppress_shape(image: np.ndarray = None, patch_size_list: list = [None, 1024, 512, 256, 128, 64, 32], return_list=False):

    if return_list:
        return patch_size_list

    imgs_list = []

    for patch_size in patch_size_list:
        if patch_size is None:
            imgs_list.append(image)
        else:
            imgs_list.append(suppress_shape(image, patch_size))

    return imgs_list

#----------------------------------------------------------------------




#======================================================================
#======================================================================
# Alignment

def suppress_band_alignment(
    image: np.ndarray,
    shifts: list[tuple[int, int]] | None = None,
    max_shift: int = 12,
    mode: str = "reflect",
    seed: int | None = None,
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


def suppress_non_visible_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Aplica supressão do espectro não visível (item 5.5), zerando as
    bandas NIR e Red Edge (equivalente a substituí-las por sua média,
    já que as imagens estão normalizadas com média ~0).

    Resultado: [B, G, R, 0, 0]

    Remove a informação discriminativa espacial específica de NIR e RE,
    mantendo RGB e toda a organização espacial das bandas visíveis.

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem [B, G, R, NIR, RE].

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), mesmo dtype de entrada, com as
        bandas NIR e RE zeradas.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")

    result = image.copy()
    result[..., 3] = 0  # NIR
    result[..., 4] = 0  # Red Edge

    return result


#======================================================================



def suppress_spatial_organization(image: np.ndarray, seed: int | None = None) -> np.ndarray:
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

def suppress_non_visible_spectrum(
    image: np.ndarray,
    suppression: float = 1.0
) -> np.ndarray:
    """
    Aplica supressão linear do espectro não visível, reduzindo as
    bandas NIR e Red Edge.

    A intensidade da supressão é controlada pelo parâmetro `suppression`:

        0.0 = nenhuma supressão (imagem original)
        1.0 = supressão completa de NIR e Red Edge
        valores intermediários = supressão linear

    A transformação aplicada é:

        NIR' = (1 - suppression) * NIR
        RE'  = (1 - suppression) * RE

    As bandas visíveis B, G e R permanecem inalteradas.

    Resultado:
        [B, G, R, NIR', RE']

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), bandas na ordem
        [B, G, R, NIR, RE].

    suppression : float, default=1.0
        Intensidade da supressão do espectro não visível.
        Deve estar no intervalo [0, 1].

        0 = imagem original.
        1 = NIR e RE completamente zerados.

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5), com o mesmo dtype da entrada.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {image.shape}"
        )

    if not (0.0 <= suppression <= 1.0):
        raise ValueError(
            f"suppression deve estar em [0, 1], recebido {suppression}"
        )

    orig_dtype = image.dtype

    # Trabalha em float para permitir valores intermediários
    result = image.astype(np.float64).copy()

    # Supressão linear das bandas não visíveis
    factor = 1.0 - suppression

    result[..., 3] *= factor  # NIR
    result[..., 4] *= factor  # Red Edge

    # Garante os limites do dtype original
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(result, info.min, info.max)

    return result.astype(orig_dtype)

#======================================================================

import numpy as np
from scipy.ndimage import gaussian_filter


def suppress_local_shape_contour(
    img: np.ndarray,
    sigma: float = 10.0
) -> np.ndarray:
    """
    Suprime progressivamente o shape local suavizando o contorno
    da planta.

    A transformação atua sobre a máscara da planta, suavizando
    irregularidades locais do contorno (serrilhamentos, pequenas
    pontas, concavidades etc.), procurando preservar o shape global.

    Funciona com imagens segmentadas normalizadas, pois utiliza
    mask_from_segmented() para obter a máscara.

    Parameters
    ----------
    img : np.ndarray
        Imagem segmentada de shape (H, W, 5).

    sigma : float, default=10.0
        Intensidade da suavização do contorno.

        Valores sugeridos para 6 níveis:
            1  -> muito fraco
            2  -> fraco
            4  -> moderado
            6  -> médio-forte
            10 -> forte (default)
            15 -> muito forte

    Returns
    -------
    np.ndarray
        Imagem com o contorno suavizado, mantendo o mesmo shape
        e dtype da entrada.
    """

    if img.ndim != 3 or img.shape[-1] != 5:
        raise ValueError(
            f"Esperado shape (H, W, 5), recebido {img.shape}"
        )

    if sigma <= 0:
        raise ValueError(
            f"sigma deve ser > 0, recebido {sigma}"
        )

    orig_dtype = img.dtype

    # ---------------------------------------------------------
    # 1. Obtém máscara 2D da planta
    # ---------------------------------------------------------
    mask = mask_from_segmented(img)[..., 0].astype(np.float64)

    # ---------------------------------------------------------
    # 2. Suaviza espacialmente a máscara
    # ---------------------------------------------------------
    smooth_mask = gaussian_filter(
        mask,
        sigma=sigma
    )

    # ---------------------------------------------------------
    # 3. Binariza novamente
    #
    # O limiar 0.5 produz uma nova fronteira mais suave.
    # ---------------------------------------------------------
    new_mask = smooth_mask >= 0.5

    # ---------------------------------------------------------
    # 4. Background original de cada banda
    #
    # Importante para imagens normalizadas:
    # o fundo NÃO é necessariamente zero.
    # ---------------------------------------------------------
    background_values = np.min(
        img,
        axis=(0, 1)
    )

    # ---------------------------------------------------------
    # 5. Constrói imagem resultante
    # ---------------------------------------------------------
    result = img.copy()

    # Pixels removidos pela suavização
    removed = (mask > 0) & (~new_mask)

    # Coloca nesses pixels o background correspondente
    # de cada banda.
    result[removed] = background_values

    # ---------------------------------------------------------
    # 6. Pixels eventualmente adicionados pela nova máscara
    #
    # Não inventamos RGB/NIR/RE para regiões que originalmente
    # eram background. Portanto eles permanecem background.
    # ---------------------------------------------------------

    return result.astype(orig_dtype)

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

from scipy import ndimage
import numpy as np


def count_connected_components(img: np.ndarray) -> int:
    """
    Conta o número de componentes conexas de uma imagem segmentada.

    Parameters
    ----------
    img : np.ndarray
        Imagem segmentada nos formatos:
            - (X, Y): imagem de uma banda
            - (X, Y, N): imagem com N bandas

        Para imagens multibanda, um pixel é considerado background
        somente quando todas as suas bandas possuem valor 0.

        Um pixel pertence ao foreground quando pelo menos uma de suas
        bandas possui valor diferente de 0.

    Returns
    -------
    int
        Número de componentes conexas presentes no foreground da imagem.

    Notes
    -----
    É utilizada conectividade 8, ou seja, pixels conectados pelas
    laterais ou pelas diagonais pertencem à mesma componente.
    """

    # Validação
    if not isinstance(img, np.ndarray):
        raise TypeError("img deve ser um numpy.ndarray.")

    if img.ndim not in (2, 3):
        raise ValueError(
            "img deve possuir dimensão (X, Y) ou (X, Y, N). "
            f"Recebido: {img.shape}"
        )

    # Cria máscara 2D de foreground
    if img.ndim == 2:
        mask = img != 0
    else:
        # Foreground se pelo menos uma banda for diferente de zero
        mask = np.any(img != 0, axis=-1)

    # Conectividade 8
    structure = np.ones((3, 3), dtype=np.uint8)

    # Identifica e conta as componentes conexas
    _, num_components = ndimage.label(
        mask,
        structure=structure
    )

    return int(num_components)


#======================================================================
import numpy as np

def mask_from_segmented(img_seg):
    """
    Retorna uma máscara com o mesmo shape da imagem segmentada.

    Funciona tanto para imagens originais quanto normalizadas.

    O fundo é identificado como os pixels que possuem,
    simultaneamente, o valor mínimo de cada uma das 5 bandas.

    1 = planta
    0 = fundo
    """

    if img_seg.ndim != 3 or img_seg.shape[2] != 5:
        raise ValueError(
            f"Esperado shape (H, W, 5), recebido {img_seg.shape}"
        )

    # Valor do background em cada banda
    background_values = np.min(img_seg, axis=(0, 1))

    # Background = mínimo simultaneamente nas 5 bandas
    background = np.all(
        np.isclose(
            img_seg,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=2
    )

    # Planta = tudo que não é background
    mask_2d = ~background

    # Replica a máscara nas 5 bandas
    mask_5d = np.repeat(
        mask_2d[:, :, np.newaxis],
        5,
        axis=2
    )

    return mask_5d.astype(np.uint8)



#======================================================================
#======================================================================
#======================================================================

from tqdm import tqdm

def features_from_transformations(DATA_DIR, metric, comparative=False):

    especies = sorted(os.listdir(DATA_DIR))

    color_list = []
    texture_list = []
    shape_list = []

    for especie in tqdm(especies):    # especie = especies[1]

        especie_dir = os.path.join(DATA_DIR, especie)

        files_dir = os.listdir(especie_dir)

        for file_name in files_dir:  # file_name = files_dir[0]

            file_dir = os.path.join(especie_dir, file_name)

            img = np.load(file_dir).astype(np.float32, copy=False)

            imgs_color = step_suppress_rgb_color(img)
            imgs_texture = step_suppress_texture(img)
            imgs_shape = step_suppress_shape(img)

            if comparative:
                color_list.append([metric(img, x) for x in imgs_color])
                texture_list.append([metric(img, x) for x in imgs_texture])
                shape_list.append([metric(img, x) for x in imgs_shape])
            else:
                color_list.append([metric(x) for x in imgs_color])
                texture_list.append([metric(x) for x in imgs_texture])
                shape_list.append([metric(x) for x in imgs_shape])
    
    color_array = np.array(color_list, dtype="float32")
    texture_array = np.array(texture_list, dtype="float32")
    shape_array = np.array(shape_list, dtype="float32")

    return (color_array, texture_array, shape_array)











