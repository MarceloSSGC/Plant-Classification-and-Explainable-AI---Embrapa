import numpy as np

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Feature Metrics ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# GPT

import numpy as np

def long_range_spatial_organization(
    img_5b: np.ndarray,
    distances=(64, 128, 192, 256),
    directions=((0, 1), (1, 0), (1, 1), (1, -1)),
    eps=1e-12,
) -> float:
    """
    Calcula M1: Long-range Spatial Organization.

    M1 é definido como a autocorrelação espacial média entre pixels
    separados por grandes distâncias, considerando:

        - múltiplas distâncias;
        - múltiplas direções;
        - as 5 bandas espectrais.

    Espera-se que M1 diminua quando a organização espacial global
    é destruída, como no Patch Shuffle.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral com shape (H, W, 5), na ordem:
        [B, G, R, NIR, RE].

    distances : tuple[int]
        Distâncias espaciais, em pixels, usadas para medir
        autocorrelação de longo alcance.

    directions : tuple[tuple[int, int]]
        Direções (dy, dx). Por padrão:
            (0, 1)  -> horizontal
            (1, 0)  -> vertical
            (1, 1)  -> diagonal principal
            (1,-1)  -> diagonal secundária

    eps : float
        Constante para estabilidade numérica.

    Returns
    -------
    float
        M1: autocorrelação espacial média de longo alcance.

        Valores maiores indicam maior organização espacial.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    H, W, C = img.shape

    correlations = []

    for band in range(C):

        I = img[..., band]

        for d in distances:

            for dy, dx in directions:

                shift_y = dy * d
                shift_x = dx * d

                # Região válida da imagem original
                y1_start = max(0, -shift_y)
                y1_end   = min(H, H - shift_y)

                x1_start = max(0, -shift_x)
                x1_end   = min(W, W - shift_x)

                # Região correspondente deslocada
                y2_start = y1_start + shift_y
                y2_end   = y1_end + shift_y

                x2_start = x1_start + shift_x
                x2_end   = x1_end + shift_x

                A = I[
                    y1_start:y1_end,
                    x1_start:x1_end
                ].ravel()

                B = I[
                    y2_start:y2_end,
                    x2_start:x2_end
                ].ravel()

                if A.size < 2:
                    continue

                # Centralização
                A = A - A.mean()
                B = B - B.mean()

                denominator = np.sqrt(
                    np.sum(A ** 2) *
                    np.sum(B ** 2)
                )

                if denominator > eps:

                    rho = np.sum(A * B) / denominator

                    correlations.append(rho)

    if len(correlations) == 0:
        return np.nan

    return float(np.mean(correlations))

#----------------------------------------------------------------------
# Claude

import numpy as np

def long_range_spatial_autocorr(
    img_5b: np.ndarray,
    d0: float = 20.0,
    d_max: float = None,
    n_bins: int = 50,
    channel: str = "luminance",
    bands: str = "BGRNIRRE",
) -> float:
    """
    Métrica M1: autocorrelação espacial de longo alcance.
    Mede organização espacial global via ACF radial (Wiener-Khinchin, FFT-based),
    integrada para distâncias d > d0.

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    d0 : float
        Distância mínima (px) a partir da qual consideramos "longo alcance".
    d_max : float ou None
        Distância máxima a considerar. Default: metade da menor dimensão.
    n_bins : int
        Número de bins radiais de distância.
    channel : str
        "luminance" (combina B,G,R) ou índice de banda específica (0-4).

    Retorna
    -------
    M1 : float
        Área sob |rho(d)| para d > d0 (quanto maior, mais organização
        espacial de longo alcance preservada).
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    # --- 1. Seleciona canal ---
    if channel == "luminance":
        B, G, R = img_5b[..., 0], img_5b[..., 1], img_5b[..., 2]
        img = 0.114 * B + 0.587 * G + 0.299 * R
    else:
        img = img_5b[..., int(channel)]

    img = img.astype(np.float64)
    img = img - img.mean()  # remove DC antes da autocorrelação

    if d_max is None:
        d_max = min(H, W) / 2.0

    # --- 2. Autocorrelação 2D via FFT (Wiener-Khinchin) ---
    # zero-padding para evitar wrap-around (correlação circular)
    Hp, Wp = 2 * H, 2 * W
    F = np.fft.rfft2(img, s=(Hp, Wp))
    power = F * np.conj(F)
    acf = np.fft.irfft2(power, s=(Hp, Wp))
    acf = np.fft.fftshift(acf)  # centraliza o lag zero

    # normaliza para correlação (rho(0) = 1)
    acf = acf / acf.max()

    # --- 3. Mapa de distâncias radiais a partir do centro ---
    cy, cx = Hp // 2, Wp // 2
    y, x = np.indices(acf.shape)
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    # --- 4. Média radial de rho(d) em bins de distância ---
    bin_edges = np.linspace(0, d_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rho_d = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if mask.any():
            rho_d[i] = acf[mask].mean()

    # --- 5. Integra |rho(d)| para d > d0 ---
    valid = bin_centers > d0
    M1 = np.trapz(np.abs(rho_d[valid]), bin_centers[valid])

    return float(M1)

#======================================================================
#======================================================================
#======================================================================

# M_2   

# GPT

import numpy as np
import cv2


def shape_descriptors(img_5b: np.ndarray) -> dict:
    """
    Calcula descritores de forma a partir de uma imagem multiespectral
    segmentada (H, W, 5), assumindo fundo = 0.

    A máscara é inferida considerando como planta todo pixel que possui
    valor diferente de zero em pelo menos uma das 5 bandas.

    Apenas a maior componente conexa é utilizada.

    Retorna:
        - area
        - perimeter
        - compactness / circularity
        - solidity
        - hu_1 ... hu_7

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem com shape (H, W, 5), bandas [B, G, R, NIR, RE].

    Returns
    -------
    dict
        Dicionário contendo os descritores geométricos.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # ------------------------------------------------------------
    # 1. Deduz máscara
    # ------------------------------------------------------------
    mask = np.any(img_5b != 0, axis=-1).astype(np.uint8)

    # ------------------------------------------------------------
    # 2. Mantém apenas maior componente conexa
    # ------------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("Nenhum objeto encontrado na imagem.")

    # label 0 = background
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = 1 + np.argmax(component_areas)

    largest_mask = (
        labels == largest_label
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # 3. Contorno
    # ------------------------------------------------------------
    contours, _ = cv2.findContours(
        largest_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise ValueError("Nenhum contorno encontrado.")

    contour = max(contours, key=cv2.contourArea)

    # ------------------------------------------------------------
    # 4. Área e perímetro
    # ------------------------------------------------------------
    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        closed=True
    )

    # ------------------------------------------------------------
    # 5. Compacidade / circularidade
    #
    # círculo perfeito -> 1
    # formas mais irregulares -> valores menores
    # ------------------------------------------------------------
    if perimeter > 0:
        compactness = (
            4.0 * np.pi * area / (perimeter ** 2)
        )
    else:
        compactness = np.nan

    # ------------------------------------------------------------
    # 6. Solidity
    #
    # área / área do convex hull
    # ------------------------------------------------------------
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = np.nan

    # ------------------------------------------------------------
    # 7. Momentos de Hu
    # ------------------------------------------------------------
    moments = cv2.moments(largest_mask)

    hu = cv2.HuMoments(moments).flatten()

    # Transformação logarítmica para melhorar escala numérica.
    #
    # Os momentos de Hu podem variar por muitas ordens de grandeza.
    hu_log = np.zeros_like(hu)

    for i, value in enumerate(hu):

        if value != 0:
            hu_log[i] = (
                -np.sign(value)
                * np.log10(abs(value))
            )
        else:
            hu_log[i] = 0.0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "compactness": float(compactness),
        "solidity": float(solidity),

        "hu_1": float(hu_log[0]),
        "hu_2": float(hu_log[1]),
        "hu_3": float(hu_log[2]),
        "hu_4": float(hu_log[3]),
        "hu_5": float(hu_log[4]),
        "hu_6": float(hu_log[5]),
        "hu_7": float(hu_log[6]),
    }

#----------------------------------------------------------------------
# Claude

import numpy as np
from scipy import ndimage
import cv2  # usado para momentos de Hu e perímetro (mais robusto que skimage)


def shape_descriptors_cl(
    img_5b: np.ndarray,
    connectivity: int = 2,
    log_hu: bool = True,
) -> dict:
    """
    Métrica M1 (alternativa): descritores de forma da maior componente conexa.
    Deriva a máscara binária a partir de img_5b (fundo = todas as bandas == 0).

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas (B, G, R, NIR, RE). Fundo assumido como vetor nulo.
    connectivity : int
        1 = 4-conectividade, 2 = 8-conectividade.
    log_hu : bool
        Se True, aplica -sign(h)*log10(|h|) aos momentos de Hu
        (forma padrão para estabilizar a escala, já que Hu tem
        magnitudes muito díspares).

    Retorna
    -------
    dict com:
        area, perimeter, compactness (circularity), solidity,
        hu_moments (array de 7 valores)
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    # --- 1. Máscara binária: foreground = qualquer banda != 0 ---
    mask = np.any(img_5b != 0, axis=-1)

    if not mask.any():
        return _empty_descriptors()

    # --- 2. Maior componente conexa ---
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, n_labels = ndimage.label(mask, structure=structure)

    if n_labels == 0:
        return _empty_descriptors()

    sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
    largest_label = np.argmax(sizes) + 1
    component = (labeled == largest_label).astype(np.uint8)

    # --- 3. Contorno externo via OpenCV ---
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return _empty_descriptors()

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if area == 0 or perimeter == 0:
        return _empty_descriptors()

    # --- 4. Compacidade / circularidade: 4*pi*A / P^2 (1.0 = círculo perfeito) ---
    compactness = (4 * np.pi * area) / (perimeter ** 2)

    # --- 5. Solidez: área / área do fecho convexo ---
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # --- 6. Momentos de Hu (invariantes a translação, escala, rotação) ---
    moments = cv2.moments(component, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()  # 7 valores

    if log_hu:
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "compactness": float(compactness),
        "solidity": float(solidity),
        "hu_moments": hu,  # array shape (7,)
    }


def _empty_descriptors() -> dict:
    return {
        "area": 0.0,
        "perimeter": 0.0,
        "compactness": 0.0,
        "solidity": 0.0,
        "hu_moments": np.full(7, np.nan),
    }


def shape_distance(desc_a: dict, desc_b: dict, hu_weight: float = 1.0) -> float:
    """
    Combina os descritores em uma distância escalar (para uso em M1 comparativo,
    ex. delta_on = shape_distance(desc_original, desc_transformada)).

    Combina diferença relativa de compactness/solidity + distância euclidiana
    dos Hu moments (já em log-scale).
    """
    d_compact = abs(desc_a["compactness"] - desc_b["compactness"])
    d_solid = abs(desc_a["solidity"] - desc_b["solidity"])
    d_hu = np.linalg.norm(desc_a["hu_moments"] - desc_b["hu_moments"])

    return float(d_compact + d_solid + hu_weight * d_hu)


#======================================================================

# Claude

import numpy as np
from skimage.metrics import structural_similarity as ssim


def coarse_ssim(
    img_a: np.ndarray,
    img_b: np.ndarray,
    downscale_factor: int = 16,
    channel: str = "luminance",
) -> float:
    """
    Métrica M1 (alternativa): SSIM entre versões coarse/downsampled.
    Mede se a organização espacial de baixa resolução (macro-estrutura)
    é preservada entre a imagem original e a transformada.

    Racional: patch shuffle destrói a macro-estrutura (SSIM coarse cai),
    enquanto blur e grayscale, em baixa resolução, ficam quase idênticos
    ao original (SSIM coarse permanece alto).

    Parâmetros
    ----------
    img_a, img_b : np.ndarray, shape (H, W, 5)
        Imagem original e imagem transformada (mesma shape).
        Bandas na ordem (B, G, R, NIR, RE).
    downscale_factor : int
        Fator de redução de resolução via average pooling
        (ex. 16 → blocos de 16x16 px viram 1 px).
    channel : str
        "luminance" (combina B,G,R) ou índice de banda específica (0-4).

    Retorna
    -------
    M1 : float
        SSIM entre as versões coarse (entre -1 e 1; 1 = idênticas).
        Alto = macro-estrutura preservada; baixo = destruída.
    """
    if img_a.shape != img_b.shape:
        raise ValueError(f"Shapes diferentes: {img_a.shape} vs {img_b.shape}")

    def to_channel(img):
        if channel == "luminance":
            B, G, R = img[..., 0], img[..., 1], img[..., 2]
            return (0.114 * B + 0.587 * G + 0.299 * R).astype(np.float64)
        else:
            return img[..., int(channel)].astype(np.float64)

    def downsample(img_2d, factor):
        H, W = img_2d.shape
        # crop para múltiplo do fator (average pooling exige blocos completos)
        H_crop = H - (H % factor)
        W_crop = W - (W % factor)
        cropped = img_2d[:H_crop, :W_crop]
        # reshape em blocos e tira a média de cada bloco
        reshaped = cropped.reshape(
            H_crop // factor, factor, W_crop // factor, factor
        )
        return reshaped.mean(axis=(1, 3))

    ch_a = to_channel(img_a)
    ch_b = to_channel(img_b)

    coarse_a = downsample(ch_a, downscale_factor)
    coarse_b = downsample(ch_b, downscale_factor)

    if coarse_a.shape[0] < 7 or coarse_a.shape[1] < 7:
        raise ValueError(
            f"Imagem coarse muito pequena {coarse_a.shape} para SSIM "
            f"(win_size padrão=7). Reduza downscale_factor."
        )

    # data_range: amplitude de valores esperada (ajuste conforme dtype/escala)
    data_range = max(coarse_a.max() - coarse_a.min(),
                      coarse_b.max() - coarse_b.min(),
                      1e-8)

    score = ssim(coarse_a, coarse_b, data_range=data_range)

    return float(score)



#======================================================================

import numpy as np
import cv2
from skimage.metrics import structural_similarity


def coarse_ssim_gpt(
    img_original: np.ndarray,
    img_transformed: np.ndarray,
    coarse_size=(64, 64),
) -> float:
    """
    Calcula SSIM entre versões coarse/downsampled de duas imagens
    multiespectrais.

    A ideia é remover grande parte dos detalhes locais e comparar
    principalmente a organização espacial em larga escala.

    Parameters
    ----------
    img_original : np.ndarray
        Imagem original (H, W, 5).

    img_transformed : np.ndarray
        Imagem transformada (H, W, 5).

    coarse_size : tuple[int, int]
        Resolução (altura, largura) usada para a representação coarse.
        Ex.: (32, 32).

    Returns
    -------
    float
        Média do SSIM calculado separadamente nas cinco bandas.

        ~1   -> estrutura coarse muito semelhante
        menor -> maior alteração da organização espacial global
    """

    if img_original.shape != img_transformed.shape:
        raise ValueError(
            "As imagens original e transformada devem ter o mesmo shape."
        )

    if img_original.ndim != 3 or img_original.shape[-1] != 5:
        raise ValueError(
            f"Esperado (H, W, 5), recebido {img_original.shape}"
        )

    img_original = img_original.astype(np.float64)
    img_transformed = img_transformed.astype(np.float64)

    coarse_h, coarse_w = coarse_size

    ssim_values = []

    for band in range(5):

        original_band = img_original[..., band]
        transformed_band = img_transformed[..., band]

        # Downsampling com INTER_AREA, adequado para redução
        original_coarse = cv2.resize(
            original_band,
            (coarse_w, coarse_h),
            interpolation=cv2.INTER_AREA
        )

        transformed_coarse = cv2.resize(
            transformed_band,
            (coarse_w, coarse_h),
            interpolation=cv2.INTER_AREA
        )

        # Data range conjunto para tornar a comparação consistente
        global_min = min(
            original_coarse.min(),
            transformed_coarse.min()
        )

        global_max = max(
            original_coarse.max(),
            transformed_coarse.max()
        )

        data_range = global_max - global_min

        # Banda constante nas duas imagens
        if data_range == 0:
            ssim_band = 1.0
        else:
            ssim_band = structural_similarity(
                original_coarse,
                transformed_coarse,
                data_range=data_range
            )

        ssim_values.append(ssim_band)

    return float(np.mean(ssim_values))

#======================================================================
#======================================================================
#======================================================================
# **M₂ (Gaussian Blur):**

import numpy as np
from scipy.ndimage import laplace


def laplacian_variance(
    img_5b: np.ndarray,
    channel: str = "luminance",
) -> float:
    """
    Métrica M2: variância do Laplaciano (luminância).
    Mede nitidez/alta frequência local. Clássica medida de "blurriness":
    quanto menor a variância, mais borrada a imagem.

    Racional: Gaussian blur é um filtro passa-baixa que suaviza bordas
    e reduz diretamente essa métrica. Patch shuffle preserva o conteúdo
    local dentro de cada patch (o Laplaciano é um operador de vizinhança
    pequena, então não "enxerga" a desorganização global). Grayscale
    calculado sobre luminância não remove alta frequência de luminância.

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    channel : str
        "luminance" (combina B,G,R) ou índice de banda específica (0-4).

    Retorna
    -------
    M2 : float
        Variância do Laplaciano. Alto = nítido/muita alta frequência;
        baixo = borrado/pouca alta frequência.
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    if channel == "luminance":
        B, G, R = img_5b[..., 0], img_5b[..., 1], img_5b[..., 2]
        img = 0.114 * B + 0.587 * G + 0.299 * R
    else:
        img = img_5b[..., int(channel)]

    img = img.astype(np.float64)

    # Laplaciano discreto (kernel padrão de 4-conectividade via scipy)
    lap = laplace(img)

    M2 = float(lap.var())

    return M2



#======================================================================

import numpy as np
import cv2


def laplacian_variance_luminance_GPT(img_5b: np.ndarray) -> float:
    """
    Calcula a Variância do Laplaciano sobre a luminância RGB.

    A métrica quantifica conteúdo espacial de alta frequência
    (bordas, detalhes finos e textura).

    Esperado:
        imagem nítida / texturizada -> valor alto
        Gaussian blur               -> valor baixo

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5), com bandas:
        [B, G, R, NIR, RE].

    Returns
    -------
    float
        Variância do Laplaciano da luminância.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # Bandas
    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    # Luminância Rec. 601
    luminance = (
        0.299 * R +
        0.587 * G +
        0.114 * B
    )

    # Laplaciano
    laplacian = cv2.Laplacian(
        luminance,
        cv2.CV_64F,
        ksize=3
    )

    # Variância do Laplaciano
    return float(np.var(laplacian))


#======================================================================

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def glcm_contrast_energy(
    img_5b: np.ndarray,
    channel: str = "luminance",
    distances: list = (1, 2),
    angles: list = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels: int = 32,
    mask_background: bool = True,
) -> dict:
    """
    Métrica M2 (alternativa): GLCM contrast/energy, offset pequeno.
    Mede textura local via matriz de co-ocorrência (Haralick).

    Racional: blur reduz contraste local entre pixels vizinhos (GLCM
    contrast cai, energy sobe pois a distribuição fica mais concentrada/
    homogênea). Com offset pequeno (1-2 px), quase todos os pares caem
    dentro do mesmo patch no shuffle (preservado); grayscale calculado
    em luminância não altera a textura de luminância.

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    channel : str
        "luminance" ou índice de banda (0-4).
    distances : list de int
        Offsets em pixels (pequenos = sensível a blur, robusto a shuffle).
    angles : list de float
        Ângulos em radianos para co-ocorrência (média sobre direções
        dá invariância rotacional aproximada).
    n_levels : int
        Número de níveis de cinza para quantização (reduz ruído e custo
        computacional da matriz GLCM).
    mask_background : bool
        Se True, restringe o cálculo ao bounding box da máscara
        (foreground = banda != 0), evitando que o fundo zerado
        domine a matriz de co-ocorrência.

    Retorna
    -------
    dict com:
        contrast : float (média sobre distances x angles)
        energy   : float (média sobre distances x angles)
        contrast_per_distance : dict {d: valor}
        energy_per_distance   : dict {d: valor}
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    if channel == "luminance":
        B, G, R = img_5b[..., 0], img_5b[..., 1], img_5b[..., 2]
        img = 0.114 * B + 0.587 * G + 0.299 * R
    else:
        img = img_5b[..., int(channel)]

    img = img.astype(np.float64)

    # --- Opcional: restringe ao bounding box do foreground ---
    if mask_background:
        mask = np.any(img_5b != 0, axis=-1)
        if mask.any():
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            img = img[y0:y1, x0:x1]

    # --- Quantização para n_levels níveis de cinza (uint8-like) ---
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-8:
        # imagem constante: textura indefinida
        return {
            "contrast": 0.0,
            "energy": 1.0,
            "contrast_per_distance": {d: 0.0 for d in distances},
            "energy_per_distance": {d: 1.0 for d in distances},
        }

    img_norm = (img - img_min) / (img_max - img_min)
    img_q = (img_norm * (n_levels - 1)).astype(np.uint8)

    contrast_per_d = {}
    energy_per_d = {}

    for d in distances:
        glcm = graycomatrix(
            img_q,
            distances=[d],
            angles=list(angles),
            levels=n_levels,
            symmetric=True,
            normed=True,
        )
        # graycoprops retorna shape (n_distances, n_angles); tiramos média
        # sobre os ângulos para aproximar invariância rotacional
        contrast_per_d[d] = float(graycoprops(glcm, "contrast").mean())
        energy_per_d[d] = float(graycoprops(glcm, "energy").mean())

    return {
        "contrast": float(np.mean(list(contrast_per_d.values()))),
        "energy": float(np.mean(list(energy_per_d.values()))),
        "contrast_per_distance": contrast_per_d,
        "energy_per_distance": energy_per_d,
    }

# energy
# energy_per_distance

def glcm_contrast_energy_contrast(
    img_5b: np.ndarray,
    channel: str = "luminance",
    distances: list = (1, 2),
    angles: list = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels: int = 32,
    mask_background: bool = True):

        data = glcm_contrast_energy(
        img_5b,
        channel,
        distances,
        angles,
        n_levels,
        mask_background)

        return data["contrast"]


def glcm_contrast_energy_test(
    img_5b: np.ndarray,
    channel: str = "luminance",
    distances: list = (1, 2),
    angles: list = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels: int = 32,
    mask_background: bool = True):

        data = glcm_contrast_energy(
        img_5b,
        channel,
        distances,
        angles,
        n_levels,
        mask_background)

        return data["energy_per_distance"][1]

#======================================================================

import numpy as np


def high_low_frequency_energy_ratio_GPT(
    img_5b: np.ndarray,
    low_radius: float = 0.10,
    high_radius: float = 0.35,
    eps: float = 1e-12
) -> float:
    """
    Calcula a razão entre energia de alta e baixa frequência da
    luminância RGB usando FFT 2D.

    M = E_high / E_low

    Esperado:
        imagem com detalhes/textura -> valor maior
        Gaussian blur               -> valor menor

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem (H, W, 5), bandas [B, G, R, NIR, RE].

    low_radius : float
        Limite radial normalizado da região de baixa frequência.
        Frequências com r <= low_radius são consideradas baixas.

    high_radius : float
        Frequência radial a partir da qual consideramos alta frequência.
        Frequências com r >= high_radius são consideradas altas.

    eps : float
        Estabilidade numérica.

    Returns
    -------
    float
        Razão E_high / E_low.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # ---------------------------------------------------------
    # 1. Luminância RGB
    # ---------------------------------------------------------
    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    luminance = (
        0.299 * R +
        0.587 * G +
        0.114 * B
    )

    # Remove componente DC / média global
    luminance = luminance - np.mean(luminance)

    # ---------------------------------------------------------
    # 2. FFT 2D
    # ---------------------------------------------------------
    fft = np.fft.fft2(luminance)
    fft = np.fft.fftshift(fft)

    # Espectro de potência
    power = np.abs(fft) ** 2

    H, W = luminance.shape

    # ---------------------------------------------------------
    # 3. Coordenadas de frequência normalizadas
    #
    # centro = frequência zero
    # bordas = frequências altas
    # ---------------------------------------------------------
    y = np.arange(H) - H // 2
    x = np.arange(W) - W // 2

    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Normaliza cada eixo aproximadamente para [-1, 1]
    yy = yy / (H / 2.0)
    xx = xx / (W / 2.0)

    radius = np.sqrt(xx**2 + yy**2)

    # ---------------------------------------------------------
    # 4. Máscaras de baixa e alta frequência
    # ---------------------------------------------------------
    low_mask = radius <= low_radius
    high_mask = radius >= high_radius

    # ---------------------------------------------------------
    # 5. Energia
    # ---------------------------------------------------------
    E_low = np.sum(power[low_mask])
    E_high = np.sum(power[high_mask])

    return float(E_high / (E_low + eps))

#======================================================================

import numpy as np


def high_low_freq_energy_ratio(
    img_5b: np.ndarray,
    channel: str = "luminance",
    cutoff_fraction: float = 0.15,
    mask_background: bool = True,
) -> dict:
    """
    Métrica M2 (alternativa): razão energia alta/baixa frequência via FFT.
    Mede quanto da energia espectral da imagem está concentrada em altas
    frequências (bordas, textura fina) vs. baixas frequências (formas
    grosseiras, tendência de iluminação).

    Racional: blur é um filtro passa-baixa — corta energia de alta
    frequência diretamente, derrubando essa razão. Patch shuffle preserva
    o conteúdo de frequência local dentro de cada patch (mistura DC entre
    patches, mas não elimina alta frequência); grayscale sobre luminância
    não altera o conteúdo espectral de luminância.

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    channel : str
        "luminance" ou índice de banda (0-4).
    cutoff_fraction : float
        Fração do raio máximo (Nyquist) usada como limiar entre baixa
        e alta frequência no espectro radial. Ex. 0.15 = frequências
        com raio normalizado < 0.15 são "baixas", o resto é "alta".
    mask_background : bool
        Se True, recorta ao bounding box do foreground antes da FFT,
        evitando que a borda abrupta fundo/planta domine o espectro
        com energia de alta frequência espúria.

    Retorna
    -------
    dict com:
        ratio       : float (energia_alta / energia_baixa)
        energy_high : float
        energy_low  : float
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    if channel == "luminance":
        B, G, R = img_5b[..., 0], img_5b[..., 1], img_5b[..., 2]
        img = 0.114 * B + 0.587 * G + 0.299 * R
    else:
        img = img_5b[..., int(channel)]

    img = img.astype(np.float64)

    if mask_background:
        mask = np.any(img_5b != 0, axis=-1)
        if mask.any():
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            img = img[y0:y1, x0:x1]

    Hc, Wc = img.shape
    img = img - img.mean()  # remove componente DC pura

    # Janela de Hann para reduzir vazamento espectral (leakage) nas bordas
    win_y = np.hanning(Hc)
    win_x = np.hanning(Wc)
    window = np.outer(win_y, win_x)
    img_win = img * window

    # --- FFT 2D e espectro de potência ---
    F = np.fft.fft2(img_win)
    F = np.fft.fftshift(F)
    power = np.abs(F) ** 2

    # --- Mapa de frequência radial normalizada (0 a ~1 no centro -> Nyquist) ---
    cy, cx = Hc // 2, Wc // 2
    y, x = np.indices((Hc, Wc))
    dist = np.sqrt(((y - cy) / Hc) ** 2 + ((x - cx) / Wc) ** 2)
    r_max = dist.max()
    r_norm = dist / r_max  # 0 (DC) a 1 (canto, maior frequência)

    low_mask = r_norm < cutoff_fraction
    high_mask = ~low_mask

    energy_low = float(power[low_mask].sum())
    energy_high = float(power[high_mask].sum())

    ratio = energy_high / energy_low if energy_low > 1e-12 else np.inf

    return {
        "ratio": ratio,
        "energy_high": energy_high,
        "energy_low": energy_low,
    }


def high_low_freq_energy_ratio_test(
                                img_5b: np.ndarray,
                                channel: str = "luminance",
                                cutoff_fraction: float = 0.15,
                                mask_background: bool = True,
                                ):

    data = high_low_freq_energy_ratio(
            img_5b,
            channel,
            cutoff_fraction,
            mask_background,
            )

    return data["energy_low"]


# cutoff_fraction = [0.05, 0.15, 0.25]
# mask_background = [True, False]

#======================================================================
#======================================================================
#======================================================================
# **M₃ (Grayscale/Dessaturação):**

# 1. Chroma média (Lab ou HSV-S)

import numpy as np
from skimage.color import rgb2lab


def mean_lab_chroma_GPT(img_5b: np.ndarray) -> float:
    """
    Calcula M3: Chroma média no espaço CIELAB.

    Mede a quantidade média de informação cromática presente
    na região segmentada da planta.

        C*_ab = sqrt(a*^2 + b*^2)

    Esperado:
        imagem RGB colorida -> chroma > 0
        grayscale           -> chroma ~ 0

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5), bandas:
        [B, G, R, NIR, RE].

        Assume-se que o fundo da imagem segmentada possui valor zero.

    Returns
    -------
    float
        Chroma média da região da planta.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # ---------------------------------------------------------
    # 1. Extrai BGR -> RGB
    # ---------------------------------------------------------
    B = img_5b[..., 0].astype(np.float64)
    G = img_5b[..., 1].astype(np.float64)
    R = img_5b[..., 2].astype(np.float64)

    rgb = np.stack([R, G, B], axis=-1)

    # ---------------------------------------------------------
    # 2. Máscara da planta
    #
    # Utilizamos apenas RGB para que alterações em NIR/RE
    # não modifiquem quais pixels entram no cálculo.
    # ---------------------------------------------------------
    mask = np.any(rgb != 0, axis=-1)

    if not np.any(mask):
        return np.nan

    # ---------------------------------------------------------
    # 3. Normalização para rgb2lab
    # ---------------------------------------------------------
    # Se os dados já estiverem em [0, 1], mantém.
    # Caso contrário, assume escala típica [0, 255].
    # ---------------------------------------------------------
    rgb_min = rgb.min()
    rgb_max = rgb.max()

    if rgb_min < 0:
        raise ValueError(
            "RGB contém valores negativos. "
            "Desnormalize a imagem antes de calcular CIELAB."
        )

    if rgb_max > 1.0:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, 1.0)

    # ---------------------------------------------------------
    # 4. RGB -> CIELAB
    # ---------------------------------------------------------
    lab = rgb2lab(rgb)

    a = lab[..., 1]
    b = lab[..., 2]

    # ---------------------------------------------------------
    # 5. Chroma por pixel
    # ---------------------------------------------------------
    chroma = np.sqrt(a**2 + b**2)

    # Somente região da planta
    return float(np.mean(chroma[mask]))


#======================================================================

import numpy as np
from skimage.color import rgb2lab


def mean_chroma(
    img_5b: np.ndarray,
    mask_background: bool = True,
    input_range: tuple = None,
) -> dict:
    """
    Métrica M3: chroma média em Lab (C* = sqrt(a*^2 + b*^2)).
    Mede quantidade de informação cromática presente na imagem.

    Racional: grayscale/dessaturação colapsa a* e b* para próximo de 0
    (chroma -> 0 por definição). Patch shuffle apenas realoca pixels
    coloridos, preservando a distribuição global de chroma. Blur suaviza
    mas não remove cor — reduz um pouco o chroma perto de bordas de alto
    contraste cromático (efeito residual esperado, não colapso).

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    mask_background : bool
        Se True, calcula a média apenas sobre pixels de foreground
        (qualquer banda != 0), evitando que o fundo zerado (que também
        é "sem cor") infle artificialmente a queda de chroma.
    input_range : tuple (min, max) ou None
        Faixa de valores das bandas RGB de entrada, usada para normalizar
        para [0, 1] antes de converter para Lab (skimage espera RGB em
        [0, 1] float). Se None, usa (img.min(), img.max()) por imagem —
        recomenda-se passar explicitamente (ex. (0, 255) ou (0, 10000)
        para refletância) para manter escala consistente entre imagens.

    Retorna
    -------
    dict com:
        mean_chroma : float (C* médio sobre pixels de foreground)
        std_chroma  : float (desvio padrão do C*, útil como diagnóstico)
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    B = img_5b[..., 0].astype(np.float64)
    G = img_5b[..., 1].astype(np.float64)
    R = img_5b[..., 2].astype(np.float64)

    rgb = np.stack([R, G, B], axis=-1)  # skimage espera ordem RGB

    # --- Normaliza para [0, 1] ---
    if input_range is None:
        lo, hi = rgb.min(), rgb.max()
    else:
        lo, hi = input_range

    if hi - lo < 1e-8:
        return {"mean_chroma": 0.0, "std_chroma": 0.0}

    rgb_norm = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)

    # --- Converte para Lab ---
    lab = rgb2lab(rgb_norm)
    a_ch = lab[..., 1]
    b_ch = lab[..., 2]
    chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)  # C*

    # --- Restringe ao foreground ---
    if mask_background:
        mask = np.any(img_5b != 0, axis=-1)
        if not mask.any():
            return {"mean_chroma": 0.0, "std_chroma": 0.0}
        chroma_vals = chroma[mask]
    else:
        chroma_vals = chroma.ravel()

    return {
        "mean_chroma": float(chroma_vals.mean()),
        "std_chroma": float(chroma_vals.std()),
    }

# mean_chroma
# std_chroma

def mean_chroma_test(
    img_5b: np.ndarray,
    mask_background: bool = True,
    input_range: tuple = None,
):
    
    data =  mean_chroma(
    img_5b,
    mask_background,
    input_range)

    return data["std_chroma"]


#======================================================================
#======================================================================
# 2. Divergência entre canais RGB (|R-G|, |G-B|, |R-B|)


def rgb_channel_divergence_GPT(img_5b: np.ndarray) -> dict:
    """
    Calcula a divergência média entre os canais RGB.

    Para cada pixel da planta:

        D_RG = |R - G|
        D_GB = |G - B|
        D_RB = |R - B|

    A métrica global é:

        M = mean(D_RG, D_GB, D_RB)

    Esperado:
        imagem colorida -> M > 0
        grayscale       -> M = 0

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5), bandas:
        [B, G, R, NIR, RE].

        Assume fundo = 0.

    Returns
    -------
    dict
        Divergência de cada par de canais e média global.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    # ---------------------------------------------------------
    # Máscara da planta
    # ---------------------------------------------------------
    mask = (R != 0) | (G != 0) | (B != 0)

    if not np.any(mask):
        return {
            "div_rg": np.nan,
            "div_gb": np.nan,
            "div_rb": np.nan,
            "rgb_divergence": np.nan
        }

    # ---------------------------------------------------------
    # Diferenças absolutas
    # ---------------------------------------------------------
    d_rg = np.abs(R - G)
    d_gb = np.abs(G - B)
    d_rb = np.abs(R - B)

    # Somente região da planta
    mean_rg = np.mean(d_rg[mask])
    mean_gb = np.mean(d_gb[mask])
    mean_rb = np.mean(d_rb[mask])

    # Métrica agregada
    mean_divergence = np.mean([
        mean_rg,
        mean_gb,
        mean_rb
    ])

    return {
        "div_rg": float(mean_rg),
        "div_gb": float(mean_gb),
        "div_rb": float(mean_rb),
        "rgb_divergence": float(mean_divergence)
    }


def rgb_channel_divergence_GPT_test(img_5b: np.ndarray):
    data = rgb_channel_divergence_GPT(img_5b)

    return data["rgb_divergence"]

#======================================================================

import numpy as np


def rgb_channel_divergence(
    img_5b: np.ndarray,
    mask_background: bool = True,
) -> dict:
    """
    Métrica M3 (alternativa): divergência entre canais RGB.
    Mede quanto os canais R, G, B diferem entre si pixel a pixel.

    Racional: em grayscale, R = G = B por definição, então a divergência
    colapsa a 0. Patch shuffle preserva a tripla (R,G,B) de cada pixel
    (apenas realoca posições), então a divergência média sobre a imagem
    não muda. Blur suaviza espacialmente mas mantém a diferença média
    entre canais aproximadamente constante (é uma média ponderada local,
    não uma mistura entre canais).

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    mask_background : bool
        Se True, calcula a média apenas sobre pixels de foreground
        (qualquer banda != 0), evitando que o fundo zerado (R=G=B=0,
        divergência=0) dilua artificialmente a métrica.

    Retorna
    -------
    dict com:
        mean_divergence : float (média de (|R-G|+|G-B|+|R-B|)/3 por pixel)
        per_pair : dict com médias individuais {"RG": ..., "GB": ..., "RB": ...}
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    B = img_5b[..., 0].astype(np.float64)
    G = img_5b[..., 1].astype(np.float64)
    R = img_5b[..., 2].astype(np.float64)

    d_rg = np.abs(R - G)
    d_gb = np.abs(G - B)
    d_rb = np.abs(R - B)

    divergence = (d_rg + d_gb + d_rb) / 3.0

    if mask_background:
        mask = np.any(img_5b != 0, axis=-1)
        if not mask.any():
            return {
                "mean_divergence": 0.0,
                "per_pair": {"RG": 0.0, "GB": 0.0, "RB": 0.0},
            }
        divergence_vals = divergence[mask]
        d_rg_vals, d_gb_vals, d_rb_vals = d_rg[mask], d_gb[mask], d_rb[mask]
    else:
        divergence_vals = divergence.ravel()
        d_rg_vals, d_gb_vals, d_rb_vals = d_rg.ravel(), d_gb.ravel(), d_rb.ravel()

    return {
        "mean_divergence": float(divergence_vals.mean()),
        "per_pair": {
            "RG": float(d_rg_vals.mean()),
            "GB": float(d_gb_vals.mean()),
            "RB": float(d_rb_vals.mean()),
        },
        "max_divergence": max(float(d_rg_vals.mean()), float(d_gb_vals.mean()), float(d_rb_vals.mean())),
    }

def rgb_channel_divergence_TEST(
    img_5b: np.ndarray,
    mask_background: bool = True,
):
    data = rgb_channel_divergence(
    img_5b,
    mask_background,
    )

    return data["max_divergence"]

#======================================================================
# 3. Entropia/variância circular do histograma de Hue

import numpy as np
from skimage.color import rgb2hsv


def hue_distribution_metrics_GPT(
    img_5b: np.ndarray,
    n_bins: int = 36,
    saturation_threshold: float = 0.05,
    eps: float = 1e-12
) -> dict:
    """
    Calcula métricas da distribuição de Hue na região segmentada:

        1. Entropia normalizada do histograma de Hue
        2. Variância circular do Hue
        3. Fração de pixels com Hue válido

    Hue é considerado válido somente quando a saturação é maior
    que saturation_threshold.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem (H, W, 5), bandas [B, G, R, NIR, RE].

        Assume-se que:
        - fundo = 0
        - RGB está em [0, 1] ou [0, 255]

    n_bins : int
        Número de bins utilizados no histograma circular de Hue.

    saturation_threshold : float
        Saturação mínima para considerar o Hue válido.
        Saturação está no intervalo [0, 1].

    eps : float
        Estabilidade numérica.

    Returns
    -------
    dict
        {
            "hue_entropy": float,
            "hue_circular_variance": float,
            "valid_hue_fraction": float
        }

        hue_entropy:
            0 -> Hue concentrado
            1 -> Hue distribuído uniformemente

        hue_circular_variance:
            0 -> matizes muito concentrados
            1 -> matizes muito dispersos

        valid_hue_fraction:
            fração da planta que possui saturação suficiente
            para que Hue seja considerado válido.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # ---------------------------------------------------------
    # 1. Extrai RGB
    # ---------------------------------------------------------
    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    rgb = np.stack([R, G, B], axis=-1)

    # ---------------------------------------------------------
    # 2. Máscara da planta
    # ---------------------------------------------------------
    plant_mask = np.any(rgb != 0, axis=-1)

    n_plant = np.sum(plant_mask)

    if n_plant == 0:
        return {
            "hue_entropy": np.nan,
            "hue_circular_variance": np.nan,
            "valid_hue_fraction": 0.0
        }

    # ---------------------------------------------------------
    # 3. Normaliza RGB para [0, 1]
    # ---------------------------------------------------------
    if rgb.min() < 0:
        raise ValueError(
            "RGB contém valores negativos. "
            "Use a imagem desnormalizada."
        )

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, 1.0)

    # ---------------------------------------------------------
    # 4. RGB -> HSV
    #
    # skimage retorna:
    # H -> [0, 1]
    # S -> [0, 1]
    # V -> [0, 1]
    # ---------------------------------------------------------
    hsv = rgb2hsv(rgb)

    hue = hsv[..., 0]
    saturation = hsv[..., 1]

    # ---------------------------------------------------------
    # 5. Seleciona pixels com Hue válido
    # ---------------------------------------------------------
    valid_mask = (
        plant_mask &
        (saturation > saturation_threshold)
    )

    n_valid = np.sum(valid_mask)

    valid_hue_fraction = n_valid / n_plant

    # Se não há pixels cromáticos, Hue não está definido.
    if n_valid == 0:
        return {
            "hue_entropy": 0.0,
            "hue_circular_variance": 0.0,
            "valid_hue_fraction": 0.0
        }

    H = hue[valid_mask]

    # =========================================================
    # 6. ENTROPIA DO HISTOGRAMA DE HUE
    # =========================================================

    hist, _ = np.histogram(
        H,
        bins=n_bins,
        range=(0.0, 1.0)
    )

    p = hist.astype(np.float64)
    p /= p.sum()

    p_nonzero = p[p > 0]

    entropy = -np.sum(
        p_nonzero * np.log2(p_nonzero + eps)
    )

    # Entropia máxima = log2(n_bins)
    # Normalização para [0, 1]
    entropy_normalized = entropy / np.log2(n_bins)

    # =========================================================
    # 7. VARIÂNCIA CIRCULAR
    # =========================================================

    # Hue [0,1] -> ângulo [0, 2pi]
    theta = 2.0 * np.pi * H

    mean_cos = np.mean(np.cos(theta))
    mean_sin = np.mean(np.sin(theta))

    # Comprimento do vetor resultante médio
    R_bar = np.sqrt(
        mean_cos**2 +
        mean_sin**2
    )

    # Variância circular
    circular_variance = 1.0 - R_bar

    return {
        "hue_entropy": float(entropy_normalized),
        "hue_circular_variance": float(circular_variance),
        "valid_hue_fraction": float(valid_hue_fraction)
    }


def hue_distribution_metrics_GPT_TEST(
                    img_5b: np.ndarray,
                    n_bins: int = 36,
                    saturation_threshold: float = 0.05,
                    eps: float = 1e-12
                ):

    data = hue_distribution_metrics_GPT(
    img_5b,
    n_bins,
    saturation_threshold,
    eps)

    return data["valid_hue_fraction"]

#======================================================================

import numpy as np
import cv2


def hue_histogram_stats(
    img_5b: np.ndarray,
    n_bins: int = 36,
    mask_background: bool = True,
    input_range: tuple = None,
    saturation_weighted: bool = True,
    sat_threshold: float = 0.05,
) -> dict:
    """
    Métrica M3 (alternativa): entropia e variância circular do histograma de Hue.
    Mede o quão concentrada/dispersa é a distribuição de matizes na imagem.

    Racional: em grayscale, S -> 0 e o Hue torna-se indefinido/instável
    (ruído numérico domina), então a distribuição de Hue tende a ficar
    artificialmente dispersa ou degenerada. Patch shuffle não altera a
    distribuição global de matizes (só realoca posições espaciais).
    Blur suaviza espacialmente mas afeta pouco a distribuição agregada
    de Hue (médias locais de cores similares tendem a preservar o matiz
    dominante).

    Como Hue é uma variável circular (0° e 360° são o mesmo ponto), usamos
    estatística circular (não estatística linear ingênua).

    Parâmetros
    ----------
    img_5b : np.ndarray, shape (H, W, 5)
        Bandas na ordem (B, G, R, NIR, RE).
    n_bins : int
        Número de bins do histograma circular de Hue.
    mask_background : bool
        Se True, restringe aos pixels de foreground.
    input_range : tuple (min, max) ou None
        Faixa de valores das bandas RGB para normalizar para [0,1]/uint8
        antes de converter para HSV. Recomenda-se fixar explicitamente.
    saturation_weighted : bool
        Se True, pondera cada pixel pela sua saturação ao construir o
        histograma/estatísticas — essencial porque Hue é ruído puro
        quando S≈0 (evita que pixels quase-acromáticos poluam a métrica
        mesmo antes da transformação de grayscale).
    sat_threshold : float
        Pixels com saturação (em [0,1]) abaixo deste limiar são
        excluídos do cálculo (Hue indefinido/instável).

    Retorna
    -------
    dict com:
        entropy          : float (entropia de Shannon do histograma, em bits)
        circular_variance : float (0 = todo concentrado num ângulo, 1 = disperso)
        n_valid_pixels    : int (pixels usados após filtro de saturação)
    """
    H, W, C = img_5b.shape
    assert C == 5, "Esperado 5 bandas (B, G, R, NIR, RE)"

    B = img_5b[..., 0].astype(np.float64)
    G = img_5b[..., 1].astype(np.float64)
    R = img_5b[..., 2].astype(np.float64)

    if input_range is None:
        lo = min(R.min(), G.min(), B.min())
        hi = max(R.max(), G.max(), B.max())
    else:
        lo, hi = input_range

    if hi - lo < 1e-8:
        return {"entropy": 0.0, "circular_variance": 1.0, "n_valid_pixels": 0}

    rgb_norm = np.stack([R, G, B], axis=-1)
    rgb_norm = np.clip((rgb_norm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    # --- Converte para HSV (OpenCV espera float32 em [0,1] -> H em [0,360)) ---
    hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2HSV)
    hue_deg = hsv[..., 0]       # [0, 360)
    sat = hsv[..., 1]           # [0, 1]

    # --- Máscara de foreground ---
    if mask_background:
        fg_mask = np.any(img_5b != 0, axis=-1)
    else:
        fg_mask = np.ones((H, W), dtype=bool)

    # --- Filtra pixels de baixa saturação (Hue indefinido) ---
    valid_mask = fg_mask & (sat > sat_threshold)

    if not valid_mask.any():
        return {"entropy": 0.0, "circular_variance": 1.0, "n_valid_pixels": 0}

    hue_vals = hue_deg[valid_mask]
    sat_vals = sat[valid_mask] if saturation_weighted else np.ones_like(hue_vals)

    # --- Variância circular (estatística circular padrão) ---
    hue_rad = np.deg2rad(hue_vals)
    C_sum = np.sum(sat_vals * np.cos(hue_rad))
    S_sum = np.sum(sat_vals * np.sin(hue_rad))
    R_bar = np.sqrt(C_sum ** 2 + S_sum ** 2) / np.sum(sat_vals)  # comprimento resultante, [0,1]
    circular_variance = 1.0 - R_bar  # 0 = concentrado, 1 = disperso

    # --- Histograma circular ponderado por saturação + entropia de Shannon ---
    bin_edges = np.linspace(0, 360, n_bins + 1)
    hist, _ = np.histogram(hue_vals, bins=bin_edges, weights=sat_vals)

    p = hist / (hist.sum() + 1e-12)
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))

    return {
        "entropy": entropy,
        "circular_variance": float(circular_variance),
        "n_valid_pixels": int(valid_mask.sum()),
    }


def hue_histogram_stats_TEST(
    img_5b: np.ndarray,
    n_bins: int = 36,
    mask_background: bool = True,
    input_range: tuple = None,
    saturation_weighted: bool = True,
    sat_threshold: float = 0.05,
):

    data = hue_histogram_stats(
        img_5b,
        n_bins,
        mask_background,
        input_range,
        saturation_weighted,
        sat_threshold)
    
    return data["n_valid_pixels"]

#======================================================================
#======================================================================
#======================================================================
#======================================================================
