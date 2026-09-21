import numpy as np

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Feature Metrics ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# Paper


def local_variance_LV(
    img_5b: np.ndarray,
    window_size: int = 11,
    min_valid_pixels: int = 2
) -> dict:
    """
    Calcula LV — Local Variance — nas 5 bandas separadamente.

    Bandas:
        [B, G, R, NIR, RE]

    A métrica é mask-aware e funciona tanto para:

        1. imagens segmentadas originais:
           background = 0;

        2. imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    Para cada banda:
        1. divide a imagem em janelas não sobrepostas;
        2. calcula a variância somente sobre pixels da planta;
        3. calcula a média das variâncias locais.

    Returns
    -------
    dict
        {
            "B": ...,
            "G": ...,
            "R": ...,
            "NIR": ...,
            "RE": ...,
            "mean_LV": ...
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    if window_size <= 0:
        raise ValueError("window_size deve ser > 0")

    if min_valid_pixels < 2:
        raise ValueError("min_valid_pixels deve ser >= 2")

    img = img_5b.astype(np.float64)

    H, W, C = img.shape

    # =========================================================
    # 1. Máscara da planta
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
        return {
            "B": 0.0,
            "G": 0.0,
            "R": 0.0,
            "NIR": 0.0,
            "RE": 0.0,
            "mean_LV": 0.0
        }

    band_names = ["B", "G", "R", "NIR", "RE"]

    results = {}

    # =========================================================
    # 2. Calcula LV para cada banda
    # =========================================================

    for band in range(C):

        image_1b = img[..., band]

        local_variances = []

        for y0 in range(0, H, window_size):

            y1 = min(
                y0 + window_size,
                H
            )

            for x0 in range(0, W, window_size):

                x1 = min(
                    x0 + window_size,
                    W
                )

                window = image_1b[
                    y0:y1,
                    x0:x1
                ]

                mask_window = plant_mask[
                    y0:y1,
                    x0:x1
                ]

                values = window[
                    mask_window
                ]

                if values.size >= min_valid_pixels:

                    local_variances.append(
                        np.var(values)
                    )

        if len(local_variances) == 0:
            lv = 0.0
        else:
            lv = float(
                np.mean(local_variances)
            )

        results[band_names[band]] = lv

    # =========================================================
    # 3. LV multiespectral agregada
    # =========================================================

    results["mean_LV"] = float(
        np.mean([
            results["B"],
            results["G"],
            results["R"],
            results["NIR"],
            results["RE"]
        ])
    )

    return results['mean_LV']

#======================================================================
import numpy as np
from scipy.ndimage import distance_transform_edt


def high_frequency_energy_HFE(
    img_5b: np.ndarray,
    radius: int = 11
) -> dict:
    """
    Calcula HFE — High-Frequency Energy Ratio — nas 5 bandas.

    Para cada banda:

        HFE = energia_alta_frequencia / energia_total

    Bandas:
        [B, G, R, NIR, RE]

    A função é mask-aware e compatível com:

        1. Imagens segmentadas originais:
           background = 0.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    O background dentro do bounding box da planta é preenchido
    temporariamente pelo pixel de planta mais próximo para evitar
    altas frequências artificiais na fronteira planta/background.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5).

    radius : int
        Raio no domínio da frequência que separa baixas e
        altas frequências.

        No artigo:
            radius = 11.

    Returns
    -------
    dict
        {
            "B": ...,
            "G": ...,
            "R": ...,
            "NIR": ...,
            "RE": ...,
            "mean_HFE": ...
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    if radius < 0:
        raise ValueError(
            f"radius deve ser >= 0, recebido {radius}"
        )

    img = img_5b.astype(np.float64)

    # =========================================================
    # 1. Máscara da planta
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
        return {
            "B": 0.0,
            "G": 0.0,
            "R": 0.0,
            "NIR": 0.0,
            "RE": 0.0,
            "mean_HFE": 0.0
        }

    # =========================================================
    # 2. Bounding box da planta
    # =========================================================

    ys, xs = np.where(plant_mask)

    y0 = ys.min()
    y1 = ys.max() + 1

    x0 = xs.min()
    x1 = xs.max() + 1

    cropped_mask = plant_mask[
        y0:y1,
        x0:x1
    ]

    background_crop = ~cropped_mask

    # =========================================================
    # 3. Pixels válidos mais próximos
    # =========================================================

    nearest_indices = None

    if np.any(background_crop):

        _, nearest_indices = distance_transform_edt(
            background_crop,
            return_indices=True
        )

    # =========================================================
    # 4. HFE por banda
    # =========================================================

    band_names = [
        "B",
        "G",
        "R",
        "NIR",
        "RE"
    ]

    results = {}

    for band in range(5):

        cropped = img[
            y0:y1,
            x0:x1,
            band
        ].copy()

        # ---------------------------------------------
        # Preenchimento mask-aware
        # ---------------------------------------------

        if nearest_indices is not None:

            nearest_y = nearest_indices[0]
            nearest_x = nearest_indices[1]

            cropped[background_crop] = cropped[
                nearest_y[background_crop],
                nearest_x[background_crop]
            ]

        # ---------------------------------------------
        # FFT 2D
        # ---------------------------------------------

        fft = np.fft.fft2(cropped)
        fft = np.fft.fftshift(fft)

        power = np.abs(fft) ** 2

        H, W = power.shape

        cy = H // 2
        cx = W // 2

        yy, xx = np.ogrid[:H, :W]

        distance = np.sqrt(
            (yy - cy) ** 2
            +
            (xx - cx) ** 2
        )

        high_frequency_mask = (
            distance > radius
        )

        # ---------------------------------------------
        # HFE
        # ---------------------------------------------

        total_energy = np.sum(power)

        if total_energy <= 1e-12:

            hfe = 0.0

        else:

            high_energy = np.sum(
                power[high_frequency_mask]
            )

            hfe = (
                high_energy
                / total_energy
            )

        results[band_names[band]] = float(hfe)

    # =========================================================
    # 5. HFE multiespectral agregada
    # =========================================================

    results["mean_HFE"] = float(
        np.mean([
            results["B"],
            results["G"],
            results["R"],
            results["NIR"],
            results["RE"]
        ])
    )

    return results['mean_HFE']


#======================================================================

import numpy as np
import cv2

from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity as ssim


def edge_ssim_ESSIM(
    img_a: np.ndarray,
    img_b: np.ndarray,
    sobel_ksize: int = 11,
    ssim_win_size: int = 7,
    channel="luminance",
) -> float:
    """
    Calcula ESSIM — Edge Structural Similarity — de forma mask-aware.

    Segue a definição do artigo:

        E(x) = sqrt(Sobel_x(x)^2 + Sobel_y(x)^2)

        ESSIM(x, x_hat) = SSIM(E(x), E(x_hat))

    Compatível com:

        1. Imagens segmentadas originais
           -> background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score
           -> background = mínimo simultâneo das 5 bandas.

    O background distante da planta não participa da média final,
    mas a fronteira planta/background é preservada, pois contém
    informação importante de shape.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Imagens com shape (H, W, 5), bandas:
        [B, G, R, NIR, RE].

    sobel_ksize : int
        Tamanho do kernel de Sobel.
        Deve ser ímpar.

        No artigo:
            k = 11.

    ssim_win_size : int
        Tamanho da janela do SSIM.
        Deve ser ímpar.

    channel : str ou int
        "luminance":
            usa 0.114 B + 0.587 G + 0.299 R

        ou:
            0 = B
            1 = G
            2 = R
            3 = NIR
            4 = RE

    Returns
    -------
    float
        ESSIM.

        Valores próximos de 1:
            estrutura de bordas preservada.

        Valores menores:
            maior alteração da estrutura de shape.
    """

    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Shapes diferentes: {img_a.shape} vs {img_b.shape}"
        )

    if img_a.ndim != 3 or img_a.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_a.shape}"
        )

    if sobel_ksize <= 0 or sobel_ksize % 2 == 0:
        raise ValueError(
            "sobel_ksize deve ser positivo e ímpar."
        )

    if ssim_win_size < 3 or ssim_win_size % 2 == 0:
        raise ValueError(
            "ssim_win_size deve ser ímpar e >= 3."
        )

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    # =========================================================
    # 1. Máscara da planta
    # =========================================================

    def get_plant_mask(img):

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

        return ~background_mask

    mask_a = get_plant_mask(img_a)
    mask_b = get_plant_mask(img_b)

    if not np.any(mask_a):
        raise ValueError(
            "Nenhuma planta encontrada em img_a."
        )

    if not np.any(mask_b):
        raise ValueError(
            "Nenhuma planta encontrada em img_b."
        )

    # =========================================================
    # 2. Converter para uma representação 2D
    # =========================================================

    def to_channel(img):

        if channel == "luminance":

            B = img[..., 0]
            G = img[..., 1]
            R = img[..., 2]

            return (
                0.114 * B
                + 0.587 * G
                + 0.299 * R
            )

        band = int(channel)

        if band < 0 or band > 4:
            raise ValueError(
                "channel deve ser 'luminance' ou índice de 0 a 4."
            )

        return img[..., band]

    image_a = to_channel(img_a)
    image_b = to_channel(img_b)

    # =========================================================
    # 3. Gradiente Sobel
    # =========================================================

    def sobel_magnitude(image):

        gx = cv2.Sobel(
            image,
            cv2.CV_64F,
            1,
            0,
            ksize=sobel_ksize,
            borderType=cv2.BORDER_REFLECT101
        )

        gy = cv2.Sobel(
            image,
            cv2.CV_64F,
            0,
            1,
            ksize=sobel_ksize,
            borderType=cv2.BORDER_REFLECT101
        )

        return np.sqrt(
            gx ** 2 + gy ** 2
        )

    edges_a = sobel_magnitude(image_a)
    edges_b = sobel_magnitude(image_b)

    # =========================================================
    # 4. SSIM entre os mapas de borda
    # =========================================================

    data_range = max(
        edges_a.max() - edges_a.min(),
        edges_b.max() - edges_b.min(),
        1e-12
    )

    _, ssim_map = ssim(
        edges_a,
        edges_b,
        data_range=data_range,
        win_size=ssim_win_size,
        full=True
    )

    # =========================================================
    # 5. Região relevante para shape
    # =========================================================
    #
    # União, e não interseção:
    #
    # Se Patch Shuffle deslocou uma parte da planta, queremos
    # que essa diferença seja contabilizada.
    #
    # Dilatação inclui alguns pixels ao redor da silhueta,
    # permitindo avaliar corretamente a estrutura da borda.

    evaluation_mask = mask_a | mask_b

    dilation_radius = ssim_win_size // 2

    evaluation_mask = binary_dilation(
        evaluation_mask,
        iterations=dilation_radius
    )

    # =========================================================
    # 6. ESSIM final
    # =========================================================

    score = np.mean(
        ssim_map[evaluation_mask]
    )

    return float(score)


#======================================================================

import numpy as np
import cv2

from scipy.ndimage import binary_dilation


def gradient_correlation_GC(
    img_a: np.ndarray,
    img_b: np.ndarray,
    sobel_ksize: int = 11,
    channel="luminance",
    eps: float = 1e-12,
) -> float:
    """
    Calcula GC — Gradient Correlation — de forma mask-aware.

    Segue a definição:

        GC(x, x_hat) =
            0.5 * [
                corr(gx(x), gx(x_hat))
                +
                corr(gy(x), gy(x_hat))
            ]

    onde gx e gy são gradientes espaciais nas direções x e y.

    Compatível automaticamente com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    O fundo distante da planta não participa da correlação.
    Entretanto, a fronteira planta/background é mantida porque
    representa informação de shape.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Imagens com shape (H, W, 5), bandas:
        [B, G, R, NIR, RE].

    sobel_ksize : int
        Tamanho do kernel de Sobel.

        No artigo:
            k = 11.

    channel : str ou int
        "luminance":
            usa
            0.114*B + 0.587*G + 0.299*R

        ou:
            0 = B
            1 = G
            2 = R
            3 = NIR
            4 = RE

    eps : float
        Constante para estabilidade numérica.

    Returns
    -------
    float
        Gradient Correlation.

        Valores altos:
            estrutura dos gradientes preservada.

        Valores baixos:
            maior alteração espacial / estrutural.
    """

    # =========================================================
    # Validação
    # =========================================================

    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Shapes diferentes: {img_a.shape} vs {img_b.shape}"
        )

    if img_a.ndim != 3 or img_a.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_a.shape}"
        )

    if sobel_ksize <= 0 or sobel_ksize % 2 == 0:
        raise ValueError(
            "sobel_ksize deve ser positivo e ímpar."
        )

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    # =========================================================
    # 1. Máscara da planta
    # =========================================================

    def get_plant_mask(img):

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

        return ~background_mask

    mask_a = get_plant_mask(img_a)
    mask_b = get_plant_mask(img_b)

    if not np.any(mask_a):
        raise ValueError(
            "Nenhuma planta encontrada em img_a."
        )

    if not np.any(mask_b):
        raise ValueError(
            "Nenhuma planta encontrada em img_b."
        )

    # =========================================================
    # 2. Representação 2D
    # =========================================================

    def to_channel(img):

        if channel == "luminance":

            B = img[..., 0]
            G = img[..., 1]
            R = img[..., 2]

            return (
                0.114 * B
                + 0.587 * G
                + 0.299 * R
            )

        band = int(channel)

        if band < 0 or band > 4:
            raise ValueError(
                "channel deve ser 'luminance' ou índice de 0 a 4."
            )

        return img[..., band]

    image_a = to_channel(img_a)
    image_b = to_channel(img_b)

    # =========================================================
    # 3. Gradientes x e y
    # =========================================================

    def gradients(image):

        gx = cv2.Sobel(
            image,
            cv2.CV_64F,
            1,
            0,
            ksize=sobel_ksize,
            borderType=cv2.BORDER_REFLECT101
        )

        gy = cv2.Sobel(
            image,
            cv2.CV_64F,
            0,
            1,
            ksize=sobel_ksize,
            borderType=cv2.BORDER_REFLECT101
        )

        return gx, gy

    gx_a, gy_a = gradients(image_a)
    gx_b, gy_b = gradients(image_b)

    # =========================================================
    # 4. Região relevante
    # =========================================================
    #
    # União das máscaras:
    #
    # se a transformação deslocar partes da planta, queremos
    # considerar tanto a posição antiga quanto a nova.
    #
    # A dilatação inclui o suporte espacial das bordas geradas
    # pelo kernel de Sobel.

    evaluation_mask = (
        mask_a | mask_b
    )

    dilation_radius = (
        sobel_ksize // 2
    )

    evaluation_mask = binary_dilation(
        evaluation_mask,
        iterations=dilation_radius
    )

    # =========================================================
    # 5. Correlação
    # =========================================================

    def correlation(a, b, mask):

        A = a[mask].astype(np.float64)
        B = b[mask].astype(np.float64)

        if A.size < 2:
            return np.nan

        A = A - A.mean()
        B = B - B.mean()

        denominator = np.sqrt(
            np.sum(A ** 2)
            *
            np.sum(B ** 2)
        )

        if denominator <= eps:

            # Ambos sem variação de gradiente
            if (
                np.std(A) <= eps
                and np.std(B) <= eps
            ):
                return 1.0

            return 0.0

        rho = (
            np.sum(A * B)
            / denominator
        )

        # Proteção contra pequenos erros numéricos
        return float(
            np.clip(rho, -1.0, 1.0)
        )

    corr_x = correlation(
        gx_a,
        gx_b,
        evaluation_mask
    )

    corr_y = correlation(
        gy_a,
        gy_b,
        evaluation_mask
    )

    # =========================================================
    # 6. GC
    # =========================================================

    if np.isnan(corr_x) or np.isnan(corr_y):
        return np.nan

    GC = 0.5 * (
        corr_x + corr_y
    )

    return float(GC)

#======================================================================
#======================================================================
#======================================================================
# Minhas

# GPT

#======================================================================

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


def shape_descriptors_TEST(img_5b: np.ndarray) -> dict:

    data = shape_descriptors(img_5b)
    return data['hu_2']

def shape_descriptors__area(img_5b: np.ndarray) -> dict:
    data = shape_descriptors(img_5b)
    return data['area']

def shape_descriptors__perimeter(img_5b: np.ndarray) -> dict:
    data = shape_descriptors(img_5b)
    return data['perimeter']

def shape_descriptors__compactness(img_5b: np.ndarray) -> dict:
    data = shape_descriptors(img_5b)
    return data['compactness']

def shape_descriptors__solidity(img_5b: np.ndarray) -> dict:
    data = shape_descriptors(img_5b)
    return data['solidity']

def shape_descriptors__hu_1(img_5b: np.ndarray) -> dict:
    data = shape_descriptors(img_5b)
    return data['hu_1']

def shape_descriptors__hu_2(img_5b: np.ndarray) -> dict:
    data = shape_descriptors(img_5b)
    return data['hu_2']

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

#----------------------------------------------------------------------

def shape_descriptors_cl_TEST(
    img_5b: np.ndarray,
    connectivity: int = 2,
    log_hu: bool = True,
) -> dict:

    data =  shape_descriptors_cl(
        img_5b,
        connectivity,
        log_hu)

    return data['area']

#----------------------------------------------------------------------

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

#----------------------------------------------------------------------

from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt


def coarse_ssim_mask_aware(
    img_a: np.ndarray,
    img_b: np.ndarray,
    downscale_factor: int = 16,
    channel="luminance",
) -> float:
    """
    Calcula SSIM entre versões coarse/downsampled de duas imagens,
    ignorando o background.

    Compatível com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    O average pooling é feito somente sobre pixels da planta.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Imagens com shape (H, W, 5), bandas:
        [B, G, R, NIR, RE].

    downscale_factor : int
        Fator de redução espacial.

    channel : str ou int
        "luminance":
            0.114*B + 0.587*G + 0.299*R

        ou índice:
            0 = B
            1 = G
            2 = R
            3 = NIR
            4 = RE

    Returns
    -------
    float
        SSIM coarse calculado apenas sobre a região válida.
    """

    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Shapes diferentes: {img_a.shape} vs {img_b.shape}"
        )

    if (
        img_a.ndim != 3
        or img_a.shape[-1] != 5
    ):
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_a.shape}"
        )

    if downscale_factor <= 0:
        raise ValueError(
            "downscale_factor deve ser > 0"
        )

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    # =========================================================
    # 1. Detectar máscara da planta
    # =========================================================

    def get_plant_mask(img):

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

        return ~background_mask

    mask_a = get_plant_mask(img_a)
    mask_b = get_plant_mask(img_b)

    if not np.any(mask_a):
        raise ValueError(
            "Nenhum pixel de planta encontrado em img_a."
        )

    if not np.any(mask_b):
        raise ValueError(
            "Nenhum pixel de planta encontrado em img_b."
        )

    # =========================================================
    # 2. Escolher canal
    # =========================================================

    def to_channel(img):

        if channel == "luminance":

            B = img[..., 0]
            G = img[..., 1]
            R = img[..., 2]

            return (
                0.114 * B
                + 0.587 * G
                + 0.299 * R
            )

        band = int(channel)

        if band < 0 or band > 4:
            raise ValueError(
                "channel deve ser 'luminance' ou índice de 0 a 4"
            )

        return img[..., band]

    ch_a = to_channel(img_a)
    ch_b = to_channel(img_b)

    # =========================================================
    # 3. Average pooling mask-aware
    # =========================================================

    def downsample_mask_aware(
        img_2d,
        mask,
        factor
    ):

        H, W = img_2d.shape

        H_crop = H - (H % factor)
        W_crop = W - (W % factor)

        img_crop = img_2d[
            :H_crop,
            :W_crop
        ]

        mask_crop = mask[
            :H_crop,
            :W_crop
        ]

        new_H = H_crop // factor
        new_W = W_crop // factor

        img_blocks = img_crop.reshape(
            new_H,
            factor,
            new_W,
            factor
        )

        mask_blocks = mask_crop.reshape(
            new_H,
            factor,
            new_W,
            factor
        )

        # Soma apenas dos pixels da planta
        weighted_sum = np.sum(
            img_blocks * mask_blocks,
            axis=(1, 3)
        )

        # Quantidade de pixels válidos por bloco
        valid_count = np.sum(
            mask_blocks,
            axis=(1, 3)
        )

        coarse = np.zeros(
            (new_H, new_W),
            dtype=np.float64
        )

        valid = valid_count > 0

        coarse[valid] = (
            weighted_sum[valid]
            / valid_count[valid]
        )

        return coarse, valid

    coarse_a, valid_a = downsample_mask_aware(
        ch_a,
        mask_a,
        downscale_factor
    )

    coarse_b, valid_b = downsample_mask_aware(
        ch_b,
        mask_b,
        downscale_factor
    )

    # Região válida nas duas imagens
    common_mask = valid_a & valid_b

    if not np.any(common_mask):
        raise ValueError(
            "Não há região válida comum após downsampling."
        )

    # =========================================================
    # 4. Preencher células coarse sem planta
    # =========================================================
    #
    # SSIM precisa de uma imagem retangular completa.
    # Portanto células inválidas são preenchidas temporariamente
    # pelo valor da célula válida mais próxima.
    #
    # Elas NÃO entram na média final do SSIM.

    def fill_invalid(coarse, valid):

        if np.all(valid):
            return coarse

        invalid = ~valid

        _, nearest = distance_transform_edt(
            invalid,
            return_indices=True
        )

        filled = coarse.copy()

        filled[invalid] = coarse[
            nearest[0][invalid],
            nearest[1][invalid]
        ]

        return filled

    coarse_a_filled = fill_invalid(
        coarse_a,
        valid_a
    )

    coarse_b_filled = fill_invalid(
        coarse_b,
        valid_b
    )

    # =========================================================
    # 5. SSIM
    # =========================================================

    if (
        coarse_a.shape[0] < 7
        or coarse_a.shape[1] < 7
    ):
        raise ValueError(
            f"Imagem coarse muito pequena "
            f"{coarse_a.shape} para SSIM."
        )

    data_range = max(
        coarse_a_filled.max()
        - coarse_a_filled.min(),

        coarse_b_filled.max()
        - coarse_b_filled.min(),

        1e-8
    )

    _, ssim_map = ssim(
        coarse_a_filled,
        coarse_b_filled,
        data_range=data_range,
        full=True
    )

    # =========================================================
    # 6. Média apenas sobre região da planta
    # =========================================================

    score = np.mean(
        ssim_map[common_mask]
    )

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

#----------------------------------------------------------------------

import numpy as np
from scipy.ndimage import laplace, distance_transform_edt


def laplacian_variance_mask_aware(
    img_5b: np.ndarray,
    channel="luminance",
) -> float:
    """
    Calcula a Variância do Laplaciano de forma mask-aware.

    Mede conteúdo espacial de alta frequência / nitidez local.

    Compatível automaticamente com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    O background não participa da métrica e sua fronteira artificial
    com a planta não é interpretada como textura.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral com shape (H, W, 5),
        bandas [B, G, R, NIR, RE].

    channel : str ou int
        "luminance":
            usa 0.114*B + 0.587*G + 0.299*R

        ou:
            0 = B
            1 = G
            2 = R
            3 = NIR
            4 = RE

    Returns
    -------
    float
        Variância do Laplaciano calculada somente sobre a planta.

        Valores maiores:
            maior quantidade de detalhes / altas frequências.

        Valores menores:
            maior suavização / blur.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # =========================================================
    # 1. Detectar background
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
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # 2. Selecionar representação 2D
    # =========================================================

    if channel == "luminance":

        B = img[..., 0]
        G = img[..., 1]
        R = img[..., 2]

        image_2d = (
            0.114 * B
            + 0.587 * G
            + 0.299 * R
        )

    else:

        band = int(channel)

        if band < 0 or band > 4:
            raise ValueError(
                "channel deve ser 'luminance' ou índice de 0 a 4."
            )

        image_2d = img[..., band]

    # =========================================================
    # 3. Preenchimento temporário do background
    # =========================================================
    #
    # Evita que a transição planta -> background seja vista
    # pelo Laplaciano como uma borda de alta frequência.

    filled = image_2d.copy()

    if np.any(background_mask):

        _, nearest_indices = distance_transform_edt(
            background_mask,
            return_indices=True
        )

        nearest_y = nearest_indices[0]
        nearest_x = nearest_indices[1]

        filled[background_mask] = image_2d[
            nearest_y[background_mask],
            nearest_x[background_mask]
        ]

    # =========================================================
    # 4. Laplaciano
    # =========================================================

    lap = laplace(filled)

    # =========================================================
    # 5. Variância somente sobre a planta
    # =========================================================

    values = lap[plant_mask]

    if values.size < 2:
        return 0.0

    return float(
        np.var(values)
    )

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

def glcm_contrast_energy_TEST(
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






# energy
# energy_per_distance

def glcm_contrast_energy__energy(
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

        return data["energy"]

def glcm_contrast_energy__energy_per_distance(
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


def glcm_contrast_energy__New(
    img_5b: np.ndarray,
    channel="luminance",
    distances=(1, 2),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels=32,
) -> dict:
    """
    Calcula GLCM Contrast e Energy de forma mask-aware.

    Compatível automaticamente com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    Somente pares de pixels em que AMBOS pertencem à planta
    são utilizados na construção da GLCM.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5),
        bandas [B, G, R, NIR, RE].

    channel : str ou int
        "luminance":
            0.114*B + 0.587*G + 0.299*R

        ou:
            0 = B
            1 = G
            2 = R
            3 = NIR
            4 = RE

    distances : sequência de int
        Distâncias espaciais entre os pares.

    angles : sequência de float
        Direções em radianos.

    n_levels : int
        Número de níveis usados na quantização.

    Returns
    -------
    dict
        {
            "contrast": ...,
            "energy": ...,
            "contrast_per_distance": {...},
            "energy_per_distance": {...}
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    if n_levels < 2:
        raise ValueError(
            "n_levels deve ser >= 2."
        )

    img_5b = img_5b.astype(np.float64)

    H, W, C = img_5b.shape

    # =========================================================
    # 1. Máscara da planta
    # =========================================================

    background_values = np.min(
        img_5b,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img_5b,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    plant_mask = ~background_mask

    if not np.any(plant_mask):
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # 2. Representação 2D
    # =========================================================

    if channel == "luminance":

        B = img_5b[..., 0]
        G = img_5b[..., 1]
        R = img_5b[..., 2]

        img = (
            0.114 * B
            + 0.587 * G
            + 0.299 * R
        )

    else:

        band = int(channel)

        if band < 0 or band > 4:
            raise ValueError(
                "channel deve ser 'luminance' ou índice de 0 a 4."
            )

        img = img_5b[..., band]

    # =========================================================
    # 3. Quantização SOMENTE usando a planta
    # =========================================================

    plant_values = img[plant_mask]

    img_min = np.min(plant_values)
    img_max = np.max(plant_values)

    if img_max - img_min < 1e-12:

        return {
            "contrast": 0.0,
            "energy": 1.0,
            "contrast_per_distance": {
                d: 0.0 for d in distances
            },
            "energy_per_distance": {
                d: 1.0 for d in distances
            },
        }

    img_q = np.zeros(
        (H, W),
        dtype=np.int32
    )

    normalized = (
        (plant_values - img_min)
        / (img_max - img_min)
    )

    quantized = np.floor(
        normalized * n_levels
    ).astype(np.int32)

    quantized = np.clip(
        quantized,
        0,
        n_levels - 1
    )

    img_q[plant_mask] = quantized

    # =========================================================
    # 4. Função para construir GLCM mascarada
    # =========================================================

    def masked_glcm(distance, angle):

        dx = int(
            np.round(
                distance * np.cos(angle)
            )
        )

        dy = int(
            np.round(
                distance * np.sin(angle)
            )
        )

        # Região central
        y1_start = max(0, -dy)
        y1_end = min(H, H - dy)

        x1_start = max(0, -dx)
        x1_end = min(W, W - dx)

        # Região deslocada
        y2_start = y1_start + dy
        y2_end = y1_end + dy

        x2_start = x1_start + dx
        x2_end = x1_end + dx

        A = img_q[
            y1_start:y1_end,
            x1_start:x1_end
        ]

        B = img_q[
            y2_start:y2_end,
            x2_start:x2_end
        ]

        mask_A = plant_mask[
            y1_start:y1_end,
            x1_start:x1_end
        ]

        mask_B = plant_mask[
            y2_start:y2_end,
            x2_start:x2_end
        ]

        # Ambos precisam pertencer à planta
        valid = (
            mask_A & mask_B
        )

        if not np.any(valid):
            return None

        values_A = A[valid]
        values_B = B[valid]

        glcm = np.zeros(
            (n_levels, n_levels),
            dtype=np.float64
        )

        np.add.at(
            glcm,
            (values_A, values_B),
            1
        )

        # GLCM simétrica
        glcm = glcm + glcm.T

        total = np.sum(glcm)

        if total == 0:
            return None

        # Normalização
        glcm /= total

        return glcm

    # =========================================================
    # 5. Contrast e Energy
    # =========================================================

    indices = np.arange(n_levels)

    I, J = np.meshgrid(
        indices,
        indices,
        indexing="ij"
    )

    contrast_per_d = {}
    energy_per_d = {}

    for d in distances:

        contrasts = []
        energies = []

        for angle in angles:

            glcm = masked_glcm(
                d,
                angle
            )

            if glcm is None:
                continue

            # Haralick Contrast
            contrast = np.sum(
                glcm * (I - J) ** 2
            )

            # Energy = sqrt(ASM)
            energy = np.sqrt(
                np.sum(glcm ** 2)
            )

            contrasts.append(
                contrast
            )

            energies.append(
                energy
            )

        if len(contrasts) > 0:

            contrast_per_d[d] = float(
                np.mean(contrasts)
            )

            energy_per_d[d] = float(
                np.mean(energies)
            )

    if len(contrast_per_d) == 0:
        return {
            "contrast": np.nan,
            "energy": np.nan,
            "contrast_per_distance": {},
            "energy_per_distance": {},
        }

    return {
        "contrast": float(
            np.mean(
                list(
                    contrast_per_d.values()
                )
            )
        ),

        "energy": float(
            np.mean(
                list(
                    energy_per_d.values()
                )
            )
        ),

        "contrast_per_distance":
            contrast_per_d,

        "energy_per_distance":
            energy_per_d,
    }



def glcm_contrast_energy__New__energy(
    img_5b: np.ndarray,
    channel="luminance",
    distances=(1, 2),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels=32):

        data = glcm_contrast_energy__New(
            img_5b,
            channel,
            distances,
            angles,
            n_levels)

        return data["energy"]


def glcm_contrast_energy__New__contrast(
    img_5b: np.ndarray,
    channel="luminance",
    distances=(1, 2),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    n_levels=32):

        data = glcm_contrast_energy__New(
            img_5b,
            channel,
            distances,
            angles,
            n_levels)

        return data["contrast"]

#======================================================================
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


def high_low_freq_energy_ratio_TEST(
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

def high_low_freq_energy_ratio__energy_low(
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

def high_low_freq_energy_ratio__energy_high(
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

    return data["energy_high"]

def high_low_freq_energy_ratio__ratio(
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

    return data["ratio"]


# cutoff_fraction = [0.05, 0.15, 0.25]
# mask_background = [True, False]

#----------------------------------------------------------------------


import numpy as np
from scipy.ndimage import distance_transform_edt


def high_low_freq_energy_ratio_new(
    img_5b: np.ndarray,
    channel="luminance",
    cutoff_fraction: float = 0.15,
) -> dict:
    """
    Calcula a razão entre energia de alta e baixa frequência via FFT,
    de forma mask-aware.

    Compatível automaticamente com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    O background não participa diretamente da análise e a fronteira
    artificial planta/background é removida antes da FFT.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral com shape (H, W, 5),
        bandas [B, G, R, NIR, RE].

    channel : str ou int
        "luminance":
            0.114*B + 0.587*G + 0.299*R

        ou:
            0 = B
            1 = G
            2 = R
            3 = NIR
            4 = RE

    cutoff_fraction : float
        Fração do raio máximo do espectro usada para separar
        baixas e altas frequências.

        Exemplo:
            0.15 -> frequências radiais abaixo de 15% são baixas.

    Returns
    -------
    dict
        {
            "ratio": energy_high / energy_low,
            "energy_high": ...,
            "energy_low": ...
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    if not 0.0 < cutoff_fraction < 1.0:
        raise ValueError(
            "cutoff_fraction deve estar entre 0 e 1."
        )

    img_5b = img_5b.astype(np.float64)

    # =========================================================
    # 1. Máscara da planta
    # =========================================================

    background_values = np.min(
        img_5b,
        axis=(0, 1)
    )

    background_mask = np.all(
        np.isclose(
            img_5b,
            background_values[None, None, :],
            rtol=1e-5,
            atol=1e-8
        ),
        axis=-1
    )

    plant_mask = ~background_mask

    if not np.any(plant_mask):
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # 2. Representação 2D
    # =========================================================

    if channel == "luminance":

        B = img_5b[..., 0]
        G = img_5b[..., 1]
        R = img_5b[..., 2]

        img = (
            0.114 * B
            + 0.587 * G
            + 0.299 * R
        )

    else:

        band = int(channel)

        if band < 0 or band > 4:
            raise ValueError(
                "channel deve ser 'luminance' ou índice de 0 a 4."
            )

        img = img_5b[..., band]

    # =========================================================
    # 3. Bounding box da planta
    # =========================================================

    ys, xs = np.where(plant_mask)

    y0 = ys.min()
    y1 = ys.max() + 1

    x0 = xs.min()
    x1 = xs.max() + 1

    cropped = img[
        y0:y1,
        x0:x1
    ].copy()

    cropped_mask = plant_mask[
        y0:y1,
        x0:x1
    ]

    Hc, Wc = cropped.shape

    if Hc < 2 or Wc < 2:
        raise ValueError(
            "Região da planta pequena demais para análise FFT."
        )

    # =========================================================
    # 4. Remover componente média da planta
    # =========================================================

    plant_mean = np.mean(
        cropped[cropped_mask]
    )

    cropped = cropped - plant_mean

    # =========================================================
    # 5. Preencher background pelo pixel de planta mais próximo
    # =========================================================
    #
    # Evita que planta -> fundo seja interpretado como
    # alta frequência artificial.

    background_crop = ~cropped_mask

    if np.any(background_crop):

        _, nearest_indices = distance_transform_edt(
            background_crop,
            return_indices=True
        )

        nearest_y = nearest_indices[0]
        nearest_x = nearest_indices[1]

        cropped[background_crop] = cropped[
            nearest_y[background_crop],
            nearest_x[background_crop]
        ]

    # =========================================================
    # 6. Janela de Hann
    # =========================================================

    win_y = np.hanning(Hc)
    win_x = np.hanning(Wc)

    window = np.outer(
        win_y,
        win_x
    )

    img_win = cropped * window

    # =========================================================
    # 7. FFT e espectro de potência
    # =========================================================

    F = np.fft.fft2(img_win)

    F = np.fft.fftshift(F)

    power = np.abs(F) ** 2

    # =========================================================
    # 8. Frequência radial normalizada
    # =========================================================

    cy = Hc // 2
    cx = Wc // 2

    y, x = np.indices(
        (Hc, Wc)
    )

    distance = np.sqrt(
        ((y - cy) / Hc) ** 2
        +
        ((x - cx) / Wc) ** 2
    )

    r_max = np.max(distance)

    if r_max <= 0:
        return {
            "ratio": 0.0,
            "energy_high": 0.0,
            "energy_low": 0.0,
        }

    r_norm = (
        distance / r_max
    )

    # =========================================================
    # 9. Baixas e altas frequências
    # =========================================================

    low_mask = (
        r_norm < cutoff_fraction
    )

    high_mask = ~low_mask

    energy_low = float(
        np.sum(power[low_mask])
    )

    energy_high = float(
        np.sum(power[high_mask])
    )

    # =========================================================
    # 10. Razão
    # =========================================================

    if energy_low > 1e-12:

        ratio = (
            energy_high
            / energy_low
        )

    else:

        ratio = np.inf

    return {
        "ratio": float(ratio),
        "energy_high": energy_high,
        "energy_low": energy_low,
    }


def high_low_freq_energy_ratio_new__ratio(
    img_5b: np.ndarray,
    channel="luminance",
    cutoff_fraction: float = 0.15,
    ):
    data = high_low_freq_energy_ratio_new(
    img_5b,
    channel,
    cutoff_fraction,
    )
    return data['ratio']

def high_low_freq_energy_ratio_new__energy_high(
    img_5b: np.ndarray,
    channel="luminance",
    cutoff_fraction: float = 0.15,
    ):
    data = high_low_freq_energy_ratio_new(
    img_5b,
    channel,
    cutoff_fraction,
    )
    return data['energy_high']

def high_low_freq_energy_ratio_new__energy_low(
    img_5b: np.ndarray,
    channel="luminance",
    cutoff_fraction: float = 0.15,
    ):
    data = high_low_freq_energy_ratio_new(
    img_5b,
    channel,
    cutoff_fraction,
    )
    return data['energy_low']

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


import numpy as np


def mean_chroma_zscore(img_5b: np.ndarray) -> float:
    """
    Mede a cromaticidade média de uma imagem segmentada
    e normalizada por Z-score.

    Não calcula CIELAB Chroma, pois os valores RGB originais
    não podem ser recuperados sem mean/std.

    A métrica mede, para cada pixel, a distância dos canais
    RGB ao eixo acromático R = G = B.

    Quanto maior o valor:
        -> maior a diferença entre R, G e B
        -> maior a informação cromática relativa

    Quanto menor o valor:
        -> R, G e B mais semelhantes
        -> imagem mais próxima de grayscale

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5), com bandas:
        [B, G, R, NIR, RE].

        Assume-se que:
        - os canais estão normalizados por Z-score;
        - o fundo segmentado permanece exatamente zero.

    Returns
    -------
    float
        Cromaticidade média da região da planta.
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    # ---------------------------------------------------------
    # 1. Máscara da planta
    # ---------------------------------------------------------
    # Usa as 5 bandas para identificar fundo.
    mask = np.any(img_5b != 0, axis=-1)

    if not np.any(mask):
        return np.nan

    # ---------------------------------------------------------
    # 2. Extrai RGB normalizado
    # ---------------------------------------------------------
    B = img_5b[..., 0].astype(np.float64)
    G = img_5b[..., 1].astype(np.float64)
    R = img_5b[..., 2].astype(np.float64)

    # ---------------------------------------------------------
    # 3. Eixo acromático
    #
    # Um pixel grayscale satisfaz aproximadamente:
    # R = G = B
    # ---------------------------------------------------------
    mean_rgb = (R + G + B) / 3.0

    # Distância ao eixo R = G = B
    chroma = np.sqrt(
        (R - mean_rgb) ** 2 +
        (G - mean_rgb) ** 2 +
        (B - mean_rgb) ** 2
    )

    # ---------------------------------------------------------
    # 4. Média somente na planta
    # ---------------------------------------------------------
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

def mean_chroma__mean_chroma(
    img_5b: np.ndarray,
    mask_background: bool = True,
    input_range: tuple = None,
):
    
    data =  mean_chroma(
    img_5b,
    mask_background,
    input_range)

    return data["mean_chroma"]


def mean_chroma__std_chroma(
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

import numpy as np
from skimage.color import rgb2lab


def mean_chroma_new(
    img_5b: np.ndarray,
    input_range=None,
) -> dict:
    """
    Calcula a chroma média CIELAB:

        C* = sqrt(a*^2 + b*^2)

    de forma mask-aware.

    Compatível com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    Importante
    ----------
    Para imagens Z-score, o resultado deve ser interpretado como
    uma medida RELATIVA de chroma no espaço normalizado, pois uma
    normalização independente por banda altera as relações físicas
    entre R, G e B.

    Para comparar imagem original e transformada, recomenda-se usar
    o MESMO input_range em ambas.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem (H, W, 5), bandas [B, G, R, NIR, RE].

    input_range : tuple ou None
        Faixa (min, max) usada para mapear RGB para [0, 1].

        Se None, a faixa é estimada apenas sobre pixels da planta.
        Para comparações rigorosas, prefira fornecer uma faixa fixa.

    Returns
    -------
    dict
        {
            "mean_chroma": float,
            "std_chroma": float
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # =========================================================
    # 1. Máscara da planta
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
        return {
            "mean_chroma": 0.0,
            "std_chroma": 0.0
        }

    # =========================================================
    # 2. RGB
    # =========================================================

    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    rgb = np.stack(
        [R, G, B],
        axis=-1
    )

    # =========================================================
    # 3. Faixa RGB
    # =========================================================

    if input_range is None:

        # Apenas pixels da planta determinam a escala.
        plant_rgb = rgb[plant_mask]

        lo = np.min(plant_rgb)
        hi = np.max(plant_rgb)

    else:

        lo, hi = input_range

    if hi - lo < 1e-12:
        return {
            "mean_chroma": 0.0,
            "std_chroma": 0.0
        }

    # =========================================================
    # 4. RGB -> [0,1]
    # =========================================================

    rgb_norm = (
        (rgb - lo)
        / (hi - lo)
    )

    rgb_norm = np.clip(
        rgb_norm,
        0.0,
        1.0
    )

    # =========================================================
    # 5. RGB -> Lab
    # =========================================================

    lab = rgb2lab(rgb_norm)

    a_ch = lab[..., 1]
    b_ch = lab[..., 2]

    chroma = np.sqrt(
        a_ch**2
        + b_ch**2
    )

    # =========================================================
    # 6. Somente planta
    # =========================================================

    chroma_values = chroma[
        plant_mask
    ]

    return {
        "mean_chroma": float(
            np.mean(chroma_values)
        ),
        "std_chroma": float(
            np.std(chroma_values)
        ),
    }


def mean_chroma_new__mean_chroma(
    img_5b: np.ndarray,
    input_range=None,
):
    data = mean_chroma_new(
    img_5b,
    input_range,
    ) 
    return data['mean_chroma']

def mean_chroma_new__std_chroma(
    img_5b: np.ndarray,
    input_range=None,
):
    data = mean_chroma_new(
    img_5b,
    input_range,
    ) 
    return data['std_chroma']


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


def rgb_channel_divergence_GPT__rgb_divergence(img_5b: np.ndarray):
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


import numpy as np
from skimage.color import rgb2hsv


def hue_distribution_metrics_GPT(
    img_5b: np.ndarray,
    n_bins: int = 36,
    saturation_threshold: float = 0.05,
    eps: float = 1e-12,
    robust_percentiles=(1.0, 99.0)
) -> dict:
    """
    Calcula métricas da distribuição de Hue em uma imagem multiespectral
    segmentada e normalizada por Z-score.

    Métricas:
        1. Entropia normalizada do histograma de Hue
        2. Variância circular do Hue
        3. Fração de pixels com Hue válido

    A imagem é assumida como:
        - shape (H, W, 5)
        - bandas [B, G, R, NIR, RE]
        - fundo = 0
        - região da planta com valores possivelmente negativos devido
          à normalização Z-score.

    Como mean/std originais não estão disponíveis, as bandas RGB são
    reescaladas separadamente para [0,1], utilizando percentis calculados
    apenas sobre os pixels da planta.

    IMPORTANTE
    ----------
    O Hue obtido não corresponde exatamente ao Hue da imagem RGB original,
    porque uma normalização Z-score independente por banda altera as
    relações relativas entre R, G e B.

    Portanto, estas métricas devem ser interpretadas como métricas de
    distribuição cromática RELATIVA da imagem normalizada.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral (H, W, 5), bandas:
        [B, G, R, NIR, RE].

    n_bins : int
        Número de bins do histograma circular de Hue.

    saturation_threshold : float
        Saturação mínima para considerar Hue válido.

    eps : float
        Estabilidade numérica.

    robust_percentiles : tuple
        Percentis inferior e superior utilizados para reescalar
        cada banda RGB para [0,1].

        Exemplo:
            (1, 99)

        reduz a influência de valores extremos.

    Returns
    -------
    dict
        {
            "hue_entropy": float,
            "hue_circular_variance": float,
            "valid_hue_fraction": float
        }
    """

    # ---------------------------------------------------------
    # 0. Validação
    # ---------------------------------------------------------
    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # ---------------------------------------------------------
    # 1. Máscara da planta
    #
    # Importante:
    # não usamos apenas RGB, pois um pixel da planta poderia,
    # em princípio, possuir RGB próximo/igual a zero após Z-score.
    # ---------------------------------------------------------
    plant_mask = np.any(img != 0, axis=-1)

    n_plant = np.count_nonzero(plant_mask)

    if n_plant == 0:
        return {
            "hue_entropy": np.nan,
            "hue_circular_variance": np.nan,
            "valid_hue_fraction": 0.0
        }

    # ---------------------------------------------------------
    # 2. Extrai RGB
    # ---------------------------------------------------------
    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    rgb_z = np.stack([R, G, B], axis=-1)

    # ---------------------------------------------------------
    # 3. Reconstrói uma representação RGB relativa [0,1]
    #
    # Cada banda é reescalada usando apenas pixels da planta.
    # ---------------------------------------------------------
    rgb = np.zeros_like(rgb_z, dtype=np.float64)

    p_low, p_high = robust_percentiles

    for c in range(3):

        channel = rgb_z[..., c]

        values = channel[plant_mask]

        low = np.percentile(values, p_low)
        high = np.percentile(values, p_high)

        if high - low > eps:

            channel_scaled = (
                (channel - low) /
                (high - low)
            )

            channel_scaled = np.clip(
                channel_scaled,
                0.0,
                1.0
            )

            # Mantém fundo exatamente zero
            channel_scaled[~plant_mask] = 0.0

            rgb[..., c] = channel_scaled

        else:
            # Banda praticamente constante na planta
            rgb[..., c] = 0.0

    # ---------------------------------------------------------
    # 4. RGB relativo -> HSV
    # ---------------------------------------------------------
    hsv = rgb2hsv(rgb)

    hue = hsv[..., 0]
    saturation = hsv[..., 1]

    # ---------------------------------------------------------
    # 5. Pixels com Hue válido
    # ---------------------------------------------------------
    valid_mask = (
        plant_mask &
        np.isfinite(hue) &
        np.isfinite(saturation) &
        (saturation > saturation_threshold)
    )

    n_valid = np.count_nonzero(valid_mask)

    valid_hue_fraction = (
        n_valid / n_plant
    )

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

    p_sum = p.sum()

    if p_sum <= eps:
        entropy_normalized = 0.0

    else:
        p /= p_sum

        p_nonzero = p[p > 0]

        entropy = -np.sum(
            p_nonzero * np.log2(p_nonzero)
        )

        entropy_normalized = (
            entropy / np.log2(n_bins)
        )

    # =========================================================
    # 7. VARIÂNCIA CIRCULAR
    # =========================================================

    # Hue [0,1] -> ângulo [0, 2π)
    theta = 2.0 * np.pi * H

    mean_cos = np.mean(np.cos(theta))
    mean_sin = np.mean(np.sin(theta))

    R_bar = np.sqrt(
        mean_cos**2 +
        mean_sin**2
    )

    circular_variance = 1.0 - R_bar

    # Pequenas correções numéricas
    entropy_normalized = np.clip(
        entropy_normalized, 0.0, 1.0
    )

    circular_variance = np.clip(
        circular_variance, 0.0, 1.0
    )

    return {
        "hue_entropy": float(entropy_normalized),
        "hue_circular_variance": float(circular_variance),
        "valid_hue_fraction": float(valid_hue_fraction)
    }


def hue_distribution_metrics_GPT__valid_hue_fraction(
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


def hue_distribution_metrics_GPT__hue_circular_variance(
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

    return data["hue_circular_variance"]

def hue_distribution_metrics_GPT__hue_entropy(
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

    return data["hue_entropy"]

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
    
    return data["circular_variance"]

#======================================================================
#======================================================================
# Espectral

import numpy as np


def spectral_intraband_variance(
    img_5b: np.ndarray
) -> dict:
    """
    Calcula a variância intrabanda das bandas espectrais NIR e RE.

    Bandas esperadas:
        [B, G, R, NIR, RE]

    A métrica é calculada somente sobre os pixels pertencentes
    à planta.

    Compatível automaticamente com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    Returns
    -------
    dict
        {
            "nir_variance": Var(NIR),
            "re_variance":  Var(RE),
            "mean_variance": média das duas variâncias
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # =========================================================
    # 1. Detectar background
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
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # 2. NIR e RE somente na região da planta
    # =========================================================

    nir = img[..., 3][plant_mask]
    re = img[..., 4][plant_mask]

    # =========================================================
    # 3. Variâncias intrabanda
    # =========================================================

    nir_variance = np.var(nir)
    re_variance = np.var(re)

    mean_variance = (
        nir_variance + re_variance
    ) / 2.0

    return {
        "nir_variance": float(nir_variance),
        "re_variance": float(re_variance),
        "mean_variance": float(mean_variance),
    }

def spectral_intraband_variance__mean_variance(
    img_5b: np.ndarray
):
    data = spectral_intraband_variance(img_5b)
    return data['mean_variance']

#======================================================================

import numpy as np


def spectral_correlation_with_green(
    img_5b: np.ndarray,
    eps: float = 1e-12
) -> dict:
    """
    Calcula a correlação de Pearson entre as bandas espectrais
    NIR/RE e a banda Green.

    Bandas esperadas:
        [B, G, R, NIR, RE]

    Métricas:

        rho(NIR, Green)
        rho(RE, Green)

    Compatível automaticamente com:

        1. Imagens segmentadas originais:
           background = 0 nas 5 bandas.

        2. Imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das 5 bandas.

    A correlação é calculada somente sobre os pixels da planta.

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem multiespectral com shape (H, W, 5),
        bandas [B, G, R, NIR, RE].

    eps : float
        Constante para verificar variância praticamente nula.

    Returns
    -------
    dict
        {
            "nir_green_correlation": rho(NIR, Green),
            "re_green_correlation": rho(RE, Green),
            "mean_correlation": média das duas correlações
        }
    """

    if img_5b.ndim != 3 or img_5b.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_5b.shape}"
        )

    img = img_5b.astype(np.float64)

    # =========================================================
    # 1. Detectar background
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
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # 2. Extrair bandas somente na planta
    # =========================================================

    green = img[..., 1][plant_mask]
    nir = img[..., 3][plant_mask]
    re = img[..., 4][plant_mask]

    # =========================================================
    # 3. Correlação de Pearson
    # =========================================================

    def pearson_correlation(x, y):

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        denominator = np.sqrt(
            np.sum(x_centered ** 2)
            *
            np.sum(y_centered ** 2)
        )

        # Correlação não é definida se uma das bandas
        # for praticamente constante.
        if denominator <= eps:
            return np.nan

        rho = (
            np.sum(
                x_centered * y_centered
            )
            / denominator
        )

        return float(
            np.clip(rho, -1.0, 1.0)
        )

    nir_green = pearson_correlation(
        nir,
        green
    )

    re_green = pearson_correlation(
        re,
        green
    )

    correlations = np.array(
        [nir_green, re_green],
        dtype=np.float64
    )

    if np.all(np.isnan(correlations)):
        mean_correlation = np.nan
    else:
        mean_correlation = np.nanmean(
            correlations
        )

    return {
        "nir_green_correlation":
            float(nir_green),

        "re_green_correlation":
            float(re_green),

        "mean_correlation":
            float(mean_correlation),
    }


def spectral_correlation_with_green__mean_correlation(
    img_5b: np.ndarray,
    eps: float = 1e-12
):
    data = spectral_correlation_with_green(
    img_5b,
    eps) 
    return data['mean_correlation']

#======================================================================

import numpy as np
from scipy.stats import wasserstein_distance


def spectral_wasserstein_distance(
    img_a: np.ndarray,
    img_b: np.ndarray,
    eps: float = 1e-12
) -> dict:
    """
    Mede a diferença espectral entre duas imagens usando
    distância de Wasserstein nas bandas NIR e RE.

    A comparação é feita entre as distribuições dos pixels da planta,
    portanto não depende da posição espacial dos pixels.

    Particularmente útil para:
        - NIR/RE -> própria média;
        - NIR/RE -> Green.

    Patch Shuffle e Patch Rotation, que apenas reorganizam pixels,
    tendem a preservar esta medida.

    Compatível com:
        - imagens segmentadas originais;
        - imagens segmentadas normalizadas por Z-score.

    Background:
        identificado como o mínimo simultâneo das 5 bandas.

    Returns
    -------
    dict
        {
            "NIR": distância Wasserstein normalizada,
            "RE": distância Wasserstein normalizada,
            "mean_spectral_distance": média NIR/RE
        }
    """

    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Shapes diferentes: {img_a.shape} vs {img_b.shape}"
        )

    if img_a.ndim != 3 or img_a.shape[-1] != 5:
        raise ValueError(
            f"Esperado array (H, W, 5), recebido {img_a.shape}"
        )

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    # =========================================================
    # Máscara
    # =========================================================

    def get_plant_mask(img):

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

        return ~background_mask

    mask_a = get_plant_mask(img_a)
    mask_b = get_plant_mask(img_b)

    if not np.any(mask_a) or not np.any(mask_b):
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # Wasserstein por banda
    # =========================================================

    results = {}

    bands = {
        "NIR": 3,
        "RE": 4
    }

    for name, band in bands.items():

        values_a = img_a[..., band][mask_a]
        values_b = img_b[..., band][mask_b]

        distance = wasserstein_distance(
            values_a,
            values_b
        )

        # Normalização pela dispersão da imagem original
        scale = np.std(values_a)

        normalized_distance = (
            distance / (scale + eps)
        )

        results[name] = float(
            normalized_distance
        )

    results["mean_spectral_distance"] = float(
        np.mean([
            results["NIR"],
            results["RE"]
        ])
    )

    return results


def spectral_wasserstein_distance__mean(
    img_a: np.ndarray,
    img_b: np.ndarray,
    eps: float = 1e-12
):
    data = spectral_wasserstein_distance(
            img_a,
            img_b,
            eps) 
    return data['mean_spectral_distance']

#======================================================================

import numpy as np


def spectral_angle_distance(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1=None,
    mask2=None,
    degrees: bool = True,
    eps: float = 1e-12,
) -> float:
    """
    Calcula a distância espectral entre duas imagens usando
    Spectral Angle Mapper (SAM).

    Para cada imagem, calcula a assinatura espectral média:

        s = [mean(B), mean(G), mean(R), mean(NIR), mean(RE)]

    considerando somente pixels pertencentes à planta.

    Depois calcula:

        theta = arccos(
            <s1, s2> / (||s1|| ||s2||)
        )

    Compatível automaticamente com:

        1. imagens segmentadas originais:
           background = 0;

        2. imagens segmentadas normalizadas por Z-score:
           background = mínimo simultâneo das bandas.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Imagens (H, W, 5), bandas:
        [B, G, R, NIR, RE].

    mask1, mask2 : np.ndarray ou None
        Máscaras booleanas opcionais.
        True = planta.

        Se None, são inferidas automaticamente.

    degrees : bool
        Se True, retorna graus.
        Caso contrário, radianos.

    eps : float
        Constante para estabilidade numérica.

    Returns
    -------
    float
        Ângulo espectral entre as assinaturas médias.

        0 = perfis espectrais iguais.
        Maior valor = maior mudança espectral.
    """

    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError(
            "As imagens devem possuir shape (H, W, C)."
        )

    if img1.shape[-1] != 5 or img2.shape[-1] != 5:
        raise ValueError(
            "Esperadas 5 bandas [B, G, R, NIR, RE]."
        )

    # =========================================================
    # Máscara automática
    # =========================================================

    def get_plant_mask(img):

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

        return ~background_mask

    if mask1 is None:
        mask1 = get_plant_mask(img1)

    if mask2 is None:
        mask2 = get_plant_mask(img2)

    if not np.any(mask1):
        raise ValueError(
            "Nenhum pixel de planta encontrado em img1."
        )

    if not np.any(mask2):
        raise ValueError(
            "Nenhum pixel de planta encontrado em img2."
        )

    # =========================================================
    # Assinaturas espectrais médias
    # =========================================================

    spectrum1 = np.mean(
        img1[mask1],
        axis=0
    )

    spectrum2 = np.mean(
        img2[mask2],
        axis=0
    )

    # =========================================================
    # Spectral Angle Mapper
    # =========================================================

    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)

    if norm1 <= eps or norm2 <= eps:
        return np.nan

    cosine = (
        np.dot(spectrum1, spectrum2)
        / (norm1 * norm2)
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = np.arccos(cosine)

    if degrees:
        angle = np.degrees(angle)

    return float(angle)

#======================================================================

import numpy as np


def spectral_angle_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1=None,
    mask2=None,
    eps: float = 1e-12,
) -> float:
    """
    Calcula similaridade espectral baseada no Spectral Angle Mapper.

    Retorna valor em [0, 1]:

        1 -> espectros idênticos
        0 -> espectros maximamente diferentes

    A assinatura espectral de cada imagem é:

        [mean(B), mean(G), mean(R), mean(NIR), mean(RE)]

    calculada somente sobre pixels da planta.

    Compatível com imagens segmentadas originais e imagens
    normalizadas por Z-score.
    """

    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError(
            "As imagens devem possuir shape (H, W, C)."
        )

    if img1.shape[-1] != 5 or img2.shape[-1] != 5:
        raise ValueError(
            "Esperadas 5 bandas [B, G, R, NIR, RE]."
        )

    # =========================================================
    # Máscara automática
    # =========================================================

    def get_plant_mask(img):

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

        return ~background_mask

    if mask1 is None:
        mask1 = get_plant_mask(img1)

    if mask2 is None:
        mask2 = get_plant_mask(img2)

    if not np.any(mask1) or not np.any(mask2):
        raise ValueError(
            "Nenhum pixel de planta encontrado."
        )

    # =========================================================
    # Assinaturas espectrais médias
    # =========================================================

    spectrum1 = np.mean(
        img1[mask1],
        axis=0
    )

    spectrum2 = np.mean(
        img2[mask2],
        axis=0
    )

    # =========================================================
    # Ângulo espectral
    # =========================================================

    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)

    if norm1 <= eps or norm2 <= eps:
        return np.nan

    cosine = (
        np.dot(spectrum1, spectrum2)
        / (norm1 * norm2)
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = np.arccos(cosine)

    # =========================================================
    # Similaridade em [0,1]
    # =========================================================

    similarity = 1.0 - (
        angle / np.pi
    )

    return float(
        np.clip(similarity, 0.0, 1.0)
    )


#======================================================================

