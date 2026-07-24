from pathlib import Path
import numpy as np
import tifffile as tiff
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
import cv2
import matplotlib.pyplot as plt
import os
import rasterio


#======================================================================
#======================================================================

def normalize_band_percentile(band, p_low=1, p_high=99, eps=1e-8):
    """
    Normaliza uma banda para [0, 1] usando percentis.
    """
    band = band.astype(np.float32)

    v_min = np.percentile(band, p_low)
    v_max = np.percentile(band, p_high)

    band_norm = (band - v_min) / (v_max - v_min + eps)
    band_norm = np.clip(band_norm, 0, 1)

    return band_norm

#-----------------------------------------------------------------------


def load_multiband_image(directory, image_name, n_bands=5, p_low=1, p_high=99):
    """
    Carrega as bandas separadas de uma imagem e retorna um array normalizado.

    Espera arquivos:
        IMG_0015_1.tif
        ...
        IMG_0015_5.tif

    Retorna
    -------
    image_norm : ndarray
        Array (H, W, B) normalizado em [0,1].

    norm_params : list
        Lista de dicionários contendo os parâmetros de normalização
        de cada banda.
    """

    directory = Path(directory)

    bands = []
    norm_params = []

    for b in range(1, n_bands + 1):

        file_path = directory / f"{image_name}_{b}.tif"

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        band = tiff.imread(file_path)

        dtype = band.dtype

        band_float = band.astype(np.float32)

        v_min = np.percentile(band_float, p_low)
        v_max = np.percentile(band_float, p_high)

        band_norm = (band_float - v_min) / (v_max - v_min + 1e-8)
        band_norm = np.clip(band_norm, 0, 1)

        bands.append(band_norm)

        norm_params.append({
            "v_min": float(v_min),
            "v_max": float(v_max),
            "dtype": dtype,
            "p_low": p_low,
            "p_high": p_high
        })

    image_norm = np.stack(bands, axis=-1)

    return image_norm, norm_params

#======================================================================

def align_bands_translation(img_norm, ref_band=2, upsample_factor=20):
    """
    Alinha todas as bandas de uma imagem utilizando apenas translação
    (phase correlation).

    Parameters
    ----------
    img_norm : ndarray
        Imagem normalizada com shape (H, W, B).

    ref_band : int
        Banda de referência (1, 2, ..., B).

    upsample_factor : int
        Precisão subpixel da estimação.

    Returns
    -------
    img_aligned : ndarray
        Imagem alinhada com mesmo shape da entrada.

    shifts : list
        Lista contendo (dy, dx) aplicado em cada banda.
    """

    if ref_band < 1 or ref_band > img_norm.shape[-1]:
        raise ValueError("ref_band inválida.")

    ref_idx = ref_band - 1
    reference = img_norm[:, :, ref_idx]

    img_aligned = np.zeros_like(img_norm)
    img_aligned[:, :, ref_idx] = reference

    shifts = []

    for b in range(img_norm.shape[-1]):

        if b == ref_idx:
            shifts.append((0.0, 0.0))
            continue

        moving = img_norm[:, :, b]

        shift_est, error, _ = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=upsample_factor
        )

        # aligned = shift(
        #     moving,
        #     shift=shift_est,
        #     order=1,
        #     mode="nearest",
        #     prefilter=True
        # )

        aligned = shift(
            moving,
            shift=shift_est,
            order=1,
            mode="reflect",
            prefilter=False
        )

        img_aligned[:, :, b] = aligned
        shifts.append(tuple(shift_est))

    return img_aligned, shifts

#-----------------------------------------------------------------------


# def align_bands_translation_exp(
#     img_norm,
#     ref_band=2,
#     method="translation",
#     upsample_factor=20,
#     ecc_iterations=1000,
#     ecc_eps=1e-7
# ):
#     """
#     Alinha todas as bandas usando uma banda de referência.

#     method:
#         "translation"  -> phase correlation, apenas deslocamento
#         "ecc_affine"   -> ECC affine, permite translação, rotação, escala e cisalhamento

#     Retorna:
#         img_aligned : ndarray (H, W, B)
#         transforms  : lista com shifts ou matrizes affine
#     """

#     if img_norm.ndim != 3:
#         raise ValueError("img_norm deve possuir shape (H, W, B).")

#     if ref_band < 1 or ref_band > img_norm.shape[-1]:
#         raise ValueError("ref_band inválida.")

#     if method not in ["translation", "ecc_affine"]:
#         raise ValueError("method deve ser 'translation' ou 'ecc_affine'.")

#     ref_idx = ref_band - 1
#     reference = img_norm[:, :, ref_idx].astype(np.float32)

#     h, w = reference.shape

#     img_aligned = np.zeros_like(img_norm, dtype=np.float32)
#     img_aligned[:, :, ref_idx] = reference

#     transforms = []

#     for b in range(img_norm.shape[-1]):

#         if b == ref_idx:
#             if method == "translation":
#                 transforms.append((0.0, 0.0))
#             else:
#                 transforms.append(np.eye(2, 3, dtype=np.float32))
#             continue

#         moving = img_norm[:, :, b].astype(np.float32)

#         if method == "translation":

#             shift_est, error, _ = phase_cross_correlation(
#                 reference,
#                 moving,
#                 upsample_factor=upsample_factor
#             )

#             aligned = shift(
#                 moving,
#                 shift=shift_est,
#                 order=1,
#                 mode="reflect",
#                 prefilter=False
#             )

#             transforms.append(tuple(shift_est))

#         elif method == "ecc_affine":

#             warp_matrix = np.eye(2, 3, dtype=np.float32)

#             criteria = (
#                 cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                 ecc_iterations,
#                 ecc_eps
#             )

#             try:
#                 cc, warp_matrix = cv2.findTransformECC(
#                     reference,
#                     moving,
#                     warp_matrix,
#                     cv2.MOTION_AFFINE,
#                     criteria
#                 )

#                 aligned = cv2.warpAffine(
#                     moving,
#                     warp_matrix,
#                     (w, h),
#                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
#                     borderMode=cv2.BORDER_REFLECT
#                 )

#             except cv2.error as e:
#                 print(f"[Aviso] ECC falhou na banda {b+1}. Mantendo banda original.")
#                 aligned = moving.copy()

#             transforms.append(warp_matrix.copy())

#         img_aligned[:, :, b] = aligned

#     return img_aligned, transforms


# import cv2
# import numpy as np
# from scipy.ndimage import shift
# from skimage.registration import phase_cross_correlation


def align_bands_translation_exp(
    img_norm,
    ref_band=2,
    method="translation",
    upsample_factor=20,
    ecc_iterations=1000,
    ecc_eps=1e-7
):
    """
    Alinha todas as bandas usando uma banda de referência.

    Parameters
    ----------
    img_norm : ndarray
        Imagem com shape (H, W, B).

    ref_band : int
        Banda de referência (indexação iniciando em 1).

    method : str
        "translation"  -> Phase Correlation (apenas translação)
        "ecc_affine"   -> ECC Affine (translação, rotação, escala e cisalhamento)

    Returns
    -------
    img_aligned : ndarray | None
        Imagem alinhada.

    transforms : list | None
        Lista contendo os deslocamentos (translation) ou matrizes affine (ECC).

    Observação
    ----------
    Caso ocorra qualquer erro durante o alinhamento de qualquer banda
    (erro do OpenCV, falha de convergência, divisão por zero, etc.),
    a função retorna:

        (None, None)
    """

    if img_norm.ndim != 3:
        raise ValueError("img_norm deve possuir shape (H, W, B).")

    if ref_band < 1 or ref_band > img_norm.shape[-1]:
        raise ValueError("ref_band inválida.")

    if method not in ["translation", "ecc_affine"]:
        raise ValueError(
            "method deve ser 'translation' ou 'ecc_affine'."
        )

    ref_idx = ref_band - 1
    reference = img_norm[:, :, ref_idx].astype(np.float32)

    h, w = reference.shape

    img_aligned = np.zeros_like(img_norm, dtype=np.float32)
    img_aligned[:, :, ref_idx] = reference

    transforms = []

    for b in range(img_norm.shape[-1]):

        if b == ref_idx:
            if method == "translation":
                transforms.append((0.0, 0.0))
            else:
                transforms.append(np.eye(2, 3, dtype=np.float32))
            continue

        try:

            moving = img_norm[:, :, b].astype(np.float32)

            if method == "translation":

                shift_est, error, _ = phase_cross_correlation(
                    reference,
                    moving,
                    upsample_factor=upsample_factor
                )

                aligned = shift(
                    moving,
                    shift=shift_est,
                    order=1,
                    mode="reflect",
                    prefilter=False
                )

                transforms.append(tuple(shift_est))

            else:  # ecc_affine

                warp_matrix = np.eye(2, 3, dtype=np.float32)

                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    ecc_iterations,
                    ecc_eps
                )

                _, warp_matrix = cv2.findTransformECC(
                    reference,
                    moving,
                    warp_matrix,
                    cv2.MOTION_AFFINE,
                    criteria
                )

                aligned = cv2.warpAffine(
                    moving,
                    warp_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT
                )

                transforms.append(warp_matrix.copy())

            # Verificações adicionais
            if not np.all(np.isfinite(aligned)):
                raise ValueError(
                    f"A banda {b+1} contém NaN ou Inf após o alinhamento."
                )

            img_aligned[:, :, b] = aligned

        except Exception as e:
            print(f"[Erro] Falha ao alinhar a banda {b+1}: {e}")
            return None, None

    return img_aligned, transforms

#======================================================================

def plot_bands(img, cmap="gray", figsize=(15, 3)):
    """
    Plota todas as bandas de uma imagem multibanda.

    Parameters
    ----------
    img : ndarray
        Imagem com shape (H, W, B).

    cmap : str
        Colormap utilizado na visualização.

    figsize : tuple
        Tamanho da figura.
    """

    n_bands = img.shape[-1]

    fig, axes = plt.subplots(1, n_bands, figsize=figsize)

    if n_bands == 1:
        axes = [axes]

    for i in range(n_bands):
        axes[i].imshow(img[:, :, i], cmap=cmap)
        axes[i].set_title(f"Banda {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

#======================================================================

def denormalize_multiband_image(image_norm, norm_params):
    """
    Retorna a imagem para a escala original.
    """

    bands = []

    for b in range(image_norm.shape[-1]):

        p = norm_params[b]

        band = image_norm[:, :, b] * (p["v_max"] - p["v_min"]) + p["v_min"]

        band = np.clip(band, p["v_min"], p["v_max"])

        band = band.astype(p["dtype"])

        bands.append(band)

    return np.stack(bands, axis=-1)


#-----------------------------------------------------------------------


#======================================================================


def save_multiband_image(img, directory, image_name, dtype=None):
    """
    Salva cada banda de uma imagem multibanda em arquivos TIFF separados.

    Parameters
    ----------
    img : ndarray
        Imagem com shape (H, W, B).

    directory : str ou Path
        Diretório onde os arquivos serão salvos.

    image_name : str
        Nome base da imagem.
        Ex.: IMG_0015

    dtype : numpy dtype, opcional
        Tipo de saída. Se None, mantém o dtype da imagem.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if img.ndim != 3:
        raise ValueError("A imagem deve possuir shape (H, W, B).")

    n_bands = img.shape[-1]

    for b in range(n_bands):

        band = img[:, :, b]

        if dtype is not None:
            band = band.astype(dtype)

        filename = directory / f"{image_name}_{b+1}.tif"

        tiff.imwrite(
            filename,
            band,
            compression="zlib"
        )


#======================================================================


def align_bands(
    img_norm,
    ref_band=2,
    method="sift",
    ratio_test=0.75,
    min_matches=10,
):
    """
    Alinha todas as bandas usando uma banda de referência.

    Parameters
    ----------
    img_norm : ndarray (H,W,B)
        Imagem multiespectral normalizada.

    ref_band : int
        Banda de referência (1...B).

    method : str
        Atualmente disponível:
            "sift"

    ratio_test : float
        Limiar do teste de Lowe.

    min_matches : int
        Número mínimo de correspondências para estimar a transformação.

    Returns
    -------
    img_aligned : ndarray
        Imagem alinhada.

    shifts : dict
        Informações do alinhamento de cada banda.
    """

    if img_norm.ndim != 3:
        raise ValueError("img_norm deve possuir shape (H,W,B).")

    if not (1 <= ref_band <= img_norm.shape[-1]):
        raise ValueError("ref_band inválida.")

    if method.lower() != "sift":
        raise ValueError("Atualmente apenas method='sift' está implementado.")

    ref_idx = ref_band - 1

    H, W, B = img_norm.shape

    img_aligned = np.zeros_like(img_norm, dtype=np.float32)

    reference = img_norm[:, :, ref_idx].astype(np.float32)

    img_aligned[:, :, ref_idx] = reference

    # ------------------------------------------------------------------
    # Conversão para uint8 (SIFT funciona melhor)
    # ------------------------------------------------------------------

    ref_uint8 = np.clip(reference * 255, 0, 255).astype(np.uint8)

    sift = cv2.SIFT_create()

    kp_ref, des_ref = sift.detectAndCompute(ref_uint8, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    shifts = {}

    shifts[ref_band] = {
        "success": True,
        "transform": np.eye(2, 3, dtype=np.float32),
        "matches": 0,
        "good_matches": 0,
        "inliers": 0,
    }

    # ------------------------------------------------------------------

    for b in range(B):

        if b == ref_idx:
            continue

        moving = img_norm[:, :, b].astype(np.float32)

        mov_uint8 = np.clip(moving * 255, 0, 255).astype(np.uint8)

        kp_mov, des_mov = sift.detectAndCompute(mov_uint8, None)

        info = {
            "success": False,
            "transform": None,
            "matches": 0,
            "good_matches": 0,
            "inliers": 0,
        }

        # Não encontrou descritores
        if des_ref is None or des_mov is None:

            img_aligned[:, :, b] = moving
            shifts[b + 1] = info
            continue

        matches = bf.knnMatch(des_mov, des_ref, k=2)

        info["matches"] = len(matches)

        # Lowe ratio test
        good = []

        for m, n in matches:

            if m.distance < ratio_test * n.distance:
                good.append(m)

        info["good_matches"] = len(good)

        if len(good) < min_matches:

            img_aligned[:, :, b] = moving
            shifts[b + 1] = info
            continue

        pts_mov = np.float32(
            [kp_mov[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        pts_ref = np.float32(
            [kp_ref[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_mov,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=5000,
            confidence=0.99,
        )

        if M is None:

            img_aligned[:, :, b] = moving
            shifts[b + 1] = info
            continue

        aligned = cv2.warpAffine(
            moving,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        img_aligned[:, :, b] = aligned

        info["success"] = True
        info["transform"] = M

        if inliers is not None:
            info["inliers"] = int(inliers.sum())

        shifts[b + 1] = info

    return img_aligned, shifts


def align_bands(
    img_norm,
    ref_band=3,
    method="sift",
    ratio_test=0.75,
    min_matches=10,
    min_inliers=8,
    min_inlier_ratio=0.30,
    max_translation_fraction=0.20,
    min_scale=0.90,
    max_scale=1.10,
    max_rotation_deg=5.0,
):
    """
    Alinha as bandas usando SIFT e transformação de similaridade:
    translação, rotação e escala uniforme.

    Transformações implausíveis são rejeitadas, mantendo-se a banda
    original como fallback.
    """

    if img_norm.ndim != 3:
        raise ValueError("img_norm deve possuir shape (H, W, B).")

    if not (1 <= ref_band <= img_norm.shape[-1]):
        raise ValueError("ref_band inválida.")

    if method.lower() != "sift":
        raise ValueError("Apenas method='sift' está implementado.")

    H, W, B = img_norm.shape
    ref_idx = ref_band - 1

    img_aligned = np.empty_like(img_norm, dtype=np.float32)

    reference = img_norm[:, :, ref_idx].astype(np.float32)
    img_aligned[:, :, ref_idx] = reference

    ref_uint8 = np.clip(reference * 255, 0, 255).astype(np.uint8)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_uint8, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    transforms = {
        ref_band: {
            "success": True,
            "reason": "reference_band",
            "transform": np.eye(2, 3, dtype=np.float32),
            "matches": 0,
            "good_matches": 0,
            "inliers": 0,
        }
    }

    for b in range(B):
        band_number = b + 1

        if b == ref_idx:
            continue

        moving = img_norm[:, :, b].astype(np.float32)

        info = {
            "success": False,
            "reason": None,
            "transform": None,
            "matches": 0,
            "good_matches": 0,
            "inliers": 0,
        }

        mov_uint8 = np.clip(moving * 255, 0, 255).astype(np.uint8)
        kp_mov, des_mov = sift.detectAndCompute(mov_uint8, None)

        if des_ref is None or des_mov is None:
            info["reason"] = "no_descriptors"
            img_aligned[:, :, b] = moving
            transforms[band_number] = info
            continue

        raw_matches = bf.knnMatch(des_mov, des_ref, k=2)
        info["matches"] = len(raw_matches)

        good = []

        for pair in raw_matches:
            # Em alguns casos o matcher pode retornar somente um vizinho.
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < ratio_test * n.distance:
                good.append(m)

        info["good_matches"] = len(good)

        if len(good) < min_matches:
            info["reason"] = "insufficient_matches"
            img_aligned[:, :, b] = moving
            transforms[band_number] = info
            continue

        pts_mov = np.float32(
            [kp_mov[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        pts_ref = np.float32(
            [kp_ref[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_mov,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.99,
            refineIters=10,
        )

        if M is None or inliers is None:
            info["reason"] = "estimation_failed"
            img_aligned[:, :, b] = moving
            transforms[band_number] = info
            continue

        M = M.astype(np.float32)

        num_inliers = int(inliers.sum())
        inlier_ratio = num_inliers / len(good)

        a = float(M[0, 0])
        c = float(M[1, 0])

        scale = np.sqrt(a**2 + c**2)
        rotation_deg = np.degrees(np.arctan2(c, a))

        tx = float(M[0, 2])
        ty = float(M[1, 2])

        max_tx = max_translation_fraction * W
        max_ty = max_translation_fraction * H

        is_valid = (
            np.all(np.isfinite(M))
            and num_inliers >= min_inliers
            and inlier_ratio >= min_inlier_ratio
            and min_scale <= scale <= max_scale
            and abs(rotation_deg) <= max_rotation_deg
            and abs(tx) <= max_tx
            and abs(ty) <= max_ty
        )

        info.update({
            "transform": M,
            "inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "scale": scale,
            "rotation_deg": rotation_deg,
            "translation": (tx, ty),
        })

        if not is_valid:
            info["reason"] = "implausible_transform"
            img_aligned[:, :, b] = moving
            transforms[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: transformação rejeitada. "
                f"Inliers={num_inliers}/{len(good)}, "
                f"escala={scale:.4f}, rotação={rotation_deg:.2f}°, "
                f"tx={tx:.2f}, ty={ty:.2f}"
            )
            continue

        aligned = cv2.warpAffine(
            moving,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        dynamic_range = float(np.max(aligned) - np.min(aligned))
        aligned_std = float(np.std(aligned))

        if (
            not np.all(np.isfinite(aligned))
            or dynamic_range < 1e-8
            or aligned_std < 1e-8
        ):
            info["reason"] = "invalid_aligned_output"
            img_aligned[:, :, b] = moving
            transforms[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: saída inválida ou constante. "
                "Mantendo banda original."
            )
            continue

        img_aligned[:, :, b] = aligned

        info["success"] = True
        info["reason"] = "ok"

        transforms[band_number] = info

    return img_aligned, transforms


#======================================================================

from typing import Any, Dict, Optional, Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch
import torch.nn.functional as F


def align_bands_LoFTR(
    img_norm: np.ndarray,
    ref_band: int = 3,
    pretrained: str = "outdoor",
    device: Optional[str] = None,
    max_image_size: int = 840,
    confidence_threshold: float = 0.50,
    min_matches: int = 20,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.25,
    ransac_threshold: float = 3.0,
    min_scale: float = 0.90,
    max_scale: float = 1.10,
    max_rotation_deg: float = 5.0,
    max_translation_fraction: float = 0.20,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Alinha as bandas de uma imagem multiespectral usando LoFTR.

    A banda indicada por `ref_band` é usada como referência. As demais
    bandas são registradas geometricamente em relação a ela.

    Quando o alinhamento não é considerado confiável, a banda original
    é mantida sem transformação.

    Parameters
    ----------
    img_norm : np.ndarray
        Imagem multibanda normalizada com shape (H, W, C).

        O intervalo recomendado é [0, 1]. Cada canal representa uma banda.

    ref_band : int
        Número da banda de referência, usando indexação iniciada em 1.

        Por exemplo:
            ref_band=3

        corresponde a:
            img_norm[:, :, 2]

    pretrained : str
        Pesos pré-treinados usados pelo LoFTR.

        Valores normalmente disponíveis no Kornia:
            "outdoor"
            "indoor_new"

    device : str ou None
        Dispositivo utilizado pelo PyTorch.

        Exemplos:
            "cuda"
            "cpu"

        Quando None, usa GPU se estiver disponível.

    max_image_size : int
        Tamanho máximo da maior dimensão usada pelo LoFTR.

        As imagens são reduzidas somente para encontrar correspondências.
        A transformação final é aplicada na resolução original.

    confidence_threshold : float
        Confiança mínima das correspondências retornadas pelo LoFTR.

    min_matches : int
        Número mínimo de correspondências aceitas antes do RANSAC.

    min_inliers : int
        Número mínimo de inliers após o RANSAC.

    min_inlier_ratio : float
        Proporção mínima de inliers em relação às correspondências.

    ransac_threshold : float
        Erro máximo, em pixels da resolução original, usado pelo RANSAC.

    min_scale, max_scale : float
        Intervalo permitido para a escala estimada.

    max_rotation_deg : float
        Rotação máxima permitida, em graus.

    max_translation_fraction : float
        Translação máxima permitida como fração da largura e da altura.

    Returns
    -------
    img_aligned : np.ndarray
        Imagem multibanda alinhada, com o mesmo shape de `img_norm`.

    shifts : dict
        Informações de alinhamento para cada banda.

        Exemplo:

        {
            4: {
                "success": True,
                "reason": "ok",
                "matches": 245,
                "inliers": 180,
                "transform": ...,
                "scale": 1.002,
                "rotation_deg": -0.12,
                "translation": (-4.2, 7.1)
            }
        }
    """

    # ---------------------------------------------------------------
    # Validação da entrada
    # ---------------------------------------------------------------

    if not isinstance(img_norm, np.ndarray):
        raise TypeError("img_norm deve ser um np.ndarray.")

    if img_norm.ndim != 3:
        raise ValueError(
            "img_norm deve possuir shape (H, W, C). "
            f"Shape recebido: {img_norm.shape}"
        )

    height, width, num_bands = img_norm.shape

    if not 1 <= ref_band <= num_bands:
        raise ValueError(
            f"ref_band deve estar entre 1 e {num_bands}. "
            f"Valor recebido: {ref_band}"
        )

    if max_image_size <= 0:
        raise ValueError("max_image_size deve ser maior que zero.")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold deve estar no intervalo [0, 1]."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)

    # Converte para float32 e evita modificar o array original.
    image = np.asarray(img_norm, dtype=np.float32)

    if not np.all(np.isfinite(image)):
        raise ValueError("img_norm contém valores NaN ou infinitos.")

    ref_index = ref_band - 1
    reference = image[:, :, ref_index]

    img_aligned = image.copy()

    # ---------------------------------------------------------------
    # Inicialização do LoFTR
    # ---------------------------------------------------------------

    matcher = KF.LoFTR(pretrained=pretrained)
    matcher = matcher.to(torch_device)
    matcher.eval()

    # A banda de referência recebe transformação identidade.
    shifts: Dict[int, Dict[str, Any]] = {
        ref_band: {
            "success": True,
            "reason": "reference_band",
            "matches": 0,
            "filtered_matches": 0,
            "inliers": 0,
            "inlier_ratio": 1.0,
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "scale": 1.0,
            "rotation_deg": 0.0,
            "translation": (0.0, 0.0),
        }
    }

    # ---------------------------------------------------------------
    # Funções auxiliares
    # ---------------------------------------------------------------

    def normalize_for_matching(band: np.ndarray) -> np.ndarray:
        """
        Normalização robusta destinada somente ao casamento de pontos.

        Ela não altera os valores que serão salvos na imagem alinhada.
        """

        band = band.astype(np.float32)

        valid = band[np.isfinite(band)]

        if valid.size == 0:
            return np.zeros_like(band, dtype=np.float32)

        lower = float(np.percentile(valid, 1.0))
        upper = float(np.percentile(valid, 99.0))

        if upper <= lower:
            lower = float(valid.min())
            upper = float(valid.max())

        if upper <= lower:
            return np.zeros_like(band, dtype=np.float32)

        normalized = (band - lower) / (upper - lower)

        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def calculate_match_size(
        original_height: int,
        original_width: int,
    ) -> Tuple[int, int]:
        """
        Redimensiona a imagem mantendo a proporção e usa dimensões
        múltiplas de 8, adequadas à arquitetura do LoFTR.
        """

        largest_side = max(original_height, original_width)

        resize_factor = min(
            1.0,
            max_image_size / float(largest_side),
        )

        new_height = int(round(original_height * resize_factor))
        new_width = int(round(original_width * resize_factor))

        # Mantém no mínimo 8 pixels e ajusta para múltiplos de 8.
        new_height = max(8, (new_height // 8) * 8)
        new_width = max(8, (new_width // 8) * 8)

        return new_height, new_width

    match_height, match_width = calculate_match_size(height, width)

    scale_x_to_original = width / float(match_width)
    scale_y_to_original = height / float(match_height)

    def band_to_tensor(band: np.ndarray) -> torch.Tensor:
        """
        Converte uma banda 2D em tensor (1, 1, H, W).
        """

        band_normalized = normalize_for_matching(band)

        tensor = torch.from_numpy(band_normalized)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(
            device=torch_device,
            dtype=torch.float32,
        )

        if tensor.shape[-2:] != (match_height, match_width):
            tensor = F.interpolate(
                tensor,
                size=(match_height, match_width),
                mode="bilinear",
                align_corners=False,
            )

        return tensor

    reference_tensor = band_to_tensor(reference)

    # ---------------------------------------------------------------
    # Alinhamento de cada banda
    # ---------------------------------------------------------------

    for band_index in range(num_bands):
        band_number = band_index + 1

        if band_index == ref_index:
            continue

        moving = image[:, :, band_index]

        info: Dict[str, Any] = {
            "success": False,
            "reason": None,
            "matches": 0,
            "filtered_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "transform": None,
            "scale": None,
            "rotation_deg": None,
            "translation": None,
        }

        # Uma banda constante não possui estrutura suficiente para registro.
        moving_range = float(np.max(moving) - np.min(moving))

        if moving_range < 1e-8:
            info["reason"] = "constant_band"
            img_aligned[:, :, band_index] = moving
            shifts[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: banda constante; "
                "mantendo a banda original."
            )
            continue

        moving_tensor = band_to_tensor(moving)

        input_dict = {
            # keypoints0 pertencem à imagem móvel.
            "image0": moving_tensor,

            # keypoints1 pertencem à imagem de referência.
            "image1": reference_tensor,
        }

        try:
            with torch.inference_mode():
                correspondences = matcher(input_dict)

        except RuntimeError as error:
            info["reason"] = f"loftr_error: {error}"
            img_aligned[:, :, band_index] = moving
            shifts[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: erro durante o LoFTR; "
                "mantendo a banda original."
            )

            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

            continue

        keypoints_moving = (
            correspondences["keypoints0"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        keypoints_reference = (
            correspondences["keypoints1"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        confidence_tensor = correspondences.get("confidence")

        if confidence_tensor is None:
            confidences = np.ones(
                len(keypoints_moving),
                dtype=np.float32,
            )
        else:
            confidences = (
                confidence_tensor
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

        info["matches"] = int(len(keypoints_moving))

        # -----------------------------------------------------------
        # Filtragem por confiança
        # -----------------------------------------------------------

        confidence_mask = confidences >= confidence_threshold

        keypoints_moving = keypoints_moving[confidence_mask]
        keypoints_reference = keypoints_reference[confidence_mask]
        confidences = confidences[confidence_mask]

        info["filtered_matches"] = int(len(keypoints_moving))

        if len(keypoints_moving) < min_matches:
            info["reason"] = "insufficient_matches"
            img_aligned[:, :, band_index] = moving
            shifts[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: somente "
                f"{len(keypoints_moving)} correspondências válidas; "
                "mantendo a banda original."
            )
            continue

        # -----------------------------------------------------------
        # Retorno das coordenadas para a resolução original
        # -----------------------------------------------------------

        keypoints_moving[:, 0] *= scale_x_to_original
        keypoints_moving[:, 1] *= scale_y_to_original

        keypoints_reference[:, 0] *= scale_x_to_original
        keypoints_reference[:, 1] *= scale_y_to_original

        # -----------------------------------------------------------
        # Estimativa da transformação
        # -----------------------------------------------------------

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            keypoints_moving,
            keypoints_reference,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )

        if transform is None or inlier_mask is None:
            info["reason"] = "transform_estimation_failed"
            img_aligned[:, :, band_index] = moving
            shifts[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: não foi possível estimar "
                "a transformação; mantendo a banda original."
            )
            continue

        transform = transform.astype(np.float32)

        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / float(len(keypoints_moving))

        # Para uma transformação de similaridade:
        #
        # [ a  b  tx ]
        # [-b  a  ty ]
        #
        # escala = sqrt(a² + b²)
        a = float(transform[0, 0])
        c = float(transform[1, 0])

        scale = float(np.sqrt(a**2 + c**2))
        rotation_deg = float(np.degrees(np.arctan2(c, a)))

        tx = float(transform[0, 2])
        ty = float(transform[1, 2])

        info.update(
            {
                "transform": transform,
                "inliers": num_inliers,
                "inlier_ratio": inlier_ratio,
                "scale": scale,
                "rotation_deg": rotation_deg,
                "translation": (tx, ty),
                "mean_match_confidence": float(confidences.mean()),
            }
        )

        # -----------------------------------------------------------
        # Validação da transformação
        # -----------------------------------------------------------

        valid_transform = (
            np.all(np.isfinite(transform))
            and num_inliers >= min_inliers
            and inlier_ratio >= min_inlier_ratio
            and min_scale <= scale <= max_scale
            and abs(rotation_deg) <= max_rotation_deg
            and abs(tx) <= max_translation_fraction * width
            and abs(ty) <= max_translation_fraction * height
        )

        if not valid_transform:
            info["reason"] = "implausible_transform"
            img_aligned[:, :, band_index] = moving
            shifts[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: transformação rejeitada. "
                f"Inliers={num_inliers}/{len(keypoints_moving)}, "
                f"escala={scale:.4f}, "
                f"rotação={rotation_deg:.2f}°, "
                f"tx={tx:.2f}, ty={ty:.2f}"
            )
            continue

        # -----------------------------------------------------------
        # Aplicação na banda em resolução original
        # -----------------------------------------------------------

        aligned = cv2.warpAffine(
            moving,
            transform,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        aligned = aligned.astype(np.float32)

        aligned_range = float(
            np.nanmax(aligned) - np.nanmin(aligned)
        )
        aligned_std = float(np.nanstd(aligned))

        valid_output = (
            np.all(np.isfinite(aligned))
            and aligned_range > 1e-8
            and aligned_std > 1e-8
        )

        if not valid_output:
            info["reason"] = "invalid_aligned_output"
            img_aligned[:, :, band_index] = moving
            shifts[band_number] = info

            print(
                f"[Aviso] Banda {band_number}: resultado do alinhamento "
                "constante ou inválido; mantendo a banda original."
            )
            continue

        img_aligned[:, :, band_index] = aligned

        info["success"] = True
        info["reason"] = "ok"

        shifts[band_number] = info

    return img_aligned, shifts

#======================================================================
import cv2
import numpy as np

from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation


def align_bands_ecc_affine_with_retry(
    img_norm,
    ref_band=2,
    upsample_factor=20,
    ecc_iterations=1000,
    ecc_eps=1e-7,
    equality_rtol=1e-5,
    equality_atol=1e-7
):
    """
    Alinha todas as bandas em relação a uma banda de referência.

    Estratégia
    ----------
    1. Tenta ECC affine.
    2. Se o ECC falhar especificamente por não convergência, utiliza somente
       translação por correlação de fase.
    3. Se ocorrer qualquer outro erro, retorna (None, None).
    4. Se a banda alinhada ficar numericamente igual à banda original,
       o resultado é rejeitado e a função retorna (None, None).

    Parameters
    ----------
    img_norm : ndarray
        Imagem com shape (H, W, B), preferencialmente normalizada entre 0 e 1.

    ref_band : int
        Banda de referência, com indexação começando em 1.

    upsample_factor : int
        Precisão subpixel da correlação de fase.

    ecc_iterations : int
        Número máximo de iterações do ECC.

    ecc_eps : float
        Critério de convergência do ECC.

    equality_rtol : float
        Tolerância relativa usada para verificar se a banda alinhada é igual
        à banda original.

    equality_atol : float
        Tolerância absoluta usada para verificar se a banda alinhada é igual
        à banda original.

    Returns
    -------
    img_aligned : ndarray | None
        Imagem alinhada, ou None em caso de falha.

    transforms : list | None
        Para bandas alinhadas por ECC, contém matrizes affine 2 x 3.

        Para bandas alinhadas por correlação de fase, contém um dicionário:
            {
                "method": "phase_correlation",
                "shift_yx": (dy, dx)
            }

        Para a banda de referência, contém a matriz identidade.

        Retorna None em caso de falha.
    """

    # ------------------------------------------------------------
    # Validação dos argumentos
    # ------------------------------------------------------------
    if not isinstance(img_norm, np.ndarray):
        raise TypeError("img_norm deve ser um array NumPy.")

    if img_norm.ndim != 3:
        raise ValueError("img_norm deve possuir shape (H, W, B).")

    if img_norm.shape[-1] < 2:
        raise ValueError("img_norm deve possuir pelo menos duas bandas.")

    if not isinstance(ref_band, (int, np.integer)):
        raise TypeError("ref_band deve ser um número inteiro.")

    if ref_band < 1 or ref_band > img_norm.shape[-1]:
        raise ValueError(
            f"ref_band inválida. Use um valor entre 1 e "
            f"{img_norm.shape[-1]}."
        )

    if upsample_factor < 1:
        raise ValueError("upsample_factor deve ser maior ou igual a 1.")

    if ecc_iterations < 1:
        raise ValueError("ecc_iterations deve ser maior ou igual a 1.")

    if ecc_eps <= 0:
        raise ValueError("ecc_eps deve ser maior que zero.")

    if not np.all(np.isfinite(img_norm)):
        print("[Erro] img_norm contém NaN ou Inf.")
        return None, None

    # ------------------------------------------------------------
    # Preparação
    # ------------------------------------------------------------
    ref_idx = ref_band - 1
    num_bands = img_norm.shape[-1]

    reference = np.asarray(
        img_norm[:, :, ref_idx],
        dtype=np.float32
    )

    h, w = reference.shape

    if np.std(reference) == 0:
        print(
            f"[Erro] A banda de referência {ref_band} é constante. "
            "Não é possível realizar o alinhamento."
        )
        return None, None

    # O array é preenchido inicialmente com NaN para impedir que bandas
    # não processadas sejam confundidas com bandas válidas.
    img_aligned = np.full(
        img_norm.shape,
        np.nan,
        dtype=np.float32
    )

    # A banda de referência necessariamente permanece igual a ela mesma.
    img_aligned[:, :, ref_idx] = reference

    transforms = [None] * num_bands
    transforms[ref_idx] = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(ecc_iterations),
        float(ecc_eps)
    )

    def is_ecc_convergence_error(error):
        """Identifica especificamente falhas de convergência do ECC."""

        if not isinstance(error, cv2.error):
            return False

        message = str(error).lower()

        convergence_messages = (
            "iterations do not converge",
            "algorithm stopped before its convergence",
            "correlation is going to be minimized",
            "images may be uncorrelated or non-overlapped"
        )

        return any(
            text in message
            for text in convergence_messages
        )

    # ------------------------------------------------------------
    # Alinhamento banda a banda
    # ------------------------------------------------------------
    for b in range(num_bands):

        if b == ref_idx:
            continue

        band_number = b + 1

        try:
            moving = np.asarray(
                img_norm[:, :, b],
                dtype=np.float32
            )

            if not np.all(np.isfinite(moving)):
                raise ValueError(
                    f"A banda {band_number} contém NaN ou Inf."
                )

            if np.std(moving) == 0:
                raise ValueError(
                    f"A banda {band_number} é constante."
                )

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            used_phase_correlation = False

            # ----------------------------------------------------
            # Primeira tentativa: ECC affine
            # ----------------------------------------------------
            try:
                correlation_coefficient, warp_matrix = (
                    cv2.findTransformECC(
                        reference,
                        moving,
                        warp_matrix,
                        cv2.MOTION_AFFINE,
                        criteria
                    )
                )

                if not np.isfinite(correlation_coefficient):
                    raise ValueError(
                        f"O coeficiente ECC da banda {band_number} "
                        "não é finito."
                    )

                if (
                    warp_matrix.shape != (2, 3)
                    or not np.all(np.isfinite(warp_matrix))
                ):
                    raise ValueError(
                        f"A matriz affine da banda {band_number} "
                        "é inválida."
                    )

                aligned = cv2.warpAffine(
                    moving,
                    warp_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT
                )

                transform_result = warp_matrix.copy()

            except cv2.error as ecc_error:

                # Qualquer erro do OpenCV que não seja de convergência
                # invalida imediatamente a imagem.
                if not is_ecc_convergence_error(ecc_error):
                    print(
                        f"[Erro] O ECC affine apresentou um erro diferente "
                        f"de não convergência na banda {band_number}: "
                        f"{ecc_error}"
                    )
                    return None, None

                print(
                    f"[Aviso] O ECC affine não convergiu para a banda "
                    f"{band_number}. Usando translação por correlação "
                    "de fase."
                )

                used_phase_correlation = True

                # ------------------------------------------------
                # Fallback: somente correlação de fase
                # ------------------------------------------------
                shift_est, phase_error, phase_difference = (
                    phase_cross_correlation(
                        reference,
                        moving,
                        upsample_factor=upsample_factor
                    )
                )

                shift_est = np.asarray(
                    shift_est,
                    dtype=np.float64
                )

                if shift_est.shape != (2,):
                    raise ValueError(
                        "A correlação de fase retornou um deslocamento "
                        f"com shape inválido para a banda {band_number}: "
                        f"{shift_est.shape}."
                    )

                if not np.all(np.isfinite(shift_est)):
                    raise ValueError(
                        "A correlação de fase retornou NaN ou Inf para "
                        f"a banda {band_number}."
                    )

                if not np.isfinite(phase_error):
                    raise ValueError(
                        "A correlação de fase retornou uma métrica de erro "
                        f"inválida para a banda {band_number}."
                    )

                dy = float(shift_est[0])
                dx = float(shift_est[1])

                aligned = shift(
                    moving,
                    shift=(dy, dx),
                    order=1,
                    mode="reflect",
                    prefilter=False
                ).astype(np.float32)

                transform_result = {
                    "method": "phase_correlation",
                    "shift_yx": (dy, dx),
                    "phase_error": float(phase_error)
                }

                print(
                    f"[Info] Banda {band_number} alinhada por correlação "
                    f"de fase: dy={dy:.4f}, dx={dx:.4f}."
                )

            # ----------------------------------------------------
            # Validação da banda produzida
            # ----------------------------------------------------
            if aligned.shape != (h, w):
                raise ValueError(
                    f"A banda {band_number} alinhada possui shape "
                    f"inválido: {aligned.shape}."
                )

            if not np.all(np.isfinite(aligned)):
                raise ValueError(
                    f"A banda {band_number} contém NaN ou Inf após "
                    "o alinhamento."
                )

            # Não altera artificialmente a imagem apenas para fazê-la
            # diferente. Se o processamento resultar em uma banda igual à
            # original, o resultado é considerado inválido.
            if np.allclose(
                aligned,
                moving,
                rtol=equality_rtol,
                atol=equality_atol,
                equal_nan=False
            ):
                method_name = (
                    "correlação de fase"
                    if used_phase_correlation
                    else "ECC affine"
                )

                raise ValueError(
                    f"O alinhamento por {method_name} produziu uma banda "
                    f"{band_number} numericamente igual à banda original."
                )

            # Somente grava após todas as verificações.
            img_aligned[:, :, b] = aligned
            transforms[b] = transform_result

        except Exception as error:
            print(
                f"[Erro] Falha ao alinhar a banda {band_number}: {error}"
            )
            return None, None

    # ------------------------------------------------------------
    # Verificação final
    # ------------------------------------------------------------
    if not np.all(np.isfinite(img_aligned)):
        print(
            "[Erro] O resultado final contém NaN, Inf ou bandas "
            "não preenchidas."
        )
        return None, None

    if any(transform is None for transform in transforms):
        print(
            "[Erro] Pelo menos uma transformação não foi calculada."
        )
        return None, None

    return img_aligned, transforms



#======================================================================


# img_norm, norm_params = load_multiband_image(
#     directory="/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment_Tests/original",
#     image_name="IMG_0015"
# )


# img_norm, norm_params = load_multiband_image(
#     directory="/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment_Tests/Trans_01",
#     image_name="IMG_0015"
# )

# for i in range(1, 6):
#     with rasterio.open(f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment_Tests/original/IMG_0015_{i}.tif") as src:
#         band = src.read()

#     print(band.min(), band.max())

# print(image_denorm.min(), image_denorm.max())



# print(img_norm.shape)  # (H, W, 5)
# print(img_norm.min(), img_norm.max())

# print(norm_params)  

# #---------------------------------------------------------------------
# img_aligned, shifts = align_bands_translation(
#     img_norm,
#     ref_band=3
# )


# img_aligned, shifts = align_bands_translation_exp(
#     img_norm,
#     ref_band=3,
#     method="translation"
# )

# img_aligned, shifts = align_bands_translation_exp(
#     img_norm,
#     ref_band=3,
#     method="ecc_affine"
# )

# img_aligned, shifts = align_bands(
#     img_norm,
#     ref_band=3,
#     method="sift"
# )


# print(img_aligned)
# print(shifts)

# #---------------------------------------------------------------------

# plot_bands(img_norm)
# plot_bands(img_aligned)
# plot_bands(img_norm - img_aligned)

# #---------------------------------------------------------------------
# image_denorm = denormalize_multiband_image(img_aligned, norm_params)

# plot_bands(image_denorm)

# #---------------------------------------------------------------------

# save_multiband_image(
#     image_denorm,
#     directory="../Alignment_Tests",
#     image_name="IMG_0015"
# )

#---------------------------------------------------------------------

#======================================================================
#======================================================================
#======================================================================
# New Dataset

DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista"
# NEW_DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned"
NEW_DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine"

os.makedirs(NEW_DATA_DIR, exist_ok=True)

#======================================================================

especies = os.listdir(DATA_DIR)

for especie in especies: # especie = especies[0]

    print("="*60 + f'\nespecie: \033[96;93m{especie}\033[0m\n')

    old_especie_dir = DATA_DIR + f"/{especie}"
    new_especie_dir = NEW_DATA_DIR + f"/{especie}"

    if not os.path.isdir(new_especie_dir):
        os.makedirs(new_especie_dir)

    files_full = [x[:-6] for x in os.listdir(old_especie_dir) if "tif" in x]
    files = list(set(files_full))
    files.sort()

    for ith, ith_file in enumerate(files): # ith, ith_file = 0, files[0]

        print(f"ith:\033[100;01m{ith} of {len(files)}\033[0m ith_file: \033[96;92m{ith_file}\033[0m")

        # ith_file_dir = old_especie_dir + f"/{ith_file}"
        if not os.path.isfile(new_especie_dir + f"/{ith_file}_1.tif") or \
        not os.path.isfile(new_especie_dir + f"/{ith_file}_2.tif") or \
        not os.path.isfile(new_especie_dir + f"/{ith_file}_3.tif") or \
        not os.path.isfile(new_especie_dir + f"/{ith_file}_4.tif") or \
        not os.path.isfile(new_especie_dir + f"/{ith_file}_5.tif"):

            img_norm, norm_params = load_multiband_image(
            directory=old_especie_dir,
            image_name=ith_file)

            # img_aligned, shifts = align_bands_LoFTR(
            #     img_norm,
            #     ref_band=3,
            # )

            img_aligned, shifts = align_bands_ecc_affine_with_retry(
                img_norm,
                ref_band=3,
            )


            # img_aligned, shifts = align_bands_translation_exp(
            #     img_norm,
            #     ref_band=3,
            #     method="ecc_affine"
            # )

            image_denorm = denormalize_multiband_image(img_aligned, norm_params)

            # plot_bands(img_norm)
            # plot_bands(img_aligned)
            # plot_bands(img_norm - img_aligned)

            save_multiband_image(
                image_denorm,
                directory=new_especie_dir,
                image_name=ith_file
            )

            print('\nSaved\n')


#======================================================================
#======================================================================


# for ith, ith_file in enumerate(files): # ith, ith_file = 0, files[0]

#     print(f"ith:\033[100;01m{ith} of {len(files)}\033[0m ith_file: \033[96;92m{ith_file}\033[0m")

#     img_norm, norm_params = load_multiband_image(
#     directory=old_especie_dir,
#     image_name=ith_file)


#     img_aligned, shifts = align_bands_LoFTR(
#         img_norm,
#         ref_band=3,
#     )

#     image_denorm = denormalize_multiband_image(img_aligned, norm_params)

#     plot_bands(img_norm)
#     plot_bands(img_aligned)
#     plot_bands(img_norm - img_aligned)

