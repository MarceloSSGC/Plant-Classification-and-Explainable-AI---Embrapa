import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

# def align_multispectral_bands(img_norm, ref_band, method='phase_correlation'):
#     """
#     Alinha as bandas de uma imagem multiespectral com base em uma banda de referência.
    
#     Parâmetros:
#     - img_norm: np.array de formato (960, 1280, 5)
#     - ref_band: int (1 a 5) indica a banda de referência
#     - method: str, método de alinhamento ('phase_correlation')
    
#     Retorna:
#     - img_aligned: np.array (960, 1280, 5) alinhado
#     - shifts: dicionário com os desvios (y, x) de cada banda em relação à referência
#     """
#     # Converte o canal 1-5 para o índice 0-4 do Python
#     ref_idx = ref_band - 1
    
#     # Extrai a banda de referência
#     reference_image = img_norm[:, :, ref_idx]
    
#     # Inicializa a imagem de saída e o dicionário de shifts
#     img_aligned = np.zeros_like(img_norm)
#     shifts = {}
    
#     # O método 'phase_correlation' é o padrão ideal para translações simples
#     if method == 'phase_correlation':
#         for i in range(img_norm.shape[2]):
#             if i == ref_idx:
#                 # A banda de referência não precisa de ajuste
#                 img_aligned[:, :, i] = reference_image
#                 shifts[f"band_{i+1}"] = (0.0, 0.0)
#             else:
#                 moving_image = img_norm[:, :, i]
                
#                 # Calcula o deslocamento necessário (retorna [shift_y, shift_x])
#                 # upsample_factor=10 permite precisão subpixel (1/10 de pixel)
#                 detected_shift, error, diffphase = phase_cross_correlation(
#                     reference_image, moving_image, upsample_factor=10
#                 )
                
#                 # Aplica o deslocamento na banda atual
#                 # cval=0 preenche as bordas vazias com zero (ou use np.nan se preferir)
#                 img_aligned[:, :, i] = shift(moving_image, shift=detected_shift, mode='constant', cval=0.0)
                
#                 # Salva o shift no dicionário (usando o nome amigável da banda)
#                 shifts[f"band_{i+1}"] = tuple(detected_shift)
#     else:
#         raise ValueError(f"Método '{method}' não reconhecido. Use 'phase_correlation'.")
        
#     return img_aligned, shifts


import numpy as np
import cv2
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

def align_multispectral_bands(img_norm, ref_band, method='phase_correlation'):
    """
    Alinha as bandas de uma imagem multiespectral com base em uma banda de referência.
    
    Parâmetros:
    - img_norm: np.array de formato (960, 1280, 5)
    - ref_band: int (1 a 5) indica a banda de referência
    - method: str, método de alinhamento ('phase_correlation' ou 'ecc')
    
    Retorna:
    - img_aligned: np.array (960, 1280, 5) alinhado
    - shifts: dicionário com os desvios (y, x) de cada banda em relação à referência
    """
    ref_idx = ref_band - 1
    reference_image = img_norm[:, :, ref_idx]
    
    img_aligned = np.zeros_like(img_norm)
    shifts = {}
    
    # --- MÉTODO 1: CORRELAÇÃO DE FASE ---
    if method == 'phase_correlation':
        for i in range(img_norm.shape[2]):
            if i == ref_idx:
                img_aligned[:, :, i] = reference_image
                shifts[f"band_{i+1}"] = (0.0, 0.0)
            else:
                moving_image = img_norm[:, :, i]
                detected_shift, error, diffphase = phase_cross_correlation(
                    reference_image, moving_image, upsample_factor=10
                )
                img_aligned[:, :, i] = shift(moving_image, shift=detected_shift, mode='constant', cval=0.0)
                shifts[f"band_{i+1}"] = tuple(detected_shift)
                
    # --- MÉTODO 2: ENHANCED CORRELATION COEFFICIENT (ECC) ---
    elif method == 'ecc':
        # O OpenCV exige float32 para o algoritmo ECC
        ref_32 = reference_image.astype(np.float32)
        
        # Define o modelo de movimento: cv2.MOTION_TRANSLATION para translação pura (x, y)
        # Se suas bandas tiverem rotação/escala leve, mude para cv2.MOTION_HOMOGRAPHY ou cv2.MOTION_AFFINE
        warp_mode = cv2.MOTION_TRANSLATION
        
        # Critério de parada: 50 iterações ou mudança menor que 1e-6
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        
        for i in range(img_norm.shape[2]):
            if i == ref_idx:
                img_aligned[:, :, i] = reference_image
                shifts[f"band_{i+1}"] = (0.0, 0.0)
            else:
                moving_image = img_norm[:, :, i].astype(np.float32)
                
                # Inicializa a matriz de transformação identidade (2x3 para translação/afim)
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                
                try:
                    # Encontra a matriz de transformação que alinha as imagens
                    _, warp_matrix = cv2.findTransformECC(
                        ref_32, moving_image, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5
                    )
                    
                    # Aplica a transformação encontrada na imagem original
                    # Usamos a广义 warpAffine do OpenCV que é extremamente rápida
                    h, w = reference_image.shape
                    img_aligned[:, :, i] = cv2.warpAffine(
                        img_norm[:, :, i], warp_matrix, (w, h), 
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                    )
                    
                    # No OpenCV, a matriz de translação guarda [deslocamento_x, deslocamento_y] na última coluna
                    # Invertemos para manter o padrão (y, x) do seu dicionário de shifts
                    shift_x = warp_matrix[0, 2]
                    shift_y = warp_matrix[1, 2]
                    shifts[f"band_{i+1}"] = (float(shift_y), float(shift_x))
                    
                except cv2.error:
                    # Caso o ECC falhe em convergir para alguma banda específica, 
                    # ele mantém a banda original e zera o shift para não quebrar o código
                    img_aligned[:, :, i] = img_norm[:, :, i]
                    shifts[f"band_{i+1}"] = (0.0, 0.0)
                    print(f"Aviso: Alinhamento ECC falhou na banda {i+1}. Mantendo original.")
                    
    else:
        raise ValueError(f"Método '{method}' não reconhecido. Use 'phase_correlation' ou 'ecc'.")
        
    return img_aligned, shifts

# Alinhando usando a banda 3 como o "ancoradouro"
img_aligned, shifts = align_multispectral_bands(img_norm, ref_band=3, method='phase_correlation')

img_aligned, shifts = align_multispectral_bands(img_norm, ref_band=3, method='ecc')


print("Deslocamentos detectados por banda:")
for banda, offset in shifts.items():
    print(f"{banda}: deslocamento em (Y, X) = {offset}")


#======================================================================
#======================================================================


import numpy as np
import cv2


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
