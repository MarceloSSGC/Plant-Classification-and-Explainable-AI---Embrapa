import os
import numpy as np
from pathlib import Path
import cv2
import tifffile as tiff
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Alignment RUN ---     \t\t\033[0m\n")

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

#======================================================================


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
#======================================================================
#======================================================================
# align_main_function

# align_method = ALIGN_METHOD
# data_dir = DATA_DIR
# align_data_dir = ALIGH_DATA_DIR

def align_main_function(align_method, data_dir, align_data_dir):

    print(f'align_method: \033[96;92m{align_method}\033[0m')
    print(f'data_dir: \033[96;92m{data_dir}\033[0m')
    print(f'align_data_dir: \033[96;92m{align_data_dir}\033[0m')

    #----------------------------------------------------------

    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir = {data_dir} doesnt exist")
    
    especies = sorted(os.listdir(data_dir))
    os.makedirs(align_data_dir, exist_ok=True)

    for especie in especies: # especie = especies[0]

        print("="*60 + f'\nespecie: \033[96;93m{especie}\033[0m\n')

        old_especie_dir = data_dir + f"/{especie}"
        new_especie_dir = align_data_dir + f"/{especie}"

        os.makedirs(new_especie_dir, exist_ok=True)

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

                # Trocando bandas de lugar, 4 <--> 5

                img_norm[:, :, [3, 4]] = img_norm[:, :, [4, 3]]

                norm_params_copy = norm_params.copy()
                norm_params[3] = norm_params_copy[4]
                norm_params[4] = norm_params_copy[3]

                if align_method == "align_bands_ecc_affine_with_retry":
                    img_aligned, shifts = align_bands_ecc_affine_with_retry(
                        img_norm,
                        ref_band=5,
                    )

                    image_denorm = denormalize_multiband_image(img_aligned, norm_params)

                else:
                    image_denorm = img_norm

                # plot_bands(img_norm)
                # plot_bands(img_aligned)
                # plot_bands(img_norm - img_aligned)

                save_multiband_image(
                    image_denorm,
                    directory=new_especie_dir,
                    image_name=ith_file
                )

                print('\nSaved\n')
