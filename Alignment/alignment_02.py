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

    method:
        "translation"  -> phase correlation, apenas deslocamento
        "ecc_affine"   -> ECC affine, permite translação, rotação, escala e cisalhamento

    Retorna:
        img_aligned : ndarray (H, W, B)
        transforms  : lista com shifts ou matrizes affine
    """

    if img_norm.ndim != 3:
        raise ValueError("img_norm deve possuir shape (H, W, B).")

    if ref_band < 1 or ref_band > img_norm.shape[-1]:
        raise ValueError("ref_band inválida.")

    if method not in ["translation", "ecc_affine"]:
        raise ValueError("method deve ser 'translation' ou 'ecc_affine'.")

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

        elif method == "ecc_affine":

            warp_matrix = np.eye(2, 3, dtype=np.float32)

            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ecc_iterations,
                ecc_eps
            )

            try:
                cc, warp_matrix = cv2.findTransformECC(
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

            except cv2.error as e:
                print(f"[Aviso] ECC falhou na banda {b+1}. Mantendo banda original.")
                aligned = moving.copy()

            transforms.append(warp_matrix.copy())

        img_aligned[:, :, b] = aligned

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
#======================================================================
#======================================================================


img_norm, norm_params = load_multiband_image(
    directory="/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment_Tests/original",
    image_name="IMG_0015"
)


img_norm, norm_params = load_multiband_image(
    directory="/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment_Tests/Trans_01",
    image_name="IMG_0015"
)

for i in range(1, 6):
    with rasterio.open(f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment_Tests/original/IMG_0015_{i}.tif") as src:
        band = src.read()

    print(band.min(), band.max())

print(image_denorm.min(), image_denorm.max())



print(img_norm.shape)  # (H, W, 5)
print(img_norm.min(), img_norm.max())

print(norm_params)  

#---------------------------------------------------------------------
img_aligned, shifts = align_bands_translation(
    img_norm,
    ref_band=2
)


img_aligned, shifts = align_bands_translation_exp(
    img_norm,
    ref_band=2,
    method="translation"
)

img_aligned, shifts = align_bands_translation_exp(
    img_norm,
    ref_band=2,
    method="ecc_affine"
)

img_aligned, shifts = align_bands(
    img_norm,
    ref_band=3,
    method="sift"
)


print(img_aligned)
print(shifts)

#---------------------------------------------------------------------

plot_bands(img_norm)
plot_bands(img_aligned)
plot_bands(img_norm - img_aligned)

#---------------------------------------------------------------------
image_denorm = denormalize_multiband_image(img_aligned, norm_params)

plot_bands(image_denorm)

#---------------------------------------------------------------------

save_multiband_image(
    image_denorm,
    directory="../Alignment_Tests",
    image_name="IMG_0015"
)

#---------------------------------------------------------------------

#======================================================================
#======================================================================
#======================================================================
