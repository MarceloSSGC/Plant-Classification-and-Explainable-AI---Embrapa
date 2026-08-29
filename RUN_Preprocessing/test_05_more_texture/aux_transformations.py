import os
import numpy as np

#======================================================================
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































































