import numpy as np

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Feature Metrics ---     \t\t\033[0m\n")

#======================================================================
#======================================================================

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

#======================================================================
#======================================================================






