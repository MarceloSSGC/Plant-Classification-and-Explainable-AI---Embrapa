from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

from time import sleep
from copy import deepcopy

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Transformations ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# Metrics

def classification_metrics_dataframe(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    class_names=None,
    zero_division=0
) -> pd.DataFrame:
    """
    Calcula métricas de classificação e retorna um DataFrame
    com uma única linha.

    Parameters
    ----------
    y_real : np.ndarray
        Classes verdadeiras, com formato (n_amostras,).

    y_pred : np.ndarray
        Classes preditas, com formato (n_amostras,).

    class_names : dict, list ou tuple, opcional
        Nomes das classes.

        Pode ser um dicionário no formato:
            {0: "classe_A", 1: "classe_B"}

        Ou uma lista:
            ["classe_A", "classe_B"]

        Caso não seja informado, serão utilizados os próprios
        valores das classes.

    zero_division : int ou float, padrão=0
        Valor usado quando precision ou recall não puderem ser
        calculados por ausência de amostras ou predições.

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha contendo as métricas gerais
        e as métricas de cada classe.
    """

    # --------------------------------------------------------------
    # Validação e padronização das entradas
    # --------------------------------------------------------------
    y_real = np.asarray(y_real).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_real.size == 0:
        raise ValueError("y_real não pode estar vazio.")

    if y_pred.size == 0:
        raise ValueError("y_pred não pode estar vazio.")

    if y_real.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_real e y_pred precisam ter o mesmo número de elementos. "
            f"Recebido: {y_real.shape[0]} e {y_pred.shape[0]}."
        )

    # Inclui classes presentes em y_real ou y_pred
    labels = np.unique(
        np.concatenate([y_real, y_pred])
    )

    # --------------------------------------------------------------
    # Define os nomes das classes
    # --------------------------------------------------------------
    if class_names is None:
        label_to_name = {
            label: str(label)
            for label in labels
        }

    elif isinstance(class_names, dict):
        label_to_name = {
            label: str(class_names.get(label, label))
            for label in labels
        }

    elif isinstance(class_names, (list, tuple)):
        label_to_name = {}

        for label in labels:
            try:
                label_to_name[label] = str(class_names[int(label)])
            except (IndexError, TypeError, ValueError):
                label_to_name[label] = str(label)

    else:
        raise TypeError(
            "class_names deve ser None, dict, list ou tuple."
        )

    # --------------------------------------------------------------
    # Métricas gerais
    # --------------------------------------------------------------
    metrics = {
        "n_amostras": y_real.shape[0],

        "acuracia": accuracy_score(
            y_real,
            y_pred
        ),

        "acuracia_balanceada": balanced_accuracy_score(
            y_real,
            y_pred
        ),

        "precision_macro": precision_score(
            y_real,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=zero_division
        ),

        "precision_micro": precision_score(
            y_real,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=zero_division
        ),

        "precision_weighted": precision_score(
            y_real,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=zero_division
        ),

        "recall_macro": recall_score(
            y_real,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=zero_division
        ),

        "recall_micro": recall_score(
            y_real,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=zero_division
        ),

        "recall_weighted": recall_score(
            y_real,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=zero_division
        ),

        "f1_macro": f1_score(
            y_real,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=zero_division
        ),

        "f1_micro": f1_score(
            y_real,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=zero_division
        ),

        "f1_weighted": f1_score(
            y_real,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=zero_division
        ),

        "cohen_kappa": cohen_kappa_score(
            y_real,
            y_pred
        ),

        "matthews_corrcoef": matthews_corrcoef(
            y_real,
            y_pred
        )
    }

    # --------------------------------------------------------------
    # Métricas por classe
    # --------------------------------------------------------------
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            y_real,
            y_pred,
            labels=labels,
            average=None,
            zero_division=zero_division
        )
    )

    for label, precision, recall, f1, n_samples in zip(
        labels,
        precision_per_class,
        recall_per_class,
        f1_per_class,
        support
    ):
        class_name = label_to_name[label]

        # Evita espaços e caracteres pouco convenientes nas colunas
        class_name = (
            class_name
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
        )

        metrics[f"precision__{class_name}"] = precision
        metrics[f"recall__{class_name}"] = recall
        metrics[f"f1__{class_name}"] = f1
        metrics[f"support__{class_name}"] = int(n_samples)

    return pd.DataFrame([metrics])

#======================================================================
#======================================================================
#======================================================================
# Color

def suppress_rgb_color(image: np.ndarray) -> np.ndarray:
    """
    Aplica a transformação de supressão de cor RGB (item 5.3).

    Recebe uma imagem multiespectral com 5 bandas na ordem:
    [Blue, Green, Red, NIR, RedEdge]

    Converte as bandas B, G, R em uma única banda grayscale (mantendo
    a mesma escala/dtype das bandas originais) e a replica nos três
    primeiros canais, preservando NIR e Red Edge inalterados.

    Resultado: [GS, GS, GS, NIR, RE]

    Parameters
    ----------
    image : np.ndarray
        Array de shape (H, W, 5), dtype float ou uint, bandas na ordem
        [B, G, R, NIR, RE].

    Returns
    -------
    np.ndarray
        Array de shape (H, W, 5) com mesmo dtype de entrada.
    """
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(f"Esperado array (H, W, 5), recebido {image.shape}")

    orig_dtype = image.dtype

    blue  = image[..., 0].astype(np.float64)
    green = image[..., 1].astype(np.float64)
    red   = image[..., 2].astype(np.float64)
    nir   = image[..., 3]
    red_edge = image[..., 4]

    # Pesos de luminosidade padrão (Rec. 601), aplicados na ordem R, G, B
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue

    # Ajusta dtype de volta ao original (evita overflow/truncamento indevido)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        grayscale = np.clip(grayscale, info.min, info.max)
    grayscale = grayscale.astype(orig_dtype)

    result = np.stack(
        [grayscale, grayscale, grayscale, nir, red_edge],
        axis=-1
    )

    return result

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

#======================================================================


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
#======================================================================
#======================================================================
