import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar PLOT ---     \t\t\033[0m\n")

#======================================================================
#======================================================================
# 5 bands

def load_5b_from_dir(image_dir: str, base_name: str):

    channels = []

    for band in range(1, 6):
        band_file = f"{base_name}_{band}.tif"
        band_path = os.path.join(image_dir, band_file)

        with rasterio.open(band_path) as src:
            img = src.read(1).astype(np.float32)

        channels.append(img)

    img = np.dstack(channels)

    return img

#======================================================================

def plot_rgb(rgb_image, bands_ch=(2, 1, 0)):

    channels = []

    for band in bands_ch:
        
        img = rgb_image[:, :, band].astype(np.float32)

        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)

    rgb = np.dstack(channels)

    plt.figure(figsize=(15, 9))
    plt.imshow(rgb)
    plt.title(f"RGB - {bands_ch}")
    plt.axis("off")
    plt.show()


#======================================================================

def plot_two_imgs(rgb_img, rgb_img_masked):
    """
    Plota a imagem original e a imagem segmentada lado a lado.

    Parameters
    ----------
    rgb_img : np.ndarray
        Imagem RGB original (H, W, 3).

    rgb_img_masked : np.ndarray
        Imagem RGB após segmentação (H, W, 3).
    """

    channels = []
    for band in range(3):
        img = rgb_img[:, :, band].astype(np.float32)
        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)
    rgb = np.dstack(channels)

    channels = []
    for band in range(3):
        img = rgb_img_masked[:, :, band].astype(np.float32)
        # Normalização individual para [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        channels.append(img)
    rgb_masked = np.dstack(channels)

    fig, ax = plt.subplots(1, 2, figsize=(15, 9))

    ax[0].imshow(rgb)
    ax[0].set_title("Imagem original")
    ax[0].axis("off")

    ax[1].imshow(rgb_masked)
    ax[1].set_title("Imagem segmentada")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

#======================================================================


def plot_ablation_accuracy(df, title="Spectral Band Ablation Study"):
    """
    Plot classification accuracy for the original model and
    for each spectral-band ablation experiment.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns:
        - 'EXP': experiment identifier ('original' or removed band)
        - 'acuracia': classification accuracy

    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.

    ax : matplotlib.axes.Axes
        Matplotlib axes object.
    """
    data = df.copy()

    # Create descriptive labels
    # data["label"] = data["EXP"].apply(
    #     lambda x: "Original" if x == "original" else f"Band {x} removed"
    # )

    data["label"] = ['Original', "Band Blue Removed", "Band Green Removed",
                     "Band Red Removed", "Band NIR Removed", "Band Red Edge Removed"]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        data["label"],
        data["acuracia"]
    )

    # Display accuracy above each bar
    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in data["acuracia"]],
        padding=3
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_ablation_4_metric(
    df,
    metric="acuracia",
    title="Spectral Band Ablation Study",
    figsize=(12, 9)
):
    """
    Plot a metric for the original model and
    for each spectral-band ablation experiment.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the ablation experiment results.

    metric : str, optional
        Name of the DataFrame column containing the metric to plot.
        Default is 'acuracia'.

    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.

    ax : matplotlib.axes.Axes
        Matplotlib axes object.
    """
    data = df.copy()

    if metric not in data.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )

    data["label"] = df["EXP"]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        data["label"],
        data[metric]
    )

    # Display metric value above each bar
    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in data[metric]],
        padding=3
    )

    # ax.set_xlabel("Experiment")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return fig, ax


#======================================================================

import matplotlib.pyplot as plt

def plotar_barras(dados):
    """
    Recebe uma lista de tuplas (x_i, y_i) e plota um gráfico
    de barras preservando a ordem original dos x_i.

    Exemplo:
        dados = [
            ("João", 10),
            ("Maria", 25),
            ("Pedro", 15)
        ]
    """
    if not dados:
        raise ValueError("A lista não pode estar vazia.")

    # Mantém exatamente a ordem recebida
    nomes = [x for x, _ in dados]
    valores = [y for _, y in dados]

    # Posições numéricas garantem que nenhuma ordenação seja feita
    posicoes = range(len(dados))

    plt.figure(figsize=(10, 6))
    plt.bar(posicoes, valores)

    plt.xticks(posicoes, nomes)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()
#======================================================================
