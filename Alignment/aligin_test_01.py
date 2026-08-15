from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import shift

#======================================================================

def plot_manual_band_alignment(
    base_dir: str,
    file_name: str,
    reference: int,
    two_bandas: tuple,
    delta_i: tuple,
    delta_j: tuple,
    plot_order: tuple,
):
    """
    Carrega três bandas de uma imagem multiespectral, translada duas delas
    em relação a uma banda de referência e plota as três como uma composição RGB.

    Parâmetros
    ----------
    base_dir : str
        Diretório contendo os arquivos TIFF.

    file_name : str
        Nome-base dos arquivos, sem "_banda.tif".
        Exemplo:
            file_name = "IMG_0003"

        Espera encontrar:
            IMG_0003_1.tif
            ...
            IMG_0003_5.tif

    reference : int
        Banda de referência. Deve estar entre 1 e 5.

    two_bandas : tuple[int, int]
        Bandas que serão transladadas.
        Exemplo:
            (4, 5)

    delta_i : tuple[float, float]
        Deslocamento (dx_i, dy_i) aplicado à banda i.

        dx > 0 -> desloca para a direita
        dx < 0 -> desloca para a esquerda
        dy > 0 -> desloca para baixo
        dy < 0 -> desloca para cima

    delta_j : tuple[float, float]
        Deslocamento (dx_j, dy_j) aplicado à banda j.

    plot_order : tuple[int, int, int]
        Ordem das bandas nos canais R, G e B da visualização.

        Exemplo:
            plot_order = (5, 4, 3)

        significa:
            R <- banda 5
            G <- banda 4
            B <- banda 3

        Deve conter exatamente:
            i, j e reference.

    Retorno
    -------
    dict
        Dicionário contendo as três bandas após as transformações:
            {
                i: banda_i_transladada,
                j: banda_j_transladada,
                reference: banda_reference
            }
    """

    # ---------------------------------------------------------
    # Validação
    # ---------------------------------------------------------

    valid_bands = {1, 2, 3, 4, 5}

    if reference not in valid_bands:
        raise ValueError("'reference' deve ser um inteiro entre 1 e 5.")

    if len(two_bandas) != 2:
        raise ValueError("'two_bandas' deve conter exatamente duas bandas.")

    i, j = two_bandas

    if i not in valid_bands or j not in valid_bands:
        raise ValueError("As bandas em 'two_bandas' devem estar entre 1 e 5.")

    if i == j:
        raise ValueError("As bandas i e j devem ser diferentes.")

    if reference in (i, j):
        raise ValueError(
            "A banda de referência não deve estar presente em 'two_bandas'."
        )

    expected_order = {i, j, reference}

    if len(plot_order) != 3 or set(plot_order) != expected_order:
        raise ValueError(
            f"'plot_order' deve ser uma permutação de "
            f"({i}, {j}, {reference})."
        )

    if len(delta_i) != 2 or len(delta_j) != 2:
        raise ValueError("'delta_i' e 'delta_j' devem ser tuplas (dx, dy).")

    # ---------------------------------------------------------
    # Caminhos
    # ---------------------------------------------------------

    base_dir = Path(base_dir)

    paths = {
        band: base_dir / f"{file_name}_{band}.tif"
        for band in (i, j, reference)
    }

    for band, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo da banda {band} não encontrado:\n{path}"
            )

    # ---------------------------------------------------------
    # Leitura
    # ---------------------------------------------------------

    band_i = tifffile.imread(paths[i]).astype(np.float32)
    band_j = tifffile.imread(paths[j]).astype(np.float32)
    band_ref = tifffile.imread(paths[reference]).astype(np.float32)

    if not (
        band_i.shape == band_j.shape == band_ref.shape
    ):
        raise ValueError(
            "As três bandas precisam possuir as mesmas dimensões. "
            f"Shapes encontrados: "
            f"banda {i}={band_i.shape}, "
            f"banda {j}={band_j.shape}, "
            f"referência={band_ref.shape}."
        )

    # ---------------------------------------------------------
    # Translação
    # ---------------------------------------------------------
    #
    # scipy.ndimage.shift recebe:
    #
    #     shift=(dy, dx)
    #
    # enquanto nossos argumentos são:
    #
    #     delta=(dx, dy)
    #

    dx_i, dy_i = delta_i
    dx_j, dy_j = delta_j

    band_i_shifted = shift(
        band_i,
        shift=(dy_i, dx_i),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    band_j_shifted = shift(
        band_j,
        shift=(dy_j, dx_j),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    # ---------------------------------------------------------
    # Dicionário com as bandas alinhadas
    # ---------------------------------------------------------

    aligned = {
        i: band_i_shifted,
        j: band_j_shifted,
        reference: band_ref,
    }

    # ---------------------------------------------------------
    # Normalização apenas para visualização
    # ---------------------------------------------------------

    def normalize_for_display(image):
        """
        Normaliza uma banda para [0, 1] usando percentis para evitar que
        poucos pixels extremos prejudiquem o contraste.
        """
        low, high = np.percentile(image, (1, 99))

        if high <= low:
            return np.zeros_like(image, dtype=np.float32)

        image = (image - low) / (high - low)

        return np.clip(image, 0, 1)

    # plot_order corresponde diretamente a R, G, B
    rgb = np.dstack(
        [
            normalize_for_display(aligned[band])
            for band in plot_order
        ]
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    band_mapping = {
        1: "Blue",
        2: "Green",
        3: "Red",
        4: "NIR",
        5: "Red Edge",
    }

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)

    plt.title(
        f"{file_name}\n"
        f"RGB = "
        f"({plot_order[0]}: {band_mapping[plot_order[0]]}, "
        f"{plot_order[1]}: {band_mapping[plot_order[1]]}, "
        f"{plot_order[2]}: {band_mapping[plot_order[2]]})\n"
        f"Banda {i}: dx={dx_i}, dy={dy_i} | "
        f"Banda {j}: dx={dx_j}, dy={dy_j} | "
        f"Referência: {reference}"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()




#======================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import shift


def plot_manual_band_alignment_both(
    base_dir: str,
    file_name: str,
    reference: int,
    two_bandas: tuple,
    delta_i: tuple,
    delta_j: tuple,
    plot_order: tuple,
):
    """
    Carrega três bandas de uma imagem multiespectral, translada duas delas
    em relação a uma banda de referência e compara lado a lado:

        Esquerda -> composição original, sem translações
        Direita  -> composição após as translações

    Parâmetros
    ----------
    base_dir : str
        Diretório contendo os arquivos TIFF.

    file_name : str
        Nome-base dos arquivos.
        Exemplo: "IMG_0003"

    reference : int
        Banda de referência. Deve estar entre 1 e 5.

    two_bandas : tuple
        Duas bandas que serão transladadas.
        Exemplo: (4, 5)

    delta_i : tuple
        Deslocamento (dx_i, dy_i) aplicado à banda i.

    delta_j : tuple
        Deslocamento (dx_j, dy_j) aplicado à banda j.

    plot_order : tuple
        Ordem das bandas nos canais R, G e B.

        Exemplo:
            plot_order = (5, 4, 3)

        significa:
            R <- banda 5
            G <- banda 4
            B <- banda 3

    Retorno
    -------
    dict
        Dicionário contendo as três bandas após as transformações.
    """

    # ---------------------------------------------------------
    # Validação
    # ---------------------------------------------------------

    valid_bands = {1, 2, 3, 4, 5}

    if reference not in valid_bands:
        raise ValueError(
            "'reference' deve ser um inteiro entre 1 e 5."
        )

    if len(two_bandas) != 2:
        raise ValueError(
            "'two_bandas' deve conter exatamente duas bandas."
        )

    i, j = two_bandas

    if i not in valid_bands or j not in valid_bands:
        raise ValueError(
            "As bandas em 'two_bandas' devem estar entre 1 e 5."
        )

    if i == j:
        raise ValueError(
            "As bandas i e j devem ser diferentes."
        )

    if reference in (i, j):
        raise ValueError(
            "A banda de referência não deve estar presente "
            "em 'two_bandas'."
        )

    expected_order = {i, j, reference}

    if len(plot_order) != 3 or set(plot_order) != expected_order:
        raise ValueError(
            f"'plot_order' deve ser uma permutação de "
            f"({i}, {j}, {reference})."
        )

    if len(delta_i) != 2 or len(delta_j) != 2:
        raise ValueError(
            "'delta_i' e 'delta_j' devem ser tuplas (dx, dy)."
        )

    # ---------------------------------------------------------
    # Caminhos
    # ---------------------------------------------------------

    base_dir = Path(base_dir)

    paths = {
        band: base_dir / f"{file_name}_{band}.tif"
        for band in (i, j, reference)
    }

    for band, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo da banda {band} não encontrado:\n{path}"
            )

    # ---------------------------------------------------------
    # Leitura
    # ---------------------------------------------------------

    band_i = tifffile.imread(
        paths[i]
    ).astype(np.float32)

    band_j = tifffile.imread(
        paths[j]
    ).astype(np.float32)

    band_ref = tifffile.imread(
        paths[reference]
    ).astype(np.float32)

    if not (
        band_i.shape == band_j.shape == band_ref.shape
    ):
        raise ValueError(
            "As três bandas precisam possuir as mesmas dimensões. "
            f"Shapes encontrados: "
            f"banda {i}={band_i.shape}, "
            f"banda {j}={band_j.shape}, "
            f"referência={band_ref.shape}."
        )

    # ---------------------------------------------------------
    # Bandas originais
    # ---------------------------------------------------------

    original = {
        i: band_i,
        j: band_j,
        reference: band_ref,
    }

    # ---------------------------------------------------------
    # Translação
    # ---------------------------------------------------------

    dx_i, dy_i = delta_i
    dx_j, dy_j = delta_j

    band_i_shifted = shift(
        band_i,
        shift=(dy_i, dx_i),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    band_j_shifted = shift(
        band_j,
        shift=(dy_j, dx_j),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    # ---------------------------------------------------------
    # Bandas alinhadas
    # ---------------------------------------------------------

    aligned = {
        i: band_i_shifted,
        j: band_j_shifted,
        reference: band_ref,
    }

    # ---------------------------------------------------------
    # Normalização apenas para visualização
    # ---------------------------------------------------------

    def normalize_for_display(image):
        low, high = np.percentile(image, (1, 99))

        if high <= low:
            return np.zeros_like(
                image,
                dtype=np.float32
            )

        image = (image - low) / (high - low)

        return np.clip(image, 0, 1)

    # ---------------------------------------------------------
    # RGB original
    # ---------------------------------------------------------

    rgb_original = np.dstack(
        [
            normalize_for_display(original[band])
            for band in plot_order
        ]
    )

    # ---------------------------------------------------------
    # RGB após alinhamento
    # ---------------------------------------------------------

    rgb_aligned = np.dstack(
        [
            normalize_for_display(aligned[band])
            for band in plot_order
        ]
    )

    # ---------------------------------------------------------
    # Informações das bandas
    # ---------------------------------------------------------

    band_mapping = {
        1: "Blue",
        2: "Green",
        3: "Red",
        4: "NIR",
        5: "Red Edge",
    }

    rgb_description = (
        f"R: {band_mapping[plot_order[0]]} | "
        f"G: {band_mapping[plot_order[1]]} | "
        f"B: {band_mapping[plot_order[2]]}"
    )

    # ---------------------------------------------------------
    # Plot lado a lado
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8)
    )

    # Original
    axes[0].imshow(rgb_original)

    axes[0].set_title(
        f"Original\n"
        f"{rgb_description}",
        fontsize=13
    )

    axes[0].axis("off")

    # Alinhada
    axes[1].imshow(rgb_aligned)

    axes[1].set_title(
        f"Após translação\n"
        f"Banda {i}: ({dx_i:.2f}, {dy_i:.2f}) | "
        f"Banda {j}: ({dx_j:.2f}, {dy_j:.2f})",
        fontsize=13
    )

    axes[1].axis("off")

    # Título geral
    fig.suptitle(
        f"{file_name} — Referência: "
        f"Banda {reference} ({band_mapping[reference]})",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

 
#======================================================================
#======================================================================


base_dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5/01_malva_branca_Agua_Boa_01"

files = sorted(set([x[:-6] for x in os.listdir(base_dir)]))

file_name = "IMG_0015"
reference = 5
two_bandas = (1, 2)
delta_i = (0, 0)
delta_j = (0, 0)
plot_order = (reference, two_bandas[1], two_bandas[0])

plot_manual_band_alignment(
    base_dir,
    file_name,
    reference,
    two_bandas,
    delta_i,
    delta_j,
    plot_order,
)



file_name = "IMG_0015"
reference = 5
two_bandas = (1, 2)
delta_i = (0, 0)
delta_j = (-10, 0)
plot_order = (reference, two_bandas[1], two_bandas[0])

plot_manual_band_alignment_both(
    base_dir,
    file_name,
    reference,
    two_bandas,
    delta_i,
    delta_j,
    plot_order,
)

#======================================================================
#======================================================================


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import shift


def interactive_band_alignment(
    base_dir: str,
    file_name: str,
    reference: int,
    two_bandas: tuple,
    plot_order: tuple,
):
    """
    Alinhamento manual interativo de duas bandas em relação a uma banda de referência.

    Controles
    ---------
    Tecla '1' : seleciona a primeira banda de two_bandas
    Tecla '2' : seleciona a segunda banda de two_bandas

    Mouse esquerdo:
        clique + arraste para mover a banda selecionada

    Tecla 'r':
        reseta os deslocamentos

    Tecla 'q':
        fecha a janela

    Ao fechar, retorna os deslocamentos finais.
    """

    base_dir = Path(base_dir)

    i, j = two_bandas

    if reference not in range(1, 6):
        raise ValueError("reference deve estar entre 1 e 5.")

    if i not in range(1, 6) or j not in range(1, 6):
        raise ValueError("As bandas devem estar entre 1 e 5.")

    if reference in (i, j):
        raise ValueError("A referência deve ser diferente de i e j.")

    if set(plot_order) != {i, j, reference}:
        raise ValueError(
            "plot_order deve ser uma permutação de "
            f"({i}, {j}, {reference})."
        )

    # ---------------------------------------------------------
    # Leitura
    # ---------------------------------------------------------

    def read_band(b):
        path = base_dir / f"{file_name}_{b}.tif"

        if not path.exists():
            raise FileNotFoundError(path)

        return tifffile.imread(path).astype(np.float32)

    bands = {
        i: read_band(i),
        j: read_band(j),
        reference: read_band(reference),
    }

    shapes = {b.shape for b in bands.values()}

    if len(shapes) != 1:
        raise ValueError("As bandas precisam ter as mesmas dimensões.")

    # ---------------------------------------------------------
    # Normalização para visualização
    # ---------------------------------------------------------

    def normalize(image):
        low, high = np.percentile(image, (1, 99))

        if high <= low:
            return np.zeros_like(image)

        image = (image - low) / (high - low)

        return np.clip(image, 0, 1)

    normalized = {
        b: normalize(img)
        for b, img in bands.items()
    }

    # ---------------------------------------------------------
    # Estado
    # ---------------------------------------------------------

    offsets = {
        i: [0.0, 0.0],  # dx, dy
        j: [0.0, 0.0],
        reference: [0.0, 0.0],
    }

    selected_band = [i]

    dragging = [False]

    mouse_start = [None]
    offset_start = [None]

    # ---------------------------------------------------------
    # Construção RGB
    # ---------------------------------------------------------

    def build_rgb():
        aligned = {}

        for band in (i, j):

            dx, dy = offsets[band]

            aligned[band] = shift(
                normalized[band],
                shift=(dy, dx),
                order=1,
                mode="constant",
                cval=0,
                prefilter=False,
            )

        aligned[reference] = normalized[reference]

        rgb = np.dstack(
            [aligned[b] for b in plot_order]
        )

        return rgb

    # ---------------------------------------------------------
    # Figura
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    image_artist = ax.imshow(build_rgb())

    ax.axis("off")

    # ---------------------------------------------------------
    # Atualização visual
    # ---------------------------------------------------------

    def update_title():

        b = selected_band[0]

        dx_i, dy_i = offsets[i]
        dx_j, dy_j = offsets[j]

        ax.set_title(
            f"Banda selecionada: {b}\n"
            f"Banda {i}: dx={dx_i:.2f}, dy={dy_i:.2f} | "
            f"Banda {j}: dx={dx_j:.2f}, dy={dy_j:.2f}\n"
            f"[1] banda {i} | [2] banda {j} | "
            f"[r] reset | [q] sair"
        )

    def redraw():

        image_artist.set_data(build_rgb())

        update_title()

        fig.canvas.draw_idle()

    update_title()

    # ---------------------------------------------------------
    # Eventos do mouse
    # ---------------------------------------------------------

    def on_press(event):

        if event.inaxes != ax:
            return

        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        dragging[0] = True

        mouse_start[0] = (
            event.xdata,
            event.ydata,
        )

        offset_start[0] = offsets[
            selected_band[0]
        ].copy()

    def on_motion(event):

        if not dragging[0]:
            return

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = mouse_start[0]

        dx_mouse = event.xdata - x0
        dy_mouse = event.ydata - y0

        b = selected_band[0]

        dx0, dy0 = offset_start[0]

        offsets[b][0] = dx0 + dx_mouse
        offsets[b][1] = dy0 + dy_mouse

        redraw()

    def on_release(event):

        dragging[0] = False

    # ---------------------------------------------------------
    # Eventos do teclado
    # ---------------------------------------------------------

    def on_key(event):

        if event.key == "1":

            selected_band[0] = i
            redraw()

        elif event.key == "2":

            selected_band[0] = j
            redraw()

        elif event.key == "r":

            offsets[i] = [0.0, 0.0]
            offsets[j] = [0.0, 0.0]

            redraw()

        elif event.key == "q":

            plt.close(fig)

    # ---------------------------------------------------------
    # Conectar eventos
    # ---------------------------------------------------------

    fig.canvas.mpl_connect(
        "button_press_event",
        on_press,
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion,
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release,
    )

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    plt.show()

    # ---------------------------------------------------------
    # Resultado
    # ---------------------------------------------------------

    return {
        i: tuple(offsets[i]),
        j: tuple(offsets[j]),
        reference: (0.0, 0.0),
    }


#======================================================================
#======================================================================


offsets = interactive_band_alignment(
    base_dir=base_dir,
    file_name=file_name,
    reference=5,
    two_bandas=(1, 2),
    plot_order=(5, 2, 1),
)

print(offsets)

#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================
import os

os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Alignment/")
print(f'os.getcwd(): {os.getcwd()}')

from band_aligner import ManualBandAligner

aligner = ManualBandAligner(
    base_dir=base_dir,
    file_name=file_name,
    reference=5,
    two_bands=(1, 2),
    plot_order=(5, 2, 1),
)
offsets = aligner.run()
