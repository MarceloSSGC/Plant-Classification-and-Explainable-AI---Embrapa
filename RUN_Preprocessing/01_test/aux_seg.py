import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile

#======================================================================
#======================================================================

print(f"\n\033[100;40m\t     --- Auxiliar Segmentation RUN---     \t\t\033[0m\n")

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

def segment_best_band_otsu_green(img_5b):
    """
    Segmentação por:
    1. seleção automática da melhor banda para Otsu;
    2. uso de Excess Green para decidir qual lado da máscara é vegetação;
    3. aplicação da máscara final às 5 bandas.

    Mapeamento esperado:
        banda 1 = Blue
        banda 2 = Green
        banda 3 = Red
        banda 4 = NIR
        banda 5 = Red Edge

    Parâmetros
    ----------
    img_5b : np.ndarray
        Imagem multiespectral com shape (H, W, 5).

    Retorna
    -------
    img_5b_seg : np.ndarray
        Imagem segmentada com shape (H, W, 5).

    mask : np.ndarray
        Máscara binária com shape (H, W).
        1 = planta
        0 = fundo.

    best_band : int
        Banda selecionada pelo critério de separabilidade.
        Retornada no padrão 1..5.
    """

    if img_5b.ndim != 3 or img_5b.shape[2] != 5:
        raise ValueError(
            f"Esperado shape (H, W, 5), recebido {img_5b.shape}"
        )

    img_float = img_5b.astype(np.float32)

    best_score = -np.inf
    best_mask = None
    best_band = None

    # -------------------------------------------------
    # 1. Avalia Otsu em cada uma das 5 bandas
    # -------------------------------------------------
    for b in range(5):

        band = img_float[:, :, b]

        threshold = threshold_otsu(band)

        mask_high = band > threshold
        mask_low = ~mask_high

        if mask_high.sum() == 0 or mask_low.sum() == 0:
            continue

        # Proporção de pixels em cada classe
        w_high = mask_high.mean()
        w_low = mask_low.mean()

        # Média radiométrica de cada classe
        mu_high = band[mask_high].mean()
        mu_low = band[mask_low].mean()

        # Variância entre classes
        score = (
            w_high *
            w_low *
            (mu_high - mu_low) ** 2
        )

        if score > best_score:
            best_score = score
            best_mask = mask_high
            best_band = b

    if best_mask is None:
        raise RuntimeError(
            "Não foi possível selecionar uma banda válida."
        )

    # -------------------------------------------------
    # 2. Excess Green
    #
    # Bandas:
    # B1 = Blue  -> índice 0
    # B2 = Green -> índice 1
    # B3 = Red   -> índice 2
    # -------------------------------------------------
    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]

    exg = 2 * G - R - B

    # -------------------------------------------------
    # 3. Mede o verdor dos dois lados do Otsu
    # -------------------------------------------------
    green_high = np.mean(exg[best_mask])
    green_low = np.mean(exg[~best_mask])

    # O lado com maior ExG é considerado planta
    if green_low > green_high:
        best_mask = ~best_mask

    # -------------------------------------------------
    # 4. Aplica a máscara às cinco bandas
    # -------------------------------------------------
    img_5b_seg = img_5b.copy()

    img_5b_seg[~best_mask] = 0

    mask = best_mask.astype(np.uint8)

    # converte índice 0..4 para banda 1..5
    best_band = best_band + 1

    return img_5b_seg, mask, best_band

#======================================================================

def plot_segmentation(rgb_img, rgb_img_masked):
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

def save_segmented_bands(img_5b_seg, base_dir, file_name):
    """
    Salva as 5 bandas segmentadas como TIFF uint16.
    """

    if img_5b_seg.ndim != 3 or img_5b_seg.shape[2] != 5:
        raise ValueError(
            f"Esperado shape (H, W, 5), recebido {img_5b_seg.shape}"
        )

    os.makedirs(base_dir, exist_ok=True)

    file_name = os.path.splitext(file_name)[0]

    for band_idx in range(5):

        band = img_5b_seg[:, :, band_idx]

        # Converte de volta para uint16
        band = np.clip(band, 0, 65535).astype(np.uint16)

        output_path = os.path.join(
            base_dir,
            f"{file_name}_{band_idx + 1}.tif"
        )

        tifffile.imwrite(
            output_path,
            band,
            photometric="minisblack",
            metadata=None
        )

#======================================================================

def plot_rgb(rgb_image, bands_ch=(3, 2, 1)):

    channels = []

    for band in range(3):
        
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
#======================================================================
#======================================================================

# segmentation_method = SEGMENTATION_METHOD
# align_data_dir = ALIGH_DATA_DIR
# seg_data_dir = SEG_DATA_DIR

def segmentation_main_function(segmentation_method, align_data_dir, seg_data_dir, plot_both=False):

    print(f'segmentation_method: \033[96;92m{segmentation_method}\033[0m')
    print(f'align_data_dir: \033[96;92m{align_data_dir}\033[0m')
    print(f'seg_data_dir: \033[96;92m{seg_data_dir}\033[0m')

    #----------------------------------------------------------

    if not os.path.isdir(align_data_dir):
        raise ValueError(f"align_data_dir = {align_data_dir} doesnt exist")
    
    os.makedirs(seg_data_dir, exist_ok=True)

    #----------------------------------------------------------

    especies = sorted(os.listdir(align_data_dir))

    for especie in especies[:31]:    # especie = especies[0]

        print("\n"+"="*60 + f'\n\t especie: {especie}')

        old_especie_dir = os.path.join(align_data_dir, especie)
        new_especie_dir = os.path.join(seg_data_dir, especie)

        if not os.path.isdir(new_especie_dir):
            os.makedirs(new_especie_dir)
        
        file_names = sorted(set([x[:-6] for x in os.listdir(old_especie_dir)]))

        for ith, ith_file in enumerate(file_names):     # ith, ith_file = 0, file_names[0]

            print(f'ith: {ith} - {len(file_names)} \t file: {ith_file} \t {especie}')

            final_dir = os.path.join(new_especie_dir, ith_file)

            if not os.path.isfile(final_dir + "_1.tif") or \
                not os.path.isfile(final_dir + "_2.tif") or \
                not os.path.isfile(final_dir + "_3.tif") or \
                not os.path.isfile(final_dir + "_4.tif") or \
                not os.path.isfile(final_dir + "_5.tif"):

                # print(f'segmentation...')
                img_5b = load_5b_from_dir(old_especie_dir, ith_file)
                # print(f"img_5b.shape : {img_5b.shape} \n")

                if segmentation_method == "best_band_otsu_green":
                    img_5b_seg, mask, best_band = segment_best_band_otsu_green(img_5b)
                else:
                    img_5b_seg = img_5b.copy()
                # plot_rgb(img_5b_seg)

                if plot_both:
                    plot_segmentation(img_5b, img_5b_seg)

                save_segmented_bands(img_5b_seg, new_especie_dir, ith_file)











