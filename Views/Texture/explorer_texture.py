import os
import numpy as np

print(f'os.getcwd(): {os.getcwd()}')

from aux_texture import *

#======================================================================
#======================================================================
#======================================================================
# Dataset

DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned_ecc_affine_interch_45_cen_5__Seg_best_band_otsu"

especies = sorted(os.listdir(DATA_DIR))

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(DATA_DIR, especie)
files = sorted(set([x[:-6]for x in os.listdir(especie_dir)]))


file_name = files[0]
# file_name = "IMG_0008"

#======================================================================
# Load

img_5b = load_5b_from_dir(especie_dir, file_name)


# Load from Augmentation

DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20_AUG/Train_Norm"
especies = sorted(os.listdir(DATA_DIR))
especie = "36_Unha_de_gato_Serra_da_Prata_06"
especie_dir = os.path.join(DATA_DIR, especie)
files = sorted(os.listdir(especie_dir))

file_name = files[0]
file_dir = os.path.join(especie_dir, file_name)

img_5b = np.load(file_dir)
print(f"img_5b.shape: {img_5b.shape}")
print(f"img_5b.mean: {img_5b.mean(axis=(0, 1))}")
print(f"img_5b.std: {img_5b.std(axis=(0, 1))}")

#======================================================================

plot_rgb(img_5b)

plot_band(img_5b[:, :, 2])

plot_band(img_5b[:, :, 3], title="NIR")

plot_5bands(img_5b)

#======================================================================
# LBP (Local Binary Pattern)

LBP = apply_lbp_1b(img_5b[:, :, 3])
plot_band(LBP, title="Canny - NIR")
plot_band(LBP, title="Canny - NIR", figsize=(20, 15))

LBP = apply_lbp_1b(img_5b[:, :, 1], radius=1, n_points=8)
plot_band(LBP, title="Canny - G")

all_LBP = apply_lbp_5b(img_5b)
plot_5bands(all_LBP)
plot_rgb(all_LBP)


LBP = apply_lbp_1b(img_5b[:, :, 1], radius=4, n_points=4)
plot_band(LBP, title="Canny - G")


# plot_rgb(keep_bigger_components(img_5b, 20))

#======================================================================
# Gabor filter bank

GABOR = apply_gabor_1b(img_5b[:, :, 3])

print(GABOR.shape)
plot_band(GABOR[:, :, 0], title="Gabor - NIR")

for i in range(GABOR.shape[-1]):
    plot_band(
        GABOR[:, :, i],
        title=f"Gabor - NIR - Filter {i}"
    )


ALL_GABOR = apply_gabor_5b(img_5b)


#======================================================================
# Wavelet decomposition

WAVELET = apply_wavelet_1b(
    img_5b[:, :, 3],
    wavelet="haar"
)

print(WAVELET.shape)
# aproximadamente (480, 640, 4)

names = ["LL", "LH", "HL", "HH"]

for i, name in enumerate(names):
    plot_band(
        WAVELET[:, :, i],
        title=f"Wavelet - NIR - {name}"
    )


all_WAVELET = apply_wavelet_5b(
    img_5b,
    wavelet="haar"
)

print(all_WAVELET.shape)
# aproximadamente (480, 640, 5, 4)


plot_5bands(
    all_WAVELET[:, :, :, 3]
)

#======================================================================
# Local Variance

LOCAL_VAR = apply_local_variance_1b(
    img_5b[:, :, 3],
    window_size=5
)

plot_band(LOCAL_VAR, title="Local Variance - NIR")



#======================================================================
# Frangi / Hessian vesselness

FRANGI = apply_frangi_1b(
    img_5b[:, :, 4],
    sigmas=range(1, 5),
    black_ridges=False
)

plot_band(
    FRANGI,
    title="Frangi - NIR"
)

#======================================================================
# DoG / LoG

img_1b = img_5b[:, :, 1]  # Green

dog = texture_filter(img_1b, method="dog", sigma1=1, sigma2=2)
log = texture_filter(img_1b, method="log", sigma1=1)

plot_band(dog, title="DoG - Green")

plot_band(log, title="DoGLoG - Green")

#======================================================================
# normalização robusta → CLAHE → Frangi multiescala

texture = texture_frangi(img_5b[:, :, 1])
plot_band(texture)

#======================================================================
# glcm

img_1b = img_5b[:, :, 3]  # ex: banda NIR

# Mapa de rugosidade/contraste
contrast_map = glcm_texture_map(img_1b, prop='contrast', window_size=15, stride=4)

# Mapa de homogeneidade (textura "lisa")
homog_map = glcm_texture_map(img_1b, prop='homogeneity', window_size=15, stride=4)

print(contrast_map.shape)  # (960, 1280) — mesma dimensão do input

plot_band(contrast_map)
plot_band(homog_map)

img_1b = img_5b[:, :, 3]  # ex: banda NIR
homog_map = glcm_texture_map(img_1b, prop='homogeneity', window_size=15)

plot_band(homog_map)

roughness = local_roughness_map(img_1b, window_size=9)
plot_band(roughness)

roughness = local_roughness_map_masked(img_1b)
plot_band(roughness)
plot_band(img_5b[:, :, 3], title="NIR")

test_2 = shuffle_patches(roughness, grid=(4, 4), seed=3)
plot_band(test_2, figsize=(20, 15))


#======================================================================
# Frangi

img_1b = img_5b[:, :, 3]  # ex: banda NIR
vessel_map = apply_frangi(img_1b)
plot_band(vessel_map, figsize=(20, 15))
plot_band(img_5b[:, :, 3], title="NIR")
plot_band(roughness, figsize=(20, 15))

#======================================================================
# sato, meijering

img_1b = img_5b[:, :, 3]  # ex: banda NIR

sato_map = apply_sato(img_1b, bg_value=0.0, black_ridges=False)
meijering_map = apply_meijering(img_1b, bg_value=0.0, black_ridges=False)

plot_band(sato_map)
plot_band(img_5b[:, :, 3], title="NIR")
plot_band(meijering_map)

#======================================================================
# tophat

img_1b = img_5b[:, :, 3]  # ex: banda NIR

tophat_map = apply_tophat(img_1b, bg_value=0.0, radius=5, mode='white')
blackhat_map = apply_tophat(img_1b, bg_value=0.0, radius=5, mode='black')

plot_band(tophat_map)
plot_band(img_5b[:, :, 3], title="NIR")
plot_band(blackhat_map)

# plot_band(clip_percentile_normalize(tophat_map, percentile=0.99))
# plot_band(clip_percentile_normalize(blackhat_map, percentile=0.99))

tophat_map = apply_tophat(img_1b, bg_value=0.0, radius=20, mode='white')
blackhat_map = apply_tophat(img_1b, bg_value=0.0, radius=15, mode='black')

plot_band(tophat_map)
plot_band(blackhat_map)

#======================================================================
# skeleton

img_1b = img_5b[:, :, 3]  # ex: banda NIR
vessel_map = apply_frangi(img_1b, bg_value=0.0, black_ridges=False)

skeleton, binary = apply_skeletonize(vessel_map, threshold=None, min_size=20)
plot_band(img_5b[:, :, 3], title="NIR")
plot_band(binary)

plot_band(skeleton + blackhat_map)

#======================================================================
# Structure tensor (tensor de estrutura local)

img_1b = img_5b[:, :, 3]  # ex: banda NIR
orientation_map, coherence_map = apply_structure_tensor(img_1b, bg_value=0.0, sigma=2)

plot_band(orientation_map)
plot_band(coherence_map)

test = stretch_border_to_background(coherence_map, size=10)
plot_band(test, figsize=(20, 15))
plot_band(coherence_map)


img_1b = img_5b[:, :, 1]  # G
orientation_map, coherence_map = apply_structure_tensor(img_1b, bg_value=0.0, sigma=1)

plot_band(orientation_map)
plot_band(coherence_map)
plot_band(keep_bigger_components(coherence_map[:, :], 15))

#======================================================================
# Entropia local (skimage.filters.rank.entropy)

img_1b = img_5b[:, :, 3]  # ex: banda NIR
entropy_map = apply_local_entropy(img_1b, bg_value=0.0, radius=5, levels=32)
plot_band(entropy_map)


img_1b = img_5b[:, :, 1]  # G
entropy_map = apply_local_entropy(img_1b, bg_value=0.0, radius=4, levels=8)
plot_band(entropy_map)

#======================================================================
#======================================================================
#======================================================================
#======================================================================
# Build imagens



best_img = {
    "01_malva_branca_Agua_Boa_01": "IMG_0020",
    "02_Vassourinha_botao_Agua_Boa_02": "IMG_0002",
    "03_brizantha_Agua_Boa_03": "IMG_0038",
    "04_cipo_fogo_Agua_Boa_04": "IMG_0067",
    "05_Salsa_Agua_Boa_05": "IMG_0094",
    "06_capim_navalha_Agua_Boa_06": "IMG_0120",
    "07_capim_capeta_Agua_Boa_07": "IMG_0099",
    "08_malicia_Agua_Boa_08": "IMG_0178",
    "09_pe_galinha_Agua_Boa_09": "IMG_0008",
    "10_carrapico_Agua_Boa_10": "IMG_0026",
    "11_apaga_fogo_Agua_Boa_11": "IMG_0059",
    "12_Andropogon_Agua_Boa_12": "IMG_0092",
    "13_Traquipoon_Agua_Boa_13": "IMG_0128",
    "14_Jaragua_Agua_Boa_14": "IMG_0045",
    "15_Quicuio_Agua_Boa_15": "IMG_0021",
    "16_Massai_Agua_Boa_16": "IMG_0009",
    "17_Ruziziensis_Agua_Boa_17": "IMG_0052",
    "20_Guanxuma_Paludo_02": "IMG_0042",
    "21_Mata_Pasto_Paludo_03": "IMG_0067",
    "23_Braquiarinha_Paludo_04": "IMG_0003",
    "24_Mombaça_Paludo_05": "IMG_0053",
    "26_Calapogonio_Paludo_07": "IMG_0073",
    "27_Mavuno_Paludo_08": "IMG_0110",
    "28_Corda_de_viola_Paludo_09": "IMG_0226",
    "29_Paiaguas_Paludo_10": "IMG_0022",
    "30_Inaja_Serra_da_Prata_01": "IMG_0006",
    "31_Cipo_Serra_da_Prata_02": "IMG_0061",
    "32_Jurubebinha_Serra_da_Prata_03": "IMG_0126",
    "33_Capim_gengibre_Serra_da_Prata_04": "IMG_0176",
    "35_Chumbinho_Serra_da_Prata_05": "IMG_0055",
    "36_Unha_de_gato_Serra_da_Prata_06": "IMG_0076"
}

especie = "01_malva_branca_Agua_Boa_01"

for especie in sorted(best_img.keys())[:3]:

    especie_dir = os.path.join(DATA_DIR, especie)

    file_name = best_img[especie]

    img_5b = load_5b_from_dir(especie_dir, file_name)

    print("\n\n"+ "="*70 + f"\n especie: {especie} - file_name: {file_name}")
    # plot_rgb(img_5b)

    LBP = apply_lbp_1b(img_5b[:, :, 3])
    plot_band(LBP, title="Canny - NIR", figsize=(20, 15))

    test = stretch_border_to_background(LBP, size=10)
    plot_band(test, figsize=(20, 15))

    test_2 = shuffle_patches(LBP, grid=(4, 4), seed=3)
    plot_band(test_2, figsize=(20, 15))

    test_3 = stretch_border_to_background(test_2, size=10)
    plot_band(test_3, figsize=(20, 15))















