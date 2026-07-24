import os
import numpy as np
import pandas as pd
import random
import rasterio
from pathlib import Path
import json
import argparse
from time import sleep

#======================================================================
# SEED

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#======================================================================
# Argparse

parser = argparse.ArgumentParser()
parser.add_argument("--BAND", type=int, required=True)
args = parser.parse_args()

BAND = args.BAND

print(f"\n BAND: {BAND}\n")
sleep(4)

# x = 1/0

#======================================================================
# Dataset

DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista"

#======================================================================
# Experiment Name and Directory

experiment_name = "01_spectral_analysis_08_07"
experiment_type = "Spectral"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/{experiment_type}/{experiment_name}"


# BAND
DIR_EXP_BAND = DIR_EXP + f"/Band_0{BAND}"

#======================================================================
# ANÁLISE EXPLORATÓRIA DOS DADOS

# Número de amostras por espécie.

species_folder = os.listdir(DATA_DIR)

all_species_list = []

for x in species_folder:    # x = species_folder[0]
    
    print(x)
    all_species_list.append(x)


sample_by_especie = {}
for ith_specie in all_species_list:       # ith_specie = species_folder[0]

    files_list = [x for x in os.listdir(DATA_DIR + f'/{ith_specie}') if 'xml' not in x]

    unique_names = []
    for x in files_list:    # x = files_list[0]
        y = x[:-6]
        if y not in unique_names:
            unique_names.append(y)
    
    print(f'{ith_specie} -- {len(unique_names)}')

    sample_by_especie.update({ith_specie: len(unique_names)})

# sample_by_especie

print(f'\n len(sample_by_especie): {len(sample_by_especie)}')

# {'01_malva_branca_Agua_Boa_01': 15,
#  '02_Vassourinha_botao_Agua_Boa_02': 33,
#  '03_brizantha_Agua_Boa_03': 112,
#  '04_cipo_fogo_Agua_Boa_04': 24,
#  '05_Salsa_Agua_Boa_05': 21,
#  '06_capim_navalha_Agua_Boa_06': 27,
#  '07_capim_capeta_Agua_Boa_07': 62,
#  '08_malicia_Agua_Boa_08': 16,
#  '09_pe_galinha_Agua_Boa_09': 18,
#  '10_carrapico_Agua_Boa_10': 30,
#  '11_apaga_fogo_Agua_Boa_11': 38,
#  '12_Andropogon_Agua_Boa_12': 30,
#  '13_Traquipoon_Agua_Boa_13': 26,
#  '14_Jaragua_Agua_Boa_14': 42,
#  '15_Quicuio_Agua_Boa_15': 27,
#  '16_Massai_Agua_Boa_16': 118,
#  '17_Ruziziensis_Agua_Boa_17': 22,
#  '20_Guanxuma_Paludo_02': 20,
#  '21_Mata_Pasto_Paludo_03': 22,
#  '23_Braquiarinha_Paludo_04': 42,
#  '24_Mombaça_Paludo_05': 48,
#  '26_Calapogonio_Paludo_07': 33,
#  '27_Mavuno_Paludo_08': 115,
#  '28_Corda_de_viola_Paludo_09': 7,
#  '29_Paiaguas_Paludo_10': 57,
#  '30_Inaja_Serra_da_Prata_01': 56,
#  '31_Cipo_Serra_da_Prata_02': 48,
#  '32_Jurubebinha_Serra_da_Prata_03': 45,
#  '33_Capim_gengibre_Serra_da_Prata_04': 63,
#  '35_Chumbinho_Serra_da_Prata_05': 30,
#  '36_Unha_de_gato_Serra_da_Prata_06': 46}

#======================================================================
# DIRETÓRIOS

partitions = ["Train", "Val", "Test"]

for part in partitions:
    os.makedirs(os.path.join(DIR_EXP_BAND, part), exist_ok=True)

#------------------------------------------------------------------------
# FUNÇÃO PARA EMPILHAR AS 5 BANDAS

def load_multispectral_image(specie_dir, sample_name):
    bands = []

    for band_id in range(1, 6):
        band_file = f"{sample_name}_{band_id}.tif"
        band_path = os.path.join(specie_dir, band_file)

        if not os.path.exists(band_path):
            print(f"Arquivo não encontrado: {band_path}")
            return None

        try:
            with rasterio.open(band_path) as src:
                band = src.read(1)
        except Exception as e:
            print(f"Erro ao ler {band_path}")
            print(e)
            return None

        bands.append(band)

    img = np.stack(bands, axis=-1)

    return img

# specie_dir, img_band
def load_one_band_image(specie_dir, img_band):

    band_path = os.path.join(specie_dir, img_band + ".tif")

    if not os.path.exists(band_path):
        print(f"Arquivo não encontrado: {band_path}")
        return None

    try:
        with rasterio.open(band_path) as src:
            band = src.read(1)

    except Exception as e:
        print(f"Erro ao ler {band_path}")
        print(e)
        return None
    
    return band

#------------------------------------------------------------------------
# SELEÇÃO IMAGENS VÁLIDA

valid_images_by_species = {}

species_folder = list(sample_by_especie.keys())


for i, specie in enumerate(species_folder, start=1):   # i, specie = 0, species_folder[0]
    
    bad_images = []
    
    specie_dir = os.path.join(DATA_DIR, specie)
    
    # if not os.path.isdir(specie_dir):
    #     print(f'Folder doesnt exist')
    #     break

    files_list = [
        x for x in os.listdir(specie_dir)
        if x.lower().endswith((".tif", ".tiff"))
    ]

    sample_names = sorted(list(set([x[:-6] for x in files_list])))

    print(f'sample_names: {len(sample_names)}')

    # if specie in imgs_to_delete.keys():
    #     bad_images = imgs_to_delete[specie]
    #     for img_name in bad_images:     # img_name = bad_images[0]
    #         if img_name in sample_names:
    #             sample_names.remove(img_name)

    # print(f'sample_names: {len(sample_names)}')

    for img_name in sample_names:   # img_name = sample_names[0]
        img_band = img_name + f"_{BAND}"

        img_test = load_one_band_image(specie_dir, img_band)
        if img_test is None:
            sample_names.remove(img_name)

    print(f'sample_names: {len(sample_names)}')

    valid_images_by_species.update({specie: sample_names})

    n_samples = len(sample_names)
    print(f"\n {i} - {specie}: -- \033[96;93m{n_samples}\033[0m")
    if len(bad_images)>0:
        print(f'Delete: {bad_images}')
    print(f'\n' + "="*30)

for i, specie in enumerate(valid_images_by_species.keys(), start=1):
    print(f'{i}-{specie} -- \033[96;93m{len(valid_images_by_species[specie])}\033[0m')

#------------------------------------------------------------------------
# DIVISÃO TRAIN / VAL / TEST E SALVAMENTO

selected_species = list(valid_images_by_species.keys())

# Testando vacuidade
print(f'os.listdir(DIR_EXP): {os.listdir(DIR_EXP_BAND)}')

if os.listdir(DIR_EXP_BAND):
    for per in os.listdir(DIR_EXP_BAND):
        print(f'{per}: {len(os.listdir(DIR_EXP_BAND + "/" + per))}')

for specie in selected_species:     # specie = selected_species[0]
    
    print(f"\nProcessando espécie: \033[96;96m{specie}\033[0m")

    specie_dir = os.path.join(DATA_DIR, specie)

    sample_names = sorted(valid_images_by_species[specie])

    random.shuffle(sample_names)

    n = len(sample_names)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_samples = sample_names[:n_train]
    val_samples = sample_names[n_train:n_train + n_val]
    test_samples = sample_names[n_train + n_val:]

    split_dict = {
        "Train": train_samples,
        "Val": val_samples,
        "Test": test_samples
    }

    print(f"Total: {n}")
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Test:  {len(test_samples)}")
    print(f'test: {test_samples}')

    for partition, samples in split_dict.items():   # partition, samples = list(split_dict.items())[0]

        output_class_dir = os.path.join(DIR_EXP_BAND, partition, specie)
        os.makedirs(output_class_dir, exist_ok=True)

        for sample_name in samples: # sample_name = samples[0]

            img_band = sample_name + f"_{BAND}"

            img = load_one_band_image(specie_dir, img_band)

            # img = load_multispectral_image(specie_dir, sample_name)

            if img is None:
                print(f"Amostra descartada: {sample_name}")
                continue

            output_path = os.path.join(output_class_dir, f"{sample_name}.npy")

            if not os.path.isfile(output_path):
                print(f'Saving image... {sample_name}')
                np.save(output_path, img)

print("\nFinalizado.")

#------------------------------------------------------------------------
# Number of Images

for partition in ["Train", "Val", "Test"]:      # partition = "Train"

    print('\n' + '='*50 + f'\n\t Partition: \033[96;92m{partition}\033[0m\n')
    per_dir = DIR_EXP_BAND + f"/{partition}"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')


#======================================================================
# Normalization

DIR_EXP_BAND = Path(DIR_EXP_BAND)

TRAIN_DIR = DIR_EXP_BAND / "Train"

#------------------------------------------------------------------------
# ETAPA 1: CALCULAR MÉDIA E DESVIO PADRÃO USANDO TRAIN

sum_pixels = 0.0
sum_sq_pixels = 0.0
count_pixels = 0

train_files = list(TRAIN_DIR.rglob("*.npy"))

print(f"Número de imagens em Train: {len(train_files)}")

for i, file_path in enumerate(train_files):
    img = np.load(file_path)  # (H, W)

    img = img.astype(np.float64)

    # Soma dos pixels
    sum_pixels += img.sum()

    # Soma dos quadrados dos pixels
    sum_sq_pixels += (img ** 2).sum()

    # Número de pixels
    h, w = img.shape
    count_pixels += h * w

    if (i + 1) % 10 == 0:
        print(f"Processadas {i + 1}/{len(train_files)} imagens")

mean = sum_pixels / count_pixels

var = (sum_sq_pixels / count_pixels) - (mean ** 2)
std = np.sqrt(var)

print("\nMédia:")
print(mean)

print("\nDesvio padrão:")
print(std)

#------------------------------------------------------------------------


norm_stats = {
    "mean": mean,
    "std": std
}

stats_path = DIR_EXP_BAND / "normalization_stats_train.json"

with open(stats_path, "w") as f:
    json.dump(norm_stats, f, indent=4)

print(f"Estatísticas salvas em: {stats_path}")

#------------------------------------------------------------------------

# ETAPA 2: NORMALIZAR TRAIN, VAL E TEST

splits = ["Train", "Val", "Test"]

for split in splits:        # split = "Train"
    input_dir = DIR_EXP_BAND / split
    output_dir = DIR_EXP_BAND / f"{split}_Norm"

    files = list(input_dir.rglob("*.npy"))

    print(f"\nNormalizando {split}: {len(files)} imagens")

    for i, file_path in enumerate(files):   # i, file_path = 0, files[0]
        relative_path = file_path.relative_to(input_dir)

        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = np.load(file_path)  # (H, W, 5)
        img = img.astype(np.float32)

        img_norm = (img - mean) / std
        img_norm = img_norm.astype(np.float32)

        np.save(output_path, img_norm)

        if (i + 1) % 10 == 0:
            print(f"{split}: {i + 1}/{len(files)} imagens normalizadas")

print("\nNormalização finalizada.")

# ============================================================
# Number of Images

DIR_EXP_BAND = str(DIR_EXP_BAND)

for partition in ["Train", "Val", "Test"]:      # partition = "Train"

    print('\n' + '='*50 + f'\n\t partition: {partition}')
    per_dir = DIR_EXP_BAND + f"/{partition}"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')

    print('\n Norm')
    per_dir = DIR_EXP_BAND + f"/{partition}_Norm"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')


#------------------------------------------------------------------------
