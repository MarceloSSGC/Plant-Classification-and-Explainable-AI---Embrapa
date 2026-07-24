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
#======================================================================

print(f"\n\t\033[100;01m --- Start Pipeline --- \033[0m\n")

#======================================================================
#======================================================================
# Argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SEED", type=int, required=True)
parser.add_argument("--ALIGN", type=int, required=True)
args = parser.parse_args()

print(f"args.ALIGN: {args.ALIGN}")
print(f"type(args.ALIGN): {type(args.ALIGN)}")

SEED = args.SEED if args.SEED is not None else 42
print(f"\n SEED: \033[96;92m{SEED}\033[0m\n")
sleep(3)

ALIGN = False if args.ALIGN == 0 else True 
print(f" ALIGN: \033[96;92m{ALIGN}\033[0m\n")
sleep(3)

# Control
# x = 1/0

# SEED = 20
# ALIGN = True

random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#======================================================================
# Dataset

if not ALIGN:
    DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista"
else:
    DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/PlantaDaninha_BoaVista_Aligned_ecc_affine"

#======================================================================
# Experiment Name and Directory

experiment_name = f"first_test__15-07__SEED_{SEED}_ALIGN_{ALIGN}"
experiment_type = "alignment_compare"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/{experiment_type}/{experiment_name}"

if not os.path.isdir(DIR_EXP):
    os.makedirs(DIR_EXP)

#-----------------------------------------------------------------------
# Infos

infos = {
"experiment_name": experiment_name,
"experiment_type": experiment_type,

"DIR_EXP": DIR_EXP,
"DATA_DIR": DATA_DIR,

"SEED": SEED,
"ALIGN": ALIGN
}

with open(DIR_EXP + "/infos.json", "w", encoding="utf-8") as arquivo:
    json.dump(infos, arquivo, ensure_ascii=False, indent=4)

#-----------------------------------------------------------------------
# Control 1

if "Aligned" in DATA_DIR and ALIGN:
    print(f"Control 1 \033[96;92mOK\033[0m")
elif "Aligned" not in DATA_DIR and not ALIGN:
    print(f"Control 1 \033[96;92mOK\033[0m")
else:
    print(f"Control 1 \033[96;91m FAIL \033[0m")
    x = 1/0

sleep(3)

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
    os.makedirs(os.path.join(DIR_EXP, part), exist_ok=True)

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

#------------------------------------------------------------------------
# SELEÇÃO IMAGENS VÁLIDAS

print(f"\n --- \033[96;96mSELEÇÃO IMAGENS VÁLIDAS \033[0m ---\n")

valid_images_by_species = {}

species_folder = sorted(list(sample_by_especie.keys()))

for i, specie in enumerate(species_folder, start=1):   # i, specie = 0, species_folder[0]
    
    bad_images = []
    
    specie_dir = os.path.join(DATA_DIR, specie)
    
    if not os.path.isdir(specie_dir):
        print(f'Folder doesnt exist')
        break

    files_list = [
        x for x in os.listdir(specie_dir)
        if x.lower().endswith((".tif", ".tiff"))
    ]

    sample_names = sorted(list(set([x[:-6] for x in files_list])))
    len_sample_names_before = len(sample_names)
    print(f'sample_names: {len(sample_names)}')

    for img_name in sample_names:   # img_name = sample_names[0]
        img_test = load_multispectral_image(specie_dir, img_name)
        if img_test is None:
            sample_names.remove(img_name)

    print(f'sample_names: {len(sample_names)}')
    len_sample_names_after = len(sample_names)

    # Control
    if len_sample_names_before != len_sample_names_after:
        print(f'\n\t \033[96;91m REMOVE: {len_sample_names_after - len_sample_names_before} \033[0m \n')
        x = 1/0

    valid_images_by_species.update({specie: sample_names})

    n_samples = len(sample_names)
    print(f"\n \033[96;96m{i} of {len(species_folder)}\033[0m - {specie}: -- \033[96;93m{n_samples}\033[0m")
    if len(bad_images)>0:
        print(f'Delete: {bad_images}')
    print(f'\n' + "="*30)

for i, specie in enumerate(valid_images_by_species.keys(), start=1):
    print(f'{i}-{specie} -- \033[96;93m{len(valid_images_by_species[specie])}\033[0m')

# Valid
# 1-01_malva_branca_Agua_Boa_01 -- 15
# 2-02_Vassourinha_botao_Agua_Boa_02 -- 33
# 3-03_brizantha_Agua_Boa_03 -- 112
# 4-04_cipo_fogo_Agua_Boa_04 -- 24
# 5-05_Salsa_Agua_Boa_05 -- 21
# 6-06_capim_navalha_Agua_Boa_06 -- 27
# 7-07_capim_capeta_Agua_Boa_07 -- 62
# 8-08_malicia_Agua_Boa_08 -- 16
# 9-09_pe_galinha_Agua_Boa_09 -- 18
# 10-10_carrapico_Agua_Boa_10 -- 30
# 11-11_apaga_fogo_Agua_Boa_11 -- 38
# 12-12_Andropogon_Agua_Boa_12 -- 30
# 13-13_Traquipoon_Agua_Boa_13 -- 26
# 14-14_Jaragua_Agua_Boa_14 -- 42
# 15-15_Quicuio_Agua_Boa_15 -- 27
# 16-16_Massai_Agua_Boa_16 -- 118
# 17-17_Ruziziensis_Agua_Boa_17 -- 22
# 18-20_Guanxuma_Paludo_02 -- 20
# 19-21_Mata_Pasto_Paludo_03 -- 22
# 20-23_Braquiarinha_Paludo_04 -- 42
# 21-24_Mombaça_Paludo_05 -- 48
# 22-26_Calapogonio_Paludo_07 -- 33
# 23-27_Mavuno_Paludo_08 -- 115
# 24-28_Corda_de_viola_Paludo_09 -- 7
# 25-29_Paiaguas_Paludo_10 -- 57
# 26-30_Inaja_Serra_da_Prata_01 -- 56
# 27-31_Cipo_Serra_da_Prata_02 -- 48
# 28-32_Jurubebinha_Serra_da_Prata_03 -- 45
# 29-33_Capim_gengibre_Serra_da_Prata_04 -- 63
# 30-35_Chumbinho_Serra_da_Prata_05 -- 30
# 31-36_Unha_de_gato_Serra_da_Prata_06 -- 46

#------------------------------------------------------------------------
# DIVISÃO TRAIN / VAL / TEST E SALVAMENTO

selected_species = list(valid_images_by_species.keys())

# Testando vacuidade
print(f'os.listdir(DIR_EXP): {os.listdir(DIR_EXP)}')

# if os.listdir(DIR_EXP):
#     for per in os.listdir(DIR_EXP):
#         print(f'{per}: {len(os.listdir(DIR_EXP + "/" + per))}')

for specie in selected_species:     # specie = selected_species[0]
    
    print(f"\nProcessando espécie: \033[96;93m{specie}\033[0m")

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

        output_class_dir = os.path.join(DIR_EXP, partition, specie)
        os.makedirs(output_class_dir, exist_ok=True)

        for sample_name in samples: # sample_name = samples[0]

            # img_band = sample_name + f"_{BAND}"

            # img = load_one_band_image(specie_dir, img_band)

            img = load_multispectral_image(specie_dir, sample_name)

            if img is None:
                print(f"Amostra descartada: {sample_name}")
                x=1/0

            output_path = os.path.join(output_class_dir, f"{sample_name}.npy")

            if not os.path.isfile(output_path):
                print(f'Saving image... {sample_name}   -   {partition}')
                np.save(output_path, img)

print("\nFinalizado.")

#------------------------------------------------------------------------
# Number of Images

for partition in ["Train", "Val", "Test"]:      # partition = "Train"

    print('\n' + '='*50 + f'\n\t Partition: \033[96;92m{partition}\033[0m\n')
    per_dir = DIR_EXP + f"/{partition}"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')


#======================================================================
# Normalization

DIR_EXP = Path(DIR_EXP)
TRAIN_DIR = DIR_EXP / "Train"

#------------------------------------------------------------------------

# ETAPA 1: CALCULAR MÉDIA E DESVIO PADRÃO POR BANDA USANDO TRAIN

sum_bands = np.zeros(5, dtype=np.float64)
sum_sq_bands = np.zeros(5, dtype=np.float64)
count_pixels = np.zeros(5, dtype=np.int64)

train_files = list(TRAIN_DIR.rglob("*.npy"))

print(f"Número de imagens em Train: {len(train_files)}")

for i, file_path in enumerate(train_files):
    img = np.load(file_path)  # (H, W, 5)

    img = img.astype(np.float64)

    # Soma por banda
    sum_bands += img.sum(axis=(0, 1))

    # Soma dos quadrados por banda
    sum_sq_bands += (img ** 2).sum(axis=(0, 1))

    # Número de pixels por banda
    h, w, c = img.shape
    count_pixels += h * w

    if (i + 1) % 10 == 0:
        print(f"Processadas {i + 1}/{len(train_files)} imagens")

mean_bands = sum_bands / count_pixels

var_bands = (sum_sq_bands / count_pixels) - (mean_bands ** 2)
std_bands = np.sqrt(var_bands)

print("\nMédia por banda:")
print(mean_bands)

print("\nDesvio padrão por banda:")
print(std_bands)

#------------------------------------------------------------------------

norm_stats = {
    "mean": mean_bands.tolist(),
    "std": std_bands.tolist()
}

infos["norm_stats"] = norm_stats
with open(DIR_EXP / "infos.json", "a") as f:
    json.dump(infos, f, indent=4)

stats_path = DIR_EXP / "normalization_stats_train.json"
with open(stats_path, "w") as f:
    json.dump(norm_stats, f, indent=4)

print(f"Estatísticas salvas em: {stats_path}")

#------------------------------------------------------------------------

# ETAPA 2: NORMALIZAR TRAIN, VAL E TEST

splits = ["Train", "Val", "Test"]

mean_bands = mean_bands.reshape(1, 1, 5)
std_bands = std_bands.reshape(1, 1, 5)

for split in splits:
    input_dir = DIR_EXP / split
    output_dir = DIR_EXP / f"{split}_Norm"

    files = list(input_dir.rglob("*.npy"))

    print(f"\nNormalizando {split}: {len(files)} imagens")

    for i, file_path in enumerate(files):
        relative_path = file_path.relative_to(input_dir)

        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = np.load(file_path)  # (H, W, 5)
        img = img.astype(np.float32)

        img_norm = (img - mean_bands) / std_bands
        img_norm = img_norm.astype(np.float32)

        np.save(output_path, img_norm)

        if (i + 1) % 10 == 0:
            print(f"{split}: {i + 1}/{len(files)} imagens normalizadas  -- {split}  --")

print("\nNormalização finalizada.")

# ============================================================

# Number of Images

DIR_EXP = str(DIR_EXP)

for partition in ["Train", "Val", "Test"]:      # partition = "Train"

    print('\n' + '='*50 + f'\n\t partition: {partition}')
    per_dir = DIR_EXP + f"/{partition}"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        n_per_len = len(os.listdir(local_dir))
        n_per_len_norm = len(os.listdir(DIR_EXP + f"/{partition}_Norm/{esp}"))

        print(f'esp: {esp}  --  {n_per_len}  --  {n_per_len_norm}')

        if n_per_len != n_per_len_norm:
            print(f"\033[96;91m n_per_len != n_per_len_norm \033[0m")

#------------------------------------------------------------------------
