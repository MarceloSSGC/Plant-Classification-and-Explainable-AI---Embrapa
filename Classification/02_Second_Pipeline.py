import os
import numpy as np
import pandas as pd
import random
import rasterio
from pathlib import Path
import json


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


#======================================================================

DATA_DIR_01 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/"
DATA_DIR_02 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_18_Paludo_Serra_Prata"

# Duplicação
# 2, 3, 7, 16, 18, 19, 22, 25, 34


DIR_EXP = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Second_Test_Daninha_Boa_Vista"

#======================================================================
# ANÁLISE EXPLORATÓRIA DOS DADOS

# Número de amostras por espécie.

species_folder_1 = os.listdir(DATA_DIR_01)
species_folder_2 = os.listdir(DATA_DIR_02)

all_species_list = []

for x in species_folder_1:
    print(x)
    all_species_list.append(x)

for x in species_folder_2:
    print(x)
    all_species_list.append(x)

sample_by_especie = {}
for ith_specie in all_species_list:       # ith_specie = species_folder[0]

    if int(ith_specie[:2]) <= 18:
        files_list = [x for x in os.listdir(DATA_DIR_01 + f'/{ith_specie}') if 'xml' not in x]
    else:
        files_list = [x for x in os.listdir(DATA_DIR_02 + f'/{ith_specie}') if 'xml' not in x]

    unique_names = []
    for x in files_list:    # x = files_list[0]
        y = x[:-6]
        if y not in unique_names:
            unique_names.append(y)
    
    print(f'{ith_specie} -- {len(unique_names)}')

    sample_by_especie.update({ith_specie: len(unique_names)})

sample_by_especie

print(f'\n len(sample_by_especie): {len(sample_by_especie)}')


# 01_malva_branca_Agua_Boa_01 -- 16
# 02_Vassourinha_botao_Agua_Boa_02 -- 5
# 03_brizantha_Agua_Boa_03 -- 26
# 04_cipo_fogo_Agua_Boa_04 -- 24
# 05_Salsa_Agua_Boa_05 -- 22
# 06_capim_navalha_Agua_Boa_06 -- 28
# 07_capim_capeta_Agua_Boa_07 -- 39
# 08_malicia_Agua_Boa_08 -- 18
# 09_pe_galinha_Agua_Boa_09 -- 23
# 10_carrapico_Agua_Boa_10 -- 30
# 11_apaga_fogo_Agua_Boa_11 -- 38
# 12_Andropogon_Agua_Boa_12 -- 31
# 13_Traquipoon_Agua_Boa_13 -- 28
# 14_Jaragua_Agua_Boa_14 -- 43
# 15_Quicuio_Agua_Boa_15 -- 27
# 16_Massai_Agua_Boa_16 -- 53
# 17_Ruziziensis_Agua_Boa_17 -- 22
# 18_Brizantha_Agua_Boa_18 -- 52
# 19_Brizantha_Paludo_01 -- 37
# 20_Guanxuma_Paludo_02 -- 20
# 21_Mata_Pasto_Paludo_03 -- 22
# 22_capim_capeta_Paludo_03 -- 29
# 23_Braquiarinha_Paludo_04 -- 45
# 24_Mombaça_Paludo_05 -- 51
# 25_Massai_Paludo_06 -- 67
# 26_Calapogonio_Paludo_07 -- 35
# 27_Mavuno_Paludo_08 -- 121
# 28_Corda_de_viola_Paludo_09 -- 9
# 29_Paiaguas_Paludo_10 -- 63
# 30_Inaja_Serra_da_Prata_01 -- 58
# 31_Cipo_Serra_da_Prata_02 -- 52
# 32_Jurubebinha_Serra_da_Prata_03 -- 46
# 33_Capim_gengibre_Serra_da_Prata_04 -- 67
# 34_vassourinha_de_Botão_Serra_da_Prata_05 -- 29
# 35_Chumbinho_Serra_da_Prata_05 -- 36
# 36_Unha_de_gato_Serra_da_Prata_06 -- 50

#======================================================================
# Delete Species

species_to_delete = ["02_Vassourinha_botao_Agua_Boa_02",
                    "03_brizantha_Agua_Boa_03",
                    "07_capim_capeta_Agua_Boa_07",
                    "16_Massai_Agua_Boa_16",
                    "18_Brizantha_Agua_Boa_18",
                    "19_Brizantha_Paludo_01",
                    "22_capim_capeta_Paludo_03",
                    "25_Massai_Paludo_06",
                    "34_vassourinha_de_Botão_Serra_da_Prata_05"
                    ]

for name in species_to_delete:
    if name in sample_by_especie.keys():
        del sample_by_especie[name]
    else:
        print(f'Not found: {name}')

print(f'\n len(sample_by_especie): {len(sample_by_especie)}')

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
# SELEÇÃO IMAGENS VÁLIDA

imgs_to_delete = {
    "01_malva_branca_Agua_Boa_01": ["IMG_0014"],
    "05_Salsa_Agua_Boa_05": ["IMG_0091"],
    "06_capim_navalha_Agua_Boa_06": ["IMG_0129"],       # Botas
    "07_capim_capeta_Agua_Boa_07": ["IMG_0137", "IMG_0138"],
    "08_malicia_Agua_Boa_08": ["IMG_0174", "IMG_0175"],
    "09_pe_galinha_Agua_Boa_09": ["IMG_0000", "IMG_0001", "IMG_0002", "IMG_0003"],
    "13_Traquipoon_Agua_Boa_13": ["IMG_0122", "IMG_0123"],
    "14_Jaragua_Agua_Boa_14": ["IMG_0009"],
    "16_Massai_Agua_Boa_16": ["IMG_0001"],
    "19_Brizantha_Paludo_01": ["IMG_0000"],
    "22_capim_capeta_Paludo_03": ["IMG_0083", "IMG_0084"],
    "23_Braquiarinha_Paludo_04": ["IMG_0112", "IMG_0113"],
    "24_Mombaça_Paludo_05": ["IMG_0043", "IMG_0044"],
    "25_Massai_Paludo_06": ["IMG_0000"],
    "27_Mavuno_Paludo_08": ["IMG_0102", "IMG_0103"],
    "29_Paiaguas_Paludo_10": ["IMG_0000", "IMG_0001"],
    "30_Inaja_Serra_da_Prata_01": ["IMG_0000", "IMG_0001"],
    "33_Capim_gengibre_Serra_da_Prata_04": ["IMG_0157", "IMG_0158", "IMG_0159", "IMG_0160"],
    "35_Chumbinho_Serra_da_Prata_05": ["IMG_0029", "IMG_0030", "IMG_0031", "IMG_0032"],
    "36_Unha_de_gato_Serra_da_Prata_06": ["IMG_0065", "IMG_0066", "IMG_0067", "IMG_0068"]
}

valid_images_by_species = {}

species_folder = list(sample_by_especie.keys())


for i, specie in enumerate(species_folder, start=1):   # i, specie = 0, species_folder[0]
    
    bad_images = []
    
    if int(specie[:2]) <= 18:
        specie_dir = os.path.join(DATA_DIR_01, specie)
    else:
        specie_dir = os.path.join(DATA_DIR_02, specie)

    # if not os.path.isdir(specie_dir):
    #     print(f'Folder doesnt exist')
    #     break

    files_list = [
        x for x in os.listdir(specie_dir)
        if x.lower().endswith((".tif", ".tiff"))
    ]

    sample_names = sorted(list(set([x[:-6] for x in files_list])))

    print(f'sample_names: {len(sample_names)}')

    if specie in imgs_to_delete.keys():
        bad_images = imgs_to_delete[specie]
        for img_name in bad_images:     # img_name = bad_images[0]
            if img_name in sample_names:
                sample_names.remove(img_name)

    print(f'sample_names: {len(sample_names)}')

    for img_name in sample_names:   # img_name = sample_names[0]
        img_test = load_multispectral_image(specie_dir, img_name)
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
print(f'os.listdir(DIR_EXP): {os.listdir(DIR_EXP)}')
if os.listdir(DIR_EXP):
    for per in os.listdir(DIR_EXP):
        print(f'{per}: {len(os.listdir(DIR_EXP + "/" + per))}')

for specie in selected_species:     # specie = selected_species[0]
    
    print(f"\nProcessando espécie: \033[96;96m{specie}\033[0m")

    if int(specie[:2]) <= 18:
        specie_dir = os.path.join(DATA_DIR_01, specie)
    else:
        specie_dir = os.path.join(DATA_DIR_02, specie)

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

            img = load_multispectral_image(specie_dir, sample_name)

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

    print('\n' + '='*50 + f'\n\t partition: {partition}')
    per_dir = DIR_EXP + f"/{partition}"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')


#======================================================================


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
            print(f"{split}: {i + 1}/{len(files)} imagens normalizadas")

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
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')

    print('\n Norm')
    per_dir = DIR_EXP + f"/{partition}_Norm"
    especies = os.listdir(per_dir)

    for esp in especies:        # esp = especies[0]
        local_dir = per_dir + f'/{esp}'
        print(f'esp: {esp}  --  {len(os.listdir(local_dir))}')


#------------------------------------------------------------------------
