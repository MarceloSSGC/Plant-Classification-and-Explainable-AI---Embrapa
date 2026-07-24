import os
import numpy as np
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

#======================================================================

DATA_DIR_01 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/"

#======================================================================
# ANÁLISE EXPLORATÓRIA DOS DADOS

# Número de amostras por espécie.

species_folder = os.listdir(DATA_DIR_01)
species_name = []

for x in species_folder:
    print(x)
    species_name.append(x[3:-12])

sample_by_especie = {}
for ith_specie in species_folder:       # ith_specie = species_folder[0]

    files_list = [x for x in os.listdir(DATA_DIR_01 + f'/{ith_specie}') if 'xml' not in x]

    unique_names = []
    for x in files_list:    # x = files_list[0]
        y = x[:-6]
        if y not in unique_names:
            unique_names.append(y)
    
    print(f'{ith_specie} -- {len(unique_names)}')

    sample_by_especie.update({ith_specie[3:-12]: len(unique_names)})


# 01_malva_branca_Agua_Boa_01       16
# 02_Vassourinha_botao_Agua_Boa_02   5
# 03_brizantha_Agua_Boa_03          26
# 04_cipo_fogo_Agua_Boa_04          24
# 05_Salsa_Agua_Boa_05              22
# 06_capim_navalha_Agua_Boa_06      28
# 07_capim_capeta_Agua_Boa_07       39
# 08_malicia_Agua_Boa_08            18
# 09_pe_galinha_Agua_Boa_09         23
# 10_carrapico_Agua_Boa_10          30
# 11_apaga_fogo_Agua_Boa_11         38
# 12_Andropogon_Agua_Boa_12         31
# 13_Traquipoon_Agua_Boa_13         28
# 14_Jaragua_Agua_Boa_14            43
# 15_Quicuio_Agua_Boa_15            27
# 16_Massai_Agua_Boa_16             53
# 17_Ruziziensis_Agua_Boa_17        22
# 18_Brizantha_Agua_Boa_18          52

#------------------------------------------------------------------------
# Resolução das imagens e Faixa de valores dos pixels

import rasterio

for ith_specie in species_folder:       # ith_specie = species_folder[0]

    print(f'\n \033[96;93m{ith_specie}\033[0m')
    files_list = [x for x in os.listdir(DATA_DIR_01 + f'/{ith_specie}') if 'xml' not in x]

    arq = files_list[np.random.randint(0, len(files_list))]

    arq_dir = DATA_DIR_01 + f'/{ith_specie}/{arq}'

    with rasterio.open(arq_dir) as src:
        img = src.read(1)
    
    print(f'img: {img.shape}')
    print(f'img max: {img.max()}')
    print(f'img min: {img.min()}')

# Todas imagnes de todos tiff tem (960, 1280)

#======================================================================


# Espécie escolhida
specie_folder = "01_malva_branca_Agua_Boa_01"
specie_dir = os.path.join(DATA_DIR_01, specie_folder)

# Lista apenas arquivos TIFF, ignorando XML
files_list = [
    x for x in os.listdir(specie_dir)
    if x.lower().endswith((".tif", ".tiff"))
]

# Identifica nomes únicos das imagens
# Exemplo:
# IMG_0016_1.tif -> IMG_0016
# IMG_0016_2.tif -> IMG_0016
unique_names = sorted(list(set([x[:-6] for x in files_list])))

print(f"Número de amostras em {specie_folder}: {len(unique_names)}")
print("Primeiras amostras:", unique_names[:5])


# Escolhe a primeira amostra
sample_name = unique_names[0]

print("Amostra escolhida:", sample_name)

bands = []

for band_id in range(1, 6):
    band_file = f"{sample_name}_{band_id}.tif"
    band_path = os.path.join(specie_dir, band_file)

    with rasterio.open(band_path) as src:
        band = src.read(1)

    bands.append(band)

# Empilhamento no formato (H, W, 5)
img_multispectral = np.stack(bands, axis=-1)

print("Shape final:", img_multispectral.shape)
print("Valor mínimo:", img_multispectral.min())
print("Valor máximo:", img_multispectral.max())
print("Tipo dos dados:", img_multispectral.dtype)

#======================================================================
#======================================================================

from pathlib import Path

#------------------------------------------------------------------------
# DIRETÓRIOS

DATA_DIR_01 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/"

# DIR_EXP = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/First_Test_Daninha_Boa_Vista"
DIR_EXP = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/First_Test_Daninha_Boa_Vista_01"

#------------------------------------------------------------------------
# CONFIGURAÇÕES

random_seed = 42
random.seed(random_seed)

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
# SELEÇÃO DAS ESPÉCIES

species_folder = sorted(os.listdir(DATA_DIR_01))

selected_species = []

for specie in species_folder:   # specie = species_folder[0]
    
    specie_dir = os.path.join(DATA_DIR_01, specie)

    if not os.path.isdir(specie_dir):
        continue

    if specie == "03_brizantha_Agua_Boa_03":
        continue

    files_list = [
        x for x in os.listdir(specie_dir)
        if x.lower().endswith((".tif", ".tiff"))
    ]

    sample_names = sorted(list(set([x[:-6] for x in files_list])))

    n_samples = len(sample_names)

    if n_samples > 25:
        selected_species.append(specie)
        print(f"Selecionada: {specie} -- {n_samples}")

print("\n Espécies selecionadas:")
for s in selected_species:
    print(s)

#------------------------------------------------------------------------
# DIVISÃO TRAIN / VAL / TEST E SALVAMENTO

for specie in selected_species:     # specie = selected_species[0]
    
    print(f"\nProcessando espécie: {specie}")

    specie_dir = os.path.join(DATA_DIR_01, specie)

    files_list = [
        x for x in os.listdir(specie_dir)
        if x.lower().endswith((".tif", ".tiff"))
    ]

    sample_names = sorted(list(set([x[:-6] for x in files_list])))

    random.shuffle(sample_names)

    n = len(sample_names)

    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
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

from pathlib import Path

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

import json

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
