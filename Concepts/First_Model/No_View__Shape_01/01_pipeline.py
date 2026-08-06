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

print(f"\n\033[100;01m\t     --- Start Model ---     \t\033[0m\n")

#======================================================================
#                            --- Params ---

interactive = True
print(f"\n\t  Interactive: \033[96;95m{interactive} \033[0m\n")


#======================================================================
#======================================================================
#======================================================================
# Interactive or Argparse

if interactive: 
    SEED = 10
    ALIGN = True
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, required=True)
    parser.add_argument("--ALIGN", type=int, required=True)
    args = parser.parse_args()

    SEED = args.SEED
    ALIGN = False if args.ALIGN == 0 else True 

print("\n" + "- "*40)
print(f"\n SEED: \033[96;91m{SEED}\033[0m")
print(f" ALIGN: \033[96;91m{ALIGN}\033[0m\n")
print("- "*40 + "\n")
sleep(3)

# Control
# x = 1/0
#======================================================================
# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#======================================================================
# Dataset & Directories

DATASET_NAME = "PlantaDaninha_BoaVista_Aligned_ecc_affine_interch_45_cen_5"
DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/"

if not ALIGN:
    DATA_DIR = os.path.join(DATA_DIR, "PlantaDaninha_BoaVista")
else:
    DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME)

#======================================================================
# Experiment Name and Directory

experiment_name = f"concepts_no_view__shape_{SEED}_ALIGN_{ALIGN}"
experiment_type = "concepts_no_view"
DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/{experiment_type}/{experiment_name}"

if not os.path.isdir(DIR_EXP):
    os.makedirs(DIR_EXP)

#======================================================================
# DataFrame SHAPE

df_shape_orign_dir = "/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Concepts/Concept_Classification/df_shape.csv"
df_shape_end_dir = os.path.join(DIR_EXP, "df_shape.csv")

df_shape = pd.read_csv(df_shape_orign_dir)

if not os.path.isfile(df_shape_end_dir):
    df_shape.to_csv(df_shape_end_dir, index=False)

#======================================================================

# Infos

DIR_INFOS = DIR_EXP + "/infos.json"

if not os.path.isfile(DIR_INFOS):
    infos = {
    "Interactive": interactive,

    "DATASET_NAME": DATASET_NAME,
    "experiment_name": experiment_name,
    "experiment_type": experiment_type,

    "DIR_EXP": DIR_EXP,
    "DATA_DIR": DATA_DIR,

    "SEED": SEED,
    "ALIGN": ALIGN,

    "TRAIN_SIZE": 0.7,
    "VAL_SIZE": 0.15,
    "TEST_SIZE": 0.15,

    "df_shape_orign_dir": df_shape_orign_dir,
    }

    with open(DIR_INFOS, "w", encoding="utf-8") as arquivo:
        json.dump(infos, arquivo, ensure_ascii=False, indent=4)

else:
    with open(DIR_INFOS, "r") as file:
        infos = json.load(file)

#-----------------------------------------------------------------------
# Control 1

if "Aligned" in DATA_DIR and ALIGN:
    # print(f"Control 1 \033[96;92mOK\033[0m")
    q=0
elif "Aligned" not in DATA_DIR and not ALIGN:
    # print(f"Control 1 \033[96;92mOK\033[0m")
    q=0
else:
    print(f"Control 1 \033[96;91m FAIL \033[0m")
    x = 1/0


#======================================================================
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


#======================================================================
# DIVISÃO TRAIN / VAL / TEST - JSON

split_file_dir = os.path.join(DIR_EXP, "split_files.json")

species = sorted(os.listdir(DATA_DIR))

if not os.path.isfile(split_file_dir):
        
    split_file_names_train = {}
    split_file_names_val = {}
    split_file_names_test = {}

    for specie in species:     # specie = species[0]
        
        print(f"\nProcessando espécie: \033[96;93m{specie}\033[0m")

        specie_dir = os.path.join(DATA_DIR, specie)

        sample_names = sorted(set([x[:-6] for x in os.listdir(specie_dir)]))

        random.shuffle(sample_names)

        n = len(sample_names)

        n_train = int(infos["TRAIN_SIZE"] * n)
        n_val = int(infos["VAL_SIZE"] * n)
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

        split_file_names_train.update({specie: train_samples})
        split_file_names_val.update({specie: val_samples})
        split_file_names_test.update({specie: test_samples})

    split_file_names = {
        "Train": split_file_names_train,
        "Val": split_file_names_val,
        "Test": split_file_names_test,
    }

    with open(split_file_dir, "w") as file:
        json.dump(split_file_names, file, indent=4) 

    print("Split Names: \033[96;92mSaved\033[0m")

else:
    with open(split_file_dir, "r", encoding="utf-8") as file:
        split_file_names = json.load(file)
    print("Split Names: \033[96;92mLoaded\033[0m")

#------------------------------------------------------------------------
# DIRETÓRIOS

partitions = ["Train", "Val", "Test"]

for part in partitions:     # part = "Train"
    os.makedirs(os.path.join(DIR_EXP, part), exist_ok=True)

#------------------------------------------------------------------------
# Split 

for specie in species:     # specie = species[0]
    
    # print(f"\nProcessando espécie: \033[96;93m{specie}\033[0m")

    for partition in partitions:   # partition = "Train"

        samples = split_file_names[partition][specie]

        output_class_dir = os.path.join(DIR_EXP, partition, specie)
        os.makedirs(output_class_dir, exist_ok=True)

        for sample_name in samples:     # sample_name = samples[0]

            output_path = os.path.join(output_class_dir, f"{sample_name}.npy")

            if not os.path.isfile(output_path):

                specie_dir = os.path.join(DATA_DIR, specie)
                img = load_multispectral_image(specie_dir, sample_name)

                if img is None:
                    print(f"Amostra descartada: {sample_name} - {specie} - {partition}")
                    x=1/0

                print(f'Saving image... {sample_name}   -   {partition}')
                np.save(output_path, img)

print("\nSplit Files: \033[96;92m Finished\033[0m\n")


#======================================================================
# Normalization

DIR_EXP = Path(DIR_EXP)
TRAIN_DIR = DIR_EXP / "Train"

#------------------------------------------------------------------------
# ETAPA 1: CALCULAR MÉDIA E DESVIO PADRÃO POR BANDA USANDO TRAIN

if "mean_bands" not in infos.keys() or "std_bands" not in infos.keys():

    print("Calculate mean and standard deviation per band using TRAIN....")

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

    infos["mean_bands"] = mean_bands.tolist()
    infos["std_bands"] = std_bands.tolist()

    with open(DIR_INFOS, "w", encoding="utf-8") as file:
        json.dump(infos, file, indent=4)
    
    print(f"Mean and STD: \033[96;92mCalculated\033[0m\n")

else:
    mean_bands = np.array(infos["mean_bands"])
    std_bands = np.array(infos["std_bands"])
    print(f"Mean and STD: \033[96;92mLoaded\033[0m\n")

#------------------------------------------------------------------------
# ETAPA 2: NORMALIZAR TRAIN, VAL E TEST

mean_bands = mean_bands.reshape(1, 1, 5)
std_bands = std_bands.reshape(1, 1, 5)

for split in partitions:    # split = "Train"
    
    input_dir = DIR_EXP / split
    output_dir = DIR_EXP / f"{split}_Norm"

    files = list(input_dir.rglob("*.npy"))

    # print(f"\nNormalizando {split}: {len(files)} imagens")

    for i, file_path in enumerate(files):   # i, file_path = 0, files[0]

        relative_path = file_path.relative_to(input_dir)

        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.isfile(output_path):

            img = np.load(file_path)  # (H, W, 5)
            img = img.astype(np.float32)

            img_norm = (img - mean_bands) / std_bands
            img_norm = img_norm.astype(np.float32)

            np.save(output_path, img_norm)

            if (i + 1) % 10 == 0:
                print(f"{split}: {i + 1}/{len(files)} imagens normalizadas  -- {split}  --")

print(f"Normalization: \033[96;92mDone\033[0m")

# ============================================================











