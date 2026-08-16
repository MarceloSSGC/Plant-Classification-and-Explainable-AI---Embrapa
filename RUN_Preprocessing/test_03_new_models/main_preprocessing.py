import os
import json

# Auxiliar
try:
    from .aux_align import align_main_function
    from .aux_seg import *
except ImportError:
        from aux_align import align_main_function
        from aux_seg import *


#======================================================================
#======================================================================

# src/preprocessing.py

def run_preprocessing(config):


    PC = config["PC"]

    #======================================================================
    # Alignment

    print(f"\n\033[100;01m\t     --- Start Alignment ---     \t\033[0m\n")

    #======================================================================
    # Base Data

    BASE_DATA_DIR = config["BASE_DATA_DIR"]

    if PC == "NITRO":
        BASE_DATA_DIR = f"/run/{BASE_DATA_DIR}"

    #======================================================================
    # Align Dataset

    ALIGN_DATASET_NAME = config["ALIGN_DATASET_NAME"]

    if PC == "NITRO":
        ALIGH_DATA_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned/{ALIGN_DATASET_NAME}"
    else:
        ALIGH_DATA_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Aligned/{ALIGN_DATASET_NAME}"


    #======================================================================
    # Alignment Method

    ALIGN_METHOD = config["ALIGN_METHOD"]

    #======================================================================
    #======================================================================
    # ALignment

    align_info_dir = os.path.join(ALIGH_DATA_DIR, "align_info.json")

    if os.path.isfile(align_info_dir):
        print(f"\nDataset Aligned ✅\n")

        with open(align_info_dir, "r", encoding="utf-8") as file:
            align_info = json.load(file)
    else:

        align_info = {
            "PC": PC,
            "ALIGN_METHOD": ALIGN_METHOD,
            "BASE_DATA_DIR": BASE_DATA_DIR,
            "ALIGN_DATASET_NAME": ALIGN_DATASET_NAME,
            "ALIGH_DATA_DIR": ALIGH_DATA_DIR
        }

        align_main_function(ALIGN_METHOD, BASE_DATA_DIR, ALIGH_DATA_DIR)

        with open(align_info_dir, "w", encoding="utf-8") as file:
            json.dump(align_info, file)

        print("\n💾 File saved\n")

    ####################################################################################
    ####################################################################################
    ####################################################################################
    # Segmentation

    print(f"\n\033[100;01m\t     --- Start Segmentation ---     \t\033[0m\n")

    #======================================================================
    # Segmentation Dataset

    SEG_DATASET_NAME = config["SEG_DATASET_NAME"]
    if PC == "NITRO":
        SEG_DATA_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Segmentation/{SEG_DATASET_NAME}"
    else:
        SEG_DATA_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/Segmentation/{SEG_DATASET_NAME}"

    #======================================================================
    # Segmentation Method

    SEGMENTATION_METHOD = config["SEGMENTATION_METHOD"]

    #======================================================================
    # Segmentation in action

    seg_info_dir = os.path.join(SEG_DATA_DIR, "seg_info.json")

    if os.path.isfile(seg_info_dir):
        print(f"\nDataset Segmented ✅\n")

        with open(seg_info_dir, "r", encoding="utf-8") as file:
            seg_info = json.load(file)
    else:

        seg_info = align_info.copy()

        seg_info_update = {
            "SEGMENTATION_METHOD": SEGMENTATION_METHOD,
            "SEG_DATASET_NAME": SEG_DATASET_NAME,
            "SEG_DATA_DIR": SEG_DATA_DIR
        }

        seg_info.update(seg_info_update)

        segmentation_main_function(SEGMENTATION_METHOD, ALIGH_DATA_DIR, SEG_DATA_DIR)

        with open(seg_info_dir, "w", encoding="utf-8") as file:
            json.dump(seg_info, file)

        print("\n💾 File saved\n")

    ####################################################################################
    ####################################################################################
    ####################################################################################
    # Split

    import random

    print(f"\n\033[100;01m\t     --- Start Split ---     \t\033[0m\n")

    #======================================================================

    INTERACTIVE = config["INTERACTIVE"]
    SEED = config["SEED"]

    print(f"\t  Interactive: \033[96;95m{INTERACTIVE} \033[0m\n")
    print(f"\t  SEED: \033[96;95m{SEED} \033[0m\n")

    #======================================================================
    #======================================================================
    # Reproducibility

    # Random
    random.seed(SEED)

    # NumPy
    np.random.seed(SEED)

    #======================================================================
    # Experiment Name and Directory

    EXPERIMENT_NAME = f"{ALIGN_METHOD}--{SEGMENTATION_METHOD}__SEED_{SEED}"
    experiment_type = "run_preprocessing"

    if PC == "NITRO":
        DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{experiment_type}/{EXPERIMENT_NAME}"
    else:
        DIR_EXP = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{experiment_type}/{EXPERIMENT_NAME}"

    if not os.path.isdir(DIR_EXP):
        os.makedirs(DIR_EXP)

    #======================================================================
    # Infos

    DIR_INFOS = DIR_EXP + "/infos.json"

    if not os.path.isfile(DIR_INFOS):
        infos = {
        "INTERACTIVE": INTERACTIVE,

        "ALIGN": align_info,
        "SEGMENTATION": seg_info,

        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "experiment_type": experiment_type,

        "DIR_EXP": DIR_EXP,
        "SEG_DATA_DIR": SEG_DATA_DIR,

        "SEED": SEED,

        "TRAIN_SIZE": 0.7,
        "VAL_SIZE": 0.15,
        "TEST_SIZE": 0.15,

        }

        with open(DIR_INFOS, "w", encoding="utf-8") as arquivo:
            json.dump(infos, arquivo, ensure_ascii=False, indent=4)

    else:
        with open(DIR_INFOS, "r") as file:
            infos = json.load(file)

    #======================================================================
    # DIVISÃO TRAIN / VAL / TEST - JSON

    split_file_dir = os.path.join(DIR_EXP, "split_files.json")

    species = sorted(os.listdir(SEG_DATA_DIR))
    if 'seg_info.json' in species:
        species.remove('seg_info.json')

    if not os.path.isfile(split_file_dir):
            
        split_file_names_train = {}
        split_file_names_val = {}
        split_file_names_test = {}

        for specie in species:     # specie = species[0]
            
            print(f"\nProcessando espécie: \033[96;93m{specie}\033[0m")

            specie_dir = os.path.join(SEG_DATA_DIR, specie)

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

    #======================================================================
    # DIRETÓRIOS

    partitions = ["Train", "Val", "Test"]

    for part in partitions:     # part = "Train"
        os.makedirs(os.path.join(DIR_EXP, part), exist_ok=True)

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

                    specie_dir = os.path.join(SEG_DATA_DIR, specie)
                    img = load_multispectral_image(specie_dir, sample_name)

                    if img is None:
                        print(f"Amostra descartada: {sample_name} - {specie} - {partition}")
                        x=1/0

                    print(f'Saving image... {sample_name}   -   {partition}')
                    np.save(output_path, img)

    print("\nSplit Files: \033[96;92m Finished\033[0m\n")

    #======================================================================
    # Normalization

    from pathlib import Path

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

    print(f"Normalization: \033[96;92mFinished\033[0m")

    ####################################################################################
    ####################################################################################
    ####################################################################################
    # Data Augmentation

    DO_AUGMENTANTION = config["DO_AUGMENTANTION"]

    if DO_AUGMENTANTION:

        # Auxiliar
        try:
            from .aux_augm import augmentation_compilation, plot_rgb

        except ImportError:
            from aux_augm import augmentation_compilation, plot_rgb


        print(f"\n\033[100;40m\t     --- Start Data Augmentation ---     \t\033[0m\n")

        #======================================================================
        # Directories

        AUG_EXPERIMENT_NAME = EXPERIMENT_NAME + "_AUG"

        if PC == "NITRO":
            AUG_DIR_EXP = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{experiment_type}/{AUG_EXPERIMENT_NAME}"
        else:
            AUG_DIR_EXP = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/{experiment_type}/{AUG_EXPERIMENT_NAME}"

        os.makedirs(AUG_DIR_EXP, exist_ok=True)

        #======================================================================
        # Augmentation Infos

        AUG_DIR_INFOS = AUG_DIR_EXP + "/aug_infos.json"

        if not os.path.isfile(AUG_DIR_INFOS):

            aug_infos = infos.copy()

            AUGMENT = dict()

            aug_rules = {"<11": "x5",
                        "<18": "x4",
                        "<28": "x3",
                        "<35": "x2"
                        }

            AUGMENT["aug_rules"] = aug_rules

            aug_params = [
                [("rotation", 180)],
                [("rotation", 90), ("scale", 1.1)],
                [("rotation", 270), ("scale", 1.1)], 
                [("rotation", 30), ("scale", 1.1)], 
                [("cutout", 39), ("cutout", 139), ("cutout", 10)]
            ]

            AUGMENT["aug_params"] = aug_params

            aug_infos["AUGMENT"] = AUGMENT
            
            with open(AUG_DIR_INFOS, "w", encoding="utf-8") as arquivo:
                json.dump(aug_infos, arquivo, ensure_ascii=False, indent=4)

        else:
            with open(AUG_DIR_INFOS, "r") as file:
                aug_infos = json.load(file)

        aug_params = aug_infos["AUGMENT"]['aug_params']

        #======================================================================
        # Augmentation Factor and Names

        def augmentation_factor(n: int)->int:
            if n < 0:
                raise ValueError("n < 0")

            if n < 11:
                return 5
            elif n < 18:
                return 4
            elif n < 28:
                return 3
            elif n < 35:
                return 2
            else:
                return 1

        #----------------------------------------------------------------------

        aug_params_names = []

        for seq in aug_params:  # seq = aug_params[1]
            step_name = "__"
            for step in seq:
                # print(step)
                step_name += "_".join([str(y) for y in step])
                step_name += "_"
            step_name = step_name[:-1]
            step_name += ".npy"
            aug_params_names.append(step_name)

        #======================================================================
        # Augmentaion in Action

        TRAIN_AUG_DIR_EXP = os.path.join(AUG_DIR_EXP, "Train")
        os.makedirs(TRAIN_AUG_DIR_EXP, exist_ok=True)

        especies = sorted(os.listdir(DIR_EXP / "Train"))

        dist_especie_period = dict()

        for especie in especies:    # especie = especies[0]

            old_especie_dir = os.path.join(DIR_EXP, "Train", especie)
            new_especie_dir = os.path.join(TRAIN_AUG_DIR_EXP, especie)
            os.makedirs(new_especie_dir, exist_ok=True)

            files = sorted(os.listdir(old_especie_dir))
            n_sample = len(files)
            dist_especie_period[especie] = n_sample

            aug_factor = augmentation_factor(n_sample)

            # print(f"especie: {especie}   --   aug_factor: {aug_factor}   --   n_sample:{n_sample}")

            for ith_file in files:  # ith_file = files[1]

                ith_file_dir = os.path.join(old_especie_dir, ith_file)

                for i in range(aug_factor+1):  # i=1

                    if i == 0:
                        ith_file_mod = ith_file
                    else:
                        ith_file_mod = ith_file.replace(".npy", aug_params_names[i-1])

                    ith_new_file_dir = os.path.join(new_especie_dir, ith_file_mod)

                    if not os.path.isfile(ith_new_file_dir):
                        # print("Carregando")
                        img = np.load(ith_file_dir)    # (H, W, 5)
                        img = img.astype(np.float32)

                        if i == 0:
                            np.save(ith_new_file_dir, img)
                        else:
                            aug_sequence = aug_params[i-1]
                            # plot_rgb(img)
                            for step in aug_sequence:   # step = aug_sequence[0]
                                img = augmentation_compilation(img, step[0], step[1])
                                # plot_rgb(img)

                            np.save(ith_new_file_dir, img)

                # # Check
                # new_files = sorted(os.listdir(new_especie_dir))
                # for file in new_files:  # file = new_files[0]
                #     print(f'\n file: {file}')
                #     temp_dir = os.path.join(new_especie_dir, file)
                #     img = np.load(temp_dir)    # (H, W, 5)
                #     img = img.astype(np.float32)
                #     plot_rgb(img)

        #======================================================================
        # Normalization

        AUG_DIR_EXP = Path(AUG_DIR_EXP)
        AUG_TRAIN_DIR = AUG_DIR_EXP / "Train"

        #------------------------------------------------------------------------
        # ETAPA 1: CALCULAR MÉDIA E DESVIO PADRÃO POR BANDA USANDO TRAIN

        if "mean_bands" not in aug_infos["AUGMENT"].keys() or "std_bands" not in aug_infos["AUGMENT"].keys():

            print("Calculate mean and standard deviation per band using TRAIN....")

            sum_bands = np.zeros(5, dtype=np.float64)
            sum_sq_bands = np.zeros(5, dtype=np.float64)
            count_pixels = np.zeros(5, dtype=np.int64)

            train_files = list(AUG_TRAIN_DIR.rglob("*.npy"))

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

            aug_infos["AUGMENT"]["mean_bands"] = mean_bands.tolist()
            aug_infos["AUGMENT"]["std_bands"] = std_bands.tolist()

            with open(AUG_DIR_INFOS, "w", encoding="utf-8") as file:
                json.dump(aug_infos, file, indent=4)
            
            print(f"Mean and STD: \033[96;92mCalculated\033[0m\n")

        else:
            mean_bands = np.array(aug_infos["AUGMENT"]["mean_bands"])
            std_bands = np.array(aug_infos["AUGMENT"]["std_bands"])
            print(f"Mean and STD: \033[96;92mLoaded\033[0m\n")

        #------------------------------------------------------------------------
        # ETAPA 2: NORMALIZAR TRAIN, VAL E TEST

        mean_bands = mean_bands.reshape(1, 1, 5)
        std_bands = std_bands.reshape(1, 1, 5)

        for split in ["Train", "Val", "Test"]:    # split = "Train"

            if split == "Train":
                input_dir = AUG_DIR_EXP / split
            else:
                input_dir = DIR_EXP / split

            output_dir = AUG_DIR_EXP / f"{split}_Norm"


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









