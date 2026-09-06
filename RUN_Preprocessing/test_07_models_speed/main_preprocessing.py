import os
import json
import random

# Auxiliar
try:
    from aux_align import align_main_function
    from aux_seg import *
    from aux_multiview_functions import *
    from aux_augm import augmentation_compilation

except ImportError:
        from RUN_Preprocessing.test_07_models_speed.aux_align import align_main_function
        from RUN_Preprocessing.test_07_models_speed.aux_seg import *
        from RUN_Preprocessing.test_07_models_speed.aux_multiview_functions import *
        from RUN_Preprocessing.test_07_models_speed.aux_augm import augmentation_compilation


#======================================================================
#======================================================================

# import yaml

# def load_config(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)
    
# yaml_test_name = "MTV_model_01.yaml"
# #Helios
# # path = f"/home/marcelo/Documents/python_projects/USP/Planta_Daninha_Embrapa/Plant-Classification-and-Explainable-AI---Embrapa/config/{yaml_test_name}"

# # Nitro
# path = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_test_name}"
# config = load_config(path)

# for x in config:
#     print(f"{x}: \033[96;96m{config[x]}\033[0m")


# raw_data 
# → alinhamento 
# → segmentação 
# → multiview
# → split 
# → normalização

#======================================================================
#======================================================================

# src/preprocessing.py

def run_preprocessing(config):


    PC = config["PC"]
    INTERACTIVE = config["INTERACTIVE"]
    SEED = config["SEED"]
    MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]

    print(f"\t  PC: \033[96;95m{PC} \033[0m\n")
    print(f"\t  Interactive: \033[96;95m{INTERACTIVE} \033[0m\n")
    print(f"\t  SEED: \033[96;95m{SEED} \033[0m\n")
    print(f"\t  MULTIVIEW_DATA_NICKNAME: \033[96;95m{MULTIVIEW_DATA_NICKNAME} \033[0m\n")

    if PC not in ["NITRO", "HELIOS", "DANTE"]:
        raise ValueError("PC not indentified")

    #======================================================================
    # PC Directory

    if PC == "NITRO":
        PC_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
    elif PC == "HELIOS":
        PC_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
    else:
        PC_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos"

    #======================================================================
    # Alignment

    print(f"\n\033[100;01m\t     --- Start Alignment ---     \t\033[0m\n")

    #======================================================================
    # Base Data

    BASE_DATA_DIR = f"{PC_DIR}/Datasets/PlantaDaninha_BoaVista"

    #======================================================================
    # Align Dataset

    ALIGN_DATA_NAME = config["ALIGN_DATA_NAME"]

    ALIGH_DATA_DIR = f"{PC_DIR}/Datasets/Aligned/{ALIGN_DATA_NAME}"

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
            "INTERACTIVE": INTERACTIVE,
            "ALIGN": {
            "ALIGN_METHOD": ALIGN_METHOD,
            "ALIGN_DATA_NAME": ALIGN_DATA_NAME,
            "BASE_DATA_DIR": BASE_DATA_DIR,
            "ALIGH_DATA_DIR": ALIGH_DATA_DIR}
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

    SEG_DATA_NICKNAME = config["SEG_DATA_NICKNAME"]
    SEG_DATA_NAME = f"{ALIGN_DATA_NAME}__{SEG_DATA_NICKNAME}"

    SEG_DATA_DIR = f"{PC_DIR}/Datasets/Segmentation/{SEG_DATA_NAME}"

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

        seg_info["SEGMENTATION"] = {
            "SEG_DATA_NICKNAME": SEG_DATA_NICKNAME,
            "SEG_DATA_NAME": SEG_DATA_NAME,
            "SEG_DATA_DIR": SEG_DATA_DIR,
            "SEGMENTATION_METHOD": SEGMENTATION_METHOD,
        }

        segmentation_main_function(SEGMENTATION_METHOD, ALIGH_DATA_DIR, SEG_DATA_DIR)

        with open(seg_info_dir, "w", encoding="utf-8") as file:
            json.dump(seg_info, file)

        print("\n💾 File saved\n")

    ####################################################################################
    ####################################################################################
    ####################################################################################
    # Multiview


    print(f"\n\033[100;01m\t     --- Start Multiview ---     \t\033[0m\n")

    #======================================================================
    # Organizing Views 

    views_dict = config["VIEWS"]

    trans_list = []
    for key in views_dict:  # key = list(views_dict.keys())[0]
        for x in views_dict[key]:   # x = list(views_dict[key])[0]
            # print(f"x: {x} - key: {key}")
            ih_view = {"type_view": key}
            ih_view.update(views_dict[key][x])
            trans_list.append(ih_view)


    view_names = views_dict_to_string(views_dict)

    #======================================================================
    # Multiview Dataset

    MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]

    MULTIVIEW_DATA_NAME = f"{SEG_DATA_NAME}__{MULTIVIEW_DATA_NICKNAME}"
    config["MULTIVIEW_DATA_NAME"] = MULTIVIEW_DATA_NAME

    MULTIVIEW_DATA_TYPE = config["MULTIVIEW_DATA_TYPE"]

    MTV_DATA_DIR = f"{PC_DIR}/Datasets/Multiview/{MULTIVIEW_DATA_TYPE}/{MULTIVIEW_DATA_NAME}"

    #======================================================================
    # Multiview in action

    mtv_info_dir = os.path.join(MTV_DATA_DIR, "mtv_info.json")

    if os.path.isfile(mtv_info_dir):
        print(f"\nDataset Multiview ✅\n")

        with open(mtv_info_dir, "r", encoding="utf-8") as file:
            mtv_info = json.load(file)
    else:

        n_bands = multiview_main_function(trans_list, SEG_DATA_DIR, MTV_DATA_DIR)
        # n_bands = multiview_main_function_PARALLEL(trans_list, SEG_DATA_DIR, MTV_DATA_DIR)

        mtv_info = seg_info.copy()
        mtv_info["MULTIVIEW_DATA_NICKNAME"] = MULTIVIEW_DATA_NICKNAME
        mtv_info["MULTIVIEW"] = {
            "MULTIVIEW_DATA_NICKNAME": MULTIVIEW_DATA_NICKNAME,
            "MULTIVIEW_DATA_NAME": MULTIVIEW_DATA_NAME,
            "MULTIVIEW_DATA_TYPE": MULTIVIEW_DATA_TYPE,
            "MTV_DATA_DIR": MTV_DATA_DIR,
            "TRANSFORMATIONS": trans_list,
            "N_BANDS": n_bands
            }


        with open(mtv_info_dir, "w", encoding="utf-8") as file:
            json.dump(mtv_info, file)

        print("\n💾 File saved\n")


    ####################################################################################
    ####################################################################################
    ####################################################################################
    # Split

    print(f"\n\033[100;01m\t     --- Start Split ---     \t\033[0m\n")

    #======================================================================
    #======================================================================
    # Reproducibility

    # Random
    random.seed(SEED)

    # NumPy
    np.random.seed(SEED)

    #======================================================================
    # Experiment Name and Directory

    TRAIN_SIZE = config["TRAIN_SIZE"]
    VAL_SIZE = config["VAL_SIZE"]

    SPLIT_DATA_NAME = config["MULTIVIEW_DATA_NAME"] + f"__SEED_{SEED}__T_{TRAIN_SIZE}_V_{VAL_SIZE}"
    SPLIT_DATE_TYPE = config["MULTIVIEW_DATA_TYPE"]

    config["SPLIT_DATA_NAME"] = SPLIT_DATA_NAME
    config["SPLIT_DATE_TYPE"] = SPLIT_DATE_TYPE

    SPLIT_DIR = f"{PC_DIR}/Datasets/Split/{SPLIT_DATE_TYPE}/{SPLIT_DATA_NAME}"

    if not os.path.isdir(SPLIT_DIR):
        os.makedirs(SPLIT_DIR)

    #======================================================================
    # Infos

    split_info_dir = os.path.join(SPLIT_DIR, "split_info.json")

    if os.path.isfile(split_info_dir):
        with open(split_info_dir, "r") as file:
            split_info = json.load(file)
    
    else:
        split_info = mtv_info.copy()
        split_info["SPLIT"] = {
            "SPLIT_DATA_NAME": SPLIT_DATA_NAME,
            "SPLIT_DATE_TYPE": SPLIT_DATE_TYPE,
            "TRAIN_SIZE": TRAIN_SIZE,
            "VAL_SIZE": VAL_SIZE,
        }

        with open(split_info_dir, "w", encoding="utf-8") as arquivo:
            json.dump(split_info, arquivo, ensure_ascii=False, indent=4)

    #======================================================================
    # DIVISÃO TRAIN / VAL / TEST - JSON

    split_file_dir = os.path.join(SPLIT_DIR, "split_files.json")

    species = sorted(os.listdir(MTV_DATA_DIR))
    if 'mtv_info.json' in species:
        species.remove('mtv_info.json')

    if not os.path.isfile(split_file_dir):
            
        split_file_names_train = {}
        split_file_names_val = {}
        split_file_names_test = {}

        for specie in species:     # specie = species[0]
            
            print(f"\nProcessando espécie: \033[96;93m{specie}\033[0m")

            specie_dir = os.path.join(MTV_DATA_DIR, specie)

            sample_names = sorted(os.listdir(specie_dir))

            random.shuffle(sample_names)

            n = len(sample_names)

            n_train = int(TRAIN_SIZE * n)
            n_val = int(VAL_SIZE * n)
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
        # print("Split Names: \033[96;92mLoaded\033[0m")

    #======================================================================
    # DIRETÓRIOS

    partitions = ["Train", "Val", "Test"]

    for part in partitions:     # part = "Train"
        os.makedirs(os.path.join(SPLIT_DIR, part), exist_ok=True)

    #======================================================================
    # Split 

    for specie in species:     # specie = species[0]
        
        # print(f"\nProcessando espécie: \033[96;93m{specie}\033[0m")

        for partition in partitions:   # partition = "Train"

            samples = split_file_names[partition][specie]

            output_class_dir = os.path.join(SPLIT_DIR, partition, specie)
            os.makedirs(output_class_dir, exist_ok=True)

            for sample_name in samples:     # sample_name = samples[0]

                output_path = os.path.join(output_class_dir, sample_name)

                if not os.path.isfile(output_path):

                    img_dir = os.path.join(MTV_DATA_DIR, specie, sample_name)
                    img = np.load(img_dir)

                    print(f'Saving image... {sample_name}  -  {partition} - {specie}')
                    np.save(output_path, img)

    # print("\nSplit Files: \033[96;92m Finished\033[0m\n")

    #======================================================================
    # Normalization

    from pathlib import Path

    SPLIT_DIR = Path(SPLIT_DIR)
    TRAIN_DIR = SPLIT_DIR / "Train"

    #------------------------------------------------------------------------
    # ETAPA 1: CALCULAR MÉDIA E DESVIO PADRÃO POR BANDA USANDO TRAIN

    N_BANDS = split_info["MULTIVIEW"]["N_BANDS"]

    if (
        "mean_bands" not in split_info["SPLIT"]
        or "std_bands" not in split_info["SPLIT"]
    ):

        print("Calculate mean and standard deviation per band using TRAIN....")

        # Acumuladores em float64 para maior estabilidade numérica
        sum_bands = np.zeros(N_BANDS, dtype=np.float64)
        sum_sq_bands = np.zeros(N_BANDS, dtype=np.float64)

        # Contagem é inteira
        count_pixels = np.zeros(N_BANDS, dtype=np.int64)

        train_files = list(TRAIN_DIR.rglob("*.npy"))

        if len(train_files) == 0:
            raise RuntimeError(f"Nenhum arquivo .npy encontrado em {TRAIN_DIR}")

        print(f"Número de imagens em Train: {len(train_files)}")

        for i, file_path in enumerate(train_files):

            # --------------------------------------------------------------
            # Carregar imagem
            # --------------------------------------------------------------

            img = np.load(file_path)  # esperado: (H, W, N_BANDS)

            # Validação do formato
            if img.ndim != 3:
                raise ValueError(
                    f"Imagem com dimensionalidade inválida:\n"
                    f"Arquivo: {file_path}\n"
                    f"Shape encontrado: {img.shape}"
                )

            if img.shape[-1] != N_BANDS:
                raise ValueError(
                    f"Número de bandas inconsistente:\n"
                    f"Arquivo: {file_path}\n"
                    f"Esperado: {N_BANDS}\n"
                    f"Encontrado: {img.shape[-1]}"
                )

            # Dados de imagem padronizados em float32
            img = img.astype(np.float32, copy=False)

            # --------------------------------------------------------------
            # Verificar NaN / Inf ANTES do cálculo das estatísticas
            # --------------------------------------------------------------

            if not np.isfinite(img).all():

                n_nan = np.isnan(img).sum()
                n_posinf = np.isposinf(img).sum()
                n_neginf = np.isneginf(img).sum()

                raise ValueError(
                    f"Valores inválidos encontrados antes da normalização:\n"
                    f"Arquivo: {file_path}\n"
                    f"NaN: {n_nan}\n"
                    f"+Inf: {n_posinf}\n"
                    f"-Inf: {n_neginf}"
                )

            # --------------------------------------------------------------
            # Estatísticas
            #
            # A imagem permanece float32, mas as reduções são feitas
            # explicitamente em float64.
            # --------------------------------------------------------------

            sum_bands += img.sum(
                axis=(0, 1),
                dtype=np.float64
            )

            sum_sq_bands += np.square(
                img,
                dtype=np.float64
            ).sum(
                axis=(0, 1),
                dtype=np.float64
            )

            # Número de pixels de cada banda
            h, w, c = img.shape
            count_pixels += h * w

            if (i + 1) % 10 == 0:
                print(
                    f"Processadas {i + 1}/{len(train_files)} imagens"
                )

        # --------------------------------------------------------------
        # Média
        # --------------------------------------------------------------

        mean_bands = sum_bands / count_pixels

        # --------------------------------------------------------------
        # Variância
        #
        # Var(X) = E[X²] - E[X]²
        # --------------------------------------------------------------

        var_bands = (
            sum_sq_bands / count_pixels
        ) - (mean_bands ** 2)

        # Proteção contra pequenos valores negativos causados apenas
        # por erro numérico de ponto flutuante.
        var_bands = np.maximum(var_bands, 0.0)

        std_bands = np.sqrt(var_bands)

        # --------------------------------------------------------------
        # Validação das estatísticas calculadas
        # --------------------------------------------------------------

        if not np.all(np.isfinite(mean_bands)):
            raise ValueError(
                f"Média contém NaN/Inf:\n{mean_bands}"
            )

        if not np.all(np.isfinite(std_bands)):
            raise ValueError(
                f"Desvio padrão contém NaN/Inf:\n{std_bands}"
            )

        # Uma banda constante ou quase constante não deve ser
        # normalizada por divisão sem antes investigar a causa.
        if np.any(std_bands < 1e-8):

            problematic_bands = np.where(std_bands < 1e-8)[0]

            raise ValueError(
                f"Desvio padrão zero ou muito pequeno.\n"
                f"Bandas problemáticas: {problematic_bands.tolist()}\n"
                f"STD: {std_bands}"
            )

        # --------------------------------------------------------------
        # Resultado
        # --------------------------------------------------------------

        print("\nMédia por banda:")
        print(mean_bands)

        print("\nDesvio padrão por banda:")
        print(std_bands)

        print("\nVariância por banda:")
        print(var_bands)

        # Salvar no JSON
        split_info["SPLIT"]["mean_bands"] = mean_bands.tolist()
        split_info["SPLIT"]["std_bands"] = std_bands.tolist()

        with open(split_info_dir, "w", encoding="utf-8") as file:
            json.dump(split_info, file, indent=4)

        print(
            f"\nETAPA 1: \033[96;92m"
            f"Mean and STD Calculated"
            f"\033[0m\n"
        )

    else:

        # Carregar explicitamente como float64 porque essas estatísticas
        # serão usadas em operações numéricas de normalização.
        mean_bands = np.asarray(
            split_info["SPLIT"]["mean_bands"],
            dtype=np.float64
        )

        std_bands = np.asarray(
            split_info["SPLIT"]["std_bands"],
            dtype=np.float64
        )

        # Também validar estatísticas previamente salvas
        if not np.all(np.isfinite(mean_bands)):
            raise ValueError(
                f"mean_bands carregado contém NaN/Inf:\n{mean_bands}"
            )

        if not np.all(np.isfinite(std_bands)):
            raise ValueError(
                f"std_bands carregado contém NaN/Inf:\n{std_bands}"
            )

        if np.any(std_bands < 1e-8):
            raise ValueError(
                f"std_bands carregado contém valores zero "
                f"ou muito pequenos:\n{std_bands}"
            )

    #------------------------------------------------------------------------
    # ETAPA 2: NORMALIZAR TRAIN, VAL E TEST

    # Estatísticas em float64
    mean_bands = np.asarray(mean_bands, dtype=np.float64).reshape(
        1, 1, N_BANDS
    )

    std_bands = np.asarray(std_bands, dtype=np.float64).reshape(
        1, 1, N_BANDS
    )

    #------------------------------------------------------------------------
    # Validar estatísticas antes de iniciar a normalização

    if not np.all(np.isfinite(mean_bands)):
        raise ValueError(
            f"mean_bands contém NaN/Inf:\n{mean_bands}"
        )

    if not np.all(np.isfinite(std_bands)):
        raise ValueError(
            f"std_bands contém NaN/Inf:\n{std_bands}"
        )

    if np.any(std_bands < 1e-8):
        problematic_bands = np.where(
            std_bands.reshape(-1) < 1e-8
        )[0]

        raise ValueError(
            f"Desvio padrão zero ou muito pequeno.\n"
            f"Bandas problemáticas: {problematic_bands.tolist()}\n"
            f"STD: {std_bands.reshape(-1)}"
        )


    #------------------------------------------------------------------------
    # Normalizar cada partição

    for split in partitions:    # "Train", "Val", "Test"

        input_dir = SPLIT_DIR / split
        output_dir = SPLIT_DIR / f"{split}_Norm"

        files = list(input_dir.rglob("*.npy"))

        if len(files) == 0:
            raise RuntimeError(
                f"Nenhum arquivo .npy encontrado em {input_dir}"
            )

        print(f"\nNormalizando {split}: {len(files)} imagens")

        for i, file_path in enumerate(files):

            relative_path = file_path.relative_to(input_dir)

            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not os.path.isfile(output_path):

                #--------------------------------------------------------------
                # Carregar imagem
                #--------------------------------------------------------------

                img = np.load(file_path)

                # Esperado: (H, W, N_BANDS)
                if img.ndim != 3:
                    raise ValueError(
                        f"Imagem com dimensionalidade inválida:\n"
                        f"Arquivo: {file_path}\n"
                        f"Shape encontrado: {img.shape}"
                    )

                if img.shape[-1] != N_BANDS:
                    raise ValueError(
                        f"Número de bandas inconsistente:\n"
                        f"Arquivo: {file_path}\n"
                        f"Esperado: {N_BANDS}\n"
                        f"Encontrado: {img.shape[-1]}"
                    )

                #--------------------------------------------------------------
                # Padronizar dados da imagem em float32
                #--------------------------------------------------------------

                img = img.astype(np.float32, copy=False)

                #--------------------------------------------------------------
                # Verificar entrada ANTES da normalização
                #--------------------------------------------------------------

                if not np.isfinite(img).all():

                    n_nan = np.isnan(img).sum()
                    n_posinf = np.isposinf(img).sum()
                    n_neginf = np.isneginf(img).sum()

                    raise ValueError(
                        f"Valores inválidos ANTES da normalização:\n"
                        f"Arquivo: {file_path}\n"
                        f"NaN: {n_nan}\n"
                        f"+Inf: {n_posinf}\n"
                        f"-Inf: {n_neginf}"
                    )

                #--------------------------------------------------------------
                # Normalização
                #
                # img         -> float32
                # mean/std    -> float64
                # img_norm    -> temporariamente float64
                #--------------------------------------------------------------

                img_norm = (img - mean_bands) / std_bands

                #--------------------------------------------------------------
                # Verificar resultado da normalização
                #--------------------------------------------------------------

                if not np.isfinite(img_norm).all():

                    n_nan = np.isnan(img_norm).sum()
                    n_posinf = np.isposinf(img_norm).sum()
                    n_neginf = np.isneginf(img_norm).sum()

                    raise ValueError(
                        f"Valores inválidos CRIADOS pela normalização:\n"
                        f"Arquivo: {file_path}\n"
                        f"NaN: {n_nan}\n"
                        f"+Inf: {n_posinf}\n"
                        f"-Inf: {n_neginf}\n"
                        f"Mean: {mean_bands.reshape(-1)}\n"
                        f"STD: {std_bands.reshape(-1)}"
                    )

                #--------------------------------------------------------------
                # Arquivo final sempre em float32
                #--------------------------------------------------------------

                img_norm = img_norm.astype(np.float32)

                # Verificação final após conversão para float32.
                # Também detecta eventual overflow na conversão.
                if not np.isfinite(img_norm).all():

                    raise ValueError(
                        f"NaN/Inf após conversão para float32:\n"
                        f"Arquivo: {file_path}\n"
                        f"Min: {np.nanmin(img_norm)}\n"
                        f"Max: {np.nanmax(img_norm)}"
                    )

                #--------------------------------------------------------------
                # Salvar
                #--------------------------------------------------------------

                np.save(output_path, img_norm)

            if (i + 1) % 10 == 0:
                print(
                    f"{split}: {i + 1}/{len(files)} imagens processadas"
                )


    print("\nSplit and Normalization ✅\n")

    # print(f"ETAPA 2: \033[96;92m Normalization Finished\033[0m")

    ####################################################################################
    ####################################################################################
    ####################################################################################
    # Data Augmentation


    print(f"\n\033[100;01m\t     --- Start Data Augmentation ---     \t\033[0m\n")

    #======================================================================
    # Directories

    AUG_DATE_NAME = SPLIT_DATA_NAME + "__AUG"
    AUG_DATE_TYPE = SPLIT_DATE_TYPE + "__AUG"

    config["AUG_DATE_NAME"] = AUG_DATE_NAME
    config["AUG_DATE_TYPE"] = AUG_DATE_TYPE

    AUG_DIR_EXP = f"{PC_DIR}/Datasets/Augmentation/{AUG_DATE_TYPE}/{AUG_DATE_NAME}"

    os.makedirs(AUG_DIR_EXP, exist_ok=True)

    #======================================================================
    # Augmentation Infos

    AUG_DIR_INFOS = AUG_DIR_EXP + "/aug_infos.json"

    if not os.path.isfile(AUG_DIR_INFOS):

        aug_infos = split_info.copy()

        aug_infos["AUGMENTATION"] = {
            "AUG_EXPERIMENT_NAME": AUG_DATE_NAME,
            "AUG_DATE_TYPE": AUG_DATE_TYPE,
            "AUG_DIR_EXP": AUG_DIR_EXP
        }

        aug_rules = {"<11": "x5",
                    "<18": "x4",
                    "<28": "x3",
                    "<35": "x2"
                    }

        aug_infos["AUGMENTATION"]["aug_rules"] = aug_rules

        aug_params = [
            [("rotation", 180)],
            [("rotation", 90), ("scale", 1.1)],
            [("rotation", 270), ("scale", 1.1)], 
            [("rotation", 30), ("scale", 1.1)], 
            [("cutout", 39), ("cutout", 139), ("cutout", 10)]
        ]

        aug_infos["AUGMENTATION"]["aug_params"] = aug_params

            
        with open(AUG_DIR_INFOS, "w", encoding="utf-8") as arquivo:
            json.dump(aug_infos, arquivo, ensure_ascii=False, indent=4)

    else:
        with open(AUG_DIR_INFOS, "r") as file:
            aug_infos = json.load(file)

    aug_params = aug_infos["AUGMENTATION"]['aug_params']

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

    especies = sorted(os.listdir(SPLIT_DIR / "Train"))

    dist_especie_period = dict()

    for especie in especies:    # especie = especies[0]

        old_especie_dir = os.path.join(SPLIT_DIR, "Train", especie)
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
            #     plot_rgb(img[:, :, [8, 6, 5]])
            #     plot_band(img[:, :, 0])

    #======================================================================
    # Normalization

    AUG_DIR_EXP = Path(AUG_DIR_EXP)
    AUG_TRAIN_DIR = AUG_DIR_EXP / "Train"


    #------------------------------------------------------------------------
    # ETAPA 1: CALCULAR MÉDIA E DESVIO PADRÃO POR BANDA USANDO TRAIN

    if (
        "mean_bands" not in aug_infos["AUGMENTATION"]
        or "std_bands" not in aug_infos["AUGMENTATION"]
    ):

        print("Calculate mean and standard deviation per band using TRAIN....")

        # Acumuladores em float64 para estabilidade numérica
        sum_bands = np.zeros(N_BANDS, dtype=np.float64)
        sum_sq_bands = np.zeros(N_BANDS, dtype=np.float64)

        # Contagem de pixels é inteira
        count_pixels = np.zeros(N_BANDS, dtype=np.int64)

        train_files = list(AUG_TRAIN_DIR.rglob("*.npy"))

        if len(train_files) == 0:
            raise RuntimeError(
                f"Nenhum arquivo .npy encontrado em {AUG_TRAIN_DIR}"
            )

        print(f"Número de imagens em Train: {len(train_files)}")

        for i, file_path in enumerate(train_files):

            #--------------------------------------------------------------
            # Carregar imagem
            #--------------------------------------------------------------

            img = np.load(file_path)  # esperado: (H, W, N_BANDS)

            # Validar dimensionalidade
            if img.ndim != 3:
                raise ValueError(
                    f"Imagem com dimensionalidade inválida:\n"
                    f"Arquivo: {file_path}\n"
                    f"Shape encontrado: {img.shape}"
                )

            # Validar número de bandas
            if img.shape[-1] != N_BANDS:
                raise ValueError(
                    f"Número de bandas inconsistente:\n"
                    f"Arquivo: {file_path}\n"
                    f"Esperado: {N_BANDS}\n"
                    f"Encontrado: {img.shape[-1]}"
                )

            #--------------------------------------------------------------
            # Padronizar imagem em float32
            #--------------------------------------------------------------

            img = img.astype(np.float32, copy=False)

            #--------------------------------------------------------------
            # Verificar NaN / Inf ANTES do cálculo das estatísticas
            #--------------------------------------------------------------

            if not np.isfinite(img).all():

                n_nan = np.isnan(img).sum()
                n_posinf = np.isposinf(img).sum()
                n_neginf = np.isneginf(img).sum()

                raise ValueError(
                    f"Valores inválidos encontrados no dataset aumentado:\n"
                    f"Arquivo: {file_path}\n"
                    f"NaN: {n_nan}\n"
                    f"+Inf: {n_posinf}\n"
                    f"-Inf: {n_neginf}"
                )

            #--------------------------------------------------------------
            # Estatísticas
            #
            # Imagem: float32
            # Acumulação/redução: float64
            #--------------------------------------------------------------

            sum_bands += img.sum(
                axis=(0, 1),
                dtype=np.float64
            )

            sum_sq_bands += np.square(
                img,
                dtype=np.float64
            ).sum(
                axis=(0, 1),
                dtype=np.float64
            )

            # Número de pixels por banda
            h, w, _ = img.shape
            count_pixels += h * w

            if (i + 1) % 10 == 0:
                print(
                    f"Processadas {i + 1}/{len(train_files)} imagens"
                )

        #--------------------------------------------------------------
        # Média por banda
        #--------------------------------------------------------------

        mean_bands = sum_bands / count_pixels

        #--------------------------------------------------------------
        # Variância e desvio padrão
        #
        # Var(X) = E[X²] - E[X]²
        #--------------------------------------------------------------

        var_bands = (
            sum_sq_bands / count_pixels
        ) - (mean_bands ** 2)

        # Evitar pequenos valores negativos causados exclusivamente
        # por erro numérico de ponto flutuante
        var_bands = np.maximum(var_bands, 0.0)

        std_bands = np.sqrt(var_bands)

        #--------------------------------------------------------------
        # Validar estatísticas calculadas
        #--------------------------------------------------------------

        if not np.all(np.isfinite(mean_bands)):
            raise ValueError(
                f"Média contém NaN/Inf:\n{mean_bands}"
            )

        if not np.all(np.isfinite(std_bands)):
            raise ValueError(
                f"Desvio padrão contém NaN/Inf:\n{std_bands}"
            )

        if np.any(std_bands < 1e-8):

            problematic_bands = np.where(
                std_bands < 1e-8
            )[0]

            raise ValueError(
                f"Desvio padrão zero ou muito pequeno.\n"
                f"Bandas problemáticas: {problematic_bands.tolist()}\n"
                f"STD: {std_bands}"
            )

        #--------------------------------------------------------------
        # Resultados
        #--------------------------------------------------------------

        print("\nMédia por banda:")
        print(mean_bands)

        print("\nDesvio padrão por banda:")
        print(std_bands)

        print("\nVariância por banda:")
        print(var_bands)

        #--------------------------------------------------------------
        # Salvar estatísticas
        #--------------------------------------------------------------

        aug_infos["AUGMENTATION"]["mean_bands"] = mean_bands.tolist()
        aug_infos["AUGMENTATION"]["std_bands"] = std_bands.tolist()

        with open(AUG_DIR_INFOS, "w", encoding="utf-8") as file:
            json.dump(aug_infos, file, indent=4)

        print(
            f"\nETAPA 1: \033[96;92m"
            f"Mean and STD Calculated"
            f"\033[0m\n"
        )

    else:

        # Estatísticas carregadas como float64
        mean_bands = np.asarray(
            aug_infos["AUGMENTATION"]["mean_bands"],
            dtype=np.float64
        )

        std_bands = np.asarray(
            aug_infos["AUGMENTATION"]["std_bands"],
            dtype=np.float64
        )

        #--------------------------------------------------------------
        # Validar estatísticas previamente armazenadas
        #--------------------------------------------------------------

        if not np.all(np.isfinite(mean_bands)):
            raise ValueError(
                f"mean_bands carregado contém NaN/Inf:\n{mean_bands}"
            )

        if not np.all(np.isfinite(std_bands)):
            raise ValueError(
                f"std_bands carregado contém NaN/Inf:\n{std_bands}"
            )

        if np.any(std_bands < 1e-8):
            raise ValueError(
                f"std_bands carregado contém valores zero "
                f"ou muito pequenos:\n{std_bands}"
            )

        
    #------------------------------------------------------------------------
    # ETAPA 2: NORMALIZAR TRAIN, VAL E TEST

    # Estatísticas em float64
    mean_bands = np.asarray(
        mean_bands,
        dtype=np.float64
    ).reshape(1, 1, N_BANDS)

    std_bands = np.asarray(
        std_bands,
        dtype=np.float64
    ).reshape(1, 1, N_BANDS)


    #------------------------------------------------------------------------
    # Validar estatísticas antes de iniciar a normalização

    if not np.all(np.isfinite(mean_bands)):
        raise ValueError(
            f"mean_bands contém NaN/Inf:\n{mean_bands}"
        )

    if not np.all(np.isfinite(std_bands)):
        raise ValueError(
            f"std_bands contém NaN/Inf:\n{std_bands}"
        )

    if np.any(std_bands < 1e-8):

        problematic_bands = np.where(
            std_bands.reshape(-1) < 1e-8
        )[0]

        raise ValueError(
            f"Desvio padrão zero ou muito pequeno.\n"
            f"Bandas problemáticas: {problematic_bands.tolist()}\n"
            f"STD: {std_bands.reshape(-1)}"
        )


    #------------------------------------------------------------------------
    # Normalizar Train, Val e Test

    for split in ["Train", "Val", "Test"]:

        # Train aumentado
        if split == "Train":
            input_dir = AUG_DIR_EXP / split

        # Val e Test originais
        else:
            input_dir = SPLIT_DIR / split

        output_dir = AUG_DIR_EXP / f"{split}_Norm"

        files = list(input_dir.rglob("*.npy"))

        if len(files) == 0:
            raise RuntimeError(
                f"Nenhum arquivo .npy encontrado em {input_dir}"
            )

        print(f"\nNormalizando {split}: {len(files)} imagens")

        for i, file_path in enumerate(files):

            relative_path = file_path.relative_to(input_dir)

            output_path = output_dir / relative_path
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if not os.path.isfile(output_path):

                #--------------------------------------------------------------
                # Carregar imagem
                #--------------------------------------------------------------

                img = np.load(file_path)

                # Esperado: (H, W, N_BANDS)
                if img.ndim != 3:
                    raise ValueError(
                        f"Imagem com dimensionalidade inválida:\n"
                        f"Arquivo: {file_path}\n"
                        f"Shape encontrado: {img.shape}"
                    )

                if img.shape[-1] != N_BANDS:
                    raise ValueError(
                        f"Número de bandas inconsistente:\n"
                        f"Arquivo: {file_path}\n"
                        f"Esperado: {N_BANDS}\n"
                        f"Encontrado: {img.shape[-1]}"
                    )

                #--------------------------------------------------------------
                # Padronizar imagem em float32
                #--------------------------------------------------------------

                img = img.astype(
                    np.float32,
                    copy=False
                )

                #--------------------------------------------------------------
                # Verificar NaN / Inf ANTES da normalização
                #--------------------------------------------------------------

                if not np.isfinite(img).all():

                    n_nan = np.isnan(img).sum()
                    n_posinf = np.isposinf(img).sum()
                    n_neginf = np.isneginf(img).sum()

                    raise ValueError(
                        f"Valores inválidos ANTES da normalização:\n"
                        f"Split: {split}\n"
                        f"Arquivo: {file_path}\n"
                        f"NaN: {n_nan}\n"
                        f"+Inf: {n_posinf}\n"
                        f"-Inf: {n_neginf}"
                    )

                #--------------------------------------------------------------
                # Normalização
                #
                # img      -> float32
                # mean/std -> float64
                # resultado temporário -> float64
                #--------------------------------------------------------------

                img_norm = (
                    img - mean_bands
                ) / std_bands

                #--------------------------------------------------------------
                # Verificar resultado ANTES da conversão para float32
                #--------------------------------------------------------------

                if not np.isfinite(img_norm).all():

                    n_nan = np.isnan(img_norm).sum()
                    n_posinf = np.isposinf(img_norm).sum()
                    n_neginf = np.isneginf(img_norm).sum()

                    raise ValueError(
                        f"Valores inválidos CRIADOS pela normalização:\n"
                        f"Split: {split}\n"
                        f"Arquivo: {file_path}\n"
                        f"NaN: {n_nan}\n"
                        f"+Inf: {n_posinf}\n"
                        f"-Inf: {n_neginf}\n"
                        f"Mean: {mean_bands.reshape(-1)}\n"
                        f"STD: {std_bands.reshape(-1)}"
                    )

                #--------------------------------------------------------------
                # Arquivo normalizado final sempre em float32
                #--------------------------------------------------------------

                img_norm = img_norm.astype(np.float32)

                #--------------------------------------------------------------
                # Verificação final após conversão para float32
                #--------------------------------------------------------------

                if not np.isfinite(img_norm).all():

                    n_nan = np.isnan(img_norm).sum()
                    n_posinf = np.isposinf(img_norm).sum()
                    n_neginf = np.isneginf(img_norm).sum()

                    raise ValueError(
                        f"Valores inválidos após conversão para float32:\n"
                        f"Split: {split}\n"
                        f"Arquivo: {file_path}\n"
                        f"NaN: {n_nan}\n"
                        f"+Inf: {n_posinf}\n"
                        f"-Inf: {n_neginf}"
                    )

                #--------------------------------------------------------------
                # Salvar
                #--------------------------------------------------------------

                np.save(
                    output_path,
                    img_norm
                )

            if (i + 1) % 10 == 0:
                print(
                    f"{split}: "
                    f"{i + 1}/{len(files)} imagens processadas"
                )


    print("\nAugmentation ✅\n")


    # ============================================================









