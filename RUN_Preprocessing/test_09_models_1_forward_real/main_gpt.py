import os
import sys
import yaml
import multiprocessing as mp

from pathlib import Path
from time import sleep
from itertools import product
from math import prod


# ============================================================
# Configurações gerais
# ============================================================

TEST_NUMBER = "test_09_models_1_forward_real"

SLEEP_BETWEEN_RUNS = 5


# ============================================================
# Diretório do projeto
# ============================================================

def find_project_root():
    """
    Procura automaticamente a raiz do projeto.

    A raiz é identificada pela presença da pasta RUN_Preprocessing.

    Isso evita hardcode de caminhos diferentes para:
        - Windows / EMBRAPA
        - NITRO
        - DANTE
        - HELIOS
        - EUROPA
        etc.
    """

    current = Path(__file__).resolve().parent

    for candidate in [current, *current.parents]:

        if (candidate / "RUN_Preprocessing").is_dir():
            return candidate

    raise FileNotFoundError(
        "Não foi possível localizar a raiz do projeto "
        "(pasta RUN_Preprocessing não encontrada)."
    )


# ============================================================
# YAML
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Grid experimental
# ============================================================

def build_grid():

    return {

        "multiview_data_nickname": [
            # "RGB_NIR_RE.yaml"
            # "RGB.yaml.yaml"
            # "RGB_entropy_v2.yaml"
            "RGB_NIR_RE_entropy_v2.yaml"
        ],

        "seed_model": list(range(6, 100, 10)),

        "epochs": [
            30
        ],

        "augmentation": [
            True, False
        ],

        "dropout": [
            0.2
        ],

        "batch_size": [
            8
        ],

        "lr": [
            1e-4
        ],

        "pretrained": [
            True
        ],

        "model_name": [
            "SmallCNN",
            "MobileNetV3Small",
            "ResNet18",
            "ConvNeXtTiny",
            "ViTTiny",
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Localiza automaticamente a raiz do projeto
    # --------------------------------------------------------

    project_root = find_project_root()

    os.chdir(project_root)

    # Garante que a raiz esteja disponível para imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"\nwork_dir:\n{os.getcwd()}\n")

    # --------------------------------------------------------
    # Imports do projeto
    #
    # Fazemos depois de definir a raiz.
    # Isso também reduz efeitos colaterais durante spawn
    # no Windows.
    # --------------------------------------------------------

    from RUN_Preprocessing.test_09_models_1_forward_real.main_preprocessing import (
        run_preprocessing
    )

    from RUN_Preprocessing.test_09_models_1_forward_real.main_run import (
        run_training
    )

    from RUN_Preprocessing.test_09_models_1_forward_real.aux_config_param import (
        config_function
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grid = build_grid()

    grid_names = list(grid.keys())
    grid_values = list(grid.values())

    total_combinations = prod(len(values) for values in grid_values)

    print("\n\nGRID:\n")

    print(grid["model_name"])

    print(
        f"\nCombinations: "
        f"\033[96m{total_combinations}\033[0m"
    )

    # --------------------------------------------------------
    # Experimentos
    # --------------------------------------------------------

    combinations = product(*grid_values)

    # experiment_idx, combination = 1, next(iter(combinations))
    for experiment_idx, combination in enumerate(combinations, start=1):

        params = dict(zip(grid_names, combination))

        # ----------------------------------------------------
        # Parâmetros da combinação atual
        # ----------------------------------------------------

        multiview_data_nickname = params["multiview_data_nickname"]
        seed_model = params["seed_model"]
        epochs = params["epochs"]
        aug_bool = params["augmentation"]
        dropout = params["dropout"]
        batch_size = params["batch_size"]
        lr = params["lr"]
        pretrained = params["pretrained"]
        model_name = params["model_name"]

        # ----------------------------------------------------
        # Config
        # ----------------------------------------------------

        config_dir = (
            project_root
            / "RUN_Preprocessing"
            / TEST_NUMBER
            / "local_config"
            / multiview_data_nickname
        )

        config = load_config(config_dir)

        expected_nickname = multiview_data_nickname.replace(".yaml", "")

        if expected_nickname != config["MULTIVIEW_DATA_NICKNAME"]:
            raise ValueError(
                f"Config incompatível: "
                f"{expected_nickname} != "
                f"{config['MULTIVIEW_DATA_NICKNAME']}"
            )

        # ----------------------------------------------------
        # Experiment Name
        # ----------------------------------------------------

        EXPERIMENT_NAME = (
            f"MTV_TEXTURE__"
            f"{config['MULTIVIEW_DATA_NICKNAME']}__"
            f"{model_name}__"
            f"DROPOUT_{dropout}_"
            f"BATCH_SIZE_{batch_size}_"
            f"lr_{lr}_"
            f"EPOCHS_{epochs}_"
            f"AUG_{aug_bool}_"
            f"SEED_{seed_model}"
        )

        # ----------------------------------------------------
        # Atualização da configuração
        # ----------------------------------------------------

        config["EXPERIMENT_NAME"] = EXPERIMENT_NAME

        config["AUGMENTATION"] = aug_bool

        config["MODEL"]["MODEL_NAME"] = model_name
        config["MODEL"]["SEED_MODEL"] = seed_model
        config["MODEL"]["PRETRAINED"] = pretrained
        config["MODEL"]["EPOCHS"] = epochs
        config["MODEL"]["DROPOUT"] = dropout
        config["MODEL"]["BATCH_SIZE"] = batch_size
        config["MODEL"]["LR"] = lr

        # ----------------------------------------------------
        # Informações
        # ----------------------------------------------------

        print("\n" + "- " * 100)

        print(
            f"\n\033[96;91m"
            f"=== PIPELINE INICIADO "
            f"[{experiment_idx}/{total_combinations}] ==="
            f"\033[0m\n"
        )

        print(
            f"\033[100;40m "
            f"{config['EXPERIMENT_NAME']} "
            f"\033[0m\n"
        )

        print(
            f"AUGMENTATION: "
            f"\033[96;95m{aug_bool}\033[0m\n"
        )

        for key, value in config["MODEL"].items():
            print(
                f"{key}: "
                f"\033[96;96m{value}\033[0m"
            )

        print(
            f"\nMULTIVIEW_DATA_NICKNAME: "
            f"\033[96;95m"
            f"{config['MULTIVIEW_DATA_NICKNAME']}"
            f"\033[0m"
        )

        print(
            f"PC: "
            f"\033[96;95m"
            f"{config['PC']}"
            f"\033[0m"
        )

        print("\nVIEWS:")

        for view_name, view_data in config["VIEWS"].items():

            for key, value in view_data.items():

                print(
                    f"{view_name}: "
                    f"\033[96;93m{value}\033[0m"
                )

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        run_preprocessing(config)

        config_function(config)

        run_training(config)

        # ----------------------------------------------------
        # Finalização
        # ----------------------------------------------------

        print(
            "\n\033[96;91m"
            "=== PIPELINE FINALIZADO ==="
            "\033[0m\n"
        )

        if SLEEP_BETWEEN_RUNS > 0:
            sleep(SLEEP_BETWEEN_RUNS)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Necessário principalmente no Windows.
    # Inofensivo no Linux.
    mp.freeze_support()

    main()