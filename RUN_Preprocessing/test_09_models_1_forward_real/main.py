import os
import gc
import yaml
import torch
from time import sleep
from itertools import product

# EMBRAPA
os.chdir("D:/Marcelo/python_projects/Planta_Daninha")

from RUN_Preprocessing.test_09_models_1_forward_real.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_09_models_1_forward_real.main_run import run_training

TEST_NUMBER = "test_09_models_1_forward_real"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU {tag}] allocated: {alloc:.0f} MB | reserved: {reserv:.0f} MB")


def main():
    print(f"\n work_dir: \n{os.getcwd()} \n")

    # ------------------------------------------------------------------
    # Grid
    multiview_data_nickname_list = ["RGB_NIR_RE.yaml"]
    seed_model_list = list(range(9, 101, 10))
    epochs_list = [30]
    augmentation_list = [True]
    dropout_list = [0.2]
    batch_size_list = [8]
    lr_list = [1e-4]
    pretrained_list = [True]
    model_name_list = ['SmallCNN', 'MobileNetV3Small', 'ResNet18', 'ConvNeXtTiny', 'ViTTiny']

    grid = list(product(
        multiview_data_nickname_list, seed_model_list, epochs_list, augmentation_list,
        dropout_list, batch_size_list, lr_list, pretrained_list, model_name_list,
    ))

    print("\n\n GRID: \n")
    print(model_name_list)
    print(f"\n Combinations: \033[96;96m{len(grid)}\033[0m")

    for i, (multiview_data_nickname, seed_model, epochs, aug_bool, dropout,
            batch_size, lr, pretrained, model_name) in enumerate(grid, start=1):

        # --------------------------------------------------------------
        # Config (recarregada a cada iteração para não herdar alterações)
        config_dir = f"RUN_Preprocessing/{TEST_NUMBER}/local_config/{multiview_data_nickname}"
        config = load_config(config_dir)

        if multiview_data_nickname.replace(".yaml", "") != config["MULTIVIEW_DATA_NICKNAME"]:
            raise ValueError(
                f"config_name ({multiview_data_nickname}) != "
                f"MULTIVIEW_DATA_NICKNAME ({config['MULTIVIEW_DATA_NICKNAME']})"
            )

        EXPERIMENT_NAME = (
            f"MTV_TEXTURE__{config['MULTIVIEW_DATA_NICKNAME']}__{model_name}__"
            f"DROPOUT_{dropout}_BATCH_SIZE_{batch_size}_lr_{lr}_EPOCHS_{epochs}_"
            f"AUG_{aug_bool}_SEED_{seed_model}"
        )

        config["EXPERIMENT_NAME"] = EXPERIMENT_NAME
        config["AUGMENTATION"] = aug_bool
        config["MODEL"]["MODEL_NAME"] = model_name
        config["MODEL"]["SEED_MODEL"] = seed_model
        config["MODEL"]["PRETRAINED"] = pretrained
        config["MODEL"]["EPOCHS"] = epochs
        config["MODEL"]["DROPOUT"] = dropout
        config["MODEL"]["BATCH_SIZE"] = batch_size
        config["MODEL"]["LR"] = lr

        # --------------------------------------------------------------
        print("\n\033[100;40m" + "- " * 100 + "\033[0m\n")
        print(f"\033[96;91m\t === PIPELINE INICIADO ({i}/{len(grid)}) === \t\033[0m \n")
        print(f"\n\033[100;40m {config['EXPERIMENT_NAME']} \033[0m\n")
        print(f"AUGMENTATION: \033[96;95m {aug_bool} \033[0m\n")

        for x in config["MODEL"]:
            print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")

        print(f"\n MULTIVIEW_DATA_NICKNAME: \033[96;95m {config['MULTIVIEW_DATA_NICKNAME']} \033[0m")
        print(f" PC: \033[96;95m {config['PC']} \033[0m")

        print("\n VIEWS:")
        for X in config["VIEWS"]:
            for y in config["VIEWS"][X]:
                print(f"{X}: \033[96;93m{config['VIEWS'][X][y]}\033[0m")

        print_gpu_mem("antes")

        # --------------------------------------------------------------
        try:
            run_preprocessing(config)
            run_training(config)
            print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")
        except torch.cuda.OutOfMemoryError:
            print(f"\n\033[96;91m OOM em {EXPERIMENT_NAME} — pulando. \033[0m\n")
        finally:
            cleanup_gpu()
            print_gpu_mem("depois")

        sleep(5)


if __name__ == "__main__":
    main()