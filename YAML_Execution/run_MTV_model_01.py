import os
import yaml
from time import sleep
from itertools import product

print(f"\n work_dir: {os.getcwd()[-50:]} \n")
# os.chdir("..")

from RUN_Preprocessing.test_04_experiments.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_04_experiments.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# #======================================================================
# #======================================================================
# # YAML File

# RUN 1

yaml_name = "MTV_model_01.yaml"


# yaml_abs_dir = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_name}"
yaml_dir = f"config/{yaml_name}"

#-----------------------------------------------------------------------
# Import config

yaml_file = load_config(yaml_dir)

for x in yaml_file:
    print(f"{x}: \033[96;96m{yaml_file[x]}\033[0m")

sleep(3)

#======================================================================
#======================================================================
# Grid

print("\n\n GRID: \n\n")

epochs_list = [30]
augmentation_list = [False, True]
dropout_list = [0.2]
pretrained_list = [True]
model_name_list = ["SmallCNN", "MobileNetV3Small", "ResNet18", "ConvNeXtTiny", "ViTTiny"]

#  ["SmallCNN", "MobileNetV3Small", "ResNet18", "ConvNeXtTiny", "ViTTiny"]

print(f"Combinations: {len(list(product(epochs_list, augmentation_list, dropout_list, pretrained_list, model_name_list)))}")

for epochs in epochs_list:                                  # epochs = 1
    for aug_bool in augmentation_list:                      # aug_bool = False
        for dropout in dropout_list:                        # dropout = 0.2
            for pretrained in pretrained_list:              # pretrained = True
                for model_name in model_name_list:          # model_name = "SmallCNN"

                    config = yaml_file.copy()

                    #-----------------------------------------------------------------------
                    # ALL PIPELINE

                    print("\n\033[100;40m" + "- "*100 + "\033[0m\n")
                    print("\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n")

                    #-----------------------------------------------------------------------


                    EXPERIMENT_NAME = f"MTV_5_BANDS__{model_name}__DROPOUT_{dropout}_PRETRAINED_{pretrained}__EPOCHS_{epochs}"

                    if aug_bool:
                        config["SPLIT_DATA_NAME"] += "_AUG"
                        config["SPLIT_NAME_TYPE"] = "Augmentation"

                        EXPERIMENT_NAME += "_AUG"

                    config["EXPERIMENT_NAME"] = EXPERIMENT_NAME


                    config["MODEL"]["MODEL_NAME"] = model_name
                    config["MODEL"]["PRETRAINED"] = pretrained
                    config["MODEL"]["EPOCHS"] = epochs
                    config["MODEL"]["DROPOUT"] = dropout

                    #-----------------------------------------------------------------------

                    print(f"\n\033[100;40m {config['EXPERIMENT_NAME']} \033[0m\n")

                    print(f"AUGMENTATION: \033[96;95m {aug_bool} \033[0m\n")

                    for x in config['MODEL']:
                        print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")

                    #-----------------------------------------------------------------------

                    # run_preprocessing(yaml_file)

                    run_training(config)

                    print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")

                    sleep(5)


#======================================================================
#======================================================================
#======================================================================
