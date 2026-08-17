import os
import yaml
from time import sleep
from itertools import product

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

sleep(2)

from RUN_Preprocessing.test_03_new_models.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_03_new_models.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# #======================================================================
# #======================================================================
# # YAML File

# RUN 1

yaml_name = "TEST_model_epoch_30_aug.yaml"

print(f"\n work_dir: {os.getcwd()} \n")
sleep(2)

# yaml_abs_dir = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_name}"
yaml_dir = f"config/{yaml_name}"

#-----------------------------------------------------------------------
# Import config

yaml_file = load_config(yaml_dir)

for x in yaml_file:
    print(f"{x}: \033[96;96m{yaml_file[x]}\033[0m")

sleep(5)

#======================================================================
#======================================================================
# Grid

print("\n\n GRID: \n\n")

model_name_list = ["SmallCNN", "MobileNetV3Small", "ResNet18"]  #  ["SmallCNN", "MobileNetV3Small", "ResNet18"]
pretrained_list = [True, False]
epochs_list = [5, 30]

for model_name in model_name_list:          # model_name = "MobileNetV3Small"
    for epochs in epochs_list:              # epochs = 1
        for pretrained in pretrained_list:  # pretrained = True
            if pretrained and model_name == "SmallCNN":
                continue

            #-----------------------------------------------------------------------
            # ALL PIPELINE

            print("\n\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n\n")

            #-----------------------------------------------------------------------

            EXPERIMENT_NAME = f"MODEL_{model_name}__PRETRAINED_{pretrained}__EPOCHS_{epochs}_AUG"

            yaml_file["EXPERIMENT_NAME"] = EXPERIMENT_NAME

            yaml_file["MODEL"]["MODEL_NAME"] = model_name
            yaml_file["MODEL"]["PRETRAINED"] = pretrained
            yaml_file["MODEL"]["EPOCHS"] = epochs

            print(f"\033[96;96m {yaml_file['EXPERIMENT_NAME']} \033[0m")
            for x in yaml_file['MODEL']:
                print(f"{x}: \033[96;96m{yaml_file['MODEL'][x]}\033[0m")

            #-----------------------------------------------------------------------

            # run_preprocessing(yaml_file)

            run_training(yaml_file)

            print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")

            sleep(5)


#======================================================================
#======================================================================
#======================================================================
