import os
import yaml
from time import sleep

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

from RUN_Multiview__Single_Output.test_01.main_preprocessing import run_preprocessing
from RUN_Multiview__Single_Output.test_01.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# #======================================================================
# #======================================================================
# # YAML File

# RUN 1

yaml_name = "Multiview_01.yaml"


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
augmentation_list = [True, False]
dropout_list = [0.2]
pretrained_list = [True]
model_name_list = ["MobileNetV3Small", "ResNet18"]  #  ["SmallCNN", "MobileNetV3Small", "ResNet18"]

for epochs in epochs_list:                                  # epochs = 1
    for aug_bool in augmentation_list:                      # aug_bool = True
        for dropout in dropout_list:                        # dropout = 0.2
            for pretrained in pretrained_list:              # pretrained = True
                for model_name in model_name_list:          # model_name = "MobileNetV3Small"

                    # if pretrained and model_name == "SmallCNN":        
                    #     continue

                    config = yaml_file.copy()

                    #-----------------------------------------------------------------------
                    # ALL PIPELINE

                    print("\n\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n\n")

                    #-----------------------------------------------------------------------


                    EXPERIMENT_NAME = f"Multiview__{model_name}__DROPOUT_{dropout}_PRETRAINED_{pretrained}__EPOCHS_{epochs}"

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
