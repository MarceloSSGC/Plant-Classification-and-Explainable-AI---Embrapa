import os
import yaml
from time import sleep
from itertools import product

os.chdir("/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa/")
print(f"\n work_dir: {os.getcwd()[-50:]} \n")

from RUN_Preprocessing.test_05_more_texture.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_05_more_texture.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# #======================================================================
# #======================================================================
# # YAML File

# RUN 1

yaml_name = "MTV_model.yaml"


yaml_abs_dir = f"/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa/RUN_Preprocessing/test_05_more_texture/{yaml_name}"
# yaml_dir = f"{yaml_name}"

#-----------------------------------------------------------------------
# Import config

yaml_file = load_config(yaml_abs_dir)

for x in yaml_file:
    print(f"{x}: \033[96;96m{yaml_file[x]}\033[0m")

sleep(3)

#======================================================================
#======================================================================
# Grid

print("\n\n GRID: \n\n")

epochs_list = [30]
augmentation_list = [True]
dropout_list = [0.2]
batch_size_list = [16]
lr_list = [1e-4]
pretrained_list = [True]
transformation_list = ["5_bands", "only_RGB", "Entropy", "LBP", "Tophat"]
model_name_list = ['SmallCNN', 'MobileNetV3Small', 'ResNet18', 'ConvNeXtTiny', 'ViTTiny']

# model_name_list = ['MobileNetV3Large', 'EfficientNetB0', 'ResNet50', 'ViTSmall', 'ViTBase']
# model_name_list = ['SmallCNN', 'MobileNetV3Small', 'MobileNetV3Large',
#                     'EfficientNetB0', 'ResNet18', 'ResNet50',
#                     'ConvNeXtTiny', 'ViTTiny', 'ViTSmall', 'ViTBase']

#  ["SmallCNN", "MobileNetV3Small", "ResNet18", "ConvNeXtTiny", "ViTTiny"]

print(model_name_list)

print(f"Combinations: {len(list(product(epochs_list, augmentation_list, dropout_list, batch_size_list, lr_list, pretrained_list, transformation_list, model_name_list)))}")

# epochs = 1
# aug_bool = False
# dropout = 0.2
# batch_size = 16
# lr = 1e-4
# pretrained = True
# transformation_name = transformation_list[1]
# model_name = "MobileNetV3Small"

for epochs in epochs_list:                                                  # epochs = 2
    for aug_bool in augmentation_list:                                      # aug_bool = True
        for dropout in dropout_list:                                        # dropout = 0.2
            for batch_size in batch_size_list:                              # batch_size = 16
                for lr in lr_list:                                          # lr = 1e-4
                    for pretrained in pretrained_list:                      # pretrained = True
                        for model_name in model_name_list:              # model_name = "MobileNetV3Small"
                            for transformation_name in transformation_list:     # transformation_name = transformation_list[0]

                                config = yaml_file.copy()

                                #-----------------------------------------------------------------------
                                # ALL PIPELINE

                                print("\n\033[100;40m" + "- "*100 + "\033[0m\n")
                                print("\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n")

                                #-----------------------------------------------------------------------


                                EXPERIMENT_NAME = f"TEST = MTV__{model_name}__DROPOUT_{dropout}_BATCH_SIZE_{batch_size}_lr_{lr}__EPOCHS_{epochs}__TRANS_{transformation_name}"

                                if aug_bool:
                                    config["SPLIT_DATA_NAME"] += "_AUG"
                                    config["SPLIT_NAME_TYPE"] = "Augmentation"

                                    EXPERIMENT_NAME += "_AUG"

                                config["EXPERIMENT_NAME"] = EXPERIMENT_NAME

                                config["TRANSFORMATION_NAME"] = transformation_name

                                config["MODEL"]["MODEL_NAME"] = model_name
                                config["MODEL"]["PRETRAINED"] = pretrained
                                config["MODEL"]["EPOCHS"] = epochs
                                config["MODEL"]["DROPOUT"] = dropout
                                config["MODEL"]["BATCH_SIZE"] = batch_size
                                config["MODEL"]["LR"] = lr

                                #-----------------------------------------------------------------------

                                print(f"\n\033[100;40m {config['EXPERIMENT_NAME']} \033[0m\n")

                                print(f"AUGMENTATION: \033[96;95m {aug_bool} \033[0m\n")
                                print(f"TRANSFORMATION_NAME: \033[96;95m {transformation_name} \033[0m\n")

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
