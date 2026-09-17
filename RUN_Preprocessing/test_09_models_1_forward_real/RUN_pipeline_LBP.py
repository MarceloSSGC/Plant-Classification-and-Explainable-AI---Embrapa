import os
import yaml
from time import sleep
from itertools import product

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Nitro
# os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista")

# DANTE
os.chdir("/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa")

from RUN_Preprocessing.test_08_models_1_forward.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_08_models_1_forward.main_run import run_training
from RUN_Preprocessing.test_08_models_1_forward.aux_config_param import config_function

test_number = "test_08_models_1_forward"


#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#======================================================================
#======================================================================
# Grid

print("\n\n GRID: \n")

# multiview_data_nickname_list = ["RGB_NIR_RE.yaml", "RGB.yaml", "RGB_entropy.yaml", "RGB_LBP.yaml"]
multiview_data_nickname_list = ["RGB_LBP.yaml"]

seed_model_list = list(range(10, 60, 10))

epochs_list = [30]
augmentation_list = [True]
dropout_list = [0.2]
batch_size_list = [8]
lr_list = [1e-4]
pretrained_list = [True]
model_name_list = ['ViTTiny', 'ConvNeXtTiny']
# model_name_list = ["MobileNetV3Small", 'SmallCNN', 'ResNet18', 'ConvNeXtTiny', 'ViTTiny']
# model_name_list = ['MobileNetV3Large', 'EfficientNetB0', 'ResNet50', 'ViTSmall', 'ViTBase']

# model_name_list = ['SmallCNN', 'MobileNetV3Small', 'MobileNetV3Large',
#                     'EfficientNetB0', 'ResNet18', 'ResNet50',
#                     'ConvNeXtTiny', 'ViTTiny', 'ViTSmall', 'ViTBase']

#  ["SmallCNN", "MobileNetV3Small", "ResNet18", "ConvNeXtTiny", "ViTTiny"]

print(model_name_list)

print(f"\n Combinations: \033[96;96m{len(list(product(multiview_data_nickname_list, seed_model_list, epochs_list, augmentation_list, dropout_list, pretrained_list, model_name_list)))}\033[0m")

seed_model = 20
multiview_data_nickname = multiview_data_nickname_list[0]
epochs = 30
aug_bool = True
dropout = 0.2
batch_size = 8
lr = 1e-4
pretrained = True
model_name = "SmallCNN"


for seed_model in seed_model_list:                                  # epochs = 1
    for epochs in epochs_list:                                  # epochs = 1
        for aug_bool in augmentation_list:                      # aug_bool = False
            for dropout in dropout_list:                        # dropout = 0.2
                for batch_size in batch_size_list:                        # batch_size = 16
                    for lr in lr_list:                         # lr = 1e-4
                        for pretrained in pretrained_list:              # pretrained = True
                            for model_name in model_name_list:          # model_name = "SmallCNN"
                                for multiview_data_nickname in multiview_data_nickname_list:          # model_name = "SmallCNN"
                                    
                                    #-----------------------------------------------------------------------
                                    # Import config

                                    config_dir = f"RUN_Preprocessing/{test_number}/local_config/{multiview_data_nickname}"
                                    config = load_config(config_dir)

                                    if multiview_data_nickname.replace(".yaml", "") != config["MULTIVIEW_DATA_NICKNAME"]:
                                        raise ValueError(f"config_name != ")

                                    #-----------------------------------------------------------------------
                                    # ALL PIPELINE

                                    print("\n\033[100;40m" + "- "*100 + "\033[0m\n")
                                    print("\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n")

                                    #-----------------------------------------------------------------------
                                    # Expemrint Name:

                                    EXPERIMENT_NAME = f"MTV_TEXTURE__{config['MULTIVIEW_DATA_NICKNAME']}__{model_name}__DROPOUT_{dropout}_BATCH_SIZE_{batch_size}_lr_{lr}_EPOCHS_{epochs}_AUG_{aug_bool}_SEED_{seed_model}"
                                    # EXPERIMENT_NAME = f"_TEST__MTV_TEXTURE__{config['MULTIVIEW_DATA_NICKNAME']}__{model_name}__DROPOUT_{dropout}_BATCH_SIZE_{batch_size}_lr_{lr}_EPOCHS_{epochs}_AUG_{aug_bool}_SEED_{seed_model}"

                                    #-----------------------------------------------------------------------

                                    config["EXPERIMENT_NAME"] = EXPERIMENT_NAME

                                    config["AUGMENTATION"] = aug_bool

                                    config["MODEL"]["MODEL_NAME"] = model_name
                                    config["MODEL"]["SEED_MODEL"] = seed_model
                                    config["MODEL"]["PRETRAINED"] = pretrained
                                    config["MODEL"]["EPOCHS"] = epochs
                                    config["MODEL"]["DROPOUT"] = dropout
                                    config["MODEL"]["BATCH_SIZE"] = batch_size
                                    config["MODEL"]["LR"] = lr

                                    #-----------------------------------------------------------------------

                                    print(f"\n\033[100;40m {config['EXPERIMENT_NAME']} \033[0m\n")

                                    print(f"AUGMENTATION: \033[96;95m {config['AUGMENTATION']} \033[0m\n")
                                    print(f"SEED_MODEL: \033[96;95m {config['MODEL']['SEED_MODEL']} \033[0m\n")

                                    for x in config['MODEL']:
                                        print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")

                                    print(f"\n MULTIVIEW_DATA_NICKNAME: \033[96;95m {config['MULTIVIEW_DATA_NICKNAME']} \033[0m")

                                    print(f"\n VIEWS:")
                                    for X in config['VIEWS']:
                                        for y in config['VIEWS'][X]:
                                            print(f"{X}: \033[96;93m{config['VIEWS'][X][y]}\033[0m")

                                    #-----------------------------------------------------------------------

                                    # run_preprocessing(config)

                                    config_function(config)
                                    config['NEW_DATA_DIR'] = False

                                    run_training(config)

                                    print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")

                                    sleep(5)


#======================================================================
#======================================================================