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

from RUN_Preprocessing.test_07_models_speed.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_07_models_speed.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#======================================================================
#======================================================================
# Grid

print("\n\n GRID: \n")

# multiview_data_nickname_list = ["RGB_&_entropy.yaml"]
# multiview_data_nickname_list = ["RGB_NIR_RE.yaml", "RGB.yaml", "RGB_entropy.yaml"]
multiview_data_nickname_list = ["RGB_NIR_RE.yaml", "RGB_entropy.yaml", "RGB.yaml", "RGB_LBP.yaml"]

seed_model_list = list(range(10, 60, 10))

epochs_list = [30]
augmentation_list = [True]
dropout_list = [0.2]
batch_size_list = [8]
lr_list = [1e-4]
pretrained_list = [True]
model_name_list = ["MobileNetV3Small", 'SmallCNN', 'ResNet18', 'ConvNeXtTiny', 'ViTTiny']
# model_name_list = ['MobileNetV3Large', 'EfficientNetB0', 'ResNet50', 'ViTSmall', 'ViTBase']

# model_name_list = ['SmallCNN', 'MobileNetV3Small', 'MobileNetV3Large',
#                     'EfficientNetB0', 'ResNet18', 'ResNet50',
#                     'ConvNeXtTiny', 'ViTTiny', 'ViTSmall', 'ViTBase']

#  ["SmallCNN", "MobileNetV3Small", "ResNet18", "ConvNeXtTiny", "ViTTiny"]

print(model_name_list)

print(f"\n Combinations: \033[96;96m{len(list(product(multiview_data_nickname_list, seed_model_list, epochs_list, augmentation_list, dropout_list, pretrained_list, model_name_list)))}\033[0m")

multiview_data_nickname = multiview_data_nickname_list[0]
epochs = 2
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

                                    config_dir = f"RUN_Preprocessing/test_07_models_speed/local_config/{multiview_data_nickname}"
                                    config = load_config(config_dir)

                                    if multiview_data_nickname.replace(".yaml", "") != config["MULTIVIEW_DATA_NICKNAME"]:
                                        raise ValueError(f"config_name != ")

                                    #-----------------------------------------------------------------------
                                    # ALL PIPELINE

                                    print("\n\033[100;40m" + "- "*100 + "\033[0m\n")
                                    print("\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n")

                                    #-----------------------------------------------------------------------

                                    EXPERIMENT_NAME = f"MTV_TEXTURE__{config['MULTIVIEW_DATA_NICKNAME']}__{model_name}__DROPOUT_{dropout}_BATCH_SIZE_{batch_size}_lr_{lr}_EPOCHS_{epochs}_AUG_{aug_bool}_SEED_{seed_model}"

                                    # if aug_bool:
                                    #     config["SPLIT_DATA_NAME"] += "_AUG"
                                    #     config["SPLIT_NAME_TYPE"] = "Augmentation"

                                    #     EXPERIMENT_NAME += "_AUG"

                                    config["EXPERIMENT_NAME"] = EXPERIMENT_NAME

                                    config["AUGMENTATION"] = aug_bool

                                    config["MODEL"]["MODEL_NAME"] = model_name
                                    config["MODEL"]["PRETRAINED"] = pretrained
                                    config["MODEL"]["EPOCHS"] = epochs
                                    config["MODEL"]["DROPOUT"] = dropout
                                    config["MODEL"]["BATCH_SIZE"] = batch_size
                                    config["MODEL"]["LR"] = lr

                                    # #-----------------------------------------------------------------------
                                    # # PC Directory

                                    # if config["PC"] == "NITRO":
                                    #     PC_DIR = f"/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
                                    # elif config["PC"] == "HELIOS":
                                    #     PC_DIR = f"/run/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos"
                                    # else:
                                    #     PC_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos"

                                    # SEG_DATA_NAME = f"{config['ALIGN_DATA_NAME']}__{config['SEG_DATA_NICKNAME']}"

                                    # config["MULTIVIEW_DATA_NAME"] = f"{SEG_DATA_NAME}__{config['MULTIVIEW_DATA_NICKNAME']}"
                                    # config["MTV_DATA_DIR"] = f"{PC_DIR}/Datasets/Multiview/{config['MULTIVIEW_DATA_TYPE']}/{config['MULTIVIEW_DATA_NAME']}"

                                    # config["SPLIT_DATA_NAME"] = config["MULTIVIEW_DATA_NAME"] + f"__SEED_{config['MTV_DATA_DIR']}__T_{config['TRAIN_SIZE']}_V_{config['VAL_SIZE']}"
                                    # config["SPLIT_DATE_TYPE"] = config["MULTIVIEW_DATA_TYPE"]
                                    # config["SPLIT_DIR"] = f"{PC_DIR}/Datasets/Split/{config['SPLIT_DATE_TYPE']}/{config['SPLIT_DATA_NAME']}"

                                    # config["AUG_DATE_NAME"] = config["SPLIT_DATA_NAME"] + "__AUG"
                                    # config["AUG_DATE_TYPE"] = config["SPLIT_DATE_TYPE"] + "__AUG"
                                    # config["AUG_DIR_EXP"] = f"{PC_DIR}/Datasets/Augmentation/{config['AUG_DATE_TYPE']}/{config['AUG_DATE_NAME']}"  


                                    # EXPERIMENT_TYPE = config["AUG_DATE_TYPE"] if aug_bool else config["SPLIT_DATE_TYPE"]

                                    # config["EXPERIMENT_TYPE"] = EXPERIMENT_TYPE

                                    # DIR_EXP = f"{PC_DIR}/Results/{EXPERIMENT_TYPE}/{config['MULTIVIEW_DATA_NICKNAME']}/{EXPERIMENT_NAME}"

                                    # if not os.path.isdir(DIR_EXP):
                                    #     os.makedirs(DIR_EXP)

                                    # with open(f"{DIR_EXP}/config.yaml", "w") as f:
                                    #     yaml.safe_dump(config, f)

                                    #-----------------------------------------------------------------------

                                    print(f"\n\033[100;40m {config['EXPERIMENT_NAME']} \033[0m\n")

                                    print(f"AUGMENTATION: \033[96;95m {aug_bool} \033[0m\n")

                                    for x in config['MODEL']:
                                        print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")

                                    print(f"\n MULTIVIEW_DATA_NICKNAME: \033[96;95m {config['MULTIVIEW_DATA_NICKNAME']} \033[0m")

                                    print(f"\n VIEWS:")
                                    for X in config['VIEWS']:
                                        for y in config['VIEWS'][X]:
                                            print(f"{X}: \033[96;93m{config['VIEWS'][X][y]}\033[0m")

                                    #-----------------------------------------------------------------------

                                    run_preprocessing(config)

                                    run_training(config)

                                    print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")

                                    sleep(5)


#======================================================================
#======================================================================
#======================================================================
