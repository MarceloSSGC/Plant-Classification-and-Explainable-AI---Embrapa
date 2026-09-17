import os
import yaml
import time

from itertools import product

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# NITRO
# os.chdir("/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista")

# HELIOS
# os.chdir("/home/marcelo/Documents/python_projects/USP/Planta_Daninha_Embrapa/Plant-Classification-and-Explainable-AI---Embrapa/")

# # DANTE
os.chdir("/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa")

from RUN_Preprocessing.test_09_models_1_forward_real.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_09_models_1_forward_real.main_run import run_training
from RUN_Preprocessing.test_09_models_1_forward_real.aux_config_param import config_function

test_number = "test_09_models_1_forward_real"
# temp_transformation

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
#======================================================================
#======================================================================
# Grid

print("\n\n GRID: \n")

# multiview_data_nickname_list = ["RGB_NIR_RE.yaml", "RGB_entropy.yaml", "RGB.yaml", "RGB_LBP.yaml", "RGB_entropy_LBP_Zoom"]
multiview_data_nickname_list = ["RGB_entropy_LBP_Zoom_1p5.yaml"]

seed_model_list = list(range(10, 60, 10))

epochs_list = [30]
augmentation_list = [True]
dropout_list = [0.2]
batch_size_list = [8]
lr_list = [1e-4]
pretrained_list = [True]
model_name_list = ['SmallCNN', 'MobileNetV3Small', 'ResNet18', 'ConvNeXtTiny', 'ViTTiny']

# model_name_list = ['SmallCNN', 'MobileNetV3Small', 'MobileNetV3Large',
#                     'EfficientNetB0', 'ResNet18', 'ResNet50',
#                     'ConvNeXtTiny', 'ViTTiny', 'ViTSmall', 'ViTBase']


print(model_name_list)

print(f"\n Combinations: \033[96;96m{len(list(product(multiview_data_nickname_list, seed_model_list, epochs_list, augmentation_list, dropout_list, pretrained_list, model_name_list)))}\033[0m")

seed_model = 30
multiview_data_nickname = multiview_data_nickname_list[0]
epochs = 30
aug_bool = True
dropout = 0.2
batch_size = 12
lr = 1e-4
pretrained = True
model_name = "SmallCNN"


for multiview_data_nickname in multiview_data_nickname_list:          # model_name = "SmallCNN"
    for seed_model in seed_model_list:                                  # epochs = 1
        for epochs in epochs_list:                                  # epochs = 1
            for aug_bool in augmentation_list:                      # aug_bool = False
                for dropout in dropout_list:                        # dropout = 0.2
                    for batch_size in batch_size_list:                        # batch_size = 16
                        for lr in lr_list:                         # lr = 1e-4
                            for pretrained in pretrained_list:              # pretrained = True
                                for model_name in model_name_list:          # model_name = "SmallCNN"
                                        
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
                                    config["warm_up_data"] = False


                                    run_training(config)

                                    print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")

                                    time.sleep(5)


#======================================================================
#======================================================================
report = """

RGB + NIR + RE

    MobileNetV3Small

        0 hour, 2 min, 33 sec
        0 hour, 1 min, 20 sec
        0 hour, 0 min, 59 sec


    SmallCNN
        0 hour, 1 min, 16 sec

    ResNet18  
        0 hour, 1 min, 25 sec
        0 hour, 1 min, 27 sec

-----------------------------------------        
RGB_entropy

    MobileNetV3Small

        0 hour, 14 min, 2 sec

    SmallCNN
        0 hour, 13 min, 59 sec

-----------------------------------------        
RGB

    SmallCNN
        0 hour, 6 min, 52 sec
        0 hour, 5 min, 41 sec

    MobileNetV3Small
        0 hour, 5 min, 53 sec

    ResNet18
        0 hour, 7 min, 8 sec

"""

#======================================================================

# 1. Reduzir 3 varreduras/época para 2 (treino calcula métricas na própria passagem; validação continua separada) — sem mudar o modelo final treinado
# 2. Print do tempo de duração de cada época
# 3. Checkpoint do "melhor modelo" por `val_loss` (menor), não por acurácia
# 4. Padronizar `_print_device_report` (relatório de GPU/device)
# 5. Padronizar `verbose` com níveis 0/1/2 + progresso por batch (nível 2)

#======================================================================


