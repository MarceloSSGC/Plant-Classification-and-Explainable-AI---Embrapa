import os
import yaml
from time import sleep
from itertools import product

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

sleep(2)
from RUN_Preprocessing.test_02_yaml.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_02_yaml.main_run import run_training

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

sleep(2)
#======================================================================
#======================================================================
# Grid







#-----------------------------------------------------------------------
# ALL PIPELINE

print("=== PIPELINE INICIADO ===")

run_preprocessing(yaml_file)

run_training(yaml_file)

print("\n\033[100;40m\t === PIPELINE FINALIZADO === \t\033[0m")

#======================================================================
#======================================================================
#======================================================================


# RUN 2

yaml_name = "test_04_epoch_30_aug.yaml"

print(f"\n work_dir: {os.getcwd()} \n")
sleep(2)

# yaml_abs_dir = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_name}"
yaml_dir = f"config/{yaml_name}"

#-----------------------------------------------------------------------
# Import config

yaml_file = load_config(yaml_dir)

for x in yaml_file:
    print(f"{x}: \033[96;96m{yaml_file[x]}\033[0m")

sleep(4)
#-----------------------------------------------------------------------
# ALL PIPELINE

print("=== PIPELINE INICIADO ===")

run_preprocessing(yaml_file)

run_training(yaml_file)

print("\n\033[100;40m\t === PIPELINE FINALIZADO === \t\033[0m")
