import os
import yaml
from time import sleep

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

sleep(2)
from RUN_Multiview__Single_Output.test_01.main_preprocessing import run_preprocessing
from RUN_Multiview__Single_Output.test_01.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# #======================================================================
# # YAML File

yaml_name = "Multiview_01.yaml"

print(f"\n work_dir: {os.getcwd()} \n")
sleep(2)

yaml_dir = f"config/{yaml_name}"

#======================================================================
# Import config

yaml_file = load_config(yaml_dir)

for x in yaml_file:
    print(f"{x}: \033[96;96m{yaml_file[x]}\033[0m")

sleep(4)
#======================================================================
#======================================================================
# ALL PIPELINE

print("=== PIPELINE INICIADO ===")

run_preprocessing(yaml_file)

run_training(yaml_file)

print("\n\033[100;40m\t === PIPELINE FINALIZADO === \t\033[0m")
