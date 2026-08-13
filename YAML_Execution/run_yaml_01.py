import os
import yaml

from RUN_Preprocessing.test_02_yaml.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_02_yaml.main_run import run_training

#======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# #======================================================================
# # YAML File

yaml_name = "test_01_experiment.yaml"
yaml_abs_dir = f"/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/config/{yaml_name}"

if not os.path.isfile(yaml_abs_dir):
    print("YAML File doesnt exist")

yaml_file = load_config(yaml_abs_dir)

for x in yaml_file:
    print(f"{x}: \033[96;96m{yaml_file[x]}\033[0m")

#======================================================================
#======================================================================

print("=== PIPELINE INICIADO ===")

run_preprocessing(yaml_file)

run_training(yaml_file)

print("=== PIPELINE FINALIZADO ===")
