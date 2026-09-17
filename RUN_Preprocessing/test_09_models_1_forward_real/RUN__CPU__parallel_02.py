import os
import yaml
from itertools import product
from joblib import Parallel, delayed

print(f"\n work_dir: {os.getcwd()[-50:]} \n")

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# DANTE
os.chdir("/home/u14696181/Documents/python_projects/Planta_Daninha_Embrapa")

from RUN_Preprocessing.test_06_texture_bands.main_preprocessing import run_preprocessing
from RUN_Preprocessing.test_06_texture_bands.main_run import run_training

# ======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(params, n_threads_per_job=1):
    """Executa um ponto do grid (uma combinação de hiperparâmetros)."""

    (multiview_data_nickname, epochs, aug_bool,
     dropout, batch_size, lr, pretrained, model_name) = params

    # Evita oversubscription de CPU: cada processo filho usa poucas threads
    # internas (BLAS/torch/tf), já que o paralelismo vem do nível de processo.
    os.environ["OMP_NUM_THREADS"] = str(n_threads_per_job)
    os.environ["MKL_NUM_THREADS"] = str(n_threads_per_job)
    try:
        import torch
        torch.set_num_threads(n_threads_per_job)
    except ImportError:
        pass

    config_dir = f"RUN_Preprocessing/test_06_texture_bands/local_config/{multiview_data_nickname}"
    config = load_config(config_dir)

    if multiview_data_nickname.replace(".yaml", "") != config["MULTIVIEW_DATA_NICKNAME"]:
        raise ValueError("config_name != MULTIVIEW_DATA_NICKNAME")

    print("\n\033[100;40m" + "- " * 100 + "\033[0m\n")
    print("\033[96;91m\t === PIPELINE INICIADO === \t\033[0m \n")

    EXPERIMENT_NAME = (
        f"MTV_TEXTURE__{config['MULTIVIEW_DATA_NICKNAME']}__{model_name}__"
        f"DROPOUT_{dropout}_BATCH_SIZE_{batch_size}_lr_{lr}_EPOCHS_{epochs}_AUG_{aug_bool}"
    )
    config["EXPERIMENT_NAME"] = EXPERIMENT_NAME
    config["AUGMENTATION"] = aug_bool
    config["MODEL"]["MODEL_NAME"] = model_name
    config["MODEL"]["PRETRAINED"] = pretrained
    config["MODEL"]["EPOCHS"] = epochs
    config["MODEL"]["DROPOUT"] = dropout
    config["MODEL"]["BATCH_SIZE"] = batch_size
    config["MODEL"]["LR"] = lr

    print(f"\n\033[100;40m {config['EXPERIMENT_NAME']} \033[0m\n")
    print(f"AUGMENTATION: \033[96;95m {aug_bool} \033[0m\n")

    for x in config["MODEL"]:
        print(f"{x}: \033[96;96m{config['MODEL'][x]}\033[0m")

    print(f"\n MULTIVIEW_DATA_NICKNAME: \033[96;95m {config['MULTIVIEW_DATA_NICKNAME']} \033[0m")

    print("\n VIEWS:")
    for X in config["VIEWS"]:
        for y in config["VIEWS"][X]:
            print(f"{X}: \033[96;93m{config['VIEWS'][X][y]}\033[0m")

    try:
        run_preprocessing(config)
        run_training(config)
        status = "OK"
    except Exception as e:
        status = f"ERRO: {e}"

    print("\n\033[96;91m\t === PIPELINE FINALIZADO === \t\033[0m \n\n")

    return {"experiment": EXPERIMENT_NAME, "status": status}


# ======================================================================
# Grid

print("\n\n GRID: \n\n")

multiview_data_nickname_list = ["RGB_NIR_RE.yaml", "RGB.yaml", "RGB_entropy.yaml"]
epochs_list = [1]
augmentation_list = [True]
dropout_list = [0.2]
batch_size_list = [8]
lr_list = [1e-4]
pretrained_list = [True]
model_name_list = ["MobileNetV3Small", "SmallCNN"]

grid = list(product(
    multiview_data_nickname_list, epochs_list, augmentation_list,
    dropout_list, batch_size_list, lr_list, pretrained_list, model_name_list
))

print(model_name_list)
print(f"Combinations: {len(grid)}")

# ======================================================================
# Execução paralela

N_JOBS = 20              # usa todos os cores disponíveis
N_THREADS_PER_JOB = 1     # threads internas por worker (evita oversubscription)

# resultados = Parallel(n_jobs=N_JOBS, backend="loky", verbose=100)(
#     delayed(run_experiment)(params, N_THREADS_PER_JOB) for params in grid
# )


from tqdm_joblib import tqdm_joblib
from tqdm import tqdm

with tqdm_joblib(tqdm(total=len(grid), desc="Grid search")):
    resultados = Parallel(n_jobs=N_JOBS, backend="loky", verbose=100)(
        delayed(run_experiment)(params, N_THREADS_PER_JOB) for params in grid
    )


print("\n\n RESUMO:")
for r in resultados:
    print(r)