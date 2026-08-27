import os
import pandas as pd
import json

# Auxiliar
from aux_plot import *
from aux_model import *
from aux_only_models import *

#======================================================================
#======================================================================

def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
    print(data.shape)

#======================================================================
#======================================================================
# Directories

test_norm_dir = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Multiview_5_BANDS/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20/Test_Norm/"

especies = sorted(os.listdir(test_norm_dir))

especie = "36_Unha_de_gato_Serra_da_Prata_06"

especie_dir = os.path.join(test_norm_dir, especie)

files = sorted(os.listdir(especie_dir))

file_name = files[4]
file_dir = os.path.join(especie_dir, file_name)

#======================================================================
# View Transformation

img_5b = np.load(file_dir).astype(np.float32)  # (H, W, N)
print(f"img_5b.shape: {img_5b.shape}")
print(f"img_5b.mean(axis=(0, 1)): \n{img_5b.mean(axis=(0, 1))}")

plot_rgb(img_5b, bands_ch=(2, 1, 0))

img_5b.mean(axis=(0, 1))

#----------------------------------------------------------------
# Color Suppression

img_5b_tr = suppress_rgb_color(img_5b)

print(f"img_5b_tr.shape: {img_5b_tr.shape}")
print(f"img_5b_tr.mean(axis=(0, 1)): \n{img_5b_tr.mean(axis=(0, 1))}")


plot_rgb(img_5b_tr)
plot_rgb(img_5b_tr, bands_ch=(2, 1, 0))
plot_rgb(img_5b, bands_ch=(2, 1, 0))

#----------------------------------------------------------------
# Texture Suppression

img_5b_tr = suppress_texture(img_5b, 5)

print(f"img_5b_tr.shape: {img_5b_tr.shape}")
print(f"img_5b_tr.mean(axis=(0, 1)): \n{img_5b_tr.mean(axis=(0, 1))}")


plot_rgb(img_5b_tr)
plot_rgb(img_5b_tr, bands_ch=(2, 1, 0))
plot_rgb(img_5b, bands_ch=(2, 1, 0))

#----------------------------------------------------------------
# Shape Suppression

img_5b_tr = suppress_shape(img_5b, 128)

print(f"img_5b_tr.shape: {img_5b_tr.shape}")
print(f"img_5b_tr.mean(axis=(0, 1)): \n{img_5b_tr.mean(axis=(0, 1))}")


plot_rgb(img_5b_tr)
plot_rgb(img_5b_tr, bands_ch=(2, 1, 0))
plot_rgb(img_5b, bands_ch=(2, 1, 0))

#----------------------------------------------------------------
# Band Alignment Suppression

img_5b_tr = suppress_band_alignment(img_5b)

print(f"img_5b_tr.shape: {img_5b_tr.shape}")
print(f"img_5b_tr.mean(axis=(0, 1)): \n{img_5b_tr.mean(axis=(0, 1))}")


plot_rgb(img_5b_tr)
plot_rgb(img_5b_tr, bands_ch=(2, 1, 0))
plot_rgb(img_5b, bands_ch=(2, 1, 0))

#----------------------------------------------------------------
# Non Visible Spectrum Suppression

img_5b_tr = suppress_non_visible_spectrum(img_5b)

print(f"img_5b_tr.shape: {img_5b_tr.shape}")
print(f"img_5b_tr.mean(axis=(0, 1)): \n{img_5b_tr.mean(axis=(0, 1))}")


plot_rgb(img_5b_tr)
plot_rgb(img_5b_tr, bands_ch=(2, 1, 0))
plot_rgb(img_5b_tr, bands_ch=(3, 1, 4))
plot_rgb(img_5b, bands_ch=(3, 1, 4))

#----------------------------------------------------------------
# Spatial Organization Suppression

img_5b_tr = suppress_spatial_organization(img_5b)

print(f"img_5b_tr.shape: {img_5b_tr.shape}")
print(f"img_5b_tr.mean(axis=(0, 1)): \n{img_5b_tr.mean(axis=(0, 1))}")


plot_rgb(img_5b_tr)
plot_rgb(img_5b_tr, bands_ch=(2, 1, 0))
plot_rgb(img_5b_tr, bands_ch=(3, 1, 4))
plot_rgb(img_5b, bands_ch=(3, 1, 4))

#----------------------------------------------------------------



#======================================================================
# ResultDir

RESULTS_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/MTV_5_BANDS"

os.listdir(RESULTS_DIR)

EXPERIMENT_NAME = "_MTV_5_BANDS__ResNet18__DROPOUT_0.2_PRETRAINED_True__EPOCHS_30"

EXP_DIR = os.path.join(RESULTS_DIR, EXPERIMENT_NAME)

os.listdir(EXP_DIR)

#======================================================================
# INFO

mdl_info_dir = os.path.join(EXP_DIR, 'mld_info.json')

with open(mdl_info_dir, "r") as f:
    mdl_info = json.load(f)

DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Multiview_5_BANDS/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20/"

#======================================================================
# Test Metrics

test_metric_dir = os.path.join(EXP_DIR, 'df_metric_test.csv')
df_metric_test = pd.read_csv(test_metric_dir)

#======================================================================
#======================================================================
# 1. Dataset PyTorch

DATA_DIR = Path(DATA_DIR)

TRAIN_DIR = DATA_DIR / "Train_Norm"
VAL_DIR   = DATA_DIR / "Val_Norm"
TEST_DIR  = DATA_DIR / "Test_Norm"

#----------------------------------------------------------------------
# 2. DataLoaders

my_transform = define_T(0, 0)

# train_dataset = WeedDataset_Transform(TRAIN_DIR)
# val_dataset = WeedDataset_Transform(VAL_DIR)
test_dataset = WeedDataset_Transform(TEST_DIR, transform=my_transform)

# print(f"\nTrain: \033[96;92m{len(train_dataset)}\033[0m")
# print(f"Val: \033[96;92m{len(val_dataset)}\033[0m")
print(f"Test: \033[96;92m{len(test_dataset)}\033[0m \n")

N_BANDS = int(mdl_info["RUN"]['N_BANDS'])
batch_size = int(df_metric_test.loc[0, "BATCH_SIZE"])
num_workers = int(df_metric_test.loc[0, "NUM_WORKERS"])
pin_memory = int(df_metric_test.loc[0, "PIN_MEMORY"])
persistent_workers = int(df_metric_test.loc[0, "PERSISTENT_WORKERS"])

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers
)

#----------------------------------------------------------------------
# Cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_cuda_is_available = torch.cuda.is_available()
torch_cuda_get_device_name = torch.cuda.get_device_name(0)

print(f"Device: \033[96;92m{device}\033[0m")
print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")


#======================================================================
# Load Model

model_dir = os.path.join(EXP_DIR, 'best_model.pt')

model = load_model_generic(model_dir, device=str(device))

#======================================================================
# Inference

print(f"\n\t --- \033[96;01m  Model Predicting --- \033[0m\n")

new_test_results = model.predict(test_loader, device=device)

y_test_real = new_test_results["true"]
y_test_pred = new_test_results["preds"]

df_metric_test_abl = classification_metrics_dataframe(y_test_real, y_test_pred, especies)

df_metric_test_abl = pd.concat([df_temp, df_metric_test_abl], axis=1)
df_metric_test

#======================================================================
# For all Bands

df_all = df_metric_test.copy()

df_all_col = list(df_all.columns)
df_all["EXP"] = "original"
df_all_col.insert(14, "EXP")

df_all = df_all[df_all_col]

df_temp = df_all.iloc[:, :15].copy()

for band in range(5):   # band = 0

    my_transform = define_T(band, 0)

    # train_dataset = WeedDataset_Transform(TRAIN_DIR)
    # val_dataset = WeedDataset_Transform(VAL_DIR)
    test_dataset = WeedDataset_Transform(TEST_DIR, transform=my_transform)

    # print(f"\nTrain: \033[96;92m{len(train_dataset)}\033[0m")
    # print(f"Val: \033[96;92m{len(val_dataset)}\033[0m")
    print(f"Test: \033[96;92m{len(test_dataset)}\033[0m \n")

    N_BANDS = int(mdl_info["RUN"]['N_BANDS'])
    batch_size = int(df_metric_test.loc[0, "BATCH_SIZE"])
    num_workers = int(df_metric_test.loc[0, "NUM_WORKERS"])
    pin_memory = int(df_metric_test.loc[0, "PIN_MEMORY"])
    persistent_workers = int(df_metric_test.loc[0, "PERSISTENT_WORKERS"])

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    new_test_results = model.predict(test_loader, device=device)

    y_test_real = new_test_results["true"]
    y_test_pred = new_test_results["preds"]

    df_metric_test_abl = classification_metrics_dataframe(y_test_real, y_test_pred, especies)

    df_temp["EXP"] = band
    df_metric_test_abl = pd.concat([df_temp, df_metric_test_abl], axis=1)

    df_all = pd.concat([df_all, df_metric_test_abl], axis=0).reset_index(drop=True)


h(df_all, 6)

