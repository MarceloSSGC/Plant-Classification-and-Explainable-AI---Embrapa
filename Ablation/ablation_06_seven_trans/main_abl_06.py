import os
import pandas as pd
import json

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

# Auxiliar
from aux_plot import *
from aux_model import *
from aux_only_models import *
from aux_transformations import *

#======================================================================
#======================================================================

def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
    print(data.shape)

def p(data):
    print(pd.DataFrame(data).to_string())
    print(data.shape)

#======================================================================
#======================================================================
# Directories

VAL_DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Multiview_5_BANDS/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20/Val_Norm/"

EXP_NAME = "MTV_TEXTURE__RGB_NIR_RE__MobileNetV3Small__DROPOUT_0.2_BATCH_SIZE_8_lr_0.0001_EPOCHS_30_AUG_True"
BAND_TYPE = "Multiview_Texture__AUG"
EXP_TYPE = "RGB_NIR_RE"
EXP_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/{BAND_TYPE}/{EXP_TYPE}/{EXP_NAME}"


ABL_NAME = "Ablation_06_seven_trans"
ABL_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Ablation/{ABL_NAME}/{BAND_TYPE}/{EXP_TYPE}/{EXP_NAME}"

#======================================================================
# All Models

EXP_LIST = sorted(os.listdir(f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/{EXP_TYPE}"))

for X in EXP_LIST:  
    for y in ["MobileNetV3Large", "EfficientNetB0", "ResNet50", "ViTBase", "ViTSmall"]:
        EXP_LIST = [x for x in EXP_LIST if y not in x]

    
# for EXP_NAME in EXP_LIST:   # EXP_NAME = EXP_LIST[7]

    # print("\n" + "="*70 + "\n")
    # print(f'EXP_NAME: {EXP_NAME}')

    # EXP_TYPE = "Multiview_Texture__AUG"
    # BAND_TYPE = "RGB_NIR_RE"
    # EXP_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/{EXP_TYPE}/{BAND_TYPE}/{EXP_NAME}"

    # do_ablation_04(TEST_DATA_DIR, EXP_DIR, ABL_DIR)


transformation = suppress_rgbnirre_color
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]
# transformation_name = "suppress_rgbnirre_color"

#======================================================================

def do_ablation_06(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
):

    #------------------------------------------------------------------
    # DIRS

    output_dir = ABL_DIR
    os.makedirs(output_dir, exist_ok=True)

    df_all_dir = f"{output_dir}/df_all_{transformation.__name__}.csv"
    # fig_dir = f"{output_dir}/ablation_study.png"

    if not os.path.isfile(df_all_dir):

        #------------------------------------------------------------------
        # INFO

        mdl_info_dir = os.path.join(EXP_DIR, 'mld_info.json')

        with open(mdl_info_dir, "r") as f:
            mdl_info = json.load(f)

        #------------------------------------------------------------------

        model_name = mdl_info["RUN"]["MODEL_CONFIG"]["MODEL_NAME"]
        if mdl_info["RUN"]["MODEL_CONFIG"]["AUGMENTATION"]:
            model_name += "_AUG"

        print(f"model_name: \033[100;40m {model_name} \033[0m")


        #------------------------------------------------------------------
        # Val Metrics

        val_metric_dir = os.path.join(EXP_DIR, 'df_metric_val.csv')
        df_metric_val = pd.read_csv(val_metric_dir)

        #------------------------------------------------------------------
        # 1. Dataset PyTorch

        DATA_DIR = Path(VAL_DATA_DIR)

        VAL_DIR  = DATA_DIR 

        #----------------------------------------------------------------------
        # Cuda

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_cuda_is_available = torch.cuda.is_available()
        torch_cuda_get_device_name = torch.cuda.get_device_name(0) if torch_cuda_is_available else None

        print(f"Device: \033[96;92m{device}\033[0m")
        print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
        print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")

        #----------------------------------------------------------------------
        # Load Model

        model_dir = os.path.join(EXP_DIR, 'best_model.pt')
        model = load_model_generic(model_dir, device=str(device))

        #----------------------------------------------------------------------
        # especies

        especies = sorted(os.listdir(VAL_DIR))

        #----------------------------------------------------------------------
        # Cases

        def define_trans(param):

            if param == 0 or param is None:
                def trans_function(img):
                    return img
            else:
                def trans_function(img):
                    return transformation(img, param)

            return trans_function

        #-------------------------

        transformations_list = [define_trans(k) for k in transformation_params]

        name_cases_list = [f"{transformation.__name__}_{k}" for k in transformation_params]

        #----------------------------------------------------------------------
        # For all Bands

        df_all = df_metric_val.copy()

        df_all_col = list(df_all.columns)
        df_all["EXP"] = "Original"
        df_all_col.insert(15, "EXP")

        df_all = df_all[df_all_col]

        df_temp = df_all.iloc[:, :22].copy()

        for i, transform in enumerate(transformations_list):   # i = 0

            print("\n"+ "="*80 + f"\n i: {i} -- transform: {transform}")
            # transform = transformations_list[0]

            val_dataset = WeedDataset_Transform(VAL_DIR, transform=transform)

            print(f"mean: \n{val_dataset[2][0].mean(axis=(1, 2))}")
            print(f"std: \n{val_dataset[2][0].std(axis=(1, 2))}")


            N_BANDS = int(mdl_info["RUN"]['N_BANDS'])
            batch_size = int(df_metric_val.loc[0, "BATCH_SIZE"])
            num_workers = int(df_metric_val.loc[0, "NUM_WORKERS"])
            pin_memory = int(df_metric_val.loc[0, "PIN_MEMORY"])
            persistent_workers = int(df_metric_val.loc[0, "PERSISTENT_WORKERS"])

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )

            new_val_results = model.predict(val_loader, device=device)

            y_val_real = new_val_results["true"]
            y_val_pred = new_val_results["preds"]

            df_metric_val_abl = classification_metrics_dataframe(y_val_real, y_val_pred, especies)

            # df_temp["EXP"] = "Only_" + "_".join([str(x) for x in sorted(set(range(5)) - set(bands))])
            df_temp["EXP"] = name_cases_list[i]
            df_metric_val_abl = pd.concat([df_temp, df_metric_val_abl], axis=1)

            df_all = pd.concat([df_all, df_metric_val_abl], axis=0).reset_index(drop=True)

        p(df_all)

        df_all.to_csv(df_all_dir, index=False)

    else:
        df_all = pd.read_csv(df_all_dir)

    return df_all

        # #------------------------------------------------------------------
        # # PLOT

        # fig_acc, ax_acc = plot_ablation_4_metric(df_all, title=f"Spectral Band Ablation Study - {model_name}", figsize=(9, 6))
        # # fig_prc_micro, ax_prc_micro = plot_ablation_metric(df_all, metric="precision_micro", title=f"Spectral Band Ablation Study - {model_name}")

        # #------------------------------------------------------------------
        # # Save

        # #-------------------------
        # # Metrics

        # df_all.to_csv(df_all_dir, index=False)

        # #-------------------------
        # # Image ACC

        # fig_acc.savefig(
        #     fig_dir,
        #     dpi=300,
        #     bbox_inches="tight"
        # )


#======================================================================
#======================================================================


#======================================================================
# Color

transformation = suppress_rgb_color
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]

df_all_color = do_ablation_06(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

df_all_color_ = df_all_color.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_all_color_["acuracia"])], norm=True, title=None, figsize=(10, 6))

#----------------------------------------------------------------------
transformation = suppress_rgbnirre_color
transformation_params = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1]

df_all_color_nirre = do_ablation_06(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

df_all_color_nirre_ = df_all_color_nirre.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_all_color_nirre_["acuracia"])], norm=True, title=None, figsize=(10, 6))


#======================================================================
# Texture

transformation = suppress_texture
transformation_params = [0, 1, 2, 3, 4, 5, 6]

df_all_texture = do_ablation_06(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

df_all_texture_ = df_all_texture.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_all_texture_["acuracia"])], norm=True, title=None, figsize=(10, 6))

#----------------------------------------------------------------------

transformation = suppress_texture_mask_aware
transformation_params = [0, 1, 2, 3, 4, 5, 6]

df_all_texture_ma = do_ablation_06(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

df_all_texture_ma_ = df_all_texture_ma.drop(0, axis=0).reset_index(drop=True)
plotar_linhas([(i, x) for i, x in enumerate(df_all_texture_ma_["acuracia"])], norm=True, title=None, figsize=(10, 6))



#======================================================================
# Shape

transformation = suppress_shape
transformation_params = [None, 1024, 512, 256, 128, 64, 32]

df_all_shape = do_ablation_06(
        VAL_DATA_DIR,
        EXP_DIR,
        ABL_DIR,
        transformation,
        transformation_params,
)

df_all_shape
plotar_linhas([(i, x) for i, x in enumerate(df_all_shape["acuracia"])], norm=True, title=None, figsize=(10, 6))

#======================================================================
#======================================================================


plotar_multiplas_linhas([df_all_color["acuracia"], df_all_texture["acuracia"],
                         df_all_shape["acuracia"]], nomes=["Color", "Texture", "Shape"], norm=True, title=None, figsize=(10, 6))