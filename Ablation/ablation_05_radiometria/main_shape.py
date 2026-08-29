import os
import pandas as pd
import json

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

# Auxiliar
from aux_plot import *
from aux_model import *
from aux_only_models import *

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

TEST_DATA_DIR = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Multiview_5_BANDS/align_bands_ecc_affine_with_retry__best_band_otsu_green__Multiview_5_BANDS__SEED_20/Test_Norm/"

EXP_NAME = "MTV_5_BANDS__ResNet18__DROPOUT_0.2_PRETRAINED_True__EPOCHS_30"
EXP_TYPE = "MTV_5_BANDS_First_Models"
EXP_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/{EXP_TYPE}/{EXP_NAME}"


ABL_NAME = "Ablation_05"
ABL_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Ablation/{ABL_NAME}"

#======================================================================
# All Models

EXP_LIST = sorted(os.listdir(f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/{EXP_TYPE}"))

for X in EXP_LIST:  
    for y in ["MobileNetV3Large", "EfficientNetB0", "ResNet50", "ViTBase", "ViTSmall"]:
        EXP_LIST = [x for x in EXP_LIST if y not in x]

    
for EXP_NAME in EXP_LIST:   # EXP_NAME = EXP_LIST[3]

    print("\n" + "="*70 + "\n")
    print(f'EXP_NAME: {EXP_NAME}')

    EXP_TYPE = "MTV_5_BANDS_First_Models"
    EXP_DIR = f"/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/{EXP_TYPE}/{EXP_NAME}"

    do_ablation_04(TEST_DATA_DIR, EXP_DIR, ABL_DIR)



#======================================================================

def do_ablation_04(
        TEST_DATA_DIR,
        EXP_DIR,
        ABL_DIR
):

    #------------------------------------------------------------------
    # DIRS

    output_dir = os.path.join(ABL_DIR, EXP_DIR.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    df_all_dir = f"{output_dir}/df_all.csv"
    fig_dir = f"{output_dir}/ablation_study.png"

    if not os.path.isfile(df_all_dir) or not os.path.isfile(fig_dir):


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
        # Test Metrics

        test_metric_dir = os.path.join(EXP_DIR, 'df_metric_test.csv')
        df_metric_test = pd.read_csv(test_metric_dir)

        #------------------------------------------------------------------
        # 1. Dataset PyTorch

        DATA_DIR = Path(TEST_DATA_DIR)

        TEST_DIR  = DATA_DIR 

        #----------------------------------------------------------------------
        # Cuda

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_cuda_is_available = torch.cuda.is_available()
        torch_cuda_get_device_name = torch.cuda.get_device_name(0)

        print(f"Device: \033[96;92m{device}\033[0m")
        print(f"torch_cuda_is_available: \033[96;92m{torch_cuda_is_available}\033[0m")
        print(f"torch_cuda_get_device_name: \033[96;92m{torch_cuda_get_device_name}\033[0m\n")

        #----------------------------------------------------------------------
        # Load Model

        model_dir = os.path.join(EXP_DIR, 'best_model.pt')
        model = load_model_generic(model_dir, device=str(device))

        #----------------------------------------------------------------------
        # especies

        especies = sorted(os.listdir(TEST_DIR))

        #----------------------------------------------------------------------

        mean_bands = np.array(mdl_info["AUGMENT"]["mean_bands"], dtype="float32")
        std_bands = np.array(mdl_info["AUGMENT"]["std_bands"], dtype="float32")

        #----------------------------------------------------------------------
        # Cases

        # def define_refle(**kwargs):
        #     def transformation(img):
        #         return apply_gain_transform(img_5b=img, **kwargs)

        #     return transformation

        # ftest = define_refle(mean_bands=mean_bands, std_bands=std_bands, a=0.6)

        # def define_supress(**kwargs):
        #     def transformation(img):
        #         return suppress_texture(img_5b=img, **kwargs)

        #     return transformation

        # supress_funct = suppress_texture(sigma = 2)


        # def final_trans(img_5b):

        #     ff_test = define_refle(mean_bands=mean_bands, std_bands=std_bands, a=1)

        #     img_5b_tr = ff_test(img_5b)


        #-------------------------
        # transformations_list = [define_trans(mean_bands=mean_bands, std_bands=std_bands, a=x) for x in [1.1, 1.25, 1.4]]

        # name_cases_list = [f"a = {x}" for x in [1.1, 1.25, 1.4]]

        #----------------------------------------------------------------------

        transformations_list = []
        name_cases_list = [f"a = {x}" for x in [1, 1.1, 1.25, 1.4]]

        for value_a in [1, 1.1, 1.2, 1.4]:

            def make_transformation(a):

                def transformation(img_5b):
                    img_5b_tr = apply_gain_transform(
                        img_5b=img_5b,
                        mean_bands=mean_bands,
                        std_bands=std_bands,
                        a=a,
                    )

                    return suppress_shape(img_5b_tr, 64)

                return transformation

            transformations_list.append(make_transformation(value_a))

        #----------------------------------------------------------------------
        # For all Bands

        df_all = df_metric_test.copy()

        df_all_col = list(df_all.columns)
        df_all["EXP"] = "Original"
        df_all_col.insert(15, "EXP")

        df_all = df_all[df_all_col]

        df_temp = df_all.iloc[:, :16].copy()

        for i, transform in enumerate(transformations_list):   # i = 0

            # transform = transformations_list[0]

            test_dataset = WeedDataset_Transform(TEST_DIR, transform=transform)

            print(f"mean: \n{test_dataset[23][0].mean(axis=(1, 2))}")
            print(f"std: \n{test_dataset[23][0].std(axis=(1, 2))}")


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

            # df_temp["EXP"] = "Only_" + "_".join([str(x) for x in sorted(set(range(5)) - set(bands))])
            df_temp["EXP"] = name_cases_list[i]
            df_metric_test_abl = pd.concat([df_temp, df_metric_test_abl], axis=1)

            df_all = pd.concat([df_all, df_metric_test_abl], axis=0).reset_index(drop=True)

        p(df_all)

        #------------------------------------------------------------------
        # PLOT

        fig_acc, ax_acc = plot_ablation_4_metric(df_all, title=f"Spectral Band Ablation Study - {model_name}", figsize=(9, 6))
        # fig_prc_micro, ax_prc_micro = plot_ablation_metric(df_all, metric="precision_micro", title=f"Spectral Band Ablation Study - {model_name}")

        #------------------------------------------------------------------
        # Save

        #-------------------------
        # Metrics

        df_all.to_csv(df_all_dir, index=False)

        #-------------------------
        # Image ACC

        fig_acc.savefig(
            fig_dir,
            dpi=300,
            bbox_inches="tight"
        )

