import os
import pandas as pd

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


# results_dir = "D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Texture\RGB"
# results_dir = "D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Texture__AUG\RGB"
# results_dir = "D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Texture__AUG\RGB_entropy_v2"



# results_list = os.listdir(results_dir)


# #======================================================================
# # Dataframe

# df = pd.DataFrame()

# for ith, ith_rst_name in enumerate(results_list):   # ith, ith_rst_name = 0, results_list[0]

#     print(f"ith: {ith} - {ith_rst_name}")

#     ith_rst_dir = os.path.join(results_dir, ith_rst_name, 'df_metric_test.csv')
#     # os.listdir(ith_rst_dir)

#     if os.path.isfile(ith_rst_dir):
#         ith_rst = pd.read_csv(ith_rst_dir)
#         df = pd.concat([df, ith_rst], axis=0).reset_index(drop=True)


# df

# inter_cols = ['NICKNAME', 'MODEL_NAME', 'SEED_MODEL', 'PRETRAINED', 'BATCH_SIZE', \
#               'EPOCHS', 'DROPOUT', 'N_BANDS', 'AUGMENTATION', 'TIME_TRAIN', 'BEST_EPOCH', \
#               'N_PARAMS', 'acuracia', 'acuracia_balanceada', 'precision_macro', 
#               'recall_macro', 'f1_macro', 'cohen_kappa', 'matthews_corrcoef']

# df[inter_cols]

# df[inter_cols].groupby("MODEL_NAME")["acuracia"].mean()


# df_[inter_cols].groupby("MODEL_NAME")["acuracia"].mean()

# p(df[df['EPOCHS'] > 1].sort_values(["MODEL_NAME", "AUGMENTATION"]).drop(["NUM_WORKERS", "PIN_MEMORY", "PERSISTENT_WORKERS"], axis=1))
# p(df[(df['EPOCHS'] > 1) & (df['AUGMENTATION'])].sort_values(["acuracia"], ascending=False).drop(["DROPOUT", "NUM_WORKERS", "PIN_MEMORY", "PERSISTENT_WORKERS"], axis=1))





import os
import pandas as pd


results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results"


results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results"




# ============================================================
# 1. Procurar todos os df_metric_test.csv recursivamente
# ============================================================

dfs = []

for root, dirs, files in os.walk(results_dir):

    if "df_metric_test.csv" in files:

        file_path = os.path.join(root, "df_metric_test.csv")

        try:
            df_i = pd.read_csv(file_path)

            # opcional: guardar origem para inspeção/debug
            df_i["SOURCE"] = file_path

            dfs.append(df_i)

        except Exception as e:
            print(f"Erro ao ler:\n{file_path}")
            print(e)


# ============================================================
# 2. Concatenar todos os resultados
# ============================================================

df = pd.concat(dfs, ignore_index=True)

# df = df[df['BATCH_SIZE'] == 4]
df = df[df['BATCH_SIZE'] == 8].reset_index(drop=True)

print(f"Arquivos encontrados: {len(dfs)}")
print(f"Resultados individuais: {len(df)}")

# df=df[df["AUGMENTATION"] == False]

# ============================================================
# 3. Agrupar
# ============================================================

df_grouped = (
    df
    .groupby(
        ["NICKNAME", "AUGMENTATION", "MODEL_NAME"],
        dropna=False
    )
    .agg(
        acuracia=("acuracia", "mean"),
        N=("acuracia", "size")
    )
    .reset_index()
)


# ============================================================
# 4. Ordenar
# ============================================================

df_grouped = df_grouped.sort_values(
    ["NICKNAME", "AUGMENTATION", "MODEL_NAME"]
).reset_index(drop=True)

print(df_grouped)

h(df_grouped[(df_grouped["AUGMENTATION"] == True) & (df_grouped["NICKNAME"] == "RGB_entropy_v2")], 20)
h(df_grouped[(df_grouped["AUGMENTATION"] == True) & (df_grouped["NICKNAME"].isin(["RGB_entropy_v2", "RGB_NIR_RE_entropy_v2"]))], 20)



df_temp = df_grouped[(df_grouped["AUGMENTATION"] == True) & (df_grouped["NICKNAME"].isin(["RGB_entropy_v2", "RGB_NIR_RE_entropy_v2"]))]

df_temp["N"].sum()


h(df[(df["AUGMENTATION"] == True) & (df["NICKNAME"] == "RGB_NIR_RE_entropy_v2")][inter_cols], 20)




#========================================================


import os
import pandas as pd


def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
    print(data.shape)

def p(data):
    print(pd.DataFrame(data).to_string())
    print(data.shape)


def create_summary_table(results_dir, batch_size=8):

    # 1. Carregar arquivos
    dfs = []

    for root, dirs, files in os.walk(results_dir):

        if "df_metric_test.csv" in files:
            file_path = os.path.join(root, "df_metric_test.csv")

            try:
                df_i = pd.read_csv(file_path)
                dfs.append(df_i)
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")

    if not dfs:
        raise ValueError("Nenhum arquivo df_metric_test.csv encontrado.")

    df = pd.concat(dfs, ignore_index=True)

    # 2. Filtrar batch size
    if batch_size is not None:
        df = df[df["BATCH_SIZE"] == batch_size].copy()

    # 3. Agrupar resultados
    df_grouped = (
        df.groupby(
            ["BATCH_SIZE", "AUGMENTATION", "MODEL_NAME", "NICKNAME"],
            dropna=False
        )
        .agg(
            acuracia=("acuracia", "mean"),
            N=("acuracia", "count")
        )
        .reset_index()
    )

    # 4. Formatar resultados
    df_grouped["RESULT"] = df_grouped.apply(
        lambda row: f"{row['acuracia']:.6f} | {int(row['N'])}",
        axis=1
    )

    # 5. Pivotar tabela
    df_table = df_grouped.pivot(
        index=["BATCH_SIZE", "AUGMENTATION", "MODEL_NAME"],
        columns="NICKNAME",
        values="RESULT"
    ).reset_index()

    # 6. Renomear e ordenar colunas
    nickname_columns = {
        "RGB": "RGB",
        "RGB_NIR_RE": "RGB + NIR + RE",
        "RGB_NIR_RE_entropy_v2": "RGB + NIR + RE + Entropia_v2",
        "RGB_entropy_v2": "RGB + Entropia_v2"
    }

    df_table = df_table.rename(columns=nickname_columns)

    column_order = [
        "BATCH_SIZE",
        "AUGMENTATION",
        "MODEL_NAME",
        *nickname_columns.values()
    ]

    df_table = df_table.reindex(columns=column_order)

    # 7. Ordenar linhas
    df_table = df_table.sort_values(
        ["BATCH_SIZE", "AUGMENTATION", "MODEL_NAME"]
    ).reset_index(drop=True)

    df_table.columns.name = None

    return df_table




results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Seed"


results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Texture__AUG"
results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Seed__AUG"

DF = create_summary_table(results_dir)
DF_aug = DF[DF["AUGMENTATION"] == True].reset_index(drop=True)
model_order = ["EfficientNetB0", "RegNetY400MF", "ShuffleNetV2x1_0", "MobileNetV3Small", "ResNet18", "SmallCNN", "ConvNeXtTiny", "ViTTiny"]
DF_aug = (
    DF_aug
    .sort_values("MODEL_NAME", key=lambda x: x.map({m: i for i, m in enumerate(model_order)}))
    .reset_index(drop=True)
)
DF_aug


results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Texture"
results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results\Multiview_Seed"

DF = create_summary_table(results_dir)
DF_noaug = DF[DF["AUGMENTATION"] == False].reset_index(drop=True)
DF_noaug = (
    DF_noaug
    .sort_values("MODEL_NAME", key=lambda x: x.map({m: i for i, m in enumerate(model_order)}))
    .reset_index(drop=True)
)
DF_noaug







def create_summary_table_std(results_dir, batch_size=8):

    # 1. Carregar arquivos
    dfs = []

    for root, dirs, files in os.walk(results_dir):
        if "df_metric_test.csv" in files:
            file_path = os.path.join(root, "df_metric_test.csv")

            try:
                df_i = pd.read_csv(file_path)
                dfs.append(df_i)
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")

    if not dfs:
        raise ValueError("Nenhum arquivo df_metric_test.csv encontrado.")

    df = pd.concat(dfs, ignore_index=True)

    # 2. Filtrar batch size
    if batch_size is not None:
        df = df[df["BATCH_SIZE"] == batch_size].copy()

    # 3. Agrupar resultados
    df_grouped = (
        df.groupby(
            ["BATCH_SIZE", "AUGMENTATION", "MODEL_NAME", "NICKNAME"],
            dropna=False
        )
        .agg(
            desvio_padrao=("acuracia", "std"),
            N=("acuracia", "count")
        )
        .reset_index()
    )

    # 4. Formatar resultados
    df_grouped["RESULT"] = df_grouped.apply(
        lambda row: f"{row['desvio_padrao']:.6f} | {int(row['N'])}",
        axis=1
    )

    # 5. Pivotar tabela
    df_table = df_grouped.pivot(
        index=["BATCH_SIZE", "AUGMENTATION", "MODEL_NAME"],
        columns="NICKNAME",
        values="RESULT"
    ).reset_index()

    # 6. Renomear e ordenar colunas
    nickname_columns = {
        "RGB": "RGB",
        "RGB_NIR_RE": "RGB + NIR + RE",
        "RGB_NIR_RE_entropy_v2": "RGB + NIR + RE + Entropia_v2",
        "RGB_entropy_v2": "RGB + Entropia_v2"
    }

    df_table = df_table.rename(columns=nickname_columns)

    column_order = [
        "BATCH_SIZE",
        "AUGMENTATION",
        "MODEL_NAME",
        *nickname_columns.values()
    ]

    df_table = df_table.reindex(columns=column_order)

    # 7. Ordenar linhas
    df_table = df_table.sort_values(
        ["BATCH_SIZE", "AUGMENTATION", "MODEL_NAME"]
    ).reset_index(drop=True)

    df_table.columns.name = None

    return df_table




model_order = ["EfficientNetB0", "RegNetY400MF", "ShuffleNetV2x1_0", "MobileNetV3Small", "ResNet18", "SmallCNN", "ConvNeXtTiny", "ViTTiny"]

results_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Results"

DF_std = create_summary_table_std(results_dir)

DF_std_aug = DF_std[DF_std["AUGMENTATION"] == True].reset_index(drop=True)

DF_std_aug = (
    DF_std_aug
    .sort_values("MODEL_NAME", key=lambda x: x.map({m: i for i, m in enumerate(model_order)}))
    .reset_index(drop=True)
)

DF_std_aug










#======================================================================
# Best Model

idx = 2

def take_param_from_df(df, idx):
    param = dict(df.iloc[idx, :29])
    return param

#======================================================================
# Loss

p(df[inter_cols])

idx = 12

param = take_param_from_df(df, idx)
for x in param:
    if x in inter_cols:
        print(f"{x}: \033[96;96m{param[x]}\033[0m")

EXPERIMENT_NAME = f"MTV_5_BANDS__{param['MODEL_NAME']}__DROPOUT_{param['DROPOUT']}_PRETRAINED_{param['PRETRAINED']}__EPOCHS_{param['EPOCHS']}"
if param["AUGMENTATION"]:
    EXPERIMENT_NAME += "_AUG"
EXPERIMENT_DIR = os.path.join(results_dir, EXPERIMENT_NAME)
LOSS_DIR = os.path.join(EXPERIMENT_DIR, 'df_loss.csv')

df_loss = pd.read_csv(LOSS_DIR)



df_loss[["train_acc", "val_acc"]].plot(ylim=(0, 1))
