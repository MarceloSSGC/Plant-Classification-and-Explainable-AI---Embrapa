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


results_dir = "/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Results/MTV_5_BANDS"

results_list = os.listdir(results_dir)


#======================================================================
# Dataframe

df = pd.DataFrame()

for ith, ith_rst_name in enumerate(results_list):   # ith, ith_rst_name = 0, results_list[0]

    print(f"ith: {ith} - {ith_rst_name}")

    ith_rst_dir = os.path.join(results_dir, ith_rst_name, 'df_metric_test.csv')
    # os.listdir(ith_rst_dir)

    if os.path.isfile(ith_rst_dir):
        ith_rst = pd.read_csv(ith_rst_dir)
        df = pd.concat([df, ith_rst], axis=0).reset_index(drop=True)


df

inter_cols = ['period', 'MODEL_NAME', 'SEED_MODEL', 'PRETRAINED', 'BATCH_SIZE', \
              'EPOCHS', 'DROPOUT', 'N_BANDS', 'AUGMENTATION', 'TIME_TRAIN', 'BEST_EPOCH', \
              'N_PARAMS', 'acuracia', 'acuracia_balanceada', 'precision_macro', 
              'recall_macro', 'f1_macro', 'cohen_kappa', 'matthews_corrcoef']

p(df[df['EPOCHS'] > 1].sort_values(["MODEL_NAME", "AUGMENTATION"]).drop(["NUM_WORKERS", "PIN_MEMORY", "PERSISTENT_WORKERS"], axis=1))
p(df[(df['EPOCHS'] > 1) & (df['AUGMENTATION'])].sort_values(["acuracia"], ascending=False).drop(["DROPOUT", "NUM_WORKERS", "PIN_MEMORY", "PERSISTENT_WORKERS"], axis=1))

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
