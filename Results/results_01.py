import os
import pandas as pd

#======================================================================
#======================================================================

def h(data, n=5):
    print(pd.DataFrame(data).iloc[:n].to_string())
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
