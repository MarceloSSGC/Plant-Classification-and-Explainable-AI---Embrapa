import pandas as pd
import numpy as np

#======================================================================




df = pd.read_csv("/home/marcelo/Documents/VSCode_python/USP/GenAI_FoudModel/Busca_03/Lucas_Cllasification/df_scopus_and_ieee_answer_AI.csv")


df.columns

df["Title"].isna().sum()
df["Abstract"].isna().sum()
df["AI_Usage_Type"].isna().sum()



df["AI_Usage_Type"].fillna("None", inplace=True)


df["50_class_selected"] = False
df["Reviewer"] = "None"

cat_list = list(df["AI_Usage_Type"].unique())
for ith, cat in enumerate(cat_list):        # ith, cat = 0, cat_list[0]

    df_cat_index = df[df["AI_Usage_Type"] == cat].index
    
    print(f'df: {len(df)}')
    print(f'df_cat: {len(df_cat_index)}')

    df_cat_index_50 = pd.Series(df_cat_index).sample(50)

    df_cat_index_50 = df_cat_index_50.sort_values().reset_index(drop=True)

    df.loc[df_cat_index_50.values, "50_class_selected"] = True

    #---------------------------------------

    idx_selected = pd.Series(df_cat_index_50.values).sort_values().reset_index(drop=True)

    idx_selected_marc = pd.Series(idx_selected.sample(17)).sort_values().reset_index(drop=True)

    idx_selected_leon_adrian = pd.Series(list(set(idx_selected) - set(idx_selected_marc)))

    idx_selected_leon = pd.Series(idx_selected_leon_adrian.sample(17)).sort_values().reset_index(drop=True)

    idx_selected_adrian = pd.Series(list(set(idx_selected_leon_adrian) - set(idx_selected_leon))).sort_values().reset_index(drop=True)

    print(f"idx_selected: {len(idx_selected)}")
    print(f"idx_selected_marc: {len(idx_selected_marc)}")
    print(f"idx_selected_leon: {len(idx_selected_leon)}")
    print(f"idx_selected_adrian: {len(idx_selected_adrian)}")

    if len(set(idx_selected_marc) | set(idx_selected_leon)) != len(set(idx_selected_marc)) + len(set(idx_selected_leon)):
        print(f" Diff")

    if len(set(idx_selected_marc) | set(idx_selected_adrian)) != len(set(idx_selected_marc)) + len(set(idx_selected_adrian)):
        print(f" Diff")

    if len(set(idx_selected_leon) | set(idx_selected_adrian)) != len(set(idx_selected_leon)) + len(set(idx_selected_adrian)):
        print(f" Diff")

    df.loc[idx_selected_marc.values, "Reviewer"] = "Marcelo"
    df.loc[idx_selected_leon.values, "Reviewer"] = "Leonardo"
    df.loc[idx_selected_adrian.values, "Reviewer"] = "Adriano"


df["50_class_selected"].value_counts()

df[["50_class_selected", "Reviewer"]].value_counts()



df["AI_Usage_Type"].value_counts()


#==========================================================================================

df["50_class_selected"] = False
df["Evaluator"] = pd.NA

rng = np.random.default_rng(42)

for cat, group in df.groupby("AI_Usage_Type", dropna=False):

    if len(group) < 50:
        print(
            f"Categoria {cat!r} possui somente {len(group)} linhas "
            "e não pode fornecer uma amostra de 50."
        )
        continue

    # Seleciona aleatoriamente 50 índices da categoria
    selected_indices = rng.choice(
        group.index.to_numpy(),
        size=50,
        replace=False,
    )

    # Marca as 50 linhas selecionadas
    df.loc[selected_indices, "50_class_selected"] = True

    # Os índices já estão aleatoriamente embaralhados
    marcelo_indices = selected_indices[:17]
    leonardo_indices = selected_indices[17:34]
    adrian_indices = selected_indices[34:50]

    df.loc[marcelo_indices, "Evaluator"] = "Marcelo"
    df.loc[leonardo_indices, "Evaluator"] = "Leonardo"
    df.loc[adrian_indices, "Evaluator"] = "Adrian"




df["AI_Usage_Type"].value_counts()


df[["50_class_selected", "Evaluator"]].value_counts()


print(df.iloc[50:70, -3:])



df_ai_50 = df[df["50_class_selected"]].reset_index(drop=False).sort_values("Evaluator").reset_index(drop=True)
df_ai_50 = df_ai_50.drop("50_class_selected", axis=1)

df_ai_50["Evaluator_Validation"] = "-"


print(df_ai_50.iloc[:, -3:])

df_ai_50.iloc[:, [0, 1, -3, -2, -1]]



df_ai_50[["AI_Usage_Type", "Evaluator"]].value_counts().sort_index()


df_ai_50.to_csv("/home/marcelo/Desktop/df_ai_50.csv", index=False)

"""
Boa noite, pessoal

Como combinado hoje a tarde, a partir da classificação feita pelo Lucas (ICMC), eu selecionei 50 artigos aleatoriamente de cada uma das 5 classes.
Cada 50 artigos eu particionei em 50 = 16 (Adriano) + 17 (Leonardo) + 17 (Marcelo), como mostra abaixo:

[imagem das distribuições por categoria]

Eu gerei um novo cvs chamado "df_ai_50.csv" para facilitar o preenchimento, então são 250 linhas. Como referência, temos as colunas:
index: posição do artigo no csv "df_scopus_and_ieee_answer_AI.csv"
AI_Usage_Type: Classificação da LLM feita pelo Lucas (ICMC)
Reviewer: Quem vai revisar e classificar
Reviewer_Validation: a classificação do revisor

[Imagem do parte do arquivo csv ]


Lembrando que a tarefa é: a partir dos metadados, classificar cada artigo indicado nas seguintes
classes: GenAI, Foundation Models, Both, None, Cannot determine.

link: (link do df_ai_50.csv)

Atenciosamente,
Marcelo


"""



