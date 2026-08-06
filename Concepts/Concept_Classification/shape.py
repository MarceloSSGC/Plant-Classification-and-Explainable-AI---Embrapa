import pandas as pd
df_shape = pd.DataFrame()

species = ['01_malva_branca_Agua_Boa_01',
            '02_Vassourinha_botao_Agua_Boa_02',
            '03_brizantha_Agua_Boa_03',
            '04_cipo_fogo_Agua_Boa_04',
            '05_Salsa_Agua_Boa_05',
            '06_capim_navalha_Agua_Boa_06',
            '07_capim_capeta_Agua_Boa_07',
            '08_malicia_Agua_Boa_08',
            '09_pe_galinha_Agua_Boa_09',
            '10_carrapico_Agua_Boa_10',
            '11_apaga_fogo_Agua_Boa_11',
            '12_Andropogon_Agua_Boa_12',
            '13_Traquipoon_Agua_Boa_13',
            '14_Jaragua_Agua_Boa_14',
            '15_Quicuio_Agua_Boa_15',
            '16_Massai_Agua_Boa_16',
            '17_Ruziziensis_Agua_Boa_17',
            '20_Guanxuma_Paludo_02',
            '21_Mata_Pasto_Paludo_03',
            '23_Braquiarinha_Paludo_04',
            '24_Mombaça_Paludo_05',
            '26_Calapogonio_Paludo_07',
            '27_Mavuno_Paludo_08',
            '28_Corda_de_viola_Paludo_09',
            '29_Paiaguas_Paludo_10',
            '30_Inaja_Serra_da_Prata_01',
            '31_Cipo_Serra_da_Prata_02',
            '32_Jurubebinha_Serra_da_Prata_03',
            '33_Capim_gengibre_Serra_da_Prata_04',
            '35_Chumbinho_Serra_da_Prata_05',
            '36_Unha_de_gato_Serra_da_Prata_06']

df_shape["especie"] = species

#======================================================================
# C_11: Folha Estreita e Filiforme / Graminóide

c_11 = [0, 0, 
        1, 0, 
        0, 1,
        1, 0,
        1, 1,
        0, 1,
        1, 1,
        1, 1,
        1, 0,
        0, 1,
        1, 0,
        1, 0,
        1, 0,
        0, 0,
        1, 0,
        0
        ]

print(f"len(c_11) == len(df_shape): {len(c_11) == len(df_shape)}")

df_shape["c_11"] = c_11

#======================================================================
# C_15: Folhagem miúda e densamente distribuída

c_15 = [
    1, 1,
    0, 0,
    0, 0,
    0, 1,
    0, 0,
    1, 0,
    0, 0,
    0, 0,
    0, 1, 
    1, 0,
    0, 0,
    0, 0, 
    0, 0, 
    0, 0, 
    0, 1, 
    0
]

print(f"len(c_15) == len(df_shape): {len(c_15) == len(df_shape)}")

df_shape["c_15"] = c_15

#======================================================================

c_16 = [
    1, 0,
    0, 1,
    0, 0, 
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 1,
    1, 0,
    0, 0,
    0, 0, 
    0, 0,
    1, 1,
    0, 1,
    1
]


print(f"len(c_16) == len(df_shape): {len(c_16) == len(df_shape)}")

df_shape["c_16"] = c_16

#======================================================================
# C_17: Folhas emergem radialmente de um único centro (touceira)

c_17 = [
    0, 0,
    1, 0, 
    0, 1, 
    1, 0,
    0, 0, 
    0, 1, 
    0, 0, 
    0, 1, 
    1, 0, 
    0, 0, 
    0, 0,
    0, 0, 
    0, 1, 
    0, 0, 
    0, 0, 
    0
]

print(f"len(c_17) == len(df_shape): {len(c_17) == len(df_shape)}")

df_shape["c_17"] = c_17

#======================================================================
# C_18: Folhas predominantemente arqueadas ou pendentes

c_18 = [
    0, 0, 
    0, 0, 
    0, 1,
    0, 0,
    1, 1,
    0, 1,
    1, 1, 
    1, 0,
    0, 0, 
    0, 1,
    1, 0,
    1, 0,
    0, 1,
    0, 0,
    1, 0,
    0
]

print(f"len(c_18) == len(df_shape): {len(c_18) == len(df_shape)}")

df_shape["c_18"] = c_18

#======================================================================
#======================================================================

df_shape[(df_shape["c_11"] == 1) & (df_shape["c_15"] == 0) & (df_shape["c_16"] == 0)]

#======================================================================
#======================================================================



df_shape_dir = "/home/marcelo/Documents/VSCode_python/Agro/SIMIDS/Planta_Daninha_Boa_Vista/Concepts/Concept_Classification/"
df_shape.to_csv(f"{df_shape_dir}/df_shape.csv")




