import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def plot_original_and_aligned(
    orgn_dir: str,
    algn_dir: str,
    img_name: str,
    bands: Tuple[int, int, int] = (3, 2, 1),
) -> None:
    """
    Exibe uma composição de três bandas da imagem original e da imagem
    alinhada, lado a lado.

    A ordem das bandas em `bands` corresponde aos canais RGB:

        bands[0] -> vermelho
        bands[1] -> verde
        bands[2] -> azul

    Por exemplo:
        bands=(3, 2, 1)

    Carrega:
        IMG_0003_3.tif como canal vermelho
        IMG_0003_2.tif como canal verde
        IMG_0003_1.tif como canal azul

    Parameters
    ----------
    orgn_dir : str
        Diretório que contém as bandas originais.

    algn_dir : str
        Diretório que contém as bandas alinhadas.

    img_name : str
        Nome-base da imagem, por exemplo "IMG_0003".

    bands : tuple[int, int, int]
        Três bandas utilizadas para construir a composição RGB.

    Raises
    ------
    ValueError
        Se `bands` não contiver exatamente três bandas ou se as bandas
        de uma mesma composição tiverem dimensões diferentes.

    FileNotFoundError
        Se algum dos arquivos TIFF não for encontrado.
    """

    if len(bands) != 3:
        raise ValueError(
            f"'bands' deve conter exatamente 3 bandas, mas recebeu {bands}."
        )

    orgn_path = Path(orgn_dir)
    algn_path = Path(algn_dir)

    def find_band_file(directory: Path, band: int) -> Path:
        """
        Procura o arquivo da banda aceitando extensões .tif e .tiff,
        independentemente de maiúsculas e minúsculas.
        """
        expected_stem = f"{img_name}_{band}".lower()

        matches = [
            file_path
            for file_path in directory.iterdir()
            if file_path.is_file()
            and file_path.stem.lower() == expected_stem
            and file_path.suffix.lower() in {".tif", ".tiff"}
        ]

        if not matches:
            raise FileNotFoundError(
                f"Arquivo da banda {band} não encontrado em '{directory}'. "
                f"Nome esperado: '{img_name}_{band}.tif'."
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Mais de um arquivo foi encontrado para a banda {band} "
                f"em '{directory}': {matches}"
            )

        return matches[0]

    def normalize_band(
        band_array: np.ndarray,
        lower_percentile: float = 2.0,
        upper_percentile: float = 98.0,
    ) -> np.ndarray:
        """
        Normaliza uma banda para o intervalo [0, 1] usando percentis.
        """
        band_array = np.asarray(band_array, dtype=np.float32)

        valid_pixels = band_array[np.isfinite(band_array)]

        if valid_pixels.size == 0:
            return np.zeros_like(band_array, dtype=np.float32)

        lower = np.percentile(valid_pixels, lower_percentile)
        upper = np.percentile(valid_pixels, upper_percentile)

        if np.isclose(lower, upper):
            minimum = valid_pixels.min()
            maximum = valid_pixels.max()

            if np.isclose(minimum, maximum):
                return np.zeros_like(band_array, dtype=np.float32)

            lower = minimum
            upper = maximum

        normalized = (band_array - lower) / (upper - lower)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Substitui NaN ou infinito por zero.
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        return normalized

    def load_rgb_composite(directory: Path) -> np.ndarray:
        """
        Carrega três bandas e cria uma composição RGB.
        """
        loaded_bands = []

        for band_number in bands:
            band_file = find_band_file(directory, band_number)
            band_array = tifffile.imread(band_file)

            # Remove dimensões unitárias, como (1, altura, largura).
            band_array = np.squeeze(band_array)

            if band_array.ndim != 2:
                raise ValueError(
                    f"A banda '{band_file}' deveria ser uma matriz 2D, "
                    f"mas possui shape {band_array.shape}."
                )

            loaded_bands.append(band_array)

        shapes = [band_array.shape for band_array in loaded_bands]

        if len(set(shapes)) != 1:
            raise ValueError(
                f"As bandas em '{directory}' possuem dimensões diferentes: "
                f"{shapes}."
            )

        normalized_bands = [
            normalize_band(band_array) for band_array in loaded_bands
        ]

        return np.stack(normalized_bands, axis=-1)

    original_rgb = load_rgb_composite(orgn_path)
    aligned_rgb = load_rgb_composite(algn_path)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 7),
        constrained_layout=True,
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title(
        f"Original — {img_name}\n"
        f"RGB = bandas {bands[0]}, {bands[1]}, {bands[2]}"
    )
    axes[0].axis("off")

    axes[1].imshow(aligned_rgb)
    axes[1].set_title(
        f"Alinhada — {img_name}\n"
        f"RGB = bandas {bands[0]}, {bands[1]}, {bands[2]}"
    )
    axes[1].axis("off")

    figure.suptitle(
        "Comparação entre as bandas originais e alinhadas",
        fontsize=15,
    )

    plt.show()

#======================================================================

DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets"
species = sorted(os.listdir(DATA_DIR + f"/PlantaDaninha_BoaVista"))


specie = "36_Unha_de_gato_Serra_da_Prata_06"


files = sorted(os.listdir(DATA_DIR + f"/PlantaDaninha_BoaVista_Aligned_ecc_affine/{specie}"))
files = sorted(list(set([x[:-6] for x in files if 'tif' in x])))


img_name = files[0]
img_name = "IMG_0095"

print(f'files:   {len(files)}')
print(f'img_name:   {img_name}')


orgn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista/{specie}"
# algn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista_Aligned/{specie}"
algn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista_Aligned_ecc_affine/{specie}"


plot_original_and_aligned(
    orgn_dir=orgn_dir,
    algn_dir=algn_dir,
    img_name=img_name,
    bands=(3, 2, 1),
)


plot_original_and_aligned(
    orgn_dir=orgn_dir,
    algn_dir=algn_dir,
    img_name=img_name,
    bands=(3, 2, 4),
)

#======================================================================


for specie in species:

    print("\n" + "="*60 + f"\n specie: {specie}")

    files = sorted(os.listdir(DATA_DIR + f"/PlantaDaninha_BoaVista/{specie}"))
    files = sorted(list(set([x[:-6] for x in files if 'tif' in x])))

    n = 8 
    img_name = files[n] if len(files) > n else files[len(files)-1]

    print(f'files:   {len(files)}')
    print(f'img_name:   {img_name}')


    orgn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista/{specie}"
    algn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista_Aligned/{specie}"


    plot_original_and_aligned(
        orgn_dir=orgn_dir,
        algn_dir=algn_dir,
        img_name=img_name,
        bands=(3, 2, 1),
    )

#======================================================================


DATA_DIR = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/Embrapa_Experimentos/Datasets/"
species = sorted(os.listdir(DATA_DIR + f"/PlantaDaninha_BoaVista"))

specie = "27_Mavuno_Paludo_08"

# files = sorted(os.listdir(DATA_DIR + f"/PlantaDaninha_BoaVista_Aligned_ecc_affine/{specie}"))
files = sorted(os.listdir(DATA_DIR + f"/PlantaDaninha_BoaVista/{specie}"))
files = sorted(list(set([x[:-6] for x in files if 'tif' in x])))

for jth, img_name in enumerate(files):  #  jth, img_name = 0, files[0]
   
    print(f'jth:   {jth} - {len(files)} \t', end="    ")
    print(f'img_name:   {img_name} \t {specie} ')

    orgn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista/{specie}"
    algn_dir = DATA_DIR + f"/PlantaDaninha_BoaVista_Aligned_ecc_affine/{specie}"


    plot_original_and_aligned(
        orgn_dir=orgn_dir,
        algn_dir=algn_dir,
        img_name=img_name,
        bands=(3, 2, 1),
    )

    # print(f"(3, 4, 5)")
    plot_original_and_aligned(
        orgn_dir=orgn_dir,
        algn_dir=algn_dir,
        img_name=img_name,
        bands=(3, 4, 5),
    )
    # print("+"*80 + "\n")


#------------------------------------------------------------
#------------------------------------------------------------
# Some Defects

# 02_Vassourinha_botao_Agua_Boa_02
# (3, 2, 1)
# 8 - 33 	    img_name:   IMG_0008
# 21 - 33 	    img_name:   IMG_0021
# 

# (3, 4, 5)
# 8 - 33 	    img_name:   IMG_0008
# 15 - 33 	    img_name:   IMG_0015
# 16 - 33 	    img_name:   IMG_0016
# 21 - 33 	    img_name:   IMG_0021
# 25 - 33 	    img_name:   IMG_0026
# 
#---------------------------------
# "03_brizantha_Agua_Boa_03"
# (3, 4, 5)
# 73 - 112 	    img_name:   IMG_0133
# 
#---------------------------------
# 07_capim_capeta_Agua_Boa_07
# (3, 2, 1)
# 11 - 62 	    img_name:   IMG_0096
# 13 - 62 	    img_name:   IMG_0098    pequeno
# 15 - 62 	    img_name:   IMG_0100
# 
# (3, 4, 5)
# 10 - 62 	    img_name:   IMG_0095
# 11 - 62 	    img_name:   IMG_0096
# 14 - 62 	    img_name:   IMG_0099    Bastante
# 15 - 62 	    img_name:   IMG_0100
# 16 - 62 	    img_name:   IMG_0102    pequeno
# 24 - 62 	    img_name:   IMG_0110

#---------------------------------
# 09_pe_galinha_Agua_Boa_09
# (3, 2, 1)
# 12 - 18 	    img_name:   IMG_0017    pequeno
# 
# (3, 4, 5)
# 12 - 18 	    img_name:   IMG_0017
# 14 - 18 	    img_name:   IMG_0019

#---------------------------------
# 11_apaga_fogo_Agua_Boa_11
# (3, 2, 1) 
# 19 - 38 	    img_name:   IMG_0072    pequeno
# 25 - 38 	    img_name:   IMG_0078    pequeno
# 
# (3, 4, 5)
# 21 - 38 	    img_name:   IMG_0074    Bastante
# 25 - 38 	    img_name:   IMG_0078    Bastante
# 28 - 38 	    img_name:   IMG_0081
# 33 - 38 	    img_name:   IMG_0086    pequeno

#---------------------------------
# 12_Andropogon_Agua_Boa_12
# (3, 2, 1)
# 10 - 30 	    img_name:   IMG_0102
# 
# (3, 4, 5)
# 19 - 30 	    img_name:   IMG_0111    bem pequeno
# 26 - 30 	    img_name:   IMG_0118    Bastante

#---------------------------------
# 13_Traquipoon_Agua_Boa_13
# (3, 2, 1)
# 17 - 26 	    img_name:   IMG_0141
# 
# (3, 4, 5)
# 17 - 26 	    img_name:   IMG_0141    Bastante MUITTO
# 
#---------------------------------
# 14_Jaragua_Agua_Boa_14
# (3, 4, 5)
# 29 - 42 	    img_name:   IMG_0044

#---------------------------------
# 16_Massai_Agua_Boa_16
# (3, 2, 1)
# 13 - 118 	    img_name:   IMG_0016

# (3, 4, 5)
# 38 - 118 	    img_name:   IMG_0048
# 43 - 118 	    img_name:   IMG_0053 
# 67 - 118 	    img_name:   IMG_2516

#---------------------------------
# 17_Ruziziensis_Agua_Boa_17
# (3, 4, 5)
# 11 - 22 	    img_name:   IMG_0058

#---------------------------------
# 24_Mombaça_Paludo_05
# (3, 2, 1)
# 1 - 48 	    img_name:   IMG_0046    pequeno
# 

#---------------------------------
# 26_Calapogonio_Paludo_07
# (3, 2, 1)
# 24 - 33 	    img_name:   IMG_0093
# 29 - 33 	    img_name:   IMG_0098    Bastante MUITO
# 30 - 33 	    img_name:   IMG_0099

# (3, 4, 5)
# 5 - 33 	    img_name:   IMG_0072    Bem pouco
# 8 - 33 	    img_name:   IMG_0075    Bem pouco
# 29 - 33 	    img_name:   IMG_0098
# 30 - 33 	    img_name:   IMG_0099

#---------------------------------
# 27_Mavuno_Paludo_08
# (3, 2, 1)
# 33 - 115 	    img_name:   IMG_0137
# 34 - 115 	    img_name:   IMG_0138
# 35 - 115 	    img_name:   IMG_0139
# 41 - 115 	    img_name:   IMG_0145
# 80 - 115 	    img_name:   IMG_0185    Bem pouco
# 89 - 115 	    img_name:   IMG_0194    Bem pouco
# 92 - 115 	    img_name:   IMG_0197
# 112 - 115 	    img_name:   IMG_0219    Bastante

# (3, 4, 5)
# 33 - 115 	    img_name:   IMG_0137 
# 35 - 115 	    img_name:   IMG_0139
# 37 - 115 	    img_name:   IMG_0141
# 41 - 115 	    img_name:   IMG_0145
# 80 - 115 	    img_name:   IMG_0185
# 95 - 115 	    img_name:   IMG_0200    bem pouco
# 107 - 115 	    img_name:   IMG_0214
# 112 - 115 	    img_name:   IMG_0219    Bastante

#---------------------------------
# 28_Corda_de_viola_Paludo_09
# (3, 2, 1)
# 5 - 7 	    img_name:   IMG_0229 

# (3, 4, 5)
# 2 - 7 	    img_name:   IMG_0226    pouco
# 5 - 7 	    img_name:   IMG_0229
# 6 - 7 	    img_name:   IMG_0230

#---------------------------------
# 29_Paiaguas_Paludo_10
# (3, 2, 1)
# 20 - 57 	    img_name:   IMG_0024    pouco
# 21 - 57 	    img_name:   IMG_0025    pouco
# 37 - 57 	    img_name:   IMG_0042    
# 38 - 57 	    img_name:   IMG_0043
# 50 - 57 	    img_name:   IMG_0055    bem pouco

# (3, 4, 5)
# 20 - 57 	    img_name:   IMG_0024
# 24 - 57 	    img_name:   IMG_0028    bem pouco
# 25 - 57 	    img_name:   IMG_0029    bem pouco
# 28 - 57 	    img_name:   IMG_0032
# 29 - 57 	    img_name:   IMG_0033    pouco
# 37 - 57 	    img_name:   IMG_0042
# 38 - 57 	    img_name:   IMG_0043
# 50 - 57 	    img_name:   IMG_0055

#---------------------------------
# 31_Cipo_Serra_da_Prata_02
# (3, 2, 1)
# 26 - 48 	    img_name:   IMG_0084

#---------------------------------
# 32_Jurubebinha_Serra_da_Prata_03
# (3, 2, 1)
# 37 - 45 	    img_name:   IMG_0149

# (3, 4, 5)
# 36 - 45 	    img_name:   IMG_0148
# 37 - 45 	    img_name:   IMG_0149
# 39 - 45 	    img_name:   IMG_0151

#---------------------------------
# 35_Chumbinho_Serra_da_Prata_05
# (3, 2, 1)
# 28 - 30 	    img_name:   IMG_0063

# (3, 4, 5)
# 8 - 30 	    img_name:   IMG_0041
# 18 - 30 	    img_name:   IMG_0051
# 19 - 30 	    img_name:   IMG_0052
# 27 - 30 	    img_name:   IMG_0062

#---------------------------------
# 36_Unha_de_gato_Serra_da_Prata_06
# (3, 4, 5)
# 13 - 46 	    img_name:   IMG_0082    Bem pouco
# 25 - 46 	    img_name:   IMG_0094
# 30 - 46 	    img_name:   IMG_0099
# 37 - 46 	    img_name:   IMG_0106
# 44 - 46 	    img_name:   IMG_0113

#------------------------------------------------------------
#------------------------------------------------------------
# specie = "06_capim_navalha_Agua_Boa_06"
# As 3/4 primeiras tem 05_Salsa_Agua_Boa_05
# 
# 07_capim_capeta_Agua_Boa_07
# jth:   5 - 62 	    img_name:   IMG_0090 Sombra
# 
# 28 - 62   IMG_0139 IMG_0140 IMG_0141 sao iguais

# 11_apaga_fogo_Agua_Boa_11
# Alinhamento nao ficou bom no geral
# 
# 27_Mavuno_Paludo_08   Algumas podem ser iguais
# 
# 
# 30_Inaja_Serra_da_Prata_01
# 7 - 56 	    img_name:   IMG_0009    pouca amostra


# 36_Unha_de_gato_Serra_da_Prata_06
# 30 - 46 	    img_name:   IMG_0099    Parece ser de outra especie
# 31 - 46 	    img_name:   IMG_0100    Parece ser de outra especie
