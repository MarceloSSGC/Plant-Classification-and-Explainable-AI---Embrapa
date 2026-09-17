# python diagnosticar_datasets_npy.py \
#   --dataset-a '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm/01_malva_branca_Agua_Boa_01'
#  \
#   --dataset-c '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB__SEED_20__T_0.75_V_0.15__AUG/Train_Norm/01_malva_branca_Agua_Boa_01'


# '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm/01_malva_branca_Agua_Boa_01'

# '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB__SEED_20__T_0.75_V_0.15__AUG/Train_Norm/01_malva_branca_Agua_Boa_01'


#======================================================================
#======================================================================


from pathlib import Path
from collections import Counter, defaultdict
import time

import numpy as np


def analisar_dataset(diretorio):
    """Analisa classes/*.npy e imprime estrutura e desempenho de leitura."""

    t_0 = time.time()

    raiz = Path(diretorio)
    if not raiz.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {raiz}")

    classes = sorted(p for p in raiz.iterdir() if p.is_dir())
    arquivos_por_classe = {
        pasta.name: sorted(pasta.glob("*.npy")) for pasta in classes
    }
    arquivos = [a for lista in arquivos_por_classe.values() for a in lista]
    if not arquivos:
        raise RuntimeError(f"Nenhum arquivo .npy encontrado em {raiz}")

    shapes = Counter()
    dtypes = Counter()
    ordens = Counter()
    tamanhos_por_shape = defaultdict(list)
    tamanhos = []
    erros = []

    t0 = time.perf_counter()
    for arquivo in arquivos:
        tamanho = arquivo.stat().st_size
        tamanhos.append(tamanho)
        try:
            imagem = np.load(arquivo, mmap_mode="r", allow_pickle=False)
            shape = tuple(imagem.shape)
            shapes[shape] += 1
            dtypes[str(imagem.dtype)] += 1
            tamanhos_por_shape[shape].append(tamanho)
            if imagem.flags.c_contiguous:
                ordens["C-contiguous"] += 1
            elif imagem.flags.f_contiguous:
                ordens["Fortran-contiguous"] += 1
            else:
                ordens["não contíguo"] += 1
        except Exception as erro:
            erros.append((arquivo, repr(erro)))
    tempo_cabecalhos = time.perf_counter() - t0

    # Reproduz a operação principal do __getitem__ informado.
    t0 = time.perf_counter()
    total_elementos = 0
    for arquivo in arquivos:
        imagem = np.load(arquivo, allow_pickle=False).astype(np.float32)
        total_elementos += imagem.size
        del imagem
    tempo_leitura = time.perf_counter() - t0
    total_bytes = sum(tamanhos)

    print("=" * 72)
    print(f"DATASET: {raiz.resolve()}")
    print("=" * 72)
    print(f"Número de classes:          {len(classes)}")
    print(f"Número de arquivos .npy:    {len(arquivos)}")
    print(f"Tamanho total:              {total_bytes / 2**30:.3f} GiB")
    print(f"Tamanho médio por arquivo:  {np.mean(tamanhos) / 2**20:.3f} MiB")
    print(f"Menor arquivo:              {min(tamanhos) / 2**20:.3f} MiB")
    print(f"Maior arquivo:              {max(tamanhos) / 2**20:.3f} MiB")

    print("\nArquivos por classe:")
    for classe, lista in arquivos_por_classe.items():
        print(f"  {classe}: {len(lista)}")

    print("\nShapes encontrados:")
    for shape, quantidade in shapes.items():
        media = np.mean(tamanhos_por_shape[shape]) / 2**20
        print(f"  {shape}: {quantidade} arquivos; média = {media:.3f} MiB")

    print("\nDtypes encontrados:")
    for dtype, quantidade in dtypes.items():
        print(f"  {dtype}: {quantidade} arquivos")

    print("\nOrganização na memória:")
    for ordem, quantidade in ordens.items():
        print(f"  {ordem}: {quantidade} arquivos")

    print("\nDesempenho de leitura:")
    print(f"  Inspeção de cabeçalhos:              {tempo_cabecalhos:.3f} s")
    print(f"  np.load + astype(float32):           {tempo_leitura:.3f} s")
    print(f"  Arquivos por segundo:                {len(arquivos) / tempo_leitura:.2f}")
    print(f"  MiB por segundo:                     {total_bytes / 2**20 / tempo_leitura:.2f}")
    print(f"  Milhões de elementos por segundo:    {total_elementos / 1e6 / tempo_leitura:.2f}")

    if erros:
        print(f"\nERROS DE LEITURA: {len(erros)}")
        for arquivo, erro in erros:
            print(f"  {arquivo}: {erro}")
    else:
        print("\nErros de leitura: nenhum")

    print("\nExecute duas vezes para observar possível efeito de cache.")
    print("=" * 72)

    t_1 = time.time()
    print(f"Tempo Total de execução da função analisar_dataset: {round(t_1 - t_0, 2)} seconds")


# Exemplos:
# analisar_dataset("/caminho/dataset_A/Train")
# analisar_dataset("/caminho/dataset_C/Train")


# RGB + NIR RE
RGB_NIRRE_DIR = '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm'

# RGB
RGB_DIR = '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB__SEED_20__T_0.75_V_0.15__AUG/Train_Norm'

# RGB_entropy
RGB_entropy_DIR = '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_entropy__SEED_20__T_0.75_V_0.15__AUG/Train_Norm'

# RGB_LBP
RGB_LBP_DIR = '/home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_LBP__SEED_20__T_0.75_V_0.15__AUG/Train_Norm'

analisar_dataset(RGB_LBP_DIR)


df -h /home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm
df -h /home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB__SEED_20__T_0.75_V_0.15__AUG/Train_Norm













RGB_out = """
========================================================================
DATASET: /home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB__SEED_20__T_0.75_V_0.15__AUG/Train_Norm
========================================================================
Número de classes:          31
Número de arquivos .npy:    2769
Tamanho total:              38.027 GiB
Tamanho médio por arquivo:  14.063 MiB
Menor arquivo:              14.063 MiB
Maior arquivo:              14.063 MiB

Arquivos por classe:
  01_malva_branca_Agua_Boa_01: 55
  02_Vassourinha_botao_Agua_Boa_02: 96
  03_brizantha_Agua_Boa_03: 168
  04_cipo_fogo_Agua_Boa_04: 72
  05_Salsa_Agua_Boa_05: 75
  06_capim_navalha_Agua_Boa_06: 80
  07_capim_capeta_Agua_Boa_07: 92
  08_malicia_Agua_Boa_08: 60
  09_pe_galinha_Agua_Boa_09: 65
  10_carrapico_Agua_Boa_10: 88
  11_apaga_fogo_Agua_Boa_11: 84
  12_Andropogon_Agua_Boa_12: 88
  13_Traquipoon_Agua_Boa_13: 76
  14_Jaragua_Agua_Boa_14: 93
  15_Quicuio_Agua_Boa_15: 80
  16_Massai_Agua_Boa_16: 176
  17_Ruziziensis_Agua_Boa_17: 80
  20_Guanxuma_Paludo_02: 75
  21_Mata_Pasto_Paludo_03: 80
  23_Braquiarinha_Paludo_04: 93
  24_Mombaça_Paludo_05: 72
  26_Calapogonio_Paludo_07: 96
  27_Mavuno_Paludo_08: 172
  28_Corda_de_viola_Paludo_09: 30
  29_Paiaguas_Paludo_10: 84
  30_Inaja_Serra_da_Prata_01: 84
  31_Cipo_Serra_da_Prata_02: 72
  32_Jurubebinha_Serra_da_Prata_03: 99
  33_Capim_gengibre_Serra_da_Prata_04: 94
  35_Chumbinho_Serra_da_Prata_05: 88
  36_Unha_de_gato_Serra_da_Prata_06: 102

Shapes encontrados:
  (960, 1280, 3): 2769 arquivos; média = 14.063 MiB

Dtypes encontrados:
  float32: 2769 arquivos

Organização na memória:
  C-contiguous: 2769 arquivos

Desempenho de leitura:
  Inspeção de cabeçalhos:              19.931 s
  np.load + astype(float32):           275.585 s
  Arquivos por segundo:                10.05
  MiB por segundo:                     141.30
  Milhões de elementos por segundo:    37.04

Erros de leitura: nenhum

Execute duas vezes para observar possível efeito de cache.
========================================================================
Tempo de execução da função analisar_dataset: 295.54 seconds

"""
# Time: 295.54 seconds


RGB_NIRRE = """
========================================================================
DATASET: /home/u14696181/Documents/Datasets/Embrapa_Experimentos/Datasets/Augmentation/Multiview_Texture__AUG/align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm
========================================================================
Número de classes:          31
Número de arquivos .npy:    2769
Tamanho total:              63.378 GiB
Tamanho médio por arquivo:  23.438 MiB
Menor arquivo:              23.438 MiB
Maior arquivo:              23.438 MiB

Arquivos por classe:
  01_malva_branca_Agua_Boa_01: 55
  02_Vassourinha_botao_Agua_Boa_02: 96
  03_brizantha_Agua_Boa_03: 168
  04_cipo_fogo_Agua_Boa_04: 72
  05_Salsa_Agua_Boa_05: 75
  06_capim_navalha_Agua_Boa_06: 80
  07_capim_capeta_Agua_Boa_07: 92
  08_malicia_Agua_Boa_08: 60
  09_pe_galinha_Agua_Boa_09: 65
  10_carrapico_Agua_Boa_10: 88
  11_apaga_fogo_Agua_Boa_11: 84
  12_Andropogon_Agua_Boa_12: 88
  13_Traquipoon_Agua_Boa_13: 76
  14_Jaragua_Agua_Boa_14: 93
  15_Quicuio_Agua_Boa_15: 80
  16_Massai_Agua_Boa_16: 176
  17_Ruziziensis_Agua_Boa_17: 80
  20_Guanxuma_Paludo_02: 75
  21_Mata_Pasto_Paludo_03: 80
  23_Braquiarinha_Paludo_04: 93
  24_Mombaça_Paludo_05: 72
  26_Calapogonio_Paludo_07: 96
  27_Mavuno_Paludo_08: 172
  28_Corda_de_viola_Paludo_09: 30
  29_Paiaguas_Paludo_10: 84
  30_Inaja_Serra_da_Prata_01: 84
  31_Cipo_Serra_da_Prata_02: 72
  32_Jurubebinha_Serra_da_Prata_03: 99
  33_Capim_gengibre_Serra_da_Prata_04: 94
  35_Chumbinho_Serra_da_Prata_05: 88
  36_Unha_de_gato_Serra_da_Prata_06: 102

Shapes encontrados:
  (960, 1280, 5): 2769 arquivos; média = 23.438 MiB

Dtypes encontrados:
  float32: 2769 arquivos

Organização na memória:
  C-contiguous: 2769 arquivos

Desempenho de leitura:
  Inspeção de cabeçalhos:              0.641 s
  np.load + astype(float32):           22.658 s
  Arquivos por segundo:                122.21
  MiB por segundo:                     2864.23
  Milhões de elementos por segundo:    750.84

Erros de leitura: nenhum

Execute duas vezes para observar possível efeito de cache.
========================================================================
Tempo de execução da função analisar_dataset: 23.32 seconds

"""



