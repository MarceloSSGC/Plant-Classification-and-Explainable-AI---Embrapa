import os
import numpy as np

#======================================================================

data_dir_01 = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/"


especies = os.listdir(data_dir_01)


#======================================================================

import re

#-----------------------------------------------------------
# 1. Existem mesmo 5 arquivos .tif para IMG_0015?

pasta = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/01_malva_branca_Agua_Boa_01/"

arquivos = os.listdir(pasta)

arquivos_img_0015 = sorted([
    arq for arq in arquivos
    if arq.startswith("IMG_0015")
])

print(arquivos_img_0015)

# Tamanhos

for arq in arquivos_img_0015:
    caminho = os.path.join(pasta, arq)
    print(arq, "| tamanho:", round(os.path.getsize(caminho)/1024, 2), "Kylobytes")


bandas_tif = []
xml_auxiliares = []
outros = []

for arq in arquivos_img_0015:
    if re.match(r"IMG_0015_[1-5]\.tif$", arq, re.IGNORECASE):
        bandas_tif.append(arq)
    elif re.match(r"IMG_0015_[1-5]\.tif\.aux\.xml$", arq, re.IGNORECASE):
        xml_auxiliares.append(arq)
    else:
        outros.append(arq)

print("Bandas TIF:", bandas_tif)
print("XML auxiliares:", xml_auxiliares)
print("Outros:", outros)


resumo_1 = """
Existe apenas um arquivo auxiliar XML associado à banda 5.
"""

#-----------------------------------------------------------
# 2. Cada .tif representa uma banda?

print("Número de bandas .tif:", len(bandas_tif))
print("Tem 5 bandas?", len(bandas_tif) == 5)

bandas_encontradas = []

for arq in bandas_tif:
    m = re.search(r"_(\d)\.tif$", arq)
    if m:
        bandas_encontradas.append(int(m.group(1)))

print("Bandas encontradas:", sorted(bandas_encontradas))
print("Bandas esperadas:", [1, 2, 3, 4, 5])
print("Está completo?", sorted(bandas_encontradas) == [1, 2, 3, 4, 5])

resumo_2 = """
Todas as bandas possuem praticamente o mesmo tamanho:
mesma resolução
mesma profundidade
mesma dimensão espacial

O XML é minúsculo → claramente apenas metadado auxiliar.
"""

#-----------------------------------------------------------
# 3. O XML é apenas auxiliar?

if len(xml_auxiliares) > 0:
    xml = xml_auxiliares[0]
    caminho_xml = os.path.join(pasta, xml)

    with open(caminho_xml, "r", encoding="utf-8", errors="ignore") as f:
        conteudo = f.read()

    print(conteudo[:3000])
else:
    print("Nenhum XML auxiliar encontrado.")


termos = [
    "STATISTICS",
    "MINIMUM",
    "MAXIMUM",
    "MEAN",
    "STDDEV",
    "WAVELENGTH",
    "BAND",
    "NIR",
    "RED",
    "GREEN",
    "BLUE"
]

if len(xml_auxiliares) > 0:
    for termo in termos:
        if termo.lower() in conteudo.lower():
            print("Encontrado:", termo)

resumo_3 = """
Não contém:
    comprimento de onda
    nome das bandas
    informação espectral física

o XML não resolve ainda quais bandas são RGB/NIR/etc
"""

#-----------------------------------------------------------
# 4. Todas as bandas têm mesmo tamanho?
# 5. Todas têm o mesmo tipo numérico?

import rasterio

for arq in bandas_tif:
    caminho = os.path.join(pasta, arq)

    with rasterio.open(caminho) as src:
        print("\nArquivo:", arq)
        print("Largura:", src.width)
        print("Altura:", src.height)
        print("Número de bandas internas:", src.count)
        print("Tipo dos pixels:", src.dtypes)
        print("CRS:", src.crs)
        print("Nodata:", src.nodata)
        print("Transform:", src.transform)

resumo_4_5 = """
Dimensão = 1280 x 960

pixel -> [b1,b2,b3,b4,b5]  que é exatamente o vetor espectral.

Número de bandas internas: 1
Cada TIFF contém: uma única banda
5 arquivos TIFF separados = 5 bandas espectrais
Tipo dos pixels: uint16

Os pixels usam: inteiros sem sinal de 16 bits
Faixa possível: 0 → 65535

Os dados:

    NÃO são imagens RGB comuns
    NÃO são imagens 8 bits tradicionais

CRS: None  -> o georreferenciamento foi removido

Transform identidade
| 1 0 0 |
| 0 1 0 |       Não existe transformação espacial relevante.
| 0 0 1 |

as imagens são apenas matrizes raster e não mapas geográficos.
"""

#-----------------------------------------------------------
# 6. Os valores parecem coerentes?

dimensoes = []

for arq in bandas_tif:
    caminho = os.path.join(pasta, arq)

    with rasterio.open(caminho) as src:
        dimensoes.append((arq, src.width, src.height, src.count, src.dtypes[0]))

for item in dimensoes:
    print(item)

print("\nDimensões únicas:")
print(set((largura, altura) for _, largura, altura, _, _ in dimensoes))

print("\nTipos únicos:")
print(set(dtype for _, _, _, _, dtype in dimensoes))


resumo_6 = """
Dimensões únicas:
{(1280, 960)}

Tipos únicos:
{'uint16'}


O que podemos afirmar

O dataset é extremamente consistente.

Isso é excelente para:

empilhamento espectral
PCA
classificação
CNNs
índices espectrais

Não será necessário:

redimensionar
alinhar
converter tipo
"""

#-----------------------------------------------------------
# 7. O XML contém estatísticas ou informação espectral?

import numpy as np

for arq in bandas_tif:
    caminho = os.path.join(pasta, arq)

    with rasterio.open(caminho) as src:
        img = src.read(1)

    print("\nArquivo:", arq)
    print("Shape:", img.shape)
    print("Mínimo:", np.nanmin(img))
    print("Máximo:", np.nanmax(img))
    print("Média:", np.nanmean(img))
    print("Desvio padrão:", np.nanstd(img))


resumo_7 = """
O que podemos afirmar

As bandas capturam respostas espectrais diferentes.

Isso é EXATAMENTE o esperado em dados multiespectrais.

Especialmente:

bandas 4 e 5 possuem resposta muito maior

Isso pode indicar:

red edge
NIR
forte refletância da vegetação

O que faz bastante sentido biologicamente.

"""

#======================================================================
#======================================================================
# Passo 1

# 1. Carregar as 5 bandas
"""
Aqui vamos ler as cinco bandas e empilhar em um cubo multiespectral.
"""

import matplotlib.pyplot as plt

pasta = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/01_malva_branca_Agua_Boa_01/"

base = "IMG_0015"

arquivos_bandas = [
    f"{base}_1.tif",
    f"{base}_2.tif",
    f"{base}_3.tif",
    f"{base}_4.tif",
    f"{base}_5.tif"
]

bandas = []

for arq in arquivos_bandas:
    caminho = os.path.join(pasta, arq)

    with rasterio.open(caminho) as src:
        img = src.read(1)

    bandas.append(img)

cubo = np.stack(bandas, axis=-1)

print("Shape do cubo:", cubo.shape)
print("Tipo dos dados:", cubo.dtype)

#-----------------------------------------------------------
# 2. Plotagem de cada banda
"""
Aqui vamos visualizar cada banda separadamente para observar contraste,
textura, saturação e diferenças visuais.
"""

plt.figure(figsize=(18, 6))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(cubo[:, :, i], cmap="gray")
    plt.title(f"Banda {i+1}")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# - A planta aparece bem em todas as bandas?
# - Alguma banda destaca melhor a vegetação?
# - Alguma banda parece saturada?
# - O fundo aparece muito forte?

#-----------------------------------------------------------
# 3. Plotagem com contraste ajustado

"""
Como os dados são uint16, a escala pode dificultar a visualização.
Aqui usamos percentis para melhorar o contraste visual.
"""

plt.figure(figsize=(18, 6))

for i in range(5):
    banda = cubo[:, :, i]

    vmin = np.percentile(banda, 2)
    vmax = np.percentile(banda, 98)

    plt.subplot(1, 5, i + 1)
    plt.imshow(banda, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"Banda {i+1}\nP2-P98")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------
# 4. Comparar histogramas das bandas
"""
Aqui vamos comparar a distribuição dos valores de pixel em cada banda.
"""

plt.figure(figsize=(10, 6))

for i in range(5):
    banda = cubo[:, :, i].ravel()
    plt.hist(banda, bins=100, alpha=0.4, label=f"Banda {i+1}")

plt.xlabel("Valor do pixel")
plt.ylabel("Frequência")
plt.title("Histogramas das bandas")
plt.legend()
plt.grid(True)
plt.show()

# - Bandas com valores mais altos
# - Bandas com maior dispersão
# - Possível saturação perto de 65535
# - Diferença entre fundo e planta

#-----------------------------------------------------------
# 5. Histogramas individuais

"""
Aqui vamos visualizar cada histograma separadamente, o que facilita 
ver detalhes de cada banda.
"""

for i in range(5):
    banda = cubo[:, :, i].ravel()

    plt.figure(figsize=(8, 4))
    plt.hist(banda, bins=100)
    plt.xlabel("Valor do pixel")
    plt.ylabel("Frequência")
    plt.title(f"Histograma - Banda {i+1}")
    plt.grid(True)
    plt.show()


#-----------------------------------------------------------
# 6. Verificar saturação

"""
Aqui vamos medir quantos pixels estão próximos do valor máximo possível 
de uint16.
"""

limite_saturacao = 65000

for i in range(5):
    banda = cubo[:, :, i]

    n_total = banda.size
    n_saturados = np.sum(banda >= limite_saturacao)
    perc_saturados = 100 * n_saturados / n_total

    print(f"Banda {i+1}")
    print(f"Pixels >= {limite_saturacao}: {n_saturados}")
    print(f"Percentual saturado: {perc_saturados:.4f}%")
    print()


# Se houver muitos pixels saturados, a interpretação espectral fica menos 
# confiável naquela banda/região.

#-----------------------------------------------------------
# 7. Composição RGB aproximada

"""
Aqui vamos tentar montar uma imagem RGB usando as bandas 1, 2 e 3.

Como ainda não sabemos quais bandas são azul, verde e vermelho, isso é apenas 
exploratório.
"""

def normalizar_para_rgb(img):
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)

    img_norm = (img - p2) / (p98 - p2)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm


R = normalizar_para_rgb(cubo[:, :, 3-1])
G = normalizar_para_rgb(cubo[:, :, 2-1])
B = normalizar_para_rgb(cubo[:, :, 1-1])

rgb = np.dstack([R, G, B])

plt.figure(figsize=(8, 6))
plt.imshow(rgb)
plt.title("Composição RGB aproximada: R=B3, G=B2, B=B1")
plt.axis("off")
plt.show()

#-----------------------------------------------------------
# 8. Composição falsa-cor

"""
Aqui vamos usar a banda 5 como canal vermelho. Se a banda 5 for NIR, a 
vegetação tende a ficar bem destacada.
"""

R = normalizar_para_rgb(cubo[:, :, 5-1])
G = normalizar_para_rgb(cubo[:, :, 3-1])
B = normalizar_para_rgb(cubo[:, :, 2-1])

falsa_cor = np.dstack([R, G, B])

plt.figure(figsize=(8, 6))
plt.imshow(falsa_cor)
plt.title("Falsa-cor exploratória: R=B5, G=B3, B=B2")
plt.axis("off")
plt.show()


# - A planta se destaca melhor?
# - O fundo fica mais separado?
# - Há regiões muito brilhantes?


#-----------------------------------------------------------
# 9. Assinatura espectral média da imagem inteira

"""
Aqui vamos calcular a média de cada banda usando todos os pixels da 
imagem. Atenção: ainda inclui fundo, sombra e solo. Portanto, não é 
a assinatura real da planta.
"""

medias = []

for i in range(5):
    medias.append(np.mean(cubo[:, :, i]))

plt.figure(figsize=(7, 5))
plt.plot([1, 2, 3, 4, 5], medias, marker="o")
plt.xlabel("Banda")
plt.ylabel("Valor médio")
plt.title("Assinatura média da imagem inteira")
plt.grid(True)
plt.show()

print("Médias por banda:", medias)

#-----------------------------------------------------------
# 10. Assinatura espectral de pixels escolhidos manualmente

pontos = {
    "ponto_1": (480, 640),
    "ponto_2": (300, 500),
    "ponto_3": (600, 700)
}

plt.figure(figsize=(7, 5))

for nome, (linha, coluna) in pontos.items():
    assinatura = cubo[linha, coluna, :]
    plt.plot([1, 2, 3, 4, 5], assinatura, marker="o", label=nome)

plt.xlabel("Banda")
plt.ylabel("Valor do pixel")
plt.title("Assinaturas espectrais de pixels individuais")
plt.legend()
plt.grid(True)
plt.show()

#-----------------------------------------------------------
# 11. Visualizar onde estão os pontos escolhidos

"""
Aqui vamos mostrar os pontos sobre uma das bandas ou sobre a falsa-cor.
"""

plt.figure(figsize=(8, 6))
plt.imshow(falsa_cor)

for nome, (linha, coluna) in pontos.items():
    plt.scatter(coluna, linha, s=80, label=nome)
    plt.text(coluna + 10, linha + 10, nome)

plt.title("Pontos escolhidos sobre a falsa-cor")
plt.axis("off")
plt.legend()
plt.show()

# Se algum ponto cair no fundo, ajuste as coordenadas.

#-----------------------------------------------------------
# 12. Primeira conclusão do Passo 1

# 1. As cinco bandas são visualmente diferentes?
# 2. A planta aparece melhor em quais bandas?
# 3. Há saturação relevante?
# 4. A falsa-cor destaca melhor a vegetação?
# 5. As bandas 4 e 5 parecem ter resposta maior?
# 6. Pixels de planta e fundo parecem ter assinaturas diferentes?



























