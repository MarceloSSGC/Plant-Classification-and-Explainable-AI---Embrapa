import os

def gerar_arvore(diretorio, arquivo_saida="arvore.txt"):
    def construir_arvore(caminho, prefixo=""):
        itens = sorted(os.listdir(caminho))
        resultado = []

        for i, item in enumerate(itens):
            caminho_completo = os.path.join(caminho, item)
            ultimo = (i == len(itens) - 1)

            if ultimo:
                conector = "└── "
                novo_prefixo = prefixo + "    "
            else:
                conector = "├── "
                novo_prefixo = prefixo + "│   "

            resultado.append(prefixo + conector + item)

            if os.path.isdir(caminho_completo):
                resultado.extend(construir_arvore(caminho_completo, novo_prefixo))

        return resultado

    # Gera a árvore
    arvore = [diretorio]
    arvore.extend(construir_arvore(diretorio))

    # Salva no arquivo txt
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write("\n".join(arvore))

    return arquivo_saida


# dir = "/media/marcelo/HD_8t/Marcelo__Seagate_8tb/Embrapa/SSH_2/PlantaDaninha_BoaVista/2019_9_17_Agua_Boa/"