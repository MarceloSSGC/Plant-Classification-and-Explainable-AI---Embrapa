#!/usr/bin/env python3
"""Compara dois datasets .npy equivalentes (A=5 bandas, C=RGB).

O script não treina modelos. Ele verifica estrutura, shape, dtype, ordem de
memória, tamanho em disco, correspondência dos arquivos, igualdade C=A[..., :3]
e velocidade de np.load + conversão para float32.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np


def arquivos_npy(raiz: Path) -> list[Path]:
    return sorted(p for p in raiz.rglob("*.npy") if p.is_file())


def resumo_dataset(raiz: Path, arquivos: list[Path]) -> dict:
    shapes, dtypes, ordens = Counter(), Counter(), Counter()
    total_bytes = 0
    erros = []

    for p in arquivos:
        total_bytes += p.stat().st_size
        try:
            x = np.load(p, mmap_mode="r", allow_pickle=False)
            shapes[str(tuple(x.shape))] += 1
            dtypes[str(x.dtype)] += 1
            ordem = "C" if x.flags.c_contiguous else "F" if x.flags.f_contiguous else "não contíguo"
            ordens[ordem] += 1
        except Exception as exc:
            erros.append({"arquivo": str(p), "erro": repr(exc)})

    tamanhos = [p.stat().st_size for p in arquivos]
    return {
        "raiz": str(raiz),
        "numero_arquivos": len(arquivos),
        "total_GiB": total_bytes / 2**30,
        "arquivo_medio_MiB": (sum(tamanhos) / len(tamanhos) / 2**20) if tamanhos else 0,
        "arquivo_min_MiB": (min(tamanhos) / 2**20) if tamanhos else 0,
        "arquivo_max_MiB": (max(tamanhos) / 2**20) if tamanhos else 0,
        "shapes": dict(shapes),
        "dtypes": dict(dtypes),
        "ordem_memoria": dict(ordens),
        "erros": erros,
    }


def mapear_relativos(raiz: Path, arquivos: list[Path]) -> dict[str, Path]:
    return {str(p.relative_to(raiz)): p for p in arquivos}


def verificar_conteudo(
    raiz_a: Path,
    raiz_c: Path,
    arquivos_a: list[Path],
    arquivos_c: list[Path],
    quantidade: int,
    seed: int,
) -> dict:
    mapa_a = mapear_relativos(raiz_a, arquivos_a)
    mapa_c = mapear_relativos(raiz_c, arquivos_c)
    comuns = sorted(mapa_a.keys() & mapa_c.keys())
    rng = random.Random(seed)
    escolhidos = rng.sample(comuns, min(quantidade, len(comuns)))
    resultados = []

    for relativo in escolhidos:
        a = np.load(mapa_a[relativo], allow_pickle=False)
        c = np.load(mapa_c[relativo], allow_pickle=False)
        shape_compativel = a.ndim == c.ndim == 3 and a.shape[:2] == c.shape[:2] and a.shape[-1] >= 3 and c.shape[-1] == 3
        iguais = bool(np.array_equal(c, a[..., :3])) if shape_compativel else False
        max_abs = float(np.max(np.abs(c.astype(np.float64) - a[..., :3].astype(np.float64)))) if shape_compativel else None
        resultados.append({
            "arquivo_relativo": relativo,
            "shape_A": list(a.shape),
            "shape_C": list(c.shape),
            "dtype_A": str(a.dtype),
            "dtype_C": str(c.dtype),
            "C_igual_A_primeiras_3": iguais,
            "diferenca_max_abs": max_abs,
        })

    return {
        "arquivos_comuns": len(comuns),
        "somente_A": len(mapa_a.keys() - mapa_c.keys()),
        "somente_C": len(mapa_c.keys() - mapa_a.keys()),
        "amostras_verificadas": resultados,
    }


def benchmark(arquivos: list[Path], max_arquivos: int, seed: int) -> dict:
    selecionados = list(arquivos)
    random.Random(seed).shuffle(selecionados)
    if max_arquivos > 0:
        selecionados = selecionados[:max_arquivos]

    inicio = time.perf_counter()
    bytes_disco = 0
    elementos = 0
    for p in selecionados:
        x = np.load(p, allow_pickle=False).astype(np.float32)
        bytes_disco += p.stat().st_size
        elementos += x.size
        del x
    duracao = time.perf_counter() - inicio
    return {
        "arquivos": len(selecionados),
        "segundos": duracao,
        "arquivos_por_segundo": len(selecionados) / duracao if duracao else None,
        "MiB_disco_por_segundo": bytes_disco / 2**20 / duracao if duracao else None,
        "milhoes_elementos_por_segundo": elementos / 1e6 / duracao if duracao else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico comparativo de datasets .npy A e C")
    parser.add_argument("--dataset-a", required=True, type=Path, help="Raiz do dataset A (5 bandas)")
    parser.add_argument("--dataset-c", required=True, type=Path, help="Raiz do dataset C (RGB)")
    parser.add_argument("--max-arquivos", type=int, default=0, help="Limite no benchmark; 0 usa todos")
    parser.add_argument("--verificar-valores", type=int, default=20, help="Pares usados para testar C=A[..., :3]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", type=Path, default=Path("diagnostico_datasets.json"))
    args = parser.parse_args()

    for raiz in (args.dataset_a, args.dataset_c):
        if not raiz.is_dir():
            parser.error(f"Diretório inexistente: {raiz}")

    arquivos_a = arquivos_npy(args.dataset_a)
    arquivos_c = arquivos_npy(args.dataset_c)
    if not arquivos_a or not arquivos_c:
        parser.error("Um dos diretórios não contém arquivos .npy")

    print("Inspecionando cabeçalhos, shapes, dtypes e tamanhos...")
    resultado = {
        "dataset_A": resumo_dataset(args.dataset_a, arquivos_a),
        "dataset_C": resumo_dataset(args.dataset_c, arquivos_c),
        "correspondencia": verificar_conteudo(
            args.dataset_a, args.dataset_c, arquivos_a, arquivos_c,
            args.verificar_valores, args.seed,
        ),
        "benchmarks": [],
        "observacao_cache": (
            "As rodadas são alternadas. A primeira leitura pode vir do disco; "
            "as seguintes podem aproveitar o cache de RAM do sistema operacional."
        ),
    }

    # A-C-C-A reduz o risco de atribuir ao dataset uma vantagem causada apenas
    # pela ordem. A repetição também torna visível o aquecimento do cache.
    ordem = [("A", arquivos_a), ("C", arquivos_c), ("C", arquivos_c), ("A", arquivos_a)]
    for rodada, (nome, arquivos) in enumerate(ordem, start=1):
        print(f"Benchmark {rodada}/4: dataset {nome}")
        medicao = benchmark(arquivos, args.max_arquivos, args.seed + rodada)
        medicao.update({"rodada": rodada, "dataset": nome})
        resultado["benchmarks"].append(medicao)

    args.saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResumo")
    for nome in ("dataset_A", "dataset_C"):
        r = resultado[nome]
        print(f"{nome}: {r['numero_arquivos']} arquivos | {r['total_GiB']:.3f} GiB | dtypes={r['dtypes']} | shapes={r['shapes']}")
    for b in resultado["benchmarks"]:
        print(f"rodada {b['rodada']} - {b['dataset']}: {b['segundos']:.3f}s | {b['MiB_disco_por_segundo']:.1f} MiB/s")
    print(f"\nRelatório completo salvo em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
