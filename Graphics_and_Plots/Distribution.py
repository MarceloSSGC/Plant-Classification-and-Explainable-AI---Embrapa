from pathlib import Path
import rasterio


def inspect_tiff(directory):
    """
    Procura um arquivo TIFF em um diretório e exibe seus metadados.

    Parameters
    ----------
    directory : str
        Caminho da pasta contendo arquivos .tif ou .tiff.
    """

    directory = Path(directory)

    # Procura TIFFs
    tiff_files = sorted(
        list(directory.glob("*.tif")) +
        list(directory.glob("*.tiff"))
    )

    if not tiff_files:
        print(f"Nenhum arquivo TIFF encontrado em:\n{directory}")
        return

    # Escolhe o primeiro arquivo
    filepath = tiff_files[0]

    print("=" * 80)
    print(f"Arquivo escolhido:\n{filepath}")
    print("=" * 80)

    with rasterio.open(filepath) as src:

        print("\n--- INFORMAÇÕES BÁSICAS ---")
        print(f"Driver:           {src.driver}")
        print(f"Largura:          {src.width}")
        print(f"Altura:           {src.height}")
        print(f"Número de bandas: {src.count}")
        print(f"Tipos de dados:   {src.dtypes}")
        print(f"Nodata:           {src.nodata}")

        print("\n--- SISTEMA DE REFERÊNCIA ---")
        print(f"CRS: {src.crs}")

        print("\n--- TRANSFORMAÇÃO GEOESPACIAL ---")
        print(src.transform)

        print("\n--- LIMITES GEOGRÁFICOS ---")
        print(f"Left:   {src.bounds.left}")
        print(f"Bottom: {src.bounds.bottom}")
        print(f"Right:  {src.bounds.right}")
        print(f"Top:    {src.bounds.top}")

        print("\n--- RESOLUÇÃO ESPACIAL ---")
        print(f"Pixel size X: {src.res[0]}")
        print(f"Pixel size Y: {src.res[1]}")

        print("\n--- TAGS GERAIS DO TIFF ---")
        tags = src.tags()

        if tags:
            for key, value in tags.items():
                print(f"{key}: {value}")
        else:
            print("Nenhuma tag geral encontrada.")

        print("\n--- TAGS POR BANDA ---")

        for band in range(1, src.count + 1):

            print(f"\nBanda {band}:")
            band_tags = src.tags(band)

            if band_tags:
                for key, value in band_tags.items():
                    print(f"  {key}: {value}")
            else:
                print("  Nenhuma tag encontrada.")

        print("\n--- DESCRIÇÃO DAS BANDAS ---")

        for band in range(1, src.count + 1):
            print(
                f"Banda {band}: "
                f"description={src.descriptions[band-1]}"
            )

        print("\n--- PERFIL COMPLETO ---")
        for key, value in src.profile.items():
            print(f"{key}: {value}")



directory = (
    r"D:\Marcelo\Datasets\Planta_Daninha\Datasets"
    r"\PlantaDaninha_BoaVista"
    r"\01_malva_branca_Agua_Boa_01"
)

inspect_tiff(directory)





from pathlib import Path


def count_samples_per_class(directory):
    """
    Conta arquivos .npy em cada pasta de classe.
    Arquivos soltos no diretório raiz, como mtv_info.json, são ignorados.
    """

    directory = Path(directory)

    if not directory.is_dir():
        raise ValueError(f"Diretório inválido: {directory}")

    counts = {}

    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue

        counts[class_dir.name] = len(list(class_dir.glob("*.npy")))

    return counts


multiview_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Datasets\Split\Multiview_Texture\align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15\Train_Norm"
aug_miltiview_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Datasets\Augmentation\Multiview_Texture__AUG\align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm"


count_samples_per_class(multiview_dir)
count_samples_per_class(aug_miltiview_dir)


species_name_map = {
    '01_malva_branca_Agua_Boa_01': 'malva branca',
    '02_Vassourinha_botao_Agua_Boa_02': 'vassourinha botão',
    '03_brizantha_Agua_Boa_03': 'brizantha',
    '04_cipo_fogo_Agua_Boa_04': 'cipó fogo',
    '05_Salsa_Agua_Boa_05': 'salsa',
    '06_capim_navalha_Agua_Boa_06': 'capim navalha',
    '07_capim_capeta_Agua_Boa_07': 'capim capeta',
    '08_malicia_Agua_Boa_08': 'malícia',
    '09_pe_galinha_Agua_Boa_09': 'pé de galinha',
    '10_carrapico_Agua_Boa_10': 'carrapicho',
    '11_apaga_fogo_Agua_Boa_11': 'apaga fogo',
    '12_Andropogon_Agua_Boa_12': 'Andropogon',
    '13_Traquipoon_Agua_Boa_13': 'Traquipoon',
    '14_Jaragua_Agua_Boa_14': 'Jaraguá',
    '15_Quicuio_Agua_Boa_15': 'Quicuio',
    '16_Massai_Agua_Boa_16': 'Massai',
    '17_Ruziziensis_Agua_Boa_17': 'Ruziziensis',
    '20_Guanxuma_Paludo_02': 'Guanxuma',
    '21_Mata_Pasto_Paludo_03': 'Mata Pasto',
    '23_Braquiarinha_Paludo_04': 'Braquiarinha',
    '24_Mombaça_Paludo_05': 'Mombaça',
    '26_Calapogonio_Paludo_07': 'Calapogonio',
    '27_Mavuno_Paludo_08': 'Mavuno',
    '28_Corda_de_viola_Paludo_09': 'Corda de viola',
    '29_Paiaguas_Paludo_10': 'Paiaguás',
    '30_Inaja_Serra_da_Prata_01': 'Inajá',
    '31_Cipo_Serra_da_Prata_02': 'Cipó',
    '32_Jurubebinha_Serra_da_Prata_03': 'Jurubebinha',
    '33_Capim_gengibre_Serra_da_Prata_04': 'Capim gengibre',
    '35_Chumbinho_Serra_da_Prata_05': 'Chumbinho',
    '36_Unha_de_gato_Serra_da_Prata_06': 'Unha de gato',
}

counts = count_samples_per_class(multiview_dir)

counts_cn = {species_name_map[x]: counts[x] for x in counts }




from pathlib import Path
import matplotlib.pyplot as plt


def plot_samples_per_class(
    counts,
    figsize=(14, 6),
    sort=False,
    save_path=None
):
    """
    Plots a vertical bar chart showing the number of samples per class.

    Parameters
    ----------
    counts : dict
        Dictionary in the format:
        {
            "class_name": number_of_samples,
            ...
        }

    figsize : tuple
        Figure size.

    sort : bool
        If True, sorts the classes by the number of samples.
        If False, keeps the original dictionary order.

    save_path : str, Path or None
        Full path where the figure will be saved, including file name
        and extension. If None, the figure is not saved.
    """

    if not counts:
        raise ValueError("The dictionary is empty.")

    if sort:
        counts = dict(sorted(counts.items(), key=lambda x: x[1]))

    classes = list(counts.keys())
    n_samples = list(counts.values())

    plt.figure(figsize=figsize)

    bars = plt.bar(
        classes,
        n_samples,
        label="Number of samples"
    )

    plt.xlabel("Species")
    plt.ylabel("Number of samples")
    plt.title("Number of Samples per Species")

    plt.xticks(rotation=90)

    # Add the number of samples above each bar
    for bar, value in zip(bars, n_samples):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        # Create parent directory if necessary
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Figure saved at:\n{save_path}")

    plt.show()

multiview_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Datasets\Split\Multiview_Texture\align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15\Train_Norm"
aug_miltiview_dir = r"D:\Marcelo\Datasets\Planta_Daninha\Datasets\Augmentation\Multiview_Texture__AUG\align_bands_ecc_affine_with_retry__best_band_otsu_green__RGB_NIR_RE__SEED_20__T_0.75_V_0.15__AUG/Train_Norm"


counts = count_samples_per_class(multiview_dir)
counts = count_samples_per_class(aug_miltiview_dir)
counts_cn = {species_name_map[x]: counts[x] for x in counts}


counts = count_samples_per_class(multiview_dir)
counts_cn = {species_name_map[x]: counts[x] for x in counts}
save_path = r"D:\Marcelo\Datasets\Planta_Daninha\Images\Distribution\samples_per_species.png"
plot_samples_per_class(counts_cn, figsize=(14, 6), sort=True, save_path=save_path)


counts = count_samples_per_class(aug_miltiview_dir)
counts_cn = {species_name_map[x]: counts[x] for x in counts}
save_path = r"D:\Marcelo\Datasets\Planta_Daninha\Images\Distribution\samples_per_species__AUG.png"
plot_samples_per_class(counts_cn, figsize=(14, 6), sort=True, save_path=save_path)

# Pasta / nome original	Nome em inglês	Nome científico

names = [
("malva branca","White mallow","Sida cordifolia"),
("Vassourinha botão","Sweet broomweed","Scoparia dulcis"),
("brizantha","Palisade grass","Brachiaria brizantha"),
("cipó fogo","Dodder","Cuscuta sp."),
("Salsa","Morning glory","Ipomoea asarifolia"),
("capim navalha","Razor grass","Paspalum virgatum"),
("capim capeta","Sandbur/Tick-trefoil","Cenchrus/Desmodium"),
("malícia","Sensitive plant","Mimosa pudica"),
("pé de galinha","Goosegrass","Eleusine indica"),
("carrapico","Sandbur/Tick-trefoil","Cenchrus/Desmodium"),
("apaga fogo","Joyweed","Alternanthera tenella"),
("Andropogon","Andropogon grass","Andropogon gayanus"),
("Traquipoon","Trachypogon grass","Trachypogon sp."),
("Jaraguá","Jaragua grass","Hyparrhenia rufa"),
("Quicuio","Kikuyu grass","Pennisetum clandestinum"),
("Massai","Massai grass","Megathyrsus maximus"),
("Ruziziensis","Ruzigrass","Brachiaria ruziziensis"),
("Guanxuma","Wireweed","Sida sp."),
("Mata Pasto","Senna","Senna spp."),
("Braquiarinha","Signalgrass","Brachiaria decumbens"),
("Mombaça","Mombaca grass","Megathyrsus maximus"),
("Calapogonio","Calopo","Calopogonium mucunoides"),
("Mavuno","Mavuno grass","Brachiaria hybrid"),
("Corda de viola","Morning glory","Ipomoea spp."),
("Paiaguás","Paiaguás grass","Brachiaria brizantha"),
("Inajá","Inajá palm","Attalea maripa"),
("Cipó","Vine species","—"),
("Jurubebinha","Nightshade","Solanum sp."),
("Capim gengibre","Wild ginger","Zingiber sp."),
("Chumbinho","Lantana","Lantana sp."),
("Unha de gato","Catclaw acacia","Acacia sp.")
]
len(names)
