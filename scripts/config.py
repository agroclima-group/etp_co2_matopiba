"""
Configuracao central do projeto
ETP sob CO2 elevado, XCO2 e SIF no MATOPIBA.

Todos os scripts importam daqui. Ajuste caminhos, periodo e parametros
apenas neste arquivo.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- caminhos
# TESTE=1 (variavel de ambiente) usa pastas *_teste com dados sinteticos,
# para conferir o pipeline sem baixar nada.
TESTE = os.environ.get("TESTE") == "1"
_suf = "_teste" if TESTE else ""
SCRIPTS = Path(__file__).resolve().parent   # pasta dos scripts
ROOT = SCRIPTS.parent                       # raiz do repositorio
DATA = ROOT / f"data{_suf}"          # dados de entrada ja processados (netCDF)
RAW = ROOT / "data_raw"           # downloads brutos (OCO-2/3, SIF)
OUT = ROOT / f"results{_suf}"      # tabelas e netCDF de saida
FIG = ROOT / "figures" / (f"generated{_suf}" + ("_en" if os.environ.get("IDIOMA") == "en" else ""))   # figuras finais
for p in (DATA, RAW, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- contas
GEE_PROJECT = os.environ.get("GEE_PROJECT", "your-gee-project-id")

# ---------------------------------------------------------------- area/periodo
# Retangulo envolvente do MATOPIBA (lon_min, lat_min, lon_max, lat_max).
# Para o recorte exato, use o shapefile oficial da Embrapa/GITE em MASK_SHP.
BBOX = (-50.8, -15.3, -41.3, -1.0)
MASK_SHP = DATA / "matopiba.gpkg"   # perimetro oficial do MATOPIBA (Embrapa/GITE; 13_limites_oficiais.py)


def recorte(obj):
    """Mantem apenas as celulas cujo centro esta dentro do MATOPIBA (NaN fora)."""
    if MASK_SHP is None or not Path(MASK_SHP).exists():
        return obj
    import numpy as np, shapely, geopandas as gpd, xarray as xr
    poli = gpd.read_file(MASK_SHP).geometry.iloc[0]
    lon, lat = np.meshgrid(obj["lon"].values, obj["lat"].values)
    m = xr.DataArray(shapely.contains_xy(poli, lon, lat), dims=("lat", "lon"),
                     coords={"lat": obj["lat"], "lon": obj["lon"]})
    return obj.where(m)
RES = 0.25                     # grade de trabalho (graus). 0.25 = mesma do XCO2/SIF e ~6x mais rapido;
                               # use 0.1 (nativa do ERA5-Land) para mapas mais finos
RES_XCO2 = 0.25                # grade de agregacao do XCO2/SIF (graus)
INICIO, FIM = "2015-01-01", "2025-12-31"

# Estacao chuvosa usada na validacao contra a ETR (sem limitacao hidrica)
MESES_CHUVOSOS = [12, 1, 2, 3, 4]
MESES_SECOS = [6, 7, 8, 9]

# ---------------------------------------------------------------- Penman-Monteith / CO2
# Sensibilidade relativa de r_s ao CO2 (ppm-1). 2,4e-4 / 0,34 reproduz a eq. 14 de Yang et al. (2019)
# (FAO-56 modificada: 0,34 u2 -> u2 [0,34 + 2,4e-4 (CO2 - 300)]). O padrao do PyEt (0,0009) e a media
# dos 16 modelos CMIP5 do mesmo artigo e entra na sensibilidade, com 0,0008 (dados de campo, Ainsworth & Rogers).
SRS = 2.4e-4 / 0.34
SRS_SENS = [0.0005, 2.4e-4 / 0.34, 0.0008, 0.0009, 0.0012]   # analise de sensibilidade
CO2_BASE = 300.0               # CO2 de referencia da formulacao (r_s ~ 70 s m-1)

# ---------------------------------------------------------------- cenarios RCP
# CO2 medio (ppm) por horizonte. VALORES APROXIMADOS - substituir pelos
# valores anuais de Meinshausen et al. (2011) (RCP database, IIASA) antes de publicar.
# Medias anuais de CO2 (ppm) em cada horizonte, interpoladas linearmente da Tabela AII.4.1
# do IPCC AR5 (Anexo II; valores de Meinshausen et al., 2011, harmonizados)
RCP_CO2 = {
    "RCP2.6": {"2021-2040": 429.2, "2041-2060": 441.9, "2081-2100": 425.9},
    "RCP4.5": {"2021-2040": 436.7, "2041-2060": 486.9, "2081-2100": 534.4},
    "RCP8.5": {"2021-2040": 452.5, "2041-2060": 546.3, "2081-2100": 850.4},
}
# Delta de temperatura: gerado por 01b_deltaT_gee.py a partir do NEX-GDDP (CMIP5)
# colunas: rcp, horizonte, mes, dT  (graus C, relativo a 2015-2025)
DELTA_T_CSV = DATA / "cmip5_deltaT.csv"
HORIZONTES = ["2021-2040", "2041-2060", "2081-2100"]

# ---------------------------------------------------------------- uso da terra
CLASSES = {1: "Cerrado nativo", 2: "Lavoura", 3: "Pastagem"}
CLASSES_EN = {1: "Native Cerrado", 2: "Cropland", 3: "Pasture"}

# ---------------------------------------------------------------- figuras
IDIOMA = os.environ.get("IDIOMA", "pt")   # "pt" ou "en" (rotulos das figuras); IDIOMA=en grava em figuras_en/
DPI = 600
LARG_DUPLA_MM = 174            # largura de coluna dupla (Elsevier)
LARG_SIMPLES_MM = 84
FONTE = ["Liberation Serif", "Times New Roman", "DejaVu Serif"]

# Paleta (daltonico-segura, validada para 3 series)
COR = {
    "etp_classica": "#eb6834",   # laranja
    "etp_co2": "#2a78d6",        # azul
    "sif": "#1baf7a",            # verde-agua
    "xco2": "#4a3aa7",           # violeta
    "neutro": "#52514e",
}
COR_CLASSES = {1: "#1baf7a", 2: "#eb6834", 3: "#2a78d6"}

# ---------------------------------------------------------------- OCO (tempo de download)
# O OCO-2 (XCO2) e obrigatorio. OCO-3 e a SIF observada do OCO-2/3 sao opcionais:
# cada um acrescenta ~10-20 h de download. A SIF principal do artigo e o GOSIF.
INCLUIR_OCO3 = False
INCLUIR_SIF_OCO = False

# ---------------------------------------------------------------- aridez (UNCCD) e SIF x chuva
# IA = P / ETP. Classes: hiperarido < 0,05; arido < 0,2; semiarido < 0,5; subumido seco < 0,65; umido >= 0,65
# Suscetivel a desertificacao: 0,05 <= IA < 0,65
LIMITES_IA = [0.05, 0.2, 0.5, 0.65]
NOMES_IA = ["Hiperárido", "Árido", "Semiárido", "Subúmido seco", "Úmido e subúmido úmido"]
NOMES_IA_EN = ["Hyper-arid", "Arid", "Semi-arid", "Dry sub-humid", "Humid"]
JANELA_SIF_MESES = 36                 # janela movel da sensibilidade da SIF a chuva (Zhang et al., 2022)
