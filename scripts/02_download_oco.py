"""
02 - XCO2 (OCO-2 e OCO-3) e SIF (OCO-2) via NASA Earthdata (earthaccess),
     agregados em grade mensal de RES_XCO2 graus.

Os arquivos Lite sao diarios e globais (~40 MB cada; ~4 mil arquivos por
produto). Para nao baixar ~150 GB, o script le por streaming apenas as
variaveis necessarias e guarda so as sondagens dentro do MATOPIBA num cache
(data_raw/<produto>/*.parquet). Se a execucao cair, basta rodar de novo:
os arquivos ja processados sao pulados.

Login: na 1a execucao, earthaccess pede usuario e senha do Earthdata e grava
~/.netrc. Antes, autorize o aplicativo "NASA GESDISC DATA ARCHIVE" no seu
perfil do Earthdata (Applications > Authorized Apps).

Saidas (data/):
    xco2_mensal.nc   xco2 (ppm), n_obs, xco2_preenchido
    sif_mensal.nc    sif (W m-2 sr-1 um-1 = mW m-2 sr-1 nm-1), n_obs, sif_preenchido
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import earthaccess
import config as C

lon0, lat0, lon1, lat1 = C.BBOX
N_THREADS = 12

PRODUTOS = {
    # nome: (short_name, versao, inicio, leitor)
    "OCO2_L2_Lite_FP": ("OCO2_L2_Lite_FP", ["11.2r", "11.1r", "11r"], C.INICIO, "fp"),
    "OCO3_L2_Lite_FP": ("OCO3_L2_Lite_FP", ["10.4r", "11r", "10.3r"], "2019-08-01", "fp"),
    "OCO2_L2_Lite_SIF": ("OCO2_L2_Lite_SIF", ["11.2r", "11r", "10r"], C.INICIO, "sif"),
    "OCO3_L2_SIF": ("OCO3_L2_Fwd_SIF", ["11", "11r", "10.4r"], "2019-08-01", "sif"),
}


def extrair(ds, leitor):
    """Seleciona as variaveis e filtra qualidade e recorte espacial."""
    if leitor == "fp":
        lat, lon = ds["latitude"].values, ds["longitude"].values
        val, q = ds["xco2"].values, ds["xco2_quality_flag"].values
        t = ds["time"].values
        tempo = pd.to_datetime(t, unit="s") if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t)
        ok = q == 0
    else:
        lat, lon = ds["Latitude"].values, ds["Longitude"].values
        val, q = ds["Daily_SIF_757nm"].values, ds["Quality_Flag"].values
        t = ds["Delta_Time"].values
        tempo = (pd.to_datetime(t, unit="s", origin="1990-01-01")
                 if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t))
        ok = q <= 1
    ok &= (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    return pd.DataFrame({"lat": lat[ok], "lon": lon[ok], "valor": val[ok], "tempo": np.asarray(tempo)[ok]})


VARS = {"fp": "latitude,longitude,xco2,xco2_quality_flag,time",
        "sif": "Latitude,Longitude,Daily_SIF_757nm,Quality_Flag,Delta_Time"}
DAP = {"ok": None, "falhas": 0}          # None = ainda testando; False = desistiu do OPeNDAP
SESSAO = {}


def url_dap(link, leitor):
    """Converte o link de download no endereco OPeNDAP (Hyrax) da GES DISC, pedindo so as
    variaveis necessarias: ~2 MB em 1 requisicao, em vez de ler o arquivo global."""
    import re
    m = re.search(r"/data/(?:OCO[23]_DATA/)?(.+)$", link)
    if not m:
        return None
    caminho = m.group(1)
    host = "oco3" if "OCO3" in caminho else "oco2"
    return f"https://{host}.gesdisc.eosdis.nasa.gov/opendap/{caminho}.nc4?{VARS[leitor]}"


def ler_dap(link, leitor):
    import io
    if "s" not in SESSAO:
        SESSAO["s"] = earthaccess.get_requests_https_session()
    u = url_dap(link, leitor)
    r = SESSAO["s"].get(u, timeout=120)
    r.raise_for_status()
    with xr.open_dataset(io.BytesIO(r.content), engine="h5netcdf") as ds:
        return extrair(ds.load(), leitor)


def processar(granulo, arquivo_fs, leitor, destino):
    if destino.exists():
        return 0
    df = None
    if DAP["ok"] is not False:
        try:
            df = ler_dap(granulo.data_links()[0], leitor)
            DAP["ok"], DAP["falhas"] = True, 0
        except Exception as e:
            DAP["falhas"] += 1
            if DAP["ok"] is None and DAP["falhas"] >= 5:
                DAP["ok"] = False
                print("  OPeNDAP indisponivel, usando leitura direta (mais lenta):", e, flush=True)
    if df is None:
        with xr.open_dataset(arquivo_fs() if callable(arquivo_fs) else arquivo_fs, engine="h5netcdf") as ds:
            df = extrair(ds, leitor)
    df.to_parquet(destino)
    return len(df)


def baixar(nome):
    short, versao, inicio, leitor = PRODUTOS[nome]
    pasta = C.RAW / nome
    pasta.mkdir(parents=True, exist_ok=True)
    res = []
    for v in (versao if isinstance(versao, list) else [versao]):   # tenta as versoes em ordem
        res = earthaccess.search_data(short_name=short, version=v,
                                      temporal=(inicio, C.FIM), bounding_box=C.BBOX)
        if res:
            print(f"{nome}: versao {v}")
            break
    if not res:
        print(f"{nome}: nenhum arquivo encontrado")
        return pd.DataFrame(columns=["lat", "lon", "valor", "tempo"])
    def dest(g):
        return pasta / (Path(g.data_links()[0]).name + ".parquet")

    pend = [g for g in res if not dest(g).exists()]
    print(f"{nome}: {len(res)} arquivos, {len(pend)} pendentes")
    for i in range(0, len(pend), 100):                       # lotes de 100
        lote = pend[i:i + 100]
        if DAP["ok"] is False:                                # tenta o OPeNDAP de novo a cada lote
            DAP["ok"], DAP["falhas"] = None, 0
        # leitura direta so e aberta se o OPeNDAP falhar (evita abrir 100 conexoes a toa)
        abrir = (lambda g: (lambda: earthaccess.open([g])[0]))
        with ThreadPoolExecutor(N_THREADS) as ex:
            futs = [ex.submit(processar, g, abrir(g), leitor, dest(g)) for g in lote]
            for fu in as_completed(futs):
                try:
                    fu.result()
                except Exception as e:                        # arquivo com problema: segue
                    print("  aviso:", e)
        print(f"  {nome}: {min(i + 100, len(pend))}/{len(pend)}", flush=True)
    arqs = sorted(pasta.glob("*.parquet"))
    if not arqs:
        return pd.DataFrame(columns=["lat", "lon", "valor", "tempo"])
    return pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)


def agregar_mensal(df, nome):
    """Media mensal por celula de RES_XCO2 graus + numero de sondagens."""
    r = C.RES_XCO2
    lats = np.arange(lat0 + r / 2, lat1, r)
    lons = np.arange(lon0 + r / 2, lon1, r)
    df = df.assign(
        ilat=np.floor((df.lat - lat0) / r).astype(int),
        ilon=np.floor((df.lon - lon0) / r).astype(int),
        mes=pd.to_datetime(df.tempo).dt.to_period("M").dt.to_timestamp(),
    )
    df = df[(df.ilat < len(lats)) & (df.ilon < len(lons))]
    g = df.groupby(["mes", "ilat", "ilon"])["valor"].agg(["mean", "count"]).reset_index()
    meses = pd.date_range(C.INICIO, C.FIM, freq="MS")
    g = g[g.mes.isin(meses)]
    media = np.full((len(meses), len(lats), len(lons)), np.nan)
    n = np.zeros_like(media)
    it = meses.get_indexer(g.mes)
    media[it, g.ilat, g.ilon] = g["mean"]
    n[it, g.ilat, g.ilon] = g["count"]
    coords = {"time": meses, "lat": lats, "lon": lons}
    return xr.Dataset({nome: (("time", "lat", "lon"), media), "n_obs": (("time", "lat", "lon"), n)}, coords)


def preencher(da):
    """Lacunas: interpolacao espacial dentro do mes (ate 3 celulas); o restante
    recebe a media regional do mes; meses vazios, interpolacao no tempo. (Alternativa: krigagem ordinaria mensal.)"""
    da = da.interpolate_na("lon", max_gap=3 * C.RES_XCO2).interpolate_na("lat", max_gap=3 * C.RES_XCO2)
    da = da.fillna(da.mean(["lat", "lon"]))
    # meses sem nenhuma sondagem: interpolacao temporal
    return da.interpolate_na("time").ffill("time").bfill("time")


if __name__ == "__main__":
    # GOSIF (sem login) primeiro; uma falha aqui nao impede o OCO-2/3
    if not (C.DATA / "gosif_mensal.nc").exists():
        try:
            import importlib
            importlib.import_module("02b_gosif").baixar_gosif()
        except Exception as e:
            print("aviso: GOSIF nao baixado:", e)

    import os
    arq_tok = C.ROOT / "token_earthdata.txt"                 # alternativa: token salvo num arquivo
    if not os.environ.get("EARTHDATA_TOKEN") and arq_tok.exists():
        os.environ["EARTHDATA_TOKEN"] = arq_tok.read_text().strip()
    if os.environ.get("EARTHDATA_TOKEN"):
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login(strategy="interactive", persist=True)

    # 1) XCO2 (obrigatorio): OCO-2, e OCO-3 se habilitado em config.py
    partes = [baixar("OCO2_L2_Lite_FP")]
    if C.INCLUIR_OCO3:
        partes.append(baixar("OCO3_L2_Lite_FP"))
    df = pd.concat(partes, ignore_index=True)
    ds = agregar_mensal(df, "xco2")
    ds["xco2_preenchido"] = preencher(ds["xco2"])
    ds.to_netcdf(C.DATA / "xco2_mensal.nc")
    print("XCO2:", len(df), "sondagens validas", flush=True)

    # 2) SIF observada (opcional, so para validar o GOSIF)
    if C.INCLUIR_SIF_OCO:
        partes = [baixar("OCO2_L2_Lite_SIF")]
        if C.INCLUIR_OCO3:
            partes.append(baixar("OCO3_L2_SIF"))
        df = pd.concat(partes, ignore_index=True)
        ds = agregar_mensal(df, "sif")
        ds["sif_preenchido"] = preencher(ds["sif"])
        ds.to_netcdf(C.DATA / "sif_mensal.nc")
        print("SIF:", len(df), "sondagens validas")
