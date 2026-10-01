"""
02b - GOSIF v2 (Li & Xiao, 2019): SIF em grade continua de 0,05 grau, mensal,
      derivada de OCO-2 + MODIS + reanalise. Download livre (sem login) em
      https://data.globalecology.unh.edu/data/GOSIF_v2/Monthly/

Recorta o MATOPIBA e agrega para a grade de RES_XCO2 (0,25 grau = 5 x 5 pixels),
a mesma do XCO2. Unidade: W m-2 um-1 sr-1 (= mW m-2 nm-1 sr-1), fator 0,0001;
32767 = agua, 32766 = neve/gelo (tratados como ausentes).

Saida: data/gosif_mensal.nc (variavel sif)
Pode ser rodado sozinho:  python 02b_gosif.py
"""
import gzip
import io
import subprocess
import sys
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import xarray as xr
import config as C

try:
    import tifffile
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tifffile"], check=True)
    import tifffile

URL = "https://data.globalecology.unh.edu/data/GOSIF_v2/Monthly/GOSIF_{a}.M{m:02d}.tif.gz"
RES_G = 0.05


def baixar_gosif():
    lon0, lat0, lon1, lat1 = C.BBOX
    r = C.RES_XCO2
    f = int(round(r / RES_G))
    lats = np.arange(lat0 + r / 2, lat1, r)
    lons = np.arange(lon0 + r / 2, lon1, r)
    ny, nx = len(lats), len(lons)
    c0 = int(round((lon0 + 180) / RES_G))
    r0 = int(round((90 - (lat0 + ny * r)) / RES_G))        # linha do topo (norte) da janela
    cache = C.RAW / "GOSIF"
    cache.mkdir(parents=True, exist_ok=True)
    meses = pd.date_range(C.INICIO, C.FIM, freq="MS")
    cubo = np.full((len(meses), ny, nx), np.nan)
    for i, t in enumerate(meses):
        npy = cache / f"{t:%Y%m}.npy"
        if npy.exists():
            bloco = np.load(npy)
        else:
            try:
                with urllib.request.urlopen(URL.format(a=t.year, m=t.month), timeout=300) as resp:
                    raw = gzip.decompress(resp.read())
            except urllib.error.HTTPError as e:
                print(f"GOSIF {t:%Y-%m}: indisponivel ({e.code})")
                continue
            img = tifffile.imread(io.BytesIO(raw))
            if img.shape != (3600, 7200):
                raise RuntimeError(f"grade GOSIF inesperada {img.shape}; esperado (3600, 7200) a partir de -180/90")
            bloco = img[r0:r0 + ny * f, c0:c0 + nx * f].astype("float32")
            bloco[bloco >= 32766] = np.nan
            bloco *= 1e-4
            np.save(npy, bloco)
            print(f"GOSIF {t:%Y-%m} ok", flush=True)
        # agrega 5x5 pixels -> 0,25 grau; linhas do norte para o sul -> inverte para lat crescente
        agg = np.nanmean(bloco.reshape(ny, f, nx, f), axis=(1, 3))[::-1]
        cubo[i] = agg
    da = xr.DataArray(cubo, {"time": meses, "lat": lats, "lon": lons}, ("time", "lat", "lon"), name="sif")
    da.attrs.update(units="mW m-2 nm-1 sr-1", fonte="GOSIF v2 (Li & Xiao, 2019)")
    da.to_netcdf(C.DATA / "gosif_mensal.nc")
    print("GOSIF salvo:", int(np.isfinite(cubo).any(axis=(1, 2)).sum()), "meses com dados")


if __name__ == "__main__":
    baixar_gosif()
