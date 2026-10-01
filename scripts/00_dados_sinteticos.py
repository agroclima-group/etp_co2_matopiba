"""
00 - Gera dados SINTETICOS no mesmo formato dos dados reais, para testar o
     pipeline e as figuras sem baixar nada. NAO usar os resultados no artigo.

Uso:   TESTE=1 python 00_dados_sinteticos.py
"""
import os
os.environ["TESTE"] = "1"
import numpy as np
import pandas as pd
import xarray as xr
import config as C

rng = np.random.default_rng(1)
lon0, lat0, lon1, lat1 = C.BBOX
res = 0.5                                        # grade grossa so para o teste
lat = np.arange(lat0 + res / 2, lat1, res)
lon = np.arange(lon0 + res / 2, lon1, res)
t = pd.date_range(C.INICIO, C.FIM, freq="D")
nt, ny, nx = len(t), len(lat), len(lon)
doy = t.dayofyear.values[:, None, None]
ano = (t.year.values - 2015)[:, None, None]
LAT = lat[None, :, None]
LON = lon[None, None, :]

# sazonalidade: seca jun-set (maior T, menor umidade), mais seco ao sul/leste
seco = np.cos(2 * np.pi * (doy - 225) / 365)               # +1 em agosto
grad = (LAT - lat0) / (lat1 - lat0)                          # 0 sul -> 1 norte
tmax = 31 + 3 * seco + 1.5 * (1 - grad) + 0.03 * ano + rng.normal(0, 1.2, (nt, ny, nx))
tmin = tmax - 10 - 3 * seco + rng.normal(0, 0.8, (nt, ny, nx))
tdew = tmin + 1 - 6 * np.clip(seco, 0, None) + rng.normal(0, 1, (nt, ny, nx))
u10 = 2.2 + 0.8 * seco + rng.normal(0, 0.4, (nt, ny, nx))
ssrd = (18 + 4 * seco + rng.normal(0, 2.5, (nt, ny, nx))) * 1e6
sp = np.full((nt, ny, nx), 97000.0)
co = {"time": t, "lat": lat, "lon": lon}
d3 = ("time", "lat", "lon")
xr.Dataset({"tmax": (d3, tmax + 273.15), "tmin": (d3, tmin + 273.15), "tdew": (d3, tdew + 273.15),
            "u10": (d3, u10), "v10": (d3, np.zeros_like(u10)), "ssrd": (d3, ssrd), "sp": (d3, sp)},
           co).astype("float32").to_netcdf(C.DATA / "era5land_diario.nc")

# uso da terra: lavoura no sul/oeste, Cerrado nativo no centro-norte, pastagem no restante
classe = np.where((grad[0] < 0.4) & (LON[0] < -45), 2, np.where(grad[0] > 0.55, 1, 3))
classe = np.broadcast_to(classe, (ny, nx)).astype("int8")
xr.DataArray(classe, {"lat": lat, "lon": lon}, ("lat", "lon"), name="classe").to_netcdf(C.DATA / "uso_terra.nc")

# mensal
tm = pd.date_range(C.INICIO, C.FIM, freq="MS")
m = tm.month.values[:, None, None]
am = (tm.year.values - 2015 + (tm.month.values - 1) / 12)[:, None, None]
chuva = np.cos(2 * np.pi * (m - 2) / 12)                   # +1 em fevereiro
fator = np.where(classe == 1, 1.0, np.where(classe == 2, 1.1, 0.8))[None]
amp = np.where(classe == 2, 1.6, 1.0)[None]
iaf = np.clip(fator * (2.0 + amp * chuva) + rng.normal(0, 0.2, (len(tm), ny, nx)), 0.2, None)
xr.DataArray(iaf, {"time": tm, "lat": lat, "lon": lon}, d3, name="iaf").to_netcdf(C.DATA / "iaf_mensal.nc")
etr = np.clip(80 + 45 * chuva * fator + rng.normal(0, 8, iaf.shape), 10, None)
xr.DataArray(etr, {"time": tm, "lat": lat, "lon": lon}, d3, name="etr").to_netcdf(C.DATA / "etr_mensal.nc")

# XCO2 e SIF em 0.25 grau
r = C.RES_XCO2
la = np.arange(lat0 + r / 2, lat1, r)
lo = np.arange(lon0 + r / 2, lon1, r)
cm = {"time": tm, "lat": la, "lon": lo}
cl25 = xr.DataArray(classe, {"lat": lat, "lon": lon}, ("lat", "lon")).interp(lat=la, lon=lo, method="nearest").ffill("lat").bfill("lat").ffill("lon").bfill("lon").values
ch = np.cos(2 * np.pi * (tm.month.values - 2) / 12)[:, None, None]
sif = np.clip((0.45 + 0.35 * ch * np.where(cl25 == 2, 1.5, 1.0)[None]) * (1 + 0.006 * am) + rng.normal(0, 0.05, (len(tm), len(la), len(lo))), 0.02, None)
xco2 = 399.5 + 2.35 * am - 0.8 * ch + rng.normal(0, 0.6, sif.shape)
nobs = rng.poisson(6, sif.shape).astype(float)
mask = rng.random(sif.shape) < 0.3
xr.Dataset({"xco2": (d3, np.where(mask, np.nan, xco2)), "n_obs": (d3, np.where(mask, 0, nobs)),
            "xco2_preenchido": (d3, xco2)}, cm).to_netcdf(C.DATA / "xco2_mensal.nc")
xr.Dataset({"sif": (d3, sif), "sif_preenchido": (d3, sif)}, cm).to_netcdf(C.DATA / "sif_mensal.nc")

# precipitacao sintetica (0,25 grau): chuvosa dez-abr, mais seca ao sul/leste; SIF responde a chuva
pm = np.clip((110 + 140 * ch) * (0.7 + 0.6 * ((la[None, :, None] - lat0) / (lat1 - lat0))) + rng.normal(0, 25, sif.shape), 0, None)
xr.DataArray(pm, cm, d3, name="prec").to_netcdf(C.DATA / "prec_mensal.nc")
gos = np.clip(0.1 + 0.0015 * pm * (1 + 0.004 * am) + rng.normal(0, 0.04, sif.shape), 0.01, None)
xr.DataArray(gos, cm, d3, name="sif").to_netcdf(C.DATA / "gosif_mensal.nc")
lp = []
for rcp, k in [("RCP2.6", 0.4), ("RCP4.5", 0.7), ("RCP8.5", 1.0)]:
    for h, b in [("2021-2040", -0.02), ("2041-2060", -0.05), ("2081-2100", -0.12)]:
        for mes in range(1, 13):
            lp.append(dict(rcp=rcp, horizonte=h, mes=mes, razaoP=round(1 + b * k, 3)))
pd.DataFrame(lp).to_csv(C.DATA / "cmip5_razaoP.csv", index=False)

# delta T sintetico (formato do arquivo real)
lin = []
for rcp, k in [("RCP2.6", 0.5), ("RCP4.5", 0.8), ("RCP8.5", 1.0)]:
    for h, base in [("2021-2040", 0.8), ("2041-2060", 1.6), ("2081-2100", 3.6)]:
        for mes in range(1, 13):
            lin.append(dict(rcp=rcp, horizonte=h, mes=mes, dT=round(base * k * (1.1 if mes in (8, 9, 10) else 1.0), 2)))
pd.DataFrame(lin).to_csv(C.DELTA_T_CSV, index=False)
print("Dados sinteticos em", C.DATA)
