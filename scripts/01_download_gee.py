"""
01 - Download via Google Earth Engine (xee):
     ERA5-Land diario, MODIS IAF (MCD15A2H), MODIS ET real (MOD16A2) e MapBiomas.

Requisitos:
    pip install earthengine-api xee xarray netCDF4
    earthengine authenticate          # uma vez
O ID do projeto Cloud vem da variavel de ambiente GEE_PROJECT (rodar_tudo.sh pergunta).

Saidas (pasta data/):
    era5land_diario.nc   tmax, tmin, tdew (K), u10, v10 (m s-1), ssrd (J m-2 dia-1), sp (Pa)
    iaf_mensal.nc        iaf (m2 m-2)
    etr_mensal.nc        etr (mm mes-1)
    uso_terra.nc         classe (1 Cerrado nativo, 2 Lavoura, 3 Pastagem, 0 outros)

Dica: o download de 11 anos diarios a 0.1 grau e grande; o script baixa ano a ano.
"""
import ee
import shapely
from xee import helpers
import xarray as xr
import pandas as pd
import config as C

ee.Initialize(project=C.GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
geom = ee.Geometry.Rectangle(list(C.BBOX))


def abrir(ic, escala=C.RES):
    # API do xee >= 0.1: grade definida por fit_geometry (crs, crs_transform, shape_2d)
    gp = helpers.fit_geometry(shapely.geometry.box(*C.BBOX), grid_crs="EPSG:4326",
                              grid_scale=(escala, -escala))
    return xr.open_dataset(ic, engine="ee", **gp)


def padronizar(ds):
    # xee devolve dimensoes (time, y, x); padroniza para (time, lat, lon)
    ren = {k: v for k, v in {"x": "lon", "y": "lat", "X": "lon", "Y": "lat"}.items() if k in ds.dims}
    ds = ds.rename(ren)
    return ds.transpose("time", "lat", "lon").sortby("lat")


# ------------------------------------------------------------------ ERA5-Land
BANDAS = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "dewpoint_temperature_2m": "tdew",
    "u_component_of_wind_10m": "u10",
    "v_component_of_wind_10m": "v10",
    "surface_solar_radiation_downwards_sum": "ssrd",
    "surface_pressure": "sp",
}
partes = []
for ano in range(pd.Timestamp(C.INICIO).year, pd.Timestamp(C.FIM).year + 1):
    arq = C.RAW / f"era5land_{ano}.nc"
    if arq.exists():                         # retomada: ano ja baixado
        partes.append(xr.open_dataset(arq)); print("ERA5-Land", ano, "ja existe"); continue
    ic = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
          .filterDate(f"{ano}-01-01", f"{ano + 1}-01-01")
          .select(list(BANDAS)))
    ds = padronizar(abrir(ic)).rename(BANDAS).load()
    ds.to_netcdf(arq)
    partes.append(ds)
    print("ERA5-Land", ano, "ok")
if not (C.DATA / "era5land_diario.nc").exists():
    xr.concat(partes, "time").to_netcdf(C.DATA / "era5land_diario.nc")


def colecao(*ids):
    """Primeira colecao existente no catalogo (os IDs do MODIS mudam entre versoes)."""
    for i in ids:
        try:
            ee.ImageCollection(i).limit(1).size().getInfo()
            print("usando", i)
            return ee.ImageCollection(i)
        except Exception:
            pass
    raise RuntimeError(f"nenhuma colecao encontrada: {ids}")

if (C.DATA / "etr_mensal.nc").exists() and (C.DATA / "iaf_mensal.nc").exists():
    print("MODIS ja baixado")
else:
    # ------------------------------------------------------------------ MODIS IAF (500 m -> grade de trabalho)
    ic = (colecao("MODIS/061/MOD15A2H", "MODIS/061/MCD15A3H", "MODIS/006/MOD15A2H")
          .filterDate(C.INICIO, C.FIM).select("Lai_500m"))
    iaf = padronizar(abrir(ic))["Lai_500m"] * 0.1
    iaf = iaf.where(iaf <= 10).resample(time="MS").mean()
    iaf.rename("iaf").to_netcdf(C.DATA / "iaf_mensal.nc")

    # ------------------------------------------------------------------ MOD16A2 (ET real, kg m-2 por 8 dias)
    ic = (colecao("MODIS/061/MOD16A2GF", "MODIS/061/MOD16A2", "MODIS/006/MOD16A2")
          .filterDate(C.INICIO, C.FIM).select("ET"))
    et8 = padronizar(abrir(ic))["ET"] * 0.1          # mm por composicao de 8 dias
    etr = (et8 / 8.0).resample(time="MS").mean()      # mm dia-1
    etr = etr * etr.time.dt.days_in_month             # mm mes-1
    etr.rename("etr").to_netcdf(C.DATA / "etr_mensal.nc")

# ------------------------------------------------------------------ MapBiomas (colecao 10)
# Confira o ID do asset na pagina da colecao vigente do MapBiomas.
MAPBIOMAS = [
    ("projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2", "classification_2024"),
    ("projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v1", "classification_2024"),
    ("projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1", "classification_2023"),
]
img = None
for asset, banda in MAPBIOMAS:
    try:
        cand = ee.Image(asset).select(banda)
        cand.bandNames().getInfo()
        img = cand
        print("usando MapBiomas", asset, banda)
        break
    except Exception:
        pass
if img is None:
    raise RuntimeError("Asset do MapBiomas nao encontrado; confira o ID em https://brasil.mapbiomas.org")
nat = img.remap([3, 4, 12], [1, 1, 1], 0)
lav = img.remap([39, 41, 20, 62], [2, 2, 2, 2], 0)
pas = img.remap([15], [3], 0)
classe = nat.max(lav).max(pas).rename("classe")
# Moda na grade de trabalho
# Baixa a classe a ~1 km (piramide do EE) e calcula a moda na grade de trabalho em Python,
# evitando o "User memory limit exceeded" do reduceResolution no servidor.
import numpy as np
fino = padronizar(abrir(ee.ImageCollection([classe.set("system:time_start", ee.Date("2024-01-01").millis())]),
                        escala=0.01))["classe"].isel(time=0).load()
f = int(round(C.RES / 0.01))


arr = fino.fillna(0).values.astype("int16")
ny, nx = arr.shape[0] // f, arr.shape[1] // f
arr = arr[: ny * f, : nx * f].reshape(ny, f, nx, f).transpose(0, 2, 1, 3).reshape(ny, nx, f * f)
cont = np.stack([(arr == k).sum(-1) for k in range(4)], -1)
moda_cls = cont.argmax(-1).astype("int8")
lat = fino.lat.values[: ny * f].reshape(ny, f).mean(1)
lon = fino.lon.values[: nx * f].reshape(nx, f).mean(1)
xr.DataArray(moda_cls, {"lat": lat, "lon": lon}, ("lat", "lon"), name="classe").to_netcdf(C.DATA / "uso_terra.nc")
print("GEE concluido.")
