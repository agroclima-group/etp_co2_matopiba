"""
01c - Precipitacao para o indice de aridez (P/ETP) e para a sensibilidade da SIF a chuva.

  a) Atual (2015-2025): CHIRPS diario (0,05 grau) somado por mes no Earth Engine,
     baixado a 0,05 grau e agregado para a grade de trabalho (media de blocos).
     Se o CHIRPS nao estiver disponivel, usa o ERA5-Land (total_precipitation_sum).
  b) Futura: razao mensal P_futuro / P_2015-2025 do NEX-GDDP (CMIP5, media dos modelos),
     mesma logica do 01b; o RCP 2.6 e escalonado a partir do RCP 4.5.

Saidas: data/prec_mensal.nc (prec, mm mes-1), data/cmip5_razaoP.csv (rcp, horizonte, mes, razaoP)
Chamado automaticamente pelo 04_cenarios.py se os arquivos nao existirem.
"""
import ee
import numpy as np
import pandas as pd
import xarray as xr
import shapely
from xee import helpers
import config as C

ee.Initialize(project=C.GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
geom = ee.Geometry.Rectangle(list(C.BBOX))


def colecao(*ids):
    for i in ids:
        try:
            ee.ImageCollection(i).limit(1).size().getInfo()
            print("usando", i, flush=True)
            return i
        except Exception:
            pass
    return None


# ------------------------------------------------------------------ a) precipitacao mensal observada
if not (C.DATA / "prec_mensal.nc").exists():
    cid = colecao("UCSB-CHG/CHIRPS/DAILY", "UCSB-CHG/CHIRPS/V3/DAILY_SAT")
    if cid:
        col, banda, fator, res_fina = ee.ImageCollection(cid).select("precipitation"), "precipitation", 1.0, 0.05
    else:
        col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("total_precipitation_sum")
        banda, fator, res_fina = "total_precipitation_sum", 1000.0, C.RES
    meses = pd.date_range(C.INICIO, C.FIM, freq="MS")
    imgs = []
    for t in meses:
        ini = ee.Date(t.strftime("%Y-%m-%d"))
        imgs.append(col.filterDate(ini, ini.advance(1, "month")).sum().rename("prec")
                    .set("system:time_start", ini.millis()))
    ic = ee.ImageCollection.fromImages(imgs)
    gp = helpers.fit_geometry(shapely.geometry.box(*C.BBOX), grid_crs="EPSG:4326",
                              grid_scale=(res_fina, -res_fina))
    ds = xr.open_dataset(ic, engine="ee", **gp)
    ren = {k: v for k, v in {"x": "lon", "y": "lat", "X": "lon", "Y": "lat"}.items() if k in ds.dims}
    p = ds.rename(ren)["prec"].transpose("time", "lat", "lon").sortby("lat").load() * fator
    f = int(round(C.RES / res_fina))
    if f > 1:                                               # media de blocos f x f -> grade de trabalho
        p = p.coarsen(lat=f, lon=f, boundary="trim").mean()
    p = p.assign_coords(time=meses[: p.sizes["time"]])
    p.attrs["units"] = "mm mes-1"
    p.rename("prec").to_netcdf(C.DATA / "prec_mensal.nc")
    print("precipitacao mensal salva:", p.shape, "media anual = %.0f mm" % float(p.sum("time").mean() / (len(meses) / 12)))

# ------------------------------------------------------------------ b) razao de precipitacao futura
if not (C.DATA / "cmip5_razaoP.csv").exists():
    nex = ee.ImageCollection("NASA/NEX-GDDP")
    PERIODOS = {"base": ("2015-01-01", "2026-01-01"), "2021-2040": ("2021-01-01", "2041-01-01"),
                "2041-2060": ("2041-01-01", "2061-01-01"), "2081-2100": ("2081-01-01", "2101-01-01")}
    RAZAO_26 = {"2021-2040": 1.0 / 1.4, "2041-2060": 1.0 / 1.4, "2081-2100": 1.0 / 1.8}
    DIAS = ee.Filter.Or(*[ee.Filter.calendarRange(d, d, "day_of_month") for d in (1, 8, 15, 22)])

    def clim(cen, ini, fim):
        ic = nex.filter(ee.Filter.eq("scenario", cen)).filterDate(ini, fim).filter(DIAS).select("pr")
        img = ee.Image.cat([ic.filter(ee.Filter.calendarRange(m, m, "month")).mean().rename(f"m{m:02d}")
                            for m in range(1, 13)])
        d = img.reduceRegion(ee.Reducer.mean(), geom, scale=50000, maxPixels=1e10, tileScale=4).getInfo()
        return {m: d[f"m{m:02d}"] for m in range(1, 13)}

    lin = []
    for cen, rcp in [("rcp45", "RCP4.5"), ("rcp85", "RCP8.5")]:
        base = clim(cen, *PERIODOS["base"])
        for h in C.HORIZONTES:
            fut = clim(cen, *PERIODOS[h])
            for m in range(1, 13):
                lin.append(dict(rcp=rcp, horizonte=h, mes=m, razaoP=round(fut[m] / base[m], 4)))
            print(rcp, h, "razao P media = %.3f" % np.mean([fut[m] / base[m] for m in fut]), flush=True)
    df = pd.DataFrame(lin)
    r26 = df[df.rcp == "RCP4.5"].copy()
    r26["rcp"] = "RCP2.6"
    r26["razaoP"] = (1 + (r26.razaoP - 1) * r26.horizonte.map(RAZAO_26)).round(4)
    pd.concat([r26, df], ignore_index=True).to_csv(C.DATA / "cmip5_razaoP.csv", index=False)
    print("razao de precipitacao salva")
