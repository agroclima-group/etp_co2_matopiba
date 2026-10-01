"""
03 - ETP diaria de Penman-Monteith (FAO-56) e ETP modificada pelo CO2
     (Yang et al., 2019) com o pacote PyEt, usando o XCO2 observado.

Tres formulacoes, todas em mm dia-1 e somadas por mes:
    etp_fao56   pyet.pm_fao56                 (FAO-56 padrao, r_s = 70 s m-1)
    etp_classica pyet.pm(co2 = 300)            (mesma equacao da ETP-CO2 sem efeito do CO2)
    etp_co2     pyet.pm(co2 = XCO2 observado)  (r_s = r_s300 [1 + S_rs (CO2 - 300)])

A diferenca etp_classica - etp_co2 isola o efeito do CO2. pm(co2=300) e
pm_fao56 diferem ~0,5% por detalhes de formulacao (resistencia aerodinamica);
essa verificacao entra no material suplementar (tabela equivalencia_pm.csv).

Atencao: nao passe r_s para pyet.pm - se r_s for informado, o CO2 e ignorado.

Saida: results/etp_mensal.nc
"""
import numpy as np
import pandas as pd
import xarray as xr
import pyet
import config as C


def preparar(ds):
    """Converte ERA5-Land para as unidades do PyEt."""
    tmax = ds.tmax - 273.15
    tmin = ds.tmin - 273.15
    tdew = ds.tdew - 273.15
    tmean = (tmax + tmin) / 2
    ea = 0.6108 * np.exp(17.27 * tdew / (tdew + 237.3))                       # kPa
    es = (0.6108 * np.exp(17.27 * tmax / (tmax + 237.3)) +
          0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))) / 2
    rh = (100 * ea / es).clip(5, 100)
    u10 = np.hypot(ds.u10, ds.v10)
    u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)                                   # FAO-56 eq. 47
    rs = ds.ssrd / 1e6                                                           # MJ m-2 dia-1
    pres = ds.sp / 1000                                                          # kPa
    # altitude a partir da pressao media (inversa da eq. 7 da FAO-56), exigida pelo PyEt para Rso
    elev = (1 - (pres.mean("time") / 101.3) ** (1 / 5.26)) * 293 / 0.0065
    lat = xr.ones_like(tmean.isel(time=0, drop=True)) * np.deg2rad(ds.lat)       # rad, (lat, lon)
    return dict(tmean=tmean, tmax=tmax, tmin=tmin, rh=rh, ea=ea, es=es,
                wind=u2, rs=rs, pressure=pres, elevation=elev, lat=lat)


def etp_pm(v, co2, srs=C.SRS, dT=0.0):
    """pyet.pm com CO2; dT aplica o metodo delta (umidade relativa constante)."""
    return pyet.pm(v["tmean"] + dT, v["wind"], rs=v["rs"], tmax=v["tmax"] + dT,
                   tmin=v["tmin"] + dT, rh=v["rh"], pressure=v["pressure"],
                   elevation=v["elevation"], lat=v["lat"], co2=co2, srs=srs)


def xco2_diario(xco2_m, alvo):
    """Leva o XCO2 mensal (0.25 grau) para a grade e o passo diario do ERA5-Land."""
    x = xco2_m.interp(lat=alvo.lat, lon=alvo.lon, method="linear")
    x = x.fillna(x.mean(["lat", "lon"]))
    return x.reindex(time=alvo.time.values, method="ffill")


if __name__ == "__main__":
    era = C.recorte(xr.open_dataset(C.DATA / "era5land_diario.nc"))
    xco2 = xr.open_dataset(C.DATA / "xco2_mensal.nc")["xco2_preenchido"]

    mensal, equiv = [], []
    for ano, ds in era.groupby("time.year"):
        v = preparar(ds.load())
        co2d = xco2_diario(xco2, v["tmean"])
        fao = pyet.pm_fao56(v["tmean"], v["wind"], rs=v["rs"], tmax=v["tmax"], tmin=v["tmin"],
                            rh=v["rh"], pressure=v["pressure"], elevation=v["elevation"], lat=v["lat"])
        cla = etp_pm(v, C.CO2_BASE)
        eco = etp_pm(v, co2d)
        vpd = v["es"] - v["ea"]
        m = xr.Dataset({
            "etp_fao56": fao.resample(time="MS").sum(min_count=1),
            "etp_classica": cla.resample(time="MS").sum(min_count=1),
            "etp_co2": eco.resample(time="MS").sum(min_count=1),
            "xco2": co2d.resample(time="MS").mean(),
            "dpv": vpd.resample(time="MS").mean(),
            "tmean": v["tmean"].resample(time="MS").mean(),
            "rs": v["rs"].resample(time="MS").mean(),
        })
        mensal.append(m)
        equiv.append({"ano": ano, "pm_fao56": float(fao.mean()), "pm_co2_300": float(cla.mean()),
                      "dif_pct": float((cla.mean() / fao.mean() - 1) * 100)})
        print(ano, "ETP classica media = %.2f mm/dia" % float(cla.mean()))

    out = xr.concat(mensal, "time")
    out["etp_classica"].attrs["units"] = out["etp_co2"].attrs["units"] = "mm mes-1"
    out["dif_co2"] = out["etp_classica"] - out["etp_co2"]
    out.attrs["srs"] = C.SRS
    out.to_netcdf(C.OUT / "etp_mensal.nc")
    pd.DataFrame(equiv).to_csv(C.OUT / "equivalencia_pm.csv", index=False)
    print("Salvo em", C.OUT / "etp_mensal.nc")
