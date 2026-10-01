"""
14 - Valores anuais (2015-2025) de cada cenario, para as barras de erro padrao das Figuras 7 e 9.

Para cada ano da base (2015-2025), calcula a ETP anual por celula na referencia e em cada
cenario/horizonte (metodo delta, igual ao 04_cenarios.py) e a area suscetivel a desertificacao
com a precipitacao do proprio ano. O erro padrao e o desvio-padrao entre os 11 anos das
diferencas pareadas (cenario - referencia no mesmo ano) dividido por raiz(11).

Saidas: results/cenarios_anual.nc, results/incerteza_anual.csv
"""
import numpy as np, pandas as pd, xarray as xr
import config as C
from importlib import import_module
E4 = import_module("04_cenarios")      # reaproveita era, dT_diario, C_REF (nao roda o __main__)
E3 = E4.E

def anual(co2, rcp=None, horiz=None):
    out = []
    for ano, ds in E4.era.groupby("time.year"):
        v = E3.preparar(ds.load())
        dT = 0.0 if rcp is None else E4.dT_diario(ds.time, rcp, horiz)
        out.append(E3.etp_pm(v, co2, dT=dT).sum("time", min_count=1).expand_dims(ano=[ano]))
    return xr.concat(out, "ano")

V = {"ref": anual(E4.C_REF), "cla_ref": anual(C.CO2_BASE)}
for rcp, hs in C.RCP_CO2.items():
    for h, cfut in hs.items():
        k = f"{rcp.replace('.', '')}_{h.replace('-', '_')}"
        V[f"ambos_{k}"] = anual(cfut, rcp, h)
        V[f"clafut_{k}"] = anual(C.CO2_BASE, rcp, h)
        print(k, flush=True)
ds = xr.Dataset(V); ds.to_netcdf(C.OUT / "cenarios_anual.nc")

# precipitacao anual por ano, na grade das ETPs
prec = xr.open_dataarray(C.DATA / "prec_mensal.nc")
alvo = dict(lat=ds.lat, lon=ds.lon)
pm = prec.interp(**alvo, method="linear").fillna(prec.interp(**alvo, method="nearest"))
razao = pd.read_csv(C.DATA / "cmip5_razaoP.csv")
area = ((C.RES * 111.32) ** 2 * np.cos(np.deg2rad(ds.lat))).broadcast_like(ds.ref.isel(ano=0))

def P_ano(rcp=None, h=None):
    if rcp is None:
        f = xr.ones_like(pm.time, dtype=float)
    else:
        r = razao[(razao.rcp == rcp) & (razao.horizonte == h)].set_index("mes")["razaoP"]
        f = xr.DataArray(r.reindex(pm.time.dt.month.values).values, {"time": pm.time}, ("time",))
    return (pm * f).groupby("time.year").sum(min_count=12).rename(year="ano")

def area_susc(ia):
    return ((ia >= 0.05) & (ia < 0.65)).astype(float).where(np.isfinite(ia)).__mul__(area).sum(["lat", "lon"])

reg = lambda a: a.mean(["lat", "lon"])
P0 = P_ano()
a_ref = {"classica": area_susc(P0 / ds.cla_ref), "co2": area_susc(P0 / ds.ref)}
lin = []
n = ds.sizes["ano"]
for rcp, hs in C.RCP_CO2.items():
    for h in hs:
        k = f"{rcp.replace('.', '')}_{h.replace('-', '_')}"
        Pf = P_ano(rcp, h)
        for etp, fut, base in [("classica", f"clafut_{k}", "cla_ref"), ("co2", f"ambos_{k}", "ref")]:
            d_etp = reg(ds[fut]) - reg(ds[base])
            d_area = (area_susc(Pf / ds[fut]) - a_ref[etp]) / 1e3
            lin.append(dict(cenario=rcp, horizonte=h, etp=etp,
                            aumento_etp_media=float(d_etp.mean()), aumento_etp_ep=float(d_etp.std(ddof=1) / np.sqrt(n)),
                            aumento_area_media_mil_km2=float(d_area.mean()),
                            aumento_area_ep_mil_km2=float(d_area.std(ddof=1) / np.sqrt(n))))
pd.DataFrame(lin).round(3).to_csv(C.OUT / "incerteza_anual.csv", index=False)
print(pd.DataFrame(lin).round(2).to_string())
