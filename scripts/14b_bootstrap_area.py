"""14b - Erro padrao da area suscetivel (climatologica) por bootstrap dos 11 anos (2015-2025).
Em cada reamostragem, os anos sao sorteados com reposicao; o IA e P medio / ETP media dos anos
sorteados (mesmo estimador climatologico do 09_aridez.py). Saida: results/incerteza_area_bootstrap.csv"""
import numpy as np, pandas as pd, xarray as xr
import config as C
ds = xr.open_dataset(C.OUT / "cenarios_anual.nc").load()
prec = xr.open_dataarray(C.DATA / "prec_mensal.nc")
alvo = dict(lat=ds.lat, lon=ds.lon)
pm = prec.interp(**alvo, method="linear").fillna(prec.interp(**alvo, method="nearest"))
razao = pd.read_csv(C.DATA / "cmip5_razaoP.csv")
area = ((C.RES * 111.32) ** 2 * np.cos(np.deg2rad(ds.lat))).broadcast_like(ds.ref.isel(ano=0)).values
def P_ano(rcp=None, h=None):
    if rcp is None:
        f = xr.ones_like(pm.time, dtype=float)
    else:
        r = razao[(razao.rcp == rcp) & (razao.horizonte == h)].set_index("mes")["razaoP"]
        f = xr.DataArray(r.reindex(pm.time.dt.month.values).values, {"time": pm.time}, ("time",))
    return (pm * f).groupby("time.year").sum(min_count=12).values      # (ano, lat, lon)
def susc(P, E):
    ia = P / E
    ok = np.isfinite(ia) & (ia >= 0.05) & (ia < 0.65)
    return (area * ok).sum()
rng = np.random.default_rng(42); B = 1000; n = ds.sizes["ano"]
idx = rng.integers(0, n, (B, n))
P0 = P_ano()
lin = []
for rcp, hs in C.RCP_CO2.items():
    for h in hs:
        k = f"{rcp.replace('.', '')}_{h.replace('-', '_')}"
        Pf = P_ano(rcp, h)
        for etp, fut, base in [("classica", f"clafut_{k}", "cla_ref"), ("co2", f"ambos_{k}", "ref")]:
            Ef, Eb = ds[fut].values, ds[base].values
            full = (susc(Pf.mean(0), Ef.mean(0)) - susc(P0.mean(0), Eb.mean(0))) / 1e3
            bs = [(susc(Pf[i].mean(0), Ef[i].mean(0)) - susc(P0[i].mean(0), Eb[i].mean(0))) / 1e3 for i in idx]
            lin.append(dict(cenario=rcp, horizonte=h, etp=etp, aumento_area_mil_km2=full, ep_bootstrap=float(np.std(bs, ddof=1))))
t = pd.DataFrame(lin).round(2); t.to_csv(C.OUT / "incerteza_area_bootstrap.csv", index=False); print(t.to_string())

# diferenca pareada ETP-PM - ETP-Y (mesmas reamostragens) no RCP 8.5, 2081-2100
k = "RCP85_2081_2100"; Pf = P_ano("RCP8.5", "2081-2100")
A = lambda i, fut, base: (susc(Pf[i].mean(0), ds[fut].values[i].mean(0)) - susc(P0[i].mean(0), ds[base].values[i].mean(0))) / 1e3
d, r = [], []
for i in idx:
    c, q = A(i, f"clafut_{k}", "cla_ref"), A(i, f"ambos_{k}", "ref")
    d.append(c - q); r.append(100 * (c / q - 1) if q > 0 else np.nan)
print("RCP8.5 2081-2100: diferenca PM-Y = %.1f mil km2 (EP %.1f; IC95 %.1f a %.1f); superestimativa %% IC95 %.0f a %.0f"
      % (np.mean(d), np.std(d, ddof=1), *np.percentile(d, [2.5, 97.5]), *np.nanpercentile(r, [2.5, 97.5])))
