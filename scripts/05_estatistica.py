"""
05 - Estatisticas que preenchem os [X] da secao de Resultados.

1. Tendencias (Mann-Kendall + declive de Sen; com n = 11 anos a correcao de autocorrelacao e instavel)
2. Magnitude da correcao pelo CO2 (media, % e IC 95% por bootstrap de anos)
3. Validacao das ETPs contra a ETR (MOD16) na estacao chuvosa: KGE, vies, RMSE
4. Modelo misto SIF/ETP ~ XCO2 de fundo + DPV + IAF + uso + mes (efeito aleatorio da celula)
5. Inclinacao SIF-DPV sob XCO2 alto (>P75) x baixo (<P25), EP agrupado por celula
6. I de Moran dos residuos

Sobre a circularidade e a causalidade reversa:
- SIF/ETP usa a ETP CLASSICA (sem CO2) no denominador; assim o XCO2 nao entra na variavel resposta.
- O ciclo sazonal do XCO2 local e CAUSADO pela fotossintese (SIF alta -> XCO2 baixo).
  Por isso o preditor e o XCO2 de fundo (media movel centrada de 12 meses da media regional),
  que representa o CO2 a que as plantas estao expostas, sem o sinal sazonal local.

Saidas: results/*.csv, results/tendencias.nc, results/resumo_resultados.json
"""
import json
import numpy as np
import pandas as pd
import xarray as xr
import pymannkendall as mk
import statsmodels.formula.api as smf
from pygam import LinearGAM, s, f
import config as cfg

rng = np.random.default_rng(42)
etp = xr.open_dataset(cfg.OUT / "etp_mensal.nc")
# SIF principal: GOSIF (grade continua); SIF do OCO-2 fica para validacao
sif_oco = xr.open_dataset(cfg.DATA / "sif_mensal.nc") if (cfg.DATA / "sif_mensal.nc").exists() else None
if (cfg.DATA / "gosif_mensal.nc").exists():
    sif = xr.open_dataarray(cfg.DATA / "gosif_mensal.nc").interpolate_na("time")
else:
    sif = sif_oco["sif_preenchido"]
iaf = xr.open_dataset(cfg.DATA / "iaf_mensal.nc")["iaf"]
etr = xr.open_dataset(cfg.DATA / "etr_mensal.nc")["etr"]
uso = xr.open_dataset(cfg.DATA / "uso_terra.nc")["classe"]
R = {}
if (cfg.DATA / "gosif_mensal.nc").exists() and sif_oco is not None:
    # validacao GOSIF x SIF observada pelo OCO-2 (so celulas-mes com >= 5 sondagens)
    obs = sif_oco["sif"].where(sif_oco["n_obs"] >= 5) if "n_obs" in sif_oco else sif_oco["sif"]
    par = xr.Dataset({"g": sif, "o": obs}).to_dataframe().dropna()
    R["gosif_vs_oco2_n"] = int(len(par))
    R["gosif_vs_oco2_r"] = float(np.corrcoef(par.g, par.o)[0, 1]) if len(par) > 2 else None
    R["gosif_vs_oco2_vies"] = float((par.g - par.o).mean()) if len(par) else None
    par.to_csv(cfg.OUT / "gosif_vs_oco2.csv")


# ------------------------------------------------------------------ 1. tendencias
def sen_mk(serie):
    serie = np.asarray(serie)
    if np.isnan(serie).any():
        return np.nan, np.nan
    r = mk.original_test(serie)
    return r.slope, r.p


anual = etp[["etp_classica", "etp_co2", "dif_co2"]].resample(time="YS").sum()
anual = anual.where(etp["etp_classica"].resample(time="YS").count() == 12)
tend = {}
for v in ["etp_classica", "etp_co2"]:
    sl, p = xr.apply_ufunc(sen_mk, anual[v], input_core_dims=[["time"]],
                           output_core_dims=[[], []], vectorize=True)
    tend[f"sen_{v}"], tend[f"p_{v}"] = sl, p
tend = xr.Dataset(tend)
tend.to_netcdf(cfg.OUT / "tendencias.nc")

reg = anual.mean(["lat", "lon"])
xco2_reg = etp["xco2"].mean(["lat", "lon"])
xa = xco2_reg.resample(time="YS").mean()
r_x = mk.original_test(xa.values)
R["xco2_inicio"], R["xco2_fim"] = float(xa[0]), float(xa[-1])
R["xco2_sen_ppm_ano"], R["xco2_p"] = r_x.slope, r_x.p
clim = xco2_reg.groupby("time.month").mean()
R["xco2_mes_min"], R["xco2_mes_max"] = int(clim.idxmin()), int(clim.idxmax())
R["xco2_amplitude_sazonal"] = float(clim.max() - clim.min())
for v in ["etp_classica", "etp_co2"]:
    r = mk.original_test(reg[v].values)
    R[f"{v}_media_mm_ano"] = float(reg[v].mean())
    R[f"{v}_dp_mm_ano"] = float(reg[v].std())
    R[f"{v}_sen_mm_ano2"], R[f"{v}_p"] = r.slope, r.p
    R[f"{v}_pct_area_tend_sig"] = float((tend[f"p_{v}"] < 0.05).mean() * 100)
R["pct_area_perde_signif"] = float(((tend.p_etp_classica < 0.05) & (tend.p_etp_co2 >= 0.05)).mean() * 100)

# ------------------------------------------------------------------ 2. magnitude da correcao
dif_anual = reg["dif_co2"].values
boot = [rng.choice(dif_anual, len(dif_anual)).mean() for _ in range(5000)]
pct = reg["dif_co2"] / reg["etp_classica"] * 100
R["dif_co2_mm_ano"] = float(dif_anual.mean())
R["dif_co2_pct"] = float(pct.mean())
R["dif_co2_ic95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
R["dif_co2_pct_inicio"], R["dif_co2_pct_fim"] = float(pct[0]), float(pct[-1])
dif_m = (etp.dif_co2 / etp.etp_classica * 100).mean(["lat", "lon"]).groupby("time.month").mean()
R["dif_pct_mes_max"], R["dif_pct_mes_min"] = int(dif_m.idxmax()), int(dif_m.idxmin())


# ------------------------------------------------------------------ 3. validacao contra ETR
def kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)


etr_g = etr.interp(lat=etp.lat, lon=etp.lon)
uso_g = uso.interp(lat=etp.lat, lon=etp.lon, method="nearest")
df_v = xr.Dataset({"cla": etp.etp_classica, "co2": etp.etp_co2, "etr": etr_g,
                   "classe": uso_g}).to_dataframe().dropna().reset_index()
df_v = df_v[df_v.time.dt.month.isin(cfg.MESES_CHUVOSOS) & df_v.classe.isin(cfg.CLASSES)]
val = []
for nome, g in [("Todas", df_v)] + [(cfg.CLASSES[k], df_v[df_v.classe == k]) for k in cfg.CLASSES]:
    for v in ["cla", "co2"]:
        val.append(dict(grupo=nome, etp=v, n=len(g), kge=kge(g[v].values, g.etr.values),
                        vies=float((g[v] - g.etr).mean()),
                        rmse=float(np.sqrt(((g[v] - g.etr) ** 2).mean()))))
val = pd.DataFrame(val).round(3)
val.to_csv(cfg.OUT / "validacao_etr.csv", index=False)
df_v.sample(min(len(df_v), 20000), random_state=1).to_csv(cfg.OUT / "pares_validacao.csv", index=False)
seca = etp.time.dt.month.isin(cfg.MESES_SECOS)
R["superestimativa_seca_pct"] = float(((etp.etp_co2.where(seca) / etr_g.where(seca)).mean() - 1) * 100)

# ------------------------------------------------------------------ 4. modelo SIF/ETP
g25 = dict(lat=sif.lat, lon=sif.lon)
fundo = xco2_reg.rolling(time=12, center=True, min_periods=6).mean()
dpv25 = etp.dpv.interp(**g25)
cla25 = etp.etp_classica.interp(**g25)
ds = xr.Dataset({
    "sif": sif, "etp": cla25, "dpv": dpv25,
    "iaf": iaf.interp(**g25), "classe": uso.interp(**g25, method="nearest"),
})
df = ds.to_dataframe().reset_index()
df["xco2_fundo"] = df.time.map(fundo.to_series())
df = df.dropna()
df = df[df.classe.isin(cfg.CLASSES) & (df.etp > 0) & (df.sif > 0)]
df["sif_etp"] = df.sif / df.etp * 100          # SIF por 100 mm de demanda
df["mes"] = df.time.dt.month
df["celula"] = df.lat.round(3).astype(str) + "_" + df.lon.round(3).astype(str)

mm = smf.mixedlm("sif_etp ~ xco2_fundo + bs(dpv, df=4) + bs(iaf, df=4) + C(classe) + C(mes)",
                 df, groups=df["celula"]).fit(reml=True)
b, se = mm.params["xco2_fundo"], mm.bse["xco2_fundo"]
media = df.sif_etp.mean()
R["sif_etp_pct_por_10ppm"] = float(b * 10 / media * 100)
R["sif_etp_pct_por_10ppm_ic95"] = [float((b - 1.96 * se) * 10 / media * 100), float((b + 1.96 * se) * 10 / media * 100)]
R["sif_etp_p"] = float(mm.pvalues["xco2_fundo"])
with open(cfg.OUT / "modelo_misto.txt", "w") as fh:
    fh.write(str(mm.summary()))

# GAM para os efeitos parciais (figura 5a)
X = df[["xco2_fundo", "dpv", "iaf", "classe", "mes"]].values
gam = LinearGAM(s(0, n_splines=8) + s(1, n_splines=8) + s(2, n_splines=8) + f(3) + f(4)).fit(X, df.sif_etp.values)
R["gam_r2_pseudo"] = float(gam.statistics_["pseudo_r2"]["explained_deviance"])
R["gam_edf_xco2"] = float(gam.statistics_["edof_per_coef"][:8].sum())
parc = []
for termo, nome in [(0, "xco2_fundo"), (1, "dpv")]:
    # grade manual (generate_X_grid do pygam zera as colunas categoricas)
    XX = np.tile(np.median(X, axis=0), (100, 1))
    XX[:, 3], XX[:, 4] = pd.Series(X[:, 3]).mode()[0], pd.Series(X[:, 4]).mode()[0]
    XX[:, termo] = np.linspace(np.percentile(X[:, termo], 1), np.percentile(X[:, termo], 99), 100)
    pdep, ci = gam.partial_dependence(term=termo, X=XX, width=0.95)
    parc.append(pd.DataFrame({"termo": nome, "x": XX[:, termo], "efeito": pdep, "li": ci[:, 0], "ls": ci[:, 1]}))
pd.concat(parc).to_csv(cfg.OUT / "efeitos_parciais.csv", index=False)

# ------------------------------------------------------------------ 5. SIF x DPV sob XCO2 alto/baixo
p25, p75 = df.xco2_fundo.quantile([0.25, 0.75])
d5 = df[(df.xco2_fundo <= p25) | (df.xco2_fundo >= p75)].copy()
d5["grupo"] = np.where(d5.xco2_fundo >= p75, "alto", "baixo")
ols = smf.ols("sif ~ dpv * C(grupo, Treatment('baixo')) + iaf + C(classe) + C(mes)", d5).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(d5.celula)[0]})
inter = [k for k in ols.params.index if k.startswith("dpv:")][0]
R["sif_dpv_incl_baixo"] = float(ols.params["dpv"])
R["sif_dpv_incl_alto"] = float(ols.params["dpv"] + ols.params[inter])
R["sif_dpv_interacao_p"] = float(ols.pvalues[inter])
R["sif_dpv_reducao_pct"] = float((1 - R["sif_dpv_incl_alto"] / R["sif_dpv_incl_baixo"]) * 100)
d5[["grupo", "dpv", "sif", "celula", "time"]].to_csv(cfg.OUT / "sif_dpv.csv", index=False)
sif_cl = df.groupby(["classe", "mes"]).sif.mean().unstack(0)
R["sif_max"], R["sif_mes_max"] = float(df.groupby("mes").sif.mean().max()), int(df.groupby("mes").sif.mean().idxmax())
for k in cfg.CLASSES:
    s_ = sif_cl[k]
    R[f"sif_queda_seca_pct_{cfg.CLASSES[k]}"] = float((1 - s_.loc[cfg.MESES_SECOS].mean() / s_.loc[cfg.MESES_CHUVOSOS].mean()) * 100)


# ------------------------------------------------------------------ 6. Moran dos residuos
def moran_grade(z):
    """I de Moran com vizinhanca torre (rook) numa grade regular 2D."""
    z = z - np.nanmean(z)
    num, w = 0.0, 0
    for dy, dx in [(0, 1), (1, 0)]:
        a, b_ = z[: z.shape[0] - dy, : z.shape[1] - dx], z[dy:, dx:]
        ok = ~np.isnan(a) & ~np.isnan(b_)
        num += 2 * np.sum(a[ok] * b_[ok])
        w += 2 * ok.sum()
    n = np.sum(~np.isnan(z))
    return n / w * num / np.nansum(z ** 2)


df["resid"] = mm.resid
rm = df.groupby(["lat", "lon"]).resid.mean().to_xarray()
R["moran_residuos"] = float(moran_grade(rm.values))

with open(cfg.OUT / "resumo_resultados.json", "w") as fh:
    json.dump(R, fh, indent=2, ensure_ascii=False, default=float)
print(json.dumps(R, indent=2, ensure_ascii=False, default=float))
print(val)
