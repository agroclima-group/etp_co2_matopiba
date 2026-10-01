"""
10 - Sensibilidade da SIF a precipitacao (theta_prec) e sua mudanca no tempo, ao longo do
     gradiente de aridez do MATOPIBA, seguindo a abordagem de Zhang et al. (2022, Nat. Commun.).

  - Anomalias mensais (ciclo sazonal removido) de SIF (GOSIF), precipitacao acumulada de
    2 meses (mes atual + anterior), temperatura e radiacao.
  - Regressao em janela movel de JANELA_SIF_MESES meses, por celula:
        univariada:    SIF' = theta * P'
        multivariada:  SIF' = theta * P' + b1 * T' + b2 * Rs'      (controla clima)
    theta em mW m-2 sr-1 nm-1 por 100 mm.
  - Media e tendencia (Sen, por decada) de theta; perfis por classe do indice de aridez
    (IA = P / ETP-CO2) com IC 95% por bootstrap (5000).
  - theta regional versus XCO2 de fundo (coerencia com o efeito do CO2; ver limitacoes).

Saidas: results/sif_chuva.nc, results/sif_chuva_resumo.json, figures/generated/Fig16_sif_chuva.(png|pdf)
"""
import json
import numpy as np
import pandas as pd
import xarray as xr
import pymannkendall as mk
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import config as cfg
from importlib import import_module

F = import_module("06_figuras")
L, PT, W2, salvar, rotulo, mapa_base = F.L, F.PT, F.W2, F.salvar, F.rotulo, F.mapa_base
rng = np.random.default_rng(7)

sif = F.sif                                                           # GOSIF (ou SIF OCO) mensal, 0,25 grau
etp = xr.open_dataset(cfg.OUT / "etp_mensal.nc")
prec = xr.open_dataarray(cfg.DATA / "prec_mensal.nc")
g = dict(lat=sif.lat, lon=sif.lon)
P = prec.interp(**g).reindex(time=sif.time)
T = etp.tmean.interp(**g).reindex(time=sif.time)
R = etp.rs.interp(**g).reindex(time=sif.time)
P2 = P.rolling(time=2, min_periods=2).sum()                          # chuva do mes + mes anterior


def anom(da):
    return da.groupby("time.month") - da.groupby("time.month").mean()


Y, X1, X2, X3 = [anom(v).drop_vars("month", errors="ignore") for v in (sif, P2 / 100.0, T, R)]
W = cfg.JANELA_SIF_MESES
nt = sif.sizes["time"]


def janela(i):
    s = slice(i, i + W)
    y = Y.isel(time=s).values.reshape(W, -1).T                       # (celulas, W)
    X = np.stack([X1.isel(time=s).values.reshape(W, -1).T,
                  X2.isel(time=s).values.reshape(W, -1).T,
                  X3.isel(time=s).values.reshape(W, -1).T], -1)        # (celulas, W, 3)
    return y, X


ncel = sif.sizes["lat"] * sif.sizes["lon"]
th_u = np.full((nt - W + 1, ncel), np.nan)
th_m = np.full((nt - W + 1, ncel), np.nan)
for i in range(nt - W + 1):
    y, X = janela(i)
    ok = np.isfinite(y) & np.isfinite(X).all(-1)
    n = ok.sum(1)
    yz = np.where(ok, y, 0.0); Xz = np.where(ok[..., None], X, 0.0)
    # univariada (com intercepto): slope = cov(x, y) / var(x)
    x = Xz[..., 0]
    mx, my = x.sum(1) / np.maximum(n, 1), yz.sum(1) / np.maximum(n, 1)
    sxy = (ok * (x - mx[:, None]) * (yz - my[:, None])).sum(1)
    sxx = (ok * (x - mx[:, None]) ** 2).sum(1)
    th_u[i] = np.where((n >= W * 0.7) & (sxx > 0), sxy / np.where(sxx > 0, sxx, 1), np.nan)
    # multivariada por equacoes normais (lote)
    A = np.concatenate([ok[..., None].astype(float), Xz], -1)          # intercepto + 3 preditores
    AtA = np.einsum("cwi,cwj->cij", A, A)
    Aty = np.einsum("cwi,cw->ci", A, yz)
    valido = (n >= W * 0.7) & (np.linalg.det(AtA) > 1e-10)
    b = np.full((ncel, 4), np.nan)
    b[valido] = np.linalg.solve(AtA[valido], Aty[valido][..., None])[..., 0]
    th_m[i] = b[:, 1]

tempo = sif.time.values[W // 2: W // 2 + nt - W + 1]
shape = (len(tempo), sif.sizes["lat"], sif.sizes["lon"])
theta = xr.Dataset({"theta_uni": (("time", "lat", "lon"), th_u.reshape(shape)),
                    "theta_multi": (("time", "lat", "lon"), th_m.reshape(shape))},
                   {"time": tempo, "lat": sif.lat, "lon": sif.lon})


def sen_dec(s):
    s = np.asarray(s)
    ok = np.isfinite(s)
    if ok.sum() < 24:
        return np.nan, np.nan
    r = mk.original_test(s[ok])
    return r.slope * 120, r.p                                          # por decada (120 meses)


for v in ("theta_uni", "theta_multi"):
    sl, p = xr.apply_ufunc(sen_dec, theta[v], input_core_dims=[["time"]],
                           output_core_dims=[[], []], vectorize=True)
    theta[f"{v}_media"] = theta[v].mean("time")
    theta[f"{v}_tend"], theta[f"{v}_p"] = sl, p

# indice de aridez atual (P / ETP-CO2) na grade da SIF
ia = (P.groupby("time.year").sum().mean("year")
      / etp.etp_co2.interp(**g).groupby("time.year").sum().mean("year"))
theta["ia"] = ia
theta.to_netcdf(cfg.OUT / "sif_chuva.nc")

# perfis por classe de IA (quantis) com bootstrap
df = theta[["theta_multi_media", "theta_multi_tend", "theta_uni_media", "theta_uni_tend", "ia"]] \
    .to_dataframe().dropna().reset_index()
bordas = np.unique(np.quantile(df.ia, np.linspace(0, 1, 9)))
df["classe"] = pd.cut(df.ia, bordas, include_lowest=True)


def boot(v):
    v = v.values
    b = [np.median(rng.choice(v, len(v))) for _ in range(5000)]
    return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)


perfil = []
for c, gdf in df.groupby("classe", observed=True):
    lin = dict(ia=float(gdf.ia.median()))
    for v in ("theta_uni_media", "theta_multi_media", "theta_uni_tend", "theta_multi_tend"):
        lin[v], lin[v + "_li"], lin[v + "_ls"] = boot(gdf[v])
    perfil.append(lin)
perfil = pd.DataFrame(perfil)
perfil.to_csv(cfg.OUT / "sif_chuva_perfil_aridez.csv", index=False)

# coerencia com o CO2: theta regional x XCO2 de fundo (colinear com o tempo; ver texto)
xco2 = etp.xco2.mean(["lat", "lon"]).rolling(time=12, center=True, min_periods=6).mean()
xr_theta = theta.theta_multi.mean(["lat", "lon"])
xc = xco2.sel(time=xr_theta.time)
ok = np.isfinite(xr_theta.values) & np.isfinite(xc.values)
from scipy import stats
rr = stats.linregress(xc.values[ok], xr_theta.values[ok]) if ok.sum() > 5 else None
seco = df.ia < 0.65 if (df.ia < 0.65).any() else df.ia <= df.ia.quantile(0.25)
res = dict(
    theta_multi_media=float(df.theta_multi_media.median()),
    theta_multi_tend_dec=float(df.theta_multi_tend.median()),
    pct_area_tend_positiva_sig=float(((theta.theta_multi_tend > 0) & (theta.theta_multi_p < 0.05)).mean() * 100),
    pct_area_tend_negativa_sig=float(((theta.theta_multi_tend < 0) & (theta.theta_multi_p < 0.05)).mean() * 100),
    tend_mais_seco=float(df[seco].theta_multi_tend.median()),
    tend_mais_umido=float(df[df.ia >= df.ia.quantile(0.75)].theta_multi_tend.median()),
    corr_tend_ia=float(np.corrcoef(np.log10(df.ia), df.theta_multi_tend)[0, 1]),
    theta_vs_xco2_incl=float(rr.slope) if rr else None,
    theta_vs_xco2_p=float(rr.pvalue) if rr else None,
    janela_meses=W,
)
json.dump(res, open(cfg.OUT / "sif_chuva_resumo.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))

# ============================================================== Fig 16 (estilo Zhang et al., 2022, Fig. 1)
un = "mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$ (100 mm)$^{-1}$"
fig = plt.figure(figsize=(W2, W2 * 0.95), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
ax = fig.add_subplot(gs[0, 0])
m = theta.theta_multi_media
lim = float(np.nanpercentile(np.abs(m), 98))
im = ax.pcolormesh(m.lon, m.lat, m, cmap="BrBG", norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
mapa_base(ax); ax.set_ylabel(L["lat"]); rotulo(ax, "a")
ax.set_title("θ$_{prec}$ médio" if PT else "Mean θ$_{prec}$", fontsize=7.5)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9, pad=0.08).set_label(un, fontsize=6.5)
ax = fig.add_subplot(gs[0, 1])
t = theta.theta_multi_tend
lim = float(np.nanpercentile(np.abs(t), 98))
im = ax.pcolormesh(t.lon, t.lat, t, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
sig = theta.theta_multi_p.where(theta.theta_multi_p < 0.05).to_dataframe().dropna().reset_index()
ax.scatter(sig.lon, sig.lat, s=0.8, color="#0b0b0b", linewidths=0)
mapa_base(ax); ax.set_yticklabels([]); rotulo(ax, "b")
ax.set_title("Tendência de θ$_{prec}$ (por década)" if PT else "θ$_{prec}$ trend (per decade)", fontsize=7.5)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9, pad=0.08).set_label(un + " — " + L["sig"], fontsize=6.5)
for j, (v, tit) in enumerate([("media", "θ$_{prec}$ médio" if PT else "Mean θ$_{prec}$"),
                              ("tend", "Tendência de θ$_{prec}$" if PT else "θ$_{prec}$ trend")]):
    ax = fig.add_subplot(gs[1, j])
    for mod, cor, nome in [("uni", cfg.COR["etp_classica"], "univariada" if PT else "univariate"),
                           ("multi", cfg.COR["etp_co2"], "multivariada" if PT else "multivariate")]:
        c = f"theta_{mod}_{v}"
        ax.fill_between(perfil.ia, perfil[c + "_li"], perfil[c + "_ls"], color=cor, alpha=0.18, lw=0)
        ax.plot(perfil.ia, perfil[c], color=cor, lw=1.8, marker="o", ms=3, label=nome)
    ax.axhline(0, color="#52514e", lw=0.6, ls="--")
    for lim_ia in cfg.LIMITES_IA:
        if perfil.ia.min() < lim_ia < perfil.ia.max():
            ax.axvline(lim_ia, color="#b4b2a9", lw=0.6)
    ax.set_xscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", labelsize=6)
    ax.set_xlabel("Índice de aridez (P/ETP)" if PT else "Aridity index (P/PET)")
    ax.set_ylabel(tit + ("\n(por década)" if v == "tend" and PT else ("\n(per decade)" if v == "tend" else "")))
    ax.grid(color="#e5e4df", lw=0.5); rotulo(ax, "cd"[j])
    if j == 0:
        ax.legend(frameon=False)
salvar(fig, "Fig16_sif_chuva")
