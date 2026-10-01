"""
11 - Padroes espaciais e temporais da SIF (GOSIF, 2015-2024) - Fig17 (texto principal)

  (a) SIF media anual
  (b) Queda relativa da SIF na estacao seca (jun-set) em relacao a chuvosa (dez-abr), %
  (c) Tendencia de Sen da SIF anual (% por decada), pontos: Mann-Kendall p < 0,05
  (d) Ciclo sazonal medio por classe de uso (mediana e intervalo interquartil entre celulas)

Saidas: results/sif_padroes.nc, results/sif_padroes_resumo.json, figures/generated/Fig17_sif_padroes.(png|pdf)
"""
import json
import numpy as np
import xarray as xr
import pymannkendall as mk
import matplotlib.pyplot as plt
import config as cfg
from importlib import import_module

M = import_module("08_mapas")                     # mapa base com limites, seta e escala
F = M.F
L, PT, W2, salvar, rotulo, CLS = F.L, F.PT, F.W2, F.salvar, F.rotulo, F.CLS

sif = F.sif.sel(time=slice("2015-01-01", "2024-12-31"))
uso = xr.open_dataset(cfg.DATA / "uso_terra.nc")["classe"].interp(lat=sif.lat, lon=sif.lon, method="nearest")
mes = sif.time.dt.month

media = sif.groupby("time.year").mean().mean("year")
chuva = sif.where(mes.isin(cfg.MESES_CHUVOSOS)).mean("time")
seca = sif.where(mes.isin(cfg.MESES_SECOS)).mean("time")
queda = (1 - seca / chuva) * 100
anual = sif.groupby("time.year").mean()


def sen(s):
    s = np.asarray(s)
    ok = np.isfinite(s)
    if ok.sum() < 8:
        return np.nan, np.nan
    r = mk.original_test(s[ok])
    return r.slope, r.p


sl, p = xr.apply_ufunc(sen, anual, input_core_dims=[["year"]], output_core_dims=[[], []], vectorize=True)
tend = sl * 10 / media * 100                      # % por decada
ds = xr.Dataset({"sif_media": media, "queda_seca_pct": queda, "tend_pct_dec": tend, "tend_p": p, "classe": uso})
ds.to_netcdf(cfg.OUT / "sif_padroes.nc")

reg = anual.mean(["lat", "lon"]).values
rr = mk.original_test(reg)
res = dict(
    sif_media_regional=float(media.mean()),
    sif_media_p5_p95=[float(v) for v in np.nanpercentile(media, [5, 95])],
    queda_seca_mediana=float(queda.median()),
    tend_regional_pct_dec=float(rr.slope * 10 / np.mean(reg) * 100), tend_regional_p=float(rr.p),
    pct_area_tend_pos_sig=float(((tend > 0) & (p < 0.05)).sum() / np.isfinite(tend).sum() * 100),
    pct_area_tend_neg_sig=float(((tend < 0) & (p < 0.05)).sum() / np.isfinite(tend).sum() * 100),
)
for k, n in cfg.CLASSES.items():
    m = uso == k
    res[f"media_{n}"] = float(media.where(m).median())
    res[f"queda_{n}"] = float(queda.where(m).median())
    res[f"tend_{n}"] = float(tend.where(m).median())
json.dump(res, open(cfg.OUT / "sif_padroes_resumo.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))

# ============================================================== Fig 17
fig = plt.figure(figsize=(W2, W2 * 1.05), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1])
ax = fig.add_subplot(gs[0, 0])
M.painel(ax, media, "Greens", "SIF média" if PT else "Mean SIF", "mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$",
         localizacao=True)
rotulo(ax, "a")
ax = fig.add_subplot(gs[0, 1])
M.painel(ax, queda, "YlOrBr", "Queda na estação seca" if PT else "Dry-season decline", "%", rot_y=False)
rotulo(ax, "b")
ax = fig.add_subplot(gs[0, 2])
lim = float(np.nanpercentile(np.abs(tend), 98))
from matplotlib.colors import TwoSlopeNorm
M.painel(ax, tend, "PiYG", "Tendência 2015–2024" if PT else "Trend 2015–2024",
         "% por década" if PT else "% per decade", TwoSlopeNorm(0, -lim, lim), rot_y=False, escala=True)
yy, xx = np.meshgrid(tend.lat, tend.lon, indexing="ij")
sig = (p.values < 0.05) & np.isfinite(tend.values)
ax.scatter(xx[sig], yy[sig], s=0.6, c="#0b0b0b", lw=0, zorder=4)
rotulo(ax, "c")

ax = fig.add_subplot(gs[1, :])
meses = np.arange(1, 13)
for i, k in enumerate(cfg.CLASSES):
    ncel = int(uso.where(uso == k).notnull().sum())
    serie = sif.where(uso == k).mean(["lat", "lon"])                       # media da classe em cada mes
    tab_m = serie.to_series().groupby([serie.time.dt.year.values, serie.time.dt.month.values]).mean().unstack()
    m, e = tab_m.mean().values, (tab_m.std(ddof=1) / np.sqrt(tab_m.notnull().sum())).values   # EP entre os anos
    ax.errorbar(meses + (i - 1) * 0.12, m, yerr=e, color=cfg.COR_CLASSES[k], lw=1.4, marker="o", ms=3,
                capsize=2, elinewidth=0.9, label=f"{CLS[k]} (n = {ncel})")
for m0, m1 in [(0.5, 4.5), (11.5, 12.5)]:
    ax.axvspan(m0, m1, color="#e5e4df", alpha=0.5, lw=0, zorder=0)
ax.axvspan(5.5, 9.5, color="#f6e8c3", alpha=0.5, lw=0, zorder=0)
nomes = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
ax.set_xticks(meses, nomes)
ax.set_xlim(0.5, 12.5)
ax.set_ylabel(L["sif"])
ax.text(2.5, ax.get_ylim()[0], " chuvosa" if PT else " wet", ha="center", va="bottom", fontsize=6.5, color="#52514e")
ax.text(7.5, ax.get_ylim()[0], " seca" if PT else " dry", ha="center", va="bottom", fontsize=6.5, color="#52514e")
ax.grid(axis="y", color="#e5e4df", lw=0.5)
ax.legend(frameon=False, loc="upper right", ncol=3)
rotulo(ax, "d")
salvar(fig, "Fig17_sif_padroes")
