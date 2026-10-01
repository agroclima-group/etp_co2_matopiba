"""
12 - Padroes espaciais e temporais do XCO2 observado (OCO-2, sem preenchimento) - Fig18

  (a) Anomalia media de XCO2 em relacao a media regional do mes, grade de 1 grau
      (celulas com >= 15 meses observados)
  (b) Ciclo sazonal regional sem tendencia: XCO2 x SIF (eixos gemeos)
  (c) Taxa de crescimento anual do XCO2 regional (ppm ano-1)
  (d) Anomalia de XCO2 por uso da terra e estacao (media e IC 95% bootstrap)

Saidas: results/xco2_padroes_resumo.json, figures/generated/Fig18_xco2_padroes.(png|pdf)
"""
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import config as cfg
from importlib import import_module

M = import_module("08_mapas")
F = M.F
L, PT, W2, salvar, rotulo, CLS = F.L, F.PT, F.W2, F.salvar, F.rotulo, F.CLS
rng = np.random.default_rng(3)

x = xr.open_dataset(cfg.DATA / "xco2_mensal.nc")["xco2"]              # apenas celulas-mes observadas
uso = xr.open_dataset(cfg.DATA / "uso_terra.nc")["classe"].interp(lat=x.lat, lon=x.lon, method="nearest")
reg = x.mean(["lat", "lon"])
an = x - reg
mes = x.time.dt.month

# (a) anomalia espacial a 1 grau
c1 = an.coarsen(lat=4, lon=4, boundary="trim").mean()
nmes = np.isfinite(c1).sum("time")
mapa = c1.where(nmes >= 15).mean("time")
df = mapa.to_dataframe(name="a").dropna().reset_index()

# (b) ciclo sazonal sem tendencia
fundo = reg.rolling(time=12, center=True, min_periods=6).mean()
ciclo_x = (reg - fundo).groupby("time.month").mean()
sif_reg = F.sif.mean(["lat", "lon"])
ciclo_s = sif_reg.groupby("time.month").mean()
r0 = float(np.corrcoef(ciclo_x, ciclo_s)[0, 1])
rlag = {k: float(np.corrcoef(np.roll(ciclo_x.values, -k), ciclo_s.values)[0, 1]) for k in range(0, 4)}

# (c) crescimento anual
anual = reg.groupby("time.year").mean()
cresc = anual.diff("year")

# (d) anomalia por uso e estacao
EST = {("Chuvosa" if PT else "Wet"): cfg.MESES_CHUVOSOS, ("Seca" if PT else "Dry"): cfg.MESES_SECOS,
       ("Out–nov" if PT else "Oct–Nov"): [10, 11]}
tab = {}
for k in cfg.CLASSES:
    for e, ms in EST.items():
        v = an.where((uso == k) & mes.isin(ms)).values
        v = v[np.isfinite(v)]
        b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        tab[(k, e)] = (float(v.mean()), *np.percentile(b, [2.5, 97.5]), len(v))

res = dict(
    anom_1grau_p5_p50_p95=[float(v) for v in np.nanpercentile(mapa, [5, 50, 95])],
    n_celulas_1grau=int(np.isfinite(mapa).sum()),
    corr_anom_lat=float(np.corrcoef(df.a, df.lat)[0, 1]), corr_anom_lon=float(np.corrcoef(df.a, df.lon)[0, 1]),
    ciclo_amplitude=float(ciclo_x.max() - ciclo_x.min()), mes_min=int(ciclo_x.idxmin()), mes_max=int(ciclo_x.idxmax()),
    corr_ciclo_xco2_sif=r0, corr_ciclo_defasagens=rlag,
    crescimento_anual={int(a): round(float(v), 2) for a, v in zip(cresc.year.values, cresc.values)},
    crescimento_medio=float(cresc.mean()),
    anom_uso_estacao={f"{cfg.CLASSES[k]}|{e}": [round(v, 3) for v in t[:3]] + [t[3]] for (k, e), t in tab.items()},
)
json.dump(res, open(cfg.OUT / "xco2_padroes_resumo.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))

# ============================================================== Fig 18
fig = plt.figure(figsize=(W2, W2 * 0.95), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[0.85, 1.15])
ax = fig.add_subplot(gs[0, 0])
anos = cresc.year.values
NINO, NINA = (2016, 2019, 2024), (2017, 2018, 2020, 2021, 2022, 2023)   # anos com El Nino / La Nina (NOAA ONI)
cores = [cfg.COR["etp_classica"] if a in NINO else cfg.COR["etp_co2"] if a in NINA else "#a9a8a2" for a in anos]
ax.bar(anos, cresc.values, color=cores, width=0.7)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cfg.COR["etp_classica"], label="El Niño"), Patch(color=cfg.COR["etp_co2"], label="La Niña"),
                   Patch(color="#a9a8a2", label="Neutro" if PT else "Neutral")],
          frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
ax.axhline(float(cresc.mean()), color="#52514e", lw=0.8, ls="--")
ax.set_ylabel("Crescimento (ppm ano$^{-1}$)" if PT else "Growth (ppm yr$^{-1}$)")
ax.set_xticks(anos[::2])
ax.grid(axis="y", color="#e5e4df", lw=0.5)
rotulo(ax, "a")

ax = fig.add_subplot(gs[0, 1])
meses = np.arange(1, 13)
ax.plot(meses, ciclo_x, color=cfg.COR["xco2"], lw=1.8, marker="o", ms=3, label="xCO$_2$")
ax.axhline(0, color="#52514e", lw=0.5)
ax.set_ylabel("xCO$_2$ − média móvel (ppm)" if PT else "xCO$_2$ − running mean (ppm)")
ax2 = ax.twinx()
ax2.plot(meses, ciclo_s, color=cfg.COR_CLASSES[1], lw=1.8, ls="--", marker="s", ms=3, label="SIF")
ax2.set_ylabel(L["sif"])
ax2.spines["right"].set_visible(True)
ax.set_xticks(meses, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", ncol=2)
ax.text(0.99, 0.04, f"r = {r0:.2f}", transform=ax.transAxes, ha="right", fontsize=7)
rotulo(ax, "b")

ax = fig.add_subplot(gs[1, 0])
lim = float(np.nanpercentile(np.abs(mapa), 98))
M.painel(ax, mapa, "RdBu_r", "Anomalia de xCO$_2$" if PT else "xCO$_2$ anomaly", "ppm",
         TwoSlopeNorm(0, -lim, lim), localizacao=True, escala=True)
rotulo(ax, "c")

ax = fig.add_subplot(gs[1, 1])
larg = 0.25
for i, k in enumerate(cfg.CLASSES):
    for j, e in enumerate(EST):
        m_, lo, hi, n = tab[(k, e)]
        xx = j + (i - 1) * larg
        ax.errorbar(xx, m_, yerr=[[m_ - lo], [hi - m_]], fmt="o", ms=3.5, color=cfg.COR_CLASSES[k],
                    capsize=2, lw=1, label=CLS[k] if j == 0 else None)
ax.axhline(0, color="#52514e", lw=0.5)
ax.set_xticks(range(len(EST)), list(EST))
ax.set_ylabel("Anomalia de xCO$_2$ (ppm)" if PT else "xCO$_2$ anomaly (ppm)")
ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
ax.grid(axis="y", color="#e5e4df", lw=0.5)
rotulo(ax, "d")
salvar(fig, "Fig18_xco2_padroes")
