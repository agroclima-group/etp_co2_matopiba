"""
07 - Figuras complementares (Figs. 7-11), de leitura imediata e pouco usuais
     na literatura de ETP:

  Fig7  Mapa bivariado: aumento da ETP pelo aquecimento x compensacao pelo CO2 (RCP 8.5, 2081-2100)
  Fig8  Diagrama de Hovmoller (latitude x tempo): correcao pelo CO2 e anomalia de SIF
  Fig9  Lacos de histerese SIF x DPV (ciclo medio anual) por uso da terra
  Fig10 Cascata (waterfall): aquecimento (+), CO2 (-), interacao e resultado, por cenario
  Fig11 Faixas ("warming stripes"): XCO2, ETP classica e ETP-CO2 por ano

Usa as saidas de 03, 04 e 05. PNG 600 dpi + PDF.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
import config as cfg
from importlib import import_module

F = import_module("06_figuras")          # reaproveita estilo, rotulos e mapa base
L, PT, W2, salvar, rotulo, mapa_base = F.L, F.PT, F.W2, F.salvar, F.rotulo, F.mapa_base
CLS = F.CLS

etp = xr.open_dataset(cfg.OUT / "etp_mensal.nc")
mapas = xr.open_dataset(cfg.OUT / "cenarios_mapas.nc")
tab = pd.read_csv(cfg.OUT / "tabela_cenarios.csv")
sif = F.sif
uso = F.uso

# ============================================================== Fig 7 - mapa bivariado
k = "RCP85_2081_2100"
aum_cla = mapas[f"clafut_{k}"] - mapas["cla_ref"]                       # aumento sem CO2
aum_co2 = mapas[f"ambos_{k}"] - mapas["ref"]                            # aumento com CO2
comp = (1 - aum_co2 / aum_cla) * 100                                    # % compensada
# terciles de cada variavel -> 3 x 3 classes
q1 = np.nanquantile(aum_cla, [1 / 3, 2 / 3])
q2 = np.nanquantile(comp, [1 / 3, 2 / 3])
c1 = np.digitize(aum_cla, q1)
c2 = np.digitize(comp, q2)
biv = (c2 * 3 + c1).astype(float)
biv[np.isnan(aum_cla.values) | np.isnan(comp.values)] = np.nan
# paleta bivariada 3x3 (J. Stevens): linhas = compensacao, colunas = aumento
PAL = ["#e8e8e8", "#ace4e4", "#5ac8c8",
       "#dfb0d6", "#a5add3", "#5698b9",
       "#be64ac", "#8c62aa", "#3b4994"]
fig = plt.figure(figsize=(W2, W2 * 0.62), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(gs[0])
ax.pcolormesh(mapas.lon, mapas.lat, biv, cmap=ListedColormap(PAL),
              norm=BoundaryNorm(np.arange(-0.5, 9.5), 9), shading="nearest")
mapa_base(ax); ax.set_ylabel(L["lat"])
ax.set_title("RCP 8.5, 2081–2100")
leg = fig.add_subplot(gs[1])
for i in range(3):
    for j in range(3):
        leg.add_patch(Rectangle((j, i), 1, 1, color=PAL[i * 3 + j], ec="white", lw=1.5))
leg.set_xlim(0, 3); leg.set_ylim(0, 3); leg.set_aspect("equal")
leg.set_xticks([0.5, 1.5, 2.5], ["baixo", "médio", "alto"] if PT else ["low", "mid", "high"])
leg.set_yticks([0.5, 1.5, 2.5], ["baixa", "média", "alta"] if PT else ["low", "mid", "high"])
leg.set_xlabel(("Aumento da ETP\npelo aquecimento\n" if PT else "PET increase\nfrom warming\n")
               + f"({q1[0]:.0f} | {q1[1]:.0f} mm ano$^{{-1}}$)")
leg.set_ylabel(("Compensação pelo CO$_2$\n" if PT else "CO$_2$ offset\n") + f"({q2[0]:.0f} | {q2[1]:.0f} %)")
for sp in leg.spines.values():
    sp.set_visible(False)
leg.tick_params(length=0)
salvar(fig, "Fig7_mapa_bivariado")

# ============================================================== Fig 8 - Hovmoller
dif_lat = etp.dif_co2.mean("lon")                                      # mm mes-1, (time, lat)
anom = sif.groupby("time.month") - sif.groupby("time.month").mean()
anom_lat = anom.mean("lon").drop_vars("month", errors="ignore")
fig, axs = plt.subplots(2, 1, figsize=(W2, W2 * 0.7), sharex=True, constrained_layout=True)
im = axs[0].pcolormesh(dif_lat.time, dif_lat.lat, dif_lat.T, cmap="Blues", shading="nearest")
fig.colorbar(im, ax=axs[0], label=L["dif_m"], pad=0.01)
lim = float(np.nanpercentile(np.abs(anom_lat), 98))
im = axs[1].pcolormesh(anom_lat.time, anom_lat.lat, anom_lat.T, cmap="BrBG",
                       norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
fig.colorbar(im, ax=axs[1], label=("Anomalia de SIF\n" if PT else "SIF anomaly\n") + "(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)", pad=0.01)
for i, ax in enumerate(axs):
    ax.set_ylabel(L["lat"]); rotulo(ax, "ab"[i])
axs[0].set_title("ETP-PM − ETP-Y (média zonal)" if PT else "CO$_2$ correction of PET (zonal mean)")
axs[1].set_title("Anomalia mensal de SIF (média zonal)" if PT else "Monthly SIF anomaly (zonal mean)")
salvar(fig, "Fig8_hovmoller")

# ============================================================== Fig 9 - histerese SIF x DPV
g25 = dict(lat=sif.lat, lon=sif.lon)
dpv = etp.dpv.interp(**g25)
uso25 = uso.interp(**g25, method="nearest")
fig, ax = plt.subplots(figsize=(W2 * 0.62, W2 * 0.5), constrained_layout=True)
MESES = "JFMAMJJASOND"
rotulou = False
for kcl in cfg.CLASSES:
    m = uso25 == kcl
    x = dpv.where(m).mean(["lat", "lon"]).groupby("time.month").mean().values
    y = sif.where(m).mean(["lat", "lon"]).groupby("time.month").mean().values
    if np.isnan(x).all():
        continue
    cor = cfg.COR_CLASSES[kcl]
    xs, ys = np.r_[x, x[0]], np.r_[y, y[0]]
    ax.plot(xs, ys, color=cor, lw=1.8, label=CLS[kcl], zorder=2)
    for i in range(12):                                                 # setas no sentido do tempo
        ax.add_patch(FancyArrowPatch((xs[i], ys[i]), (xs[i + 1], ys[i + 1]), arrowstyle="-|>",
                                     mutation_scale=7, color=cor, lw=0, zorder=3))
    ax.scatter(x, y, s=16, color=cor, edgecolor="white", linewidth=0.8, zorder=4)
    if not rotulou:                                                     # meses 1, 4, 7, 10 num unico laco
        for i in (0, 3, 6, 9):
            ax.annotate(str(i + 1), (x[i], y[i]), xytext=(5, 4), textcoords="offset points",
                        fontsize=6.5, color="#52514e")
        rotulou = True
ax.set_xlabel(L["dpv"]); ax.set_ylabel(L["sif"])
ax.grid(color="#e5e4df", lw=0.5); ax.legend(frameon=False)
ax.text(0.99, 0.70, "números = mês" if PT else "numbers = month", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.5, color="#52514e")
salvar(fig, "Fig9_histerese_sif_dpv")

# ============================================================== Fig 10 - cascata
cen = tab[tab.cenario != "Referencia"]
fig, axs = plt.subplots(1, 3, figsize=(W2, W2 * 0.42), sharey=True, constrained_layout=True)
COR_T, COR_C, COR_I, COR_R = cfg.COR["etp_classica"], cfg.COR["etp_co2"], "#b4b2a9", cfg.COR["neutro"]
for i, (ax, rcp) in enumerate(zip(axs, cfg.RCP_CO2)):
    t = cen[cen.cenario == rcp].set_index("horizonte").loc[cfg.HORIZONTES]
    for j, (h, r) in enumerate(t.iterrows()):
        x0 = j * 4.2
        passos = [("T", r.efeito_T, COR_T), ("CO$_2$", r.efeito_CO2, COR_C), ("T×CO$_2$", r.interacao, COR_I)]
        base = 0.0
        for n, (nome, val, cor) in enumerate(passos):
            ax.bar(x0 + n, val, bottom=base, width=0.8, color=cor)
            if n < 2:
                ax.plot([x0 + n + 0.4, x0 + n + 0.6], [base + val] * 2, color="#52514e", lw=0.6)
            base += val
        ax.bar(x0 + 3, base, width=0.8, color=COR_R)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xticks([])
    ax.set_title(rcp.replace("RCP", "RCP ")); rotulo(ax, "abc"[i])
    ax.grid(axis="y", color="#e5e4df", lw=0.5)
    for j, h in enumerate(cfg.HORIZONTES):
        ax.text(j * 4.2 + 1.5, -0.04, h.replace("-", "–"), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color="#52514e")
axs[0].set_ylabel("Variação da ETP (mm ano$^{-1}$)" if PT else "PET change (mm yr$^{-1}$)")
topo = float(cen[["efeito_T"]].max().iloc[0])
for ax in axs:
    ax.set_ylim(top=topo * 1.12)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COR_T, label="Aquecimento" if PT else "Warming"),
                    Patch(color=COR_C, label="Efeito do CO$_2$" if PT else "CO$_2$ effect"),
                    Patch(color=COR_I, label="Interação" if PT else "Interaction"),
                    Patch(color=COR_R, label="Variação líquida" if PT else "Net change")],
           loc="outside lower center", ncol=4, frameon=False)
salvar(fig, "Fig10_cascata_cenarios")

# ============================================================== Fig 11 - faixas
anual = etp[["etp_classica", "etp_co2", "xco2"]].resample(time="YS").mean().mean(["lat", "lon"])
anual[["etp_classica", "etp_co2"]] = anual[["etp_classica", "etp_co2"]] * 12
anos = anual.time.dt.year.values
fig, axs = plt.subplots(3, 1, figsize=(W2, W2 * 0.36), sharex=True, constrained_layout=True)
lin = [("xco2", "xCO$_2$", "Purples"),
       ("etp_classica", L["cla"], "Oranges"),
       ("etp_co2", L["co2"], "Blues")]
vmin = float(min(anual.etp_classica.min(), anual.etp_co2.min()))
vmax = float(max(anual.etp_classica.max(), anual.etp_co2.max()))
for ax, (v, nome, cm) in zip(axs, lin):
    if ax is not axs[-1]:
        ax.tick_params(bottom=False)
    val = anual[v].values
    lims = (val.min(), val.max()) if v == "xco2" else (vmin, vmax)       # ETPs na mesma escala
    ax.imshow(val[None, :], aspect="auto", cmap=cm, vmin=lims[0], vmax=lims[1],
              extent=(anos[0] - 0.5, anos[-1] + 0.5, 0, 1))
    ax.set_yticks([]); ax.text(-0.01, 0.5, nome, transform=ax.transAxes, ha="right", va="center")
    ax.text(1.01, 0.5, f"{val[0]:.0f}→{val[-1]:.0f}", transform=ax.transAxes, ha="left", va="center",
            fontsize=6.5, color="#52514e")
    for sp in ax.spines.values():
        sp.set_visible(False)
axs[-1].set_xticks(anos[::2])
salvar(fig, "Fig11_faixas")
