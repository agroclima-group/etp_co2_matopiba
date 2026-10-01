"""
08 - Pranchas de mapas (Figs. 12-14) com acabamento cartografico:
     limites estaduais (geobr), mapa de localizacao do Brasil, seta de norte e escala.

  Fig12  Climatologia 2015-2025: ETP classica, ETP-CO2, correcao pelo CO2,
         XCO2, SIF e DPV (6 mapas)
  Fig13  Compensacao pelo CO2 (%) em cada cenario x horizonte (3 x 3 mapas)
  Fig14  Sensibilidade local: inclinacao de SIF/ETP em funcao do XCO2 de fundo,
         por celula (% por 10 ppm), com pontos onde p < 0,05

Se o geobr nao conseguir baixar os limites, os mapas saem sem contornos.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import config as cfg
from importlib import import_module

F = import_module("06_figuras")
L, PT, W2, salvar, rotulo = F.L, F.PT, F.W2, F.salvar, F.rotulo

UF, BR = F.UF_ALL, F.BR_ALL

etp = xr.open_dataset(cfg.OUT / "etp_mensal.nc")
mapas = xr.open_dataset(cfg.OUT / "cenarios_mapas.nc")
xco2 = xr.open_dataset(cfg.DATA / "xco2_mensal.nc")["xco2_preenchido"]
sif = F.sif
lon0, lat0, lon1, lat1 = F.EXT


def base(ax, localizacao=False, escala=False, rot_y=True):
    """Limites, eixos, seta de norte, escala e mapa de localizacao."""
    if UF is not None:
        UF.boundary.plot(ax=ax, color="#3d3d3a", linewidth=0.5, zorder=5)
        for _, r in UF[UF.abbrev_state.isin(["MA", "TO", "PI", "BA"])].iterrows():
            p = r.geometry.representative_point()
            if lon0 < p.x < lon1 and lat0 < p.y < lat1:
                ax.text(p.x, p.y, r.abbrev_state, fontsize=6.5, ha="center", va="center",
                        color="#3d3d3a", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    F.contorno_matopiba(ax)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}°O" if PT else f"{abs(v):.0f}°W")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}°S")
    if not rot_y:
        ax.set_yticklabels([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    if escala:                                                   # seta de norte + barra de 200 km
        ax.annotate("N", xy=(0.92, 0.42), xytext=(0.92, 0.30), xycoords="axes fraction",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=0.8))
        km_grau = 111.32 * np.cos(np.deg2rad((lat0 + lat1) / 2))
        comp = 200 / km_grau
        x, y = lon0 + 0.5, lat0 + 0.6
        ax.plot([x, x + comp], [y, y], color="#0b0b0b", lw=2, solid_capstyle="butt", zorder=7)
        ax.text(x + comp / 2, y + 0.25, "200 km", ha="center", va="bottom", fontsize=6, zorder=7)
    if localizacao and BR is not None:
        ins = ax.inset_axes([0.01, 0.77, 0.26, 0.22])
        BR.plot(ax=ins, color="#e5e4df", edgecolor="#52514e", linewidth=0.4)
        if F.MATOPIBA is not None:
            F.MATOPIBA.plot(ax=ins, color="#b8b7b0", edgecolor="#0b0b0b", linewidth=0.3)
        ins.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, ec="#e34948", lw=1))
        ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")


def painel(ax, da, cmap, titulo, unidade, norm=None, **kw):
    im = ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    base(ax, **kw)
    ax.set_title(titulo, fontsize=7.5)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.08, aspect=25)
    cb.set_label(unidade, fontsize=6.5); cb.ax.tick_params(labelsize=6)
    return im


# ============================================================== Fig 12 - climatologia
anual = etp[["etp_classica", "etp_co2", "dif_co2"]].resample(time="YS").sum(min_count=1).mean("time")
fig, axs = plt.subplots(2, 3, figsize=(W2, W2 * 1.0), constrained_layout=True)
mm = "mm ano$^{-1}$" if PT else "mm yr$^{-1}$"
vmin = float(min(anual.etp_classica.min(), anual.etp_co2.min()))
vmax = float(max(anual.etp_classica.max(), anual.etp_co2.max()))
from matplotlib.colors import Normalize
nE = Normalize(vmin, vmax)
painel(axs[0, 0], anual.etp_classica, "YlOrRd", L["cla"], mm, nE, localizacao=True)
painel(axs[0, 1], anual.etp_co2, "YlOrRd", L["co2"], mm, nE, rot_y=False)
painel(axs[0, 2], anual.dif_co2, "Blues", "ETP-PM − ETP-Y" if PT else "PET-PM − PET-Y", mm, rot_y=False)
painel(axs[1, 0], xco2.mean("time"), "Purples", "xCO$_2$", "ppm", escala=True)
painel(axs[1, 1], sif.mean("time"), "Greens", "SIF", "mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$", rot_y=False)
painel(axs[1, 2], etp.dpv.mean("time"), "OrRd", "DPV" if PT else "VPD", "kPa", rot_y=False)
for i, ax in enumerate(axs.flat):
    rotulo(ax, "abcdef"[i])
salvar(fig, "Fig12_mapas_climatologia")

# ============================================================== Fig 13 - compensacao 3 x 3
fig, axs = plt.subplots(3, 3, figsize=(W2, W2 * 1.35), constrained_layout=True)
comps = {}
for rcp in cfg.RCP_CO2:
    for h in cfg.HORIZONTES:
        k = f"{rcp.replace('.', '')}_{h.replace('-', '_')}"
        a_cla = mapas[f"clafut_{k}"] - mapas["cla_ref"]
        a_co2 = mapas[f"ambos_{k}"] - mapas["ref"]
        comps[(rcp, h)] = (1 - a_co2 / a_cla) * 100
vmax = float(np.nanpercentile(np.concatenate([c.values.ravel() for c in comps.values()]), 99))
for i, rcp in enumerate(cfg.RCP_CO2):
    for j, h in enumerate(cfg.HORIZONTES):
        ax = axs[i, j]
        im = ax.pcolormesh(mapas.lon, mapas.lat, comps[(rcp, h)], cmap="viridis", vmin=0, vmax=vmax,
                           shading="nearest", rasterized=True)
        base(ax, escala=(i == 2 and j == 0), rot_y=(j == 0))
        if i == 0:
            ax.set_title(h.replace("-", "–"), fontsize=8)
        if j == 0:
            ax.text(-0.32, 0.5, rcp.replace("RCP", "RCP "), transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=8, fontweight="bold")
        med = float(comps[(rcp, h)].median())
        ax.text(0.03, 0.97, f"{med:.0f}%", transform=ax.transAxes, fontsize=7, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=8)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.6, aspect=35)
cb.set_label("Compensação pelo CO$_2$ (% do aumento da ETP-PM)" if PT else "CO$_2$ offset (% of PET-PM increase)")
salvar(fig, "Fig13_mapas_compensacao_cenarios")

# ============================================================== Fig 14 - sensibilidade local
g25 = dict(lat=sif.lat, lon=sif.lon)
razao = sif / etp.etp_classica.interp(**g25) * 100
reg = etp.xco2.mean(["lat", "lon"])
fundo = reg.rolling(time=12, center=True, min_periods=6).mean()
# remove o ciclo sazonal de SIF/ETP (anomalia mensal) antes da regressao
anom = razao.groupby("time.month") - razao.groupby("time.month").mean()
x = fundo.values
incl = np.full(razao.shape[1:], np.nan)
pval = np.full(razao.shape[1:], np.nan)
media = razao.mean("time").values
for a in range(razao.shape[1]):
    for b in range(razao.shape[2]):
        y = anom.values[:, a, b]
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() > 24:
            r = stats.linregress(x[ok], y[ok])
            incl[a, b] = r.slope * 10 / media[a, b] * 100
            pval[a, b] = r.pvalue
sens = xr.DataArray(incl, {"lat": razao.lat, "lon": razao.lon}, ("lat", "lon"))
lim = float(np.nanpercentile(np.abs(incl), 98))
fig, ax = plt.subplots(figsize=(W2 * 0.6, W2 * 0.75), constrained_layout=True)
im = ax.pcolormesh(sens.lon, sens.lat, sens, cmap="PuOr", norm=TwoSlopeNorm(0, -lim, lim),
                   shading="nearest", rasterized=True)
yy, xx = np.where(pval < 0.05)
ax.scatter(razao.lon.values[xx], razao.lat.values[yy], s=1.2, color="#0b0b0b", linewidths=0, zorder=6)
base(ax, localizacao=True, escala=True)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.06)
cb.set_label(("Sensibilidade de SIF/ETP ao xCO$_2$ (% por 10 ppm)" if PT
              else "Sensitivity of SIF/PET to xCO$_2$ (% per 10 ppm)") + " — " + L["sig"])
salvar(fig, "Fig14_mapa_sensibilidade")
sens.to_netcdf(cfg.OUT / "sensibilidade_local.nc")
