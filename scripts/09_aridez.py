"""
09 - Indice de aridez (IA = P / ETP) com a ETP classica e com a ETP-CO2, no periodo atual
     e nos cenarios RCP, e area suscetivel a desertificacao pelas faixas da UNCCD
     (0,05 <= IA < 0,65). Aplica a recomendacao de Yang et al. (2019): indices de aridez
     calculados sem o efeito do CO2 exageram o ressecamento futuro.

Entradas: results/cenarios_mapas.nc (04), data/prec_mensal.nc e data/cmip5_razaoP.csv (01c)
Saidas:   results/aridez_areas.csv, results/aridez_mapas.nc, figures/generated/Fig15_aridez.(png|pdf)
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import config as cfg
from importlib import import_module

F = import_module("06_figuras")
L, PT, W2, salvar, rotulo, mapa_base = F.L, F.PT, F.W2, F.salvar, F.rotulo, F.mapa_base

mapas = xr.open_dataset(cfg.OUT / "cenarios_mapas.nc")
prec = xr.open_dataarray(cfg.DATA / "prec_mensal.nc")
razao = pd.read_csv(cfg.DATA / "cmip5_razaoP.csv")
alvo = dict(lat=mapas.lat, lon=mapas.lon)

# climatologia mensal de P (mm mes-1) na grade das ETPs
clim = prec.groupby("time.month").mean().interp(**alvo, method="linear")
clim = clim.fillna(prec.groupby("time.month").mean().interp(**alvo, method="nearest"))
P_ref = clim.sum("month", min_count=12)


def P_fut(rcp, h):
    r = razao[(razao.rcp == rcp) & (razao.horizonte == h)].set_index("mes")["razaoP"]
    fator = xr.DataArray(r.reindex(range(1, 13)).values, {"month": np.arange(1, 13)}, ("month",))
    return (clim * fator).sum("month", min_count=12)


limites = [0.0] + cfg.LIMITES_IA + [np.inf]
nomes = cfg.NOMES_IA if PT else cfg.NOMES_IA_EN
area_cel = (cfg.RES * 111.32) ** 2 * np.cos(np.deg2rad(mapas.lat))          # km2 por celula
area_cel = area_cel.broadcast_like(P_ref)


def resumo(ia, cenario, horizonte, etp):
    cls = np.digitize(ia.values, cfg.LIMITES_IA)
    ok = np.isfinite(ia.values)
    lin = dict(cenario=cenario, horizonte=horizonte, etp=etp, ia_medio=float(ia.mean()))
    for k, n in enumerate(cfg.NOMES_IA):
        lin[f"area_{n}_km2"] = float(area_cel.values[ok & (cls == k)].sum())
    lin["area_suscetivel_km2"] = float(area_cel.values[ok & (ia.values >= 0.05) & (ia.values < 0.65)].sum())
    return lin


linhas, grades = [], {}
ia = P_ref / mapas["cla_ref"]; grades["atual_classica"] = ia
linhas.append(resumo(ia, "Referencia", "2015-2025", "classica"))
ia = P_ref / mapas["ref"]; grades["atual_co2"] = ia
linhas.append(resumo(ia, "Referencia", "2015-2025", "co2"))
for rcp in cfg.RCP_CO2:
    for h in cfg.HORIZONTES:
        k = f"{rcp.replace('.', '')}_{h.replace('-', '_')}"
        pf = P_fut(rcp, h)
        for etp, var in [("classica", f"clafut_{k}"), ("co2", f"ambos_{k}")]:
            ia = pf / mapas[var]
            grades[f"{k}_{etp}"] = ia
            linhas.append(resumo(ia, rcp, h, etp))
tab = pd.DataFrame(linhas)
tab.round(3).to_csv(cfg.OUT / "aridez_areas.csv", index=False)
xr.Dataset(grades).to_netcdf(cfg.OUT / "aridez_mapas.nc")
print(tab[["cenario", "horizonte", "etp", "ia_medio", "area_suscetivel_km2"]].round(2).to_string())

# ============================================================== Fig 15
CORES = ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac"]      # hiperarido -> umido
cmap, norm = ListedColormap(CORES), BoundaryNorm(np.arange(-0.5, 5.5), 5)
k85 = "RCP85_2081_2100"
paineis = [(grades["atual_co2"], "2015–2025"),
           (grades[f"{k85}_classica"], ("RCP 8.5, 2081–2100\n" + (L["cla"]))),
           (grades[f"{k85}_co2"], ("RCP 8.5, 2081–2100\n" + L["co2"]))]
fig = plt.figure(figsize=(W2, W2 * 1.02), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1])
for i, (g, tit) in enumerate(paineis):
    ax = fig.add_subplot(gs[0, i])
    ax.pcolormesh(g.lon, g.lat, np.digitize(g.values, cfg.LIMITES_IA).astype(float)
                  * np.where(np.isfinite(g.values), 1, np.nan), cmap=cmap, norm=norm, shading="nearest")
    mapa_base(ax)
    if i == 0:
        ax.set_ylabel(L["lat"])
    else:
        ax.set_yticklabels([])
    ax.set_title(tit, fontsize=7.5); rotulo(ax, "abc"[i])
fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(CORES, nomes)], loc="outside upper center",
           ncol=5, frameon=False, fontsize=6.5)
ax = fig.add_subplot(gs[1, :])
fut = tab[tab.cenario != "Referencia"]
boot = pd.read_csv(cfg.OUT / "incerteza_area_bootstrap.csv")       # erro padrao por bootstrap dos anos (14b)
base = tab[tab.cenario == "Referencia"].set_index("etp")["area_suscetivel_km2"]
rot, x = [], []
for j, (rcp, h) in enumerate([(r, h) for r in cfg.RCP_CO2 for h in cfg.HORIZONTES]):
    t = fut[(fut.cenario == rcp) & (fut.horizonte == h)].set_index("etp")
    xj = j + (j // 3) * 0.6
    dc = (t.loc["classica", "area_suscetivel_km2"] - base["classica"]) / 1e3
    dq = (t.loc["co2", "area_suscetivel_km2"] - base["co2"]) / 1e3
    b = boot[(boot.cenario == rcp) & (boot.horizonte == h)].set_index("etp")["ep_bootstrap"]
    ekw = dict(elinewidth=0.9, capsize=2.5, ecolor="#0b0b0b")
    ax.bar(xj - 0.19, dc, 0.36, color=cfg.COR["etp_classica"], label=L["cla"] if j == 0 else None,
           yerr=b["classica"], error_kw=ekw)
    ax.bar(xj + 0.19, dq, 0.36, color=cfg.COR["etp_co2"], label=L["co2"] if j == 0 else None,
           yerr=b["co2"], error_kw=ekw)
    ax.text(xj - 0.19, dc + b["classica"] + 1, f"{dc:.0f}", ha="center", va="bottom", fontsize=5.5, color="#52514e")
    ax.text(xj + 0.19, dq + b["co2"] + 1, f"{dq:.0f}", ha="center", va="bottom", fontsize=5.5, color="#52514e")
    x.append(xj); rot.append(f"{rcp.replace('RCP', 'RCP ')}\n{h.replace('-', '–')}")
ax.set_xticks(x, rot, fontsize=6)
ax.set_ylabel(("Aumento da área suscetível\nà desertificação (10$^3$ km$^2$)" if PT
               else "Increase in area susceptible\nto desertification (10$^3$ km$^2$)"))
ax.axhline(0, color="#52514e", lw=0.6)
ax.grid(axis="y", color="#e5e4df", lw=0.5); ax.legend(frameon=False, loc="upper left")
rotulo(ax, "d")
salvar(fig, "Fig15_aridez")
