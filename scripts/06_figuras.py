"""
06 - Figuras 1 a 6 do manuscrito (PNG 600 dpi + PDF vetorial).

Estilo: fonte serifada (Liberation Serif / Times New Roman), largura de
coluna dupla Elsevier (174 mm), paleta segura para daltonismo.
Rotulos em portugues ou ingles: config.IDIOMA = "pt" | "en".

Limites estaduais: se o pacote geobr estiver instalado (pip install geobr),
os contornos de MA, TO, PI e BA sao desenhados nos mapas.
"""
import json
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import Patch
import config as cfg

MM = 1 / 25.4
W2 = cfg.LARG_DUPLA_MM * MM
mpl.rcParams.update({
    "font.family": "serif", "font.serif": cfg.FONTE, "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": cfg.DPI, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
})
PT = cfg.IDIOMA == "pt"
L = {
    "lon": "Longitude (°)", "lat": "Latitude (°)",
    "uso": "Uso da terra (MapBiomas)" if PT else "Land use (MapBiomas)",
    "nobs": "Sondagens OCO-2 válidas" if PT else "Valid OCO-2 soundings",
    "outros": "Outros" if PT else "Other",
    "xco2": "xCO$_2$ (ppm)", "fundo": "Média móvel de 12 meses" if PT else "12-month running mean",
    "mensal": "Média mensal" if PT else "Monthly mean",
    "etp_m": "ETP (mm mês$^{-1}$)" if PT else "PET (mm month$^{-1}$)",
    "cla": "ETP-PM" if PT else "PET-PM", "co2": "ETP-Y" if PT else "PET-Y",
    "dif_m": "ETP-PM − ETP-Y\n(mm mês$^{-1}$)" if PT else "PET-PM − PET-Y\n(mm month$^{-1}$)",
    "sif": "SIF (mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)",
    "dif_a": "ETP-PM − ETP-Y\n(mm ano$^{-1}$)" if PT else "PET-PM − PET-Y\n(mm yr$^{-1}$)",
    "sen": "Tendência (mm ano$^{-2}$)" if PT else "Trend (mm yr$^{-2}$)",
    "etr": "ETR MOD16 (mm mês$^{-1}$)" if PT else "MOD16 actual ET (mm month$^{-1}$)",
    "n": "n.º de pares" if PT else "no. of pairs",
    "fx": "Média móvel do xCO$_2$ (ppm)" if PT else "12-month running mean of xCO$_2$ (ppm)",
    "ef": "Efeito parcial em SIF/ETP" if PT else "Partial effect on SIF/PET",
    "dpv": "DPV (kPa)" if PT else "VPD (kPa)",
    "alto": "xCO$_2$ alto (> P75)" if PT else "High xCO$_2$ (> P75)",
    "baixo": "xCO$_2$ baixo (< P25)" if PT else "Low xCO$_2$ (< P25)",
    "aum": "Aumento da ETP (mm ano$^{-1}$)" if PT else "PET increase (mm yr$^{-1}$)",
    "comp": "comp." if PT else "offset",
    "sig": "pontos: p < 0,05" if PT else "dots: p < 0.05",
}
CLS = cfg.CLASSES if PT else cfg.CLASSES_EN


def salvar(fig, nome):
    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIG / f"{nome}.{ext}")
    plt.close(fig)
    print("ok", nome)


def rotulo(ax, letra):
    ax.text(0.0, 1.02, f"({letra})", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", fontsize=8)


def _limites():
    """Limites para os mapas. Ordem de preferencia: arquivos oficiais salvos por 13_limites_oficiais.py
    (IBGE 2020 + perimetro do MATOPIBA), geobr ao vivo, arquivo simplificado offline."""
    import geopandas as gpd
    uf = br = mato = None
    if (cfg.DATA / "estados_ibge.gpkg").exists():
        uf = gpd.read_file(cfg.DATA / "estados_ibge.gpkg")
        br = gpd.read_file(cfg.DATA / "brasil_ibge.gpkg") if (cfg.DATA / "brasil_ibge.gpkg").exists() else uf.dissolve()
    else:
        try:
            import geobr
            uf, br = geobr.read_state(year=2020), geobr.read_country(year=2020)
        except Exception:
            try:
                uf = gpd.read_file(cfg.DATA / "brasil_estados.geojson").rename(columns={"sigla": "abbrev_state"})
                br = uf.dissolve()
            except Exception:
                pass
    if (cfg.DATA / "matopiba.gpkg").exists():
        mato = gpd.read_file(cfg.DATA / "matopiba.gpkg")
    # simplifica os limites (~500 m, invisivel na escala dos mapas) para economizar memoria
    for g in (uf, br):
        if g is not None:
            g["geometry"] = g.geometry.simplify(0.005, preserve_topology=True)
    return uf, br, mato


UF_ALL, BR_ALL, MATOPIBA = _limites()
ESTADOS = None if UF_ALL is None else UF_ALL[UF_ALL.abbrev_state.isin(["MA", "TO", "PI", "BA"])]


# extensao dos mapas: perimetro do MATOPIBA com pequena margem
if MATOPIBA is not None:
    _b = MATOPIBA.total_bounds
    EXT = (_b[0] - 0.15, _b[1] - 0.15, _b[2] + 0.15, _b[3] + 0.15)
else:
    EXT = tuple(cfg.BBOX)


def contorno_matopiba(ax):
    """Mascara branca fora do MATOPIBA (recorte) + perimetro tracejado."""
    if MATOPIBA is not None:
        import geopandas as gpd
        from shapely.geometry import box
        fora = box(-80, -40, -20, 10).difference(MATOPIBA.geometry.iloc[0])
        gpd.GeoSeries([fora], crs=MATOPIBA.crs).plot(ax=ax, facecolor="white", edgecolor="none",
                                                     zorder=4.5, rasterized=True)
        MATOPIBA.boundary.plot(ax=ax, color="#0b0b0b", linewidth=0.9, linestyle=(0, (4, 2)), zorder=6)


def mapa_base(ax):
    if ESTADOS is not None:
        ESTADOS.boundary.plot(ax=ax, color="#52514e", linewidth=0.4, zorder=5)
    contorno_matopiba(ax)
    ax.set_xlim(EXT[0], EXT[2])
    ax.set_ylim(EXT[1], EXT[3])
    ax.set_aspect("equal")
    ax.set_xlabel(L["lon"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


NOMES_UF = {"MA": "Maranhão", "TO": "Tocantins", "PI": "Piauí", "BA": "Bahia"}


def rotulos_estados(ax, completo=False, fs=6.5):
    """Nome de cada estado na parte dele que fica dentro do MATOPIBA."""
    if ESTADOS is None or MATOPIBA is None:
        return
    poli = MATOPIBA.geometry.iloc[0]
    for _, r in ESTADOS.iterrows():
        g = r.geometry.intersection(poli)
        if g.is_empty:
            continue
        p = g.representative_point()
        ax.text(p.x, p.y, NOMES_UF[r.abbrev_state] if completo else r.abbrev_state, fontsize=fs,
                ha="center", va="center", color="#0b0b0b", zorder=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


def localizacao(ax, pos=(0.01, 0.74, 0.30, 0.25)):
    """Mapa do Brasil com os estados e o MATOPIBA em destaque."""
    if UF_ALL is None:
        return
    ins = ax.inset_axes(list(pos))
    UF_ALL.plot(ax=ins, color="#e5e4df", edgecolor="#8a8984", linewidth=0.2)
    ESTADOS.plot(ax=ins, color="#c9c8c2", edgecolor="#52514e", linewidth=0.3)
    if MATOPIBA is not None:
        MATOPIBA.plot(ax=ins, color="#e34948", edgecolor="#0b0b0b", linewidth=0.3, alpha=0.85)
    ins.text(-60, -6, "Brasil" if PT else "Brazil", fontsize=5.5, ha="center", color="#3d3d3a")
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white"); ins.set_zorder(9)
    for sp in ins.spines.values():
        sp.set_linewidth(0.5)


etp = xr.open_dataset(cfg.OUT / "etp_mensal.nc")
xco2 = xr.open_dataset(cfg.DATA / "xco2_mensal.nc")
sif = (xr.open_dataarray(cfg.DATA / "gosif_mensal.nc").interpolate_na("time")
       if (cfg.DATA / "gosif_mensal.nc").exists()
       else xr.open_dataset(cfg.DATA / "sif_mensal.nc")["sif_preenchido"])
uso = xr.open_dataset(cfg.DATA / "uso_terra.nc")["classe"]
tend = xr.open_dataset(cfg.OUT / "tendencias.nc")
R = json.load(open(cfg.OUT / "resumo_resultados.json"))

def figuras_1a6():
    """Figuras 1 a 6 (so roda quando este arquivo e executado diretamente)."""
    # ============================================================== Figura 1
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.62), constrained_layout=True)
    cores = ["#d9d8d3"] + [cfg.COR_CLASSES[k] for k in cfg.CLASSES]
    cmap = ListedColormap(cores)
    axs[0].pcolormesh(uso.lon, uso.lat, uso.where(uso.isin(list(cfg.CLASSES)), 0),
                      cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 4), 4), shading="nearest")
    mapa_base(axs[0]); axs[0].set_ylabel(L["lat"]); axs[0].set_title(L["uso"]); rotulo(axs[0], "a")
    rotulos_estados(axs[0], completo=True); localizacao(axs[0])
    axs[0].legend(handles=[Patch(color=cfg.COR_CLASSES[k], label=CLS[k]) for k in cfg.CLASSES]
                  + [Patch(color=cores[0], label=L["outros"]),
                     Line2D([], [], color="#0b0b0b", lw=0.9, ls=(0, (4, 2)), label="Limite do MATOPIBA" if PT else "MATOPIBA boundary")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=5, frameon=False,
                  handlelength=1.2, handletextpad=0.4, columnspacing=0.9, fontsize=6.5)
    n = xco2["n_obs"].sum("time")
    im = axs[1].pcolormesh(n.lon, n.lat, n.where(n > 0), cmap="Blues", shading="nearest")
    mapa_base(axs[1]); axs[1].set_title(L["nobs"]); rotulo(axs[1], "b")
    rotulos_estados(axs[1], completo=True)
    fig.colorbar(im, ax=axs[1], shrink=0.8, label=L["nobs"])
    salvar(fig, "Fig1_area_estudo")

    # ============================================================== Figura 2
    reg = etp.mean(["lat", "lon"])
    x_reg = reg.xco2
    fundo = x_reg.rolling(time=12, center=True, min_periods=6).mean()
    fig, axs = plt.subplots(4, 1, figsize=(W2, W2 * 0.95), sharex=True, constrained_layout=True)
    axs[0].plot(x_reg.time, x_reg, color=cfg.COR["xco2"], lw=1, alpha=0.55, label=L["mensal"])
    axs[0].plot(fundo.time, fundo, color=cfg.COR["xco2"], lw=2, label=L["fundo"])
    axs[0].set_ylabel(L["xco2"]); axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axs[1].plot(reg.time, reg.etp_classica, color=cfg.COR["etp_classica"], lw=1.5, label=L["cla"])
    axs[1].plot(reg.time, reg.etp_co2, color=cfg.COR["etp_co2"], lw=1.5, label=L["co2"])
    axs[1].set_ylabel(L["etp_m"]); axs[1].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axs[2].bar(pd.to_datetime(reg.time.values), reg.dif_co2.values, width=20, color=cfg.COR["neutro"])
    axs[2].set_ylabel(L["dif_m"])
    uso25 = uso.interp(lat=sif.lat, lon=sif.lon, method="nearest")
    for k in cfg.CLASSES:
        sk = sif.where(uso25 == k).mean(["lat", "lon"])
        axs[3].plot(sk.time, sk, color=cfg.COR_CLASSES[k], lw=1.5, label=CLS[k])
    axs[3].set_ylabel(L["sif"]); axs[3].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    for i, ax in enumerate(axs):
        rotulo(ax, "abcd"[i]); ax.grid(axis="y", color="#e5e4df", lw=0.5)
        if i != 2:                                   # espaco para a legenda
            y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.35 * (y1 - y0))
    salvar(fig, "Fig2_series_temporais")

    # ============================================================== Figura 3
    anual = etp[["etp_classica", "dif_co2"]].resample(time="YS").sum(min_count=1)
    dif = anual.dif_co2.mean("time")
    fig, axs = plt.subplots(1, 3, figsize=(W2, W2 * 0.5), constrained_layout=True)
    im = axs[0].pcolormesh(dif.lon, dif.lat, dif, cmap="Blues", shading="nearest")
    fig.colorbar(im, ax=axs[0], orientation="horizontal", shrink=0.9, label=L["dif_a"])
    axs[0].set_title("Correção pelo CO$_2$" if PT else "CO$_2$ correction")
    lim = float(np.nanmax(np.abs([tend.sen_etp_classica, tend.sen_etp_co2])))
    norm = TwoSlopeNorm(0, -lim, lim)
    for ax, v, tit in [(axs[1], "etp_classica", L["cla"]), (axs[2], "etp_co2", L["co2"])]:
        im2 = ax.pcolormesh(tend.lon, tend.lat, tend[f"sen_{v}"], cmap="RdBu_r", norm=norm, shading="nearest")
        sig = tend[f"p_{v}"].where(tend[f"p_{v}"] < 0.05).to_dataframe().dropna().reset_index()
        ax.scatter(sig.lon, sig.lat, s=0.8, color="#0b0b0b", linewidths=0)
        ax.set_title(tit)
    fig.colorbar(im2, ax=axs[1:], orientation="horizontal", shrink=0.6, label=f'{L["sen"]} — {L["sig"]}')
    for i, ax in enumerate(axs):
        mapa_base(ax); rotulo(ax, "abc"[i])
    axs[0].set_ylabel(L["lat"])
    salvar(fig, "Fig3_mapas_diferenca_tendencia")

    # ============================================================== Figura 4
    pv = pd.read_csv(cfg.OUT / "pares_validacao.csv")
    val = pd.read_csv(cfg.OUT / "validacao_etr.csv").query("grupo == 'Todas'").set_index("etp")
    lo_, hi_ = np.percentile(pd.concat([pv.etr, pv.cla, pv.co2]), [0.5, 99.5])
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.46), sharey=True, constrained_layout=True)
    for i, (ax, v) in enumerate(zip(axs, ["cla", "co2"])):
        hb = ax.hexbin(pv[v], pv.etr, gridsize=40, cmap="Greys", mincnt=1, extent=(lo_, hi_, lo_, hi_), linewidths=0)
        ax.plot([lo_, hi_], [lo_, hi_], color=cfg.COR[f"etp_{'classica' if v == 'cla' else 'co2'}"], lw=1.5)
        ax.set_xlabel(L[v] + (" (mm mês$^{-1}$)" if PT else " (mm month$^{-1}$)"))
        k = val.loc[v]
        dec = "," if PT else "."
        ax.text(0.97, 0.05, f"KGE = {k.kge:.2f}\n{'viés' if PT else 'bias'} = {k.vies:.1f} mm\nRMSE = {k.rmse:.1f} mm".replace(".", dec),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.set_aspect("equal"); rotulo(ax, "ab"[i])
    axs[0].set_ylabel(L["etr"])
    fig.colorbar(hb, ax=axs, shrink=0.8, label=L["n"])
    salvar(fig, "Fig4_validacao_etr")

    # ============================================================== Figura 5
    pe = pd.read_csv(cfg.OUT / "efeitos_parciais.csv").query("termo == 'xco2_fundo'")
    sd = pd.read_csv(cfg.OUT / "sif_dpv.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.42), constrained_layout=True)
    axs[0].fill_between(pe.x, pe.li, pe.ls, color=cfg.COR["xco2"], alpha=0.2, lw=0)
    axs[0].plot(pe.x, pe.efeito, color=cfg.COR["xco2"], lw=2)
    axs[0].axhline(0, color="#52514e", lw=0.6, ls="--")
    axs[0].set_xlabel(L["fx"]); axs[0].set_ylabel(L["ef"])
    bins = np.quantile(sd.dpv, np.linspace(0, 1, 13))
    for grp, cor in [("baixo", cfg.COR["etp_classica"]), ("alto", cfg.COR["etp_co2"])]:
        g = sd[sd.grupo == grp].copy()
        g["b"] = pd.cut(g.dpv, bins, include_lowest=True)
        # erro padrao entre celulas (media de cada celula no intervalo de DPV), unidade independente do teste
        c = g.groupby(["b", "celula"], observed=True).agg(x=("dpv", "mean"), y=("sif", "mean")).reset_index()
        a = c.groupby("b", observed=True).agg(x=("x", "mean"), y=("y", "mean"), e=("y", "sem"))
        zero = np.zeros(len(a))
        yerr = [zero, a.e.values] if grp == "alto" else [a.e.values, zero]    # alto: barra para cima; baixo: para baixo
        axs[1].errorbar(a.x, a.y, yerr=yerr, color=cor, marker="o", ms=3.5, lw=1.5, capsize=2.5, elinewidth=0.9,
                        label=L[grp])
    axs[1].set_xlabel(L["dpv"]); axs[1].set_ylabel(L["sif"]); axs[1].legend(frameon=False)
    for i, ax in enumerate(axs):
        rotulo(ax, "ab"[i]); ax.grid(color="#e5e4df", lw=0.5)
    salvar(fig, "Fig5_sif_xco2_dpv")

    # ============================================================== Figura 6
    tab = pd.read_csv(cfg.OUT / "tabela_cenarios.csv").query("cenario != 'Referencia'")
    inc = pd.read_csv(cfg.OUT / "incerteza_anual.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, W2 * 0.4), sharey=True, constrained_layout=True)
    larg = 0.38
    for i, (ax, rcp) in enumerate(zip(axs, cfg.RCP_CO2)):
        t = tab[tab.cenario == rcp].set_index("horizonte").loc[cfg.HORIZONTES]
        x = np.arange(len(t))
        ep = inc[inc.cenario == rcp]
        e_c = ep[ep.etp == "classica"].set_index("horizonte").loc[cfg.HORIZONTES, "aumento_etp_ep"].values
        e_y = ep[ep.etp == "co2"].set_index("horizonte").loc[cfg.HORIZONTES, "aumento_etp_ep"].values
        ekw = dict(elinewidth=0.9, capsize=2.5, ecolor="#0b0b0b")
        ax.bar(x - larg / 2 - 0.01, t.aumento_classica, larg, color=cfg.COR["etp_classica"], label=L["cla"],
               yerr=e_c, error_kw=ekw)
        ax.bar(x + larg / 2 + 0.01, t.aumento_co2, larg, color=cfg.COR["etp_co2"], label=L["co2"],
               yerr=e_y, error_kw=ekw)
        for xi, (a, c) in enumerate(zip(t.aumento_classica, t.compensacao_pct)):
            txt = f"{c:.0f}%"
            ax.text(xi, a * 1.02, f"{L['comp']} {txt}", ha="center", va="bottom", fontsize=6.5, color="#52514e")
        ax.set_xticks(x, [h.replace("-", "–") for h in t.index], rotation=0)
        ax.set_title(rcp.replace("RCP", "RCP ")); ax.grid(axis="y", color="#e5e4df", lw=0.5)
        ax.axhline(0, color="#52514e", lw=0.6); rotulo(ax, "abc"[i])
    axs[0].set_ylabel(L["aum"])
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.06, 0.93))
    ymax = tab.aumento_classica.max()
    for ax in axs:
        ax.set_ylim(min(0, tab.aumento_co2.min() * 1.1), ymax * 1.18)
    salvar(fig, "Fig6_cenarios_rcp")


if __name__ in ("__main__", "x"):
    figuras_1a6()
    # demais figuras: cada script roda em um processo separado, que devolve a memoria ao terminar
    import gc, subprocess, sys
    plt.close("all"); gc.collect()
    for s in ("07_figuras_inovadoras.py", "08_mapas.py", "09_aridez.py", "10_sif_chuva.py", "11_sif.py", "12_xco2.py"):
        if s not in ("07_figuras_inovadoras.py", "08_mapas.py") and not (cfg.DATA / "prec_mensal.nc").exists():
            continue
        subprocess.run([sys.executable, str(cfg.SCRIPTS / s)], check=True)
