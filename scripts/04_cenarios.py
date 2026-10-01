"""
04 - Projecoes da ETP sob RCP 2.6, 4.5 e 8.5 (metodo delta) com
     experimento fatorial temperatura x CO2 e sensibilidade de S_rs.

Base: ERA5-Land 2015-2025. Para cada cenario/horizonte soma-se o dT mensal
(mediana CMIP5) a Tmax/Tmin, mantendo a umidade relativa; o CO2 vem de
config.RCP_CO2.

Definicoes (medias regionais, mm ano-1):
    ref        = pm(T,      C_ref)     C_ref = XCO2 medio observado 2015-2025
    so_T       = pm(T + dT, C_ref)
    so_CO2     = pm(T,      C_fut)
    ambos      = pm(T + dT, C_fut)     -> ETP-CO2 futura
    cla_ref    = pm(T,      300)       -> ETP classica atual
    cla_fut    = pm(T + dT, 300)       -> ETP classica futura
    compensacao (%) = 100 * [1 - (ambos - ref) / (cla_fut - cla_ref)]

Saidas: results/tabela_cenarios.csv, results/cenarios_mapas.nc (mapas por celula),
        results/sensibilidade_srs.csv
"""
import numpy as np
import pandas as pd
import xarray as xr
import config as C
from importlib import import_module

E = import_module("03_etp")

# precipitacao (indice de aridez): baixa do Earth Engine se ainda nao existir
if not C.TESTE and not ((C.DATA / "prec_mensal.nc").exists() and (C.DATA / "cmip5_razaoP.csv").exists()):
    try:
        import runpy
        runpy.run_path(str(C.SCRIPTS / "01c_precipitacao_gee.py"), run_name="__prec__")
    except Exception as e:
        print("aviso: precipitacao nao baixada (indice de aridez sera pulado):", e)

era = C.recorte(xr.open_dataset(C.DATA / "era5land_diario.nc"))
C_REF = float(xr.open_dataset(C.DATA / "xco2_mensal.nc")["xco2_preenchido"].mean())
dT_tab = pd.read_csv(C.DELTA_T_CSV)          # rcp, horizonte, mes, dT
anos = np.unique(era.time.dt.year)
n_anos = len(anos)


def dT_diario(tempo, rcp, horiz):
    t = dT_tab[(dT_tab.rcp == rcp) & (dT_tab.horizonte == horiz)].set_index("mes")["dT"]
    return xr.DataArray(t.reindex(tempo.dt.month.values).values, coords={"time": tempo}, dims="time")


MAPAS = {}


def total_anual(co2, rcp=None, horiz=None, srs=C.SRS, nome=None):
    """ETP media regional (mm ano-1) na base 2015-2025 com dT e CO2 dados.
    Se nome for dado, guarda tambem o mapa (mm ano-1 por celula) em MAPAS."""
    soma = None
    for _, ds in era.groupby("time.year"):
        v = E.preparar(ds.load())
        dT = 0.0 if rcp is None else dT_diario(ds.time, rcp, horiz)
        a = E.etp_pm(v, co2, srs=srs, dT=dT).sum("time", min_count=1)
        soma = a if soma is None else soma + a
    mapa = soma / n_anos
    if nome:
        MAPAS[nome] = mapa
    return float(mapa.mean())


if __name__ == "__main__":
    ref = total_anual(C_REF, nome="ref")
    cla_ref = total_anual(C.CO2_BASE, nome="cla_ref")
    linhas = [dict(cenario="Referencia", horizonte="2015-2025", co2=C_REF,
                   etp_classica=cla_ref, etp_co2=ref)]
    for rcp, hs in C.RCP_CO2.items():
        for h, cfut in hs.items():
            k = f"{rcp}_{h}"
            so_T = total_anual(C_REF, rcp, h, nome=f"soT_{k}")
            so_CO2 = total_anual(cfut, nome=f"soCO2_{k}")
            ambos = total_anual(cfut, rcp, h, nome=f"ambos_{k}")
            cla_fut = total_anual(C.CO2_BASE, rcp, h, nome=f"clafut_{k}")
            linhas.append(dict(
                cenario=rcp, horizonte=h, co2=cfut,
                etp_classica=cla_fut, etp_co2=ambos,
                aumento_classica=cla_fut - cla_ref, aumento_co2=ambos - ref,
                efeito_T=so_T - ref, efeito_CO2=so_CO2 - ref,
                interacao=ambos - so_T - so_CO2 + ref,
                compensacao_pct=100 * (1 - (ambos - ref) / (cla_fut - cla_ref)),
            ))
            print(rcp, h, "compensacao = %.1f%%" % linhas[-1]["compensacao_pct"])
    tab = pd.DataFrame(linhas)
    tab["diferenca"] = tab.etp_classica - tab.etp_co2
    tab.round(2).to_csv(C.OUT / "tabela_cenarios.csv", index=False)
    xr.Dataset({k.replace(".", "").replace("-", "_"): v for k, v in MAPAS.items()}).to_netcdf(C.OUT / "cenarios_mapas.nc")

    # Sensibilidade de S_rs no cenario mais extremo
    rcp, h = "RCP8.5", "2081-2100"
    cfut = C.RCP_CO2[rcp][h]
    cla_fut = total_anual(C.CO2_BASE, rcp, h)
    sens = []
    for s in C.SRS_SENS:
        r = total_anual(C_REF, srs=s)
        a = total_anual(cfut, rcp, h, srs=s)
        sens.append(dict(srs=s, compensacao_pct=100 * (1 - (a - r) / (cla_fut - cla_ref))))
    pd.DataFrame(sens).round({"srs": 6, "compensacao_pct": 2}).to_csv(C.OUT / "sensibilidade_srs.csv", index=False)
    print(pd.DataFrame(sens))
