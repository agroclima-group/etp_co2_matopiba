"""
01b - Delta de temperatura mensal por cenario e horizonte a partir do
      NASA NEX-GDDP (CMIP5 regionalizado, 0,25 grau, 21 modelos) no Earth Engine.

dT(cenario, horizonte, mes) = Tmedia_conjunto(horizonte) - Tmedia_conjunto(2015-2025),
media regional sobre o MATOPIBA, Tmedia = (tasmax + tasmin) / 2, media do conjunto de modelos.

O NEX-GDDP (CMIP5) traz apenas RCP 4.5 e RCP 8.5. O RCP 2.6 e obtido por
escalonamento de padrao (pattern scaling) a partir do RCP 4.5, usando a razao
do aquecimento medio global do IPCC AR5 (WG1, Tabela SPM.2):
    2046-2065: RCP2.6 1,0 / RCP4.5 1,4 ;  2081-2100: 1,0 / 1,8
Para 2021-2040 usa-se a razao de 2046-2065 (aproximacao). Isso e declarado
no texto como limitacao; a alternativa e trocar todo o conjunto para o
NEX-GDDP-CMIP6 (SSP1-2.6, SSP2-4.5, SSP5-8.5).

Saida: data/cmip5_deltaT.csv (rcp, horizonte, mes, dT)
"""
import ee
import pandas as pd
import config as C

ee.Initialize(project=C.GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
geom = ee.Geometry.Rectangle(list(C.BBOX))
nex = ee.ImageCollection("NASA/NEX-GDDP")
PERIODOS = {"base": ("2015-01-01", "2026-01-01"),
            "2021-2040": ("2021-01-01", "2041-01-01"),
            "2041-2060": ("2041-01-01", "2061-01-01"),
            "2081-2100": ("2081-01-01", "2101-01-01")}
RAZAO_26 = {"2021-2040": 1.0 / 1.4, "2041-2060": 1.0 / 1.4, "2081-2100": 1.0 / 1.8}


# 4 dias por mes (1, 8, 15, 22): climatologia mensal com ~1/7 do custo de usar todos os dias
DIAS = ee.Filter.Or(*[ee.Filter.calendarRange(d, d, "day_of_month") for d in (1, 8, 15, 22)])


def climatologia(cenario, ini, fim):
    """Uma imagem de 12 bandas (T media de cada mes) e UMA reducao regional."""
    ic = (nex.filter(ee.Filter.eq("scenario", cenario)).filterDate(ini, fim)
          .filter(DIAS).select(["tasmax", "tasmin"]))
    bandas = []
    for m in range(1, 13):
        img = ic.filter(ee.Filter.calendarRange(m, m, "month")).mean()
        bandas.append(img.expression("(b('tasmax') + b('tasmin')) / 2 - 273.15").rename(f"m{m:02d}"))
    img = ee.Image.cat(bandas)
    d = img.reduceRegion(ee.Reducer.mean(), geom, scale=50000, maxPixels=1e10, tileScale=4).getInfo()
    return {m: d[f"m{m:02d}"] for m in range(1, 13)}


linhas = []
for rcp_nex, rcp in [("rcp45", "RCP4.5"), ("rcp85", "RCP8.5")]:
    base = climatologia(rcp_nex, *PERIODOS["base"])
    print(rcp, "base ok", flush=True)
    for h in C.HORIZONTES:
        fut = climatologia(rcp_nex, *PERIODOS[h])
        for mes in range(1, 13):
            linhas.append(dict(rcp=rcp, horizonte=h, mes=mes, dT=round(fut[mes] - base[mes], 3)))
        print(rcp, h, "dT medio = %.2f C" % (sum(fut[m] - base[m] for m in fut) / 12), flush=True)

df = pd.DataFrame(linhas)
r26 = df[df.rcp == "RCP4.5"].copy()
r26["rcp"] = "RCP2.6"
r26["dT"] = (r26.dT * r26.horizonte.map(RAZAO_26)).round(3)
df = pd.concat([r26, df], ignore_index=True)
df.to_csv(C.DELTA_T_CSV, index=False)
print(df.groupby(["rcp", "horizonte"]).dT.mean().round(2))
