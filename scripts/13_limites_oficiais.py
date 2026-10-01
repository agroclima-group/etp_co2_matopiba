"""
13 - Baixa os limites oficiais para os mapas (roda no Mac, precisa de internet):

  data/estados_ibge.gpkg   Limites estaduais do IBGE (malha 2020) via pacote geobr
  data/brasil_ibge.gpkg    Contorno do Brasil (IBGE 2020) via geobr
  data/matopiba.gpkg       Perimetro do MATOPIBA (delimitacao Embrapa/GITE, 337 municipios),
                            obtido da relacao 19040398 do OpenStreetMap (API Overpass)

Uso: python scripts/13_limites_oficiais.py
"""
import json
import sys
import urllib.parse
import urllib.request
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, polygonize, unary_union
import config as cfg

ok = True

# ------------------------------------------------------------------ IBGE (geobr)
try:
    import geobr
    uf = geobr.read_state(year=2020, simplified=False)
    uf = uf.rename(columns={"abbrev_state": "abbrev_state"})
    uf.to_file(cfg.DATA / "estados_ibge.gpkg", driver="GPKG")
    br = geobr.read_country(year=2020, simplified=False)
    br.to_file(cfg.DATA / "brasil_ibge.gpkg", driver="GPKG")
    print("IBGE: limites estaduais e do Brasil salvos", len(uf), "estados")
except Exception as e:
    ok = False
    print("ERRO ao baixar a malha do IBGE:", e)

# ------------------------------------------------------------------ MATOPIBA (OSM / Overpass)
consulta = "[out:json][timeout:120];relation(19040398);out geom;"
servidores = ["https://overpass-api.de/api/interpreter",
              "https://overpass.kumi.systems/api/interpreter",
              "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
dado = None
for s in servidores:
    try:
        req = urllib.request.Request(s, data=urllib.parse.urlencode({"data": consulta}).encode(),
                                     headers={"User-Agent": "etp_co2_matopiba/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            dado = json.load(r)
        if dado.get("elements"):
            print("Overpass:", s)
            break
    except Exception as e:
        print("  falhou", s, e)
try:
    rel = dado["elements"][0]
    linhas = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
              for m in rel["members"] if m["type"] == "way" and m.get("role", "outer") in ("outer", "")]
    poli = unary_union(list(polygonize(linemerge(linhas))))
    g = gpd.GeoDataFrame({"nome": ["MATOPIBA"]}, geometry=[poli], crs="EPSG:4326")
    area = g.to_crs("EPSG:5880").area.iloc[0] / 1e4
    g.to_file(cfg.DATA / "matopiba.gpkg", driver="GPKG")
    print(f"MATOPIBA salvo: {area/1e6:.1f} milhoes de ha (referencia Embrapa: 73,2 milhoes de ha)")
except Exception as e:
    ok = False
    print("ERRO ao montar o perimetro do MATOPIBA:", e)

print("\n>> Pronto." if ok else "\n>> Terminou com erros (veja acima).")
sys.exit(0 if ok else 1)
