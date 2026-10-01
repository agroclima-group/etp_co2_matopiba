# Effect of atmospheric CO₂ on potential evapotranspiration in MATOPIBA, Brazil

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23087114.svg)](https://doi.org/10.5281/zenodo.23087114)

Code and derived data for the article:

> Meneses, K.C., Santos, G.A.A., Silva, M.V., Rolim, G.S., La Scala Jr., N. *Effect of atmospheric CO₂ on potential evapotranspiration in the drylands of MATOPIBA, Brazil: evidence from satellite-observed xCO₂ and SIF.* Manuscript in preparation.

**Authors**

- Kamila Cunha de Meneses¹ (corresponding: meneses.kamila@ufma.br)
- Gustavo André de Araújo dos Santos¹
- Marcos Vinicius da Silva¹
- Glauco de Souza Rolim²
- Newton La Scala Junior²

¹ Federal University of Maranhão (UFMA), Chapadinha Science Center (CCCh), Chapadinha, MA, Brazil
² São Paulo State University (UNESP), School of Agricultural and Veterinary Sciences (FCAV), Jaboticabal, SP, Brazil

## What the code does

Two estimates of potential evapotranspiration (PET) are computed with [PyEt](https://github.com/pyet-org/pyet) on a 0.25° grid over MATOPIBA (964 cells, ≈736,000 km²):

- **PET-PM**: FAO-56 Penman-Monteith with fixed surface resistance (`pyet.pm(co2=300)`).
- **PET-Y**: Penman-Monteith modified by Yang et al. (2019), with surface resistance increasing with CO₂ (`pyet.pm(co2=xCO₂)`), driven by OCO-2 xCO₂.

The pipeline covers the observed period (2015–2025: xCO₂, PET, SIF from GOSIF, aridity index) and future scenarios (RCP 2.6, 4.5 and 8.5; 2021–2040, 2041–2060, 2081–2100) with the delta method, including the CO₂ offset and the area susceptible to desertification (UNCCD classes).

## Repository structure

```
scripts/     Python pipeline (run in numerical order) and run_all.sh
data_raw/    Raw downloads clipped to MATOPIBA: ERA5-Land daily by year, OCO-2 Lite soundings (parquet), GOSIF (npy)
data/        Processed monthly grids (NetCDF), MATOPIBA boundary (matopiba.gpkg) and CMIP5 regional deltas
results/     Derived tables (CSV/JSON) and gridded outputs (NetCDF) used in the article
figures/     Final figures of the article (Figure_1–9, Figure_S1–S4; PNG and PDF)
```

All inputs used in the article are included (≈330 MB), so scripts 03 onwards can be run directly on them. The only exception is `data/era5land_diario.nc` (243 MB, above GitHub's file limit), which can be rebuilt from the yearly files in `data_raw/` (`01_download_gee.py`, or `xr.open_mfdataset("data_raw/era5land_*.nc")`). The download scripts skip files that already exist. Figures produced by the scripts are written to `figures/generated/`.

## Data sources

| Variable | Source | Script |
| --- | --- | --- |
| Meteorology (daily) | ERA5-Land, via Google Earth Engine | `01_download_gee.py` |
| LAI, AET, land use | MODIS MCD15A2H, MOD16A2, MapBiomas | `01_download_gee.py` |
| ΔT and precipitation ratio | NASA NEX-GDDP (CMIP5) | `01b_deltaT_gee.py`, `01c_precipitacao_gee.py` |
| Precipitation | CHIRPS | `01c_precipitacao_gee.py` |
| xCO₂ | OCO-2 Lite (NASA GES DISC, Earthdata) | `02_download_oco.py` |
| SIF | GOSIF v2 (Li and Xiao, 2019) | `02b_gosif.py` |
| Boundaries | IBGE 2020; MATOPIBA (Embrapa/GITE) | `13_limites_oficiais.py` |

## How to run

1. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate etp_co2
   ```
2. Accounts (free):
   - Google Earth Engine project registered for non-commercial use; then `earthengine authenticate` and `export GEE_PROJECT=<your-project-id>`.
   - NASA Earthdata account with the application *NASA GESDISC DATA ARCHIVE* authorised. Credentials are read from `~/.netrc` or from the `EARTHDATA_TOKEN` environment variable. **Never commit credentials.**
3. Run the full pipeline from the repository root:
   ```bash
   bash scripts/run_all.sh
   ```
   or run the scripts one by one (`python scripts/03_etp.py`, …). Completed steps are skipped on re-runs.

To test the pipeline without downloading anything: `TESTE=1 python scripts/00_dados_sinteticos.py` (synthetic data; do not use the results). Figure labels in English: `IDIOMA=en`.

| Script | Output |
| --- | --- |
| `03_etp.py` | Monthly PET-PM and PET-Y (`results/etp_mensal.nc`, not versioned, 15 MB) |
| `04_cenarios.py` | Scenario table and maps, S_rs sensitivity |
| `05_estatistica.py` | Trends, AET comparison, mixed model, SIF–VPD, Moran's I |
| `06_figuras.py` – `12_xco2.py` | Figures and summaries |
| `09_aridez.py` | Aridity index and susceptible area |
| `14_incerteza_anual.py`, `14b_bootstrap_area.py` | Standard errors and bootstrap CI |

Comments in the code are in Portuguese.

## Citation

If you use this code or data, please cite the article above and the Zenodo record:

> Meneses, K.C., Santos, G.A.A., Silva, M.V., Rolim, G.S., La Scala Jr., N. (2026). *Code and data for: Ignoring plant responses to rising CO₂ overestimates dryland expansion under high emissions in a Brazilian Cerrado agricultural frontier (MATOPIBA)* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23087114

## License

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0. Source datasets remain subject to the terms of their providers.
