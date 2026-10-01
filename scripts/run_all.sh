#!/bin/bash
# Roda o pipeline completo no Mac (Anaconda/Miniconda).
# Uso:  abra o Terminal nesta pasta e digite:   bash rodar_tudo.sh
# Se cair no meio, rode de novo: as etapas ja concluidas sao puladas.
set -eo pipefail
cd "$(dirname "$0")/.."   # raiz do repositorio
# impede o repouso do Mac enquanto o script roda
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &
mkdir -p logs
# localiza o Anaconda/Miniconda
for c in "$HOME/anaconda3" "$HOME/opt/anaconda3" "/opt/anaconda3" "$HOME/miniconda3" "$HOME/opt/miniconda3" "/opt/miniconda3" "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniconda/base"; do
  if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
command -v conda >/dev/null || { echo "Anaconda nao encontrado. Instale em https://www.anaconda.com/download"; exit 1; }

if ! conda env list | grep -q "^etp_co2 "; then
  echo ">> Criando o ambiente etp_co2 (so na primeira vez, ~5 min)"
  conda env create -f environment.yml
fi
conda activate etp_co2

GEE_PROJECT="${GEE_PROJECT:-}"
if [ -z "$GEE_PROJECT" ]; then
  read -p "ID do seu projeto Google Cloud cadastrado no Earth Engine: " GEE_PROJECT
fi
export GEE_PROJECT

if [ ! -f "$HOME/.config/earthengine/credentials" ]; then
  echo ">> Login do Earth Engine (abre o navegador)"
  earthengine authenticate
fi

etapa () {   # etapa <script> <arquivo que indica conclusao>
  if [ -f "$2" ]; then echo ">> $1 ja concluido, pulando"; return; fi
  echo ">> $1  ($(date '+%H:%M'))"
  python "scripts/$1" 2>&1 | tee "logs/${1%.py}.log"
}

etapa 01_download_gee.py   data/uso_terra.nc
etapa 01b_deltaT_gee.py    data/cmip5_deltaT.csv
if [ ! -f data/xco2_mensal.nc ] && [ -z "$EARTHDATA_TOKEN" ] && [ ! -s token_earthdata.txt ] && ! grep -q urs.earthdata "$HOME/.netrc" 2>/dev/null; then
  echo ""
  echo ">> NASA Earthdata: cole o TOKEN gerado em https://urs.earthdata.nasa.gov/profile (aba Generate Token)"
  echo "   e tecle Enter. (Deixe em branco para digitar usuario e senha.) O texto nao aparece ao colar."
  read -s -p "Token: " EARTHDATA_TOKEN; echo ""
  export EARTHDATA_TOKEN
fi
etapa 02_download_oco.py   data/xco2_mensal.nc      # pede usuario/senha do Earthdata na 1a vez
etapa 03_etp.py            results/etp_mensal.nc
etapa 04_cenarios.py       results/sensibilidade_srs.csv
etapa 05_estatistica.py    results/resumo_resultados.json
python scripts/06_figuras.py 2>&1 | tee logs/06_figuras.log

echo ""
echo ">> Pronto. Resultados em results/ e figuras em figures/generated/."
