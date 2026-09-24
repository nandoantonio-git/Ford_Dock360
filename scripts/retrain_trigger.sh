#!/usr/bin/env bash
# Gatilho seguro de retreino trimestral em modo candidato isolado.
# Uso:
#   ./scripts/retrain_trigger.sh 2024-01-31 \
#     --authorized-by "Fernando RM555201" \
#     --authorization-ticket "Sprint3-demo" \
#     --scope "Gerar candidato isolado"

set -euo pipefail

CADENCIA_MINIMA_MESES=3
JANELA_CHURN_MESES=18
CORTE_ATUAL_FILE="src/pipeline/config.py"
CANDIDATE_DATA_DIR="data/processed/candidate"
CANDIDATE_MODEL_DIR="models/candidate"

usage() {
  sed -n '2,8p' "$0"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

NOVO_CORTE="$1"
shift

AUTHORIZED_BY=""
AUTHORIZATION_TICKET=""
SCOPE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --authorized-by)
      AUTHORIZED_BY="${2:-}"
      shift 2
      ;;
    --authorization-ticket)
      AUTHORIZATION_TICKET="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Argumento desconhecido: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$AUTHORIZED_BY" ] || [ -z "$AUTHORIZATION_TICKET" ] || [ -z "$SCOPE" ]; then
  echo "Rejeitado: autorizacao explicita obrigatoria (--authorized-by, --authorization-ticket, --scope)." >&2
  exit 1
fi

CORTE_ATUAL=$(python3 - <<'PY'
import re
from pathlib import Path
text = Path('src/pipeline/config.py').read_text(encoding='utf-8')
m = re.search("DATA_CORTE\\s*=\\s*['\"]([0-9-]+)['\"]", text)
print(m.group(1) if m else '')
PY
)

if [ -z "$CORTE_ATUAL" ]; then
  echo "Nao foi possivel localizar DATA_CORTE atual em $CORTE_ATUAL_FILE" >&2
  exit 1
fi

DIFF_MESES=$(python3 - <<PY
from datetime import datetime
atual = datetime.strptime('$CORTE_ATUAL', '%Y-%m-%d')
novo = datetime.strptime('$NOVO_CORTE', '%Y-%m-%d')
print((novo.year - atual.year) * 12 + (novo.month - atual.month))
PY
)

if [ "$DIFF_MESES" -lt "$CADENCIA_MINIMA_MESES" ]; then
  echo "Rejeitado: novo corte esta a apenas $DIFF_MESES mes(es) do corte atual." >&2
  echo "Cadencia minima: $CADENCIA_MINIMA_MESES meses." >&2
  exit 1
fi

echo "== Retreino candidato iniciado =="
echo "Corte atual: $CORTE_ATUAL"
echo "Novo corte: $NOVO_CORTE"
echo "Janela churn: ${JANELA_CHURN_MESES}m"
echo "Autorizado por: $AUTHORIZED_BY"
echo "Ticket: $AUTHORIZATION_TICKET"
echo "Escopo: $SCOPE"

mkdir -p "$CANDIDATE_DATA_DIR" "$CANDIDATE_MODEL_DIR"

python3 -m src.pipeline.feature_engineering_real \
  --data-corte "$NOVO_CORTE" \
  --janela-churn-meses "$JANELA_CHURN_MESES" \
  --output-dir "$CANDIDATE_DATA_DIR" \
  --fail-on-immature-window

python3 -m src.pipeline.clustering_real \
  --input-dir "$CANDIDATE_DATA_DIR" \
  --output-dir "$CANDIDATE_MODEL_DIR"

python3 -m src.pipeline.train_churn_real \
  --input-dir "$CANDIDATE_DATA_DIR" \
  --output-dir "$CANDIDATE_MODEL_DIR"

echo "Modelo candidato treinado em $CANDIDATE_MODEL_DIR"
echo "Dados candidatos gerados em $CANDIDATE_DATA_DIR"
echo "Proximo passo: python3 -m src.pipeline.promotion_gate --candidate-dir $CANDIDATE_MODEL_DIR"
