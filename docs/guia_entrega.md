# Guia de leitura e execução — Sprint 3

## Para compreender a solução

1. Leia `docs/ia_ml_requisitos.md`: problema, preparação, comparação e conclusão.
2. Consulte `reports/model_comparison_churn.csv` para os resultados dos candidatos.
3. Abra `notebooks/00_pipeline_completo.ipynb` para inspecionar a execução salva.
4. Consulte `docs/release_sprint3.md` para notas e gráficos comentados.
5. Leia o README para contratos da API e comandos do pipeline.

O pacote de código não contém dados Ford, segredos ou modelos binários.
Os resultados salvos permitem a análise acadêmica, mas não substituem a
reprodução com dados autorizados nem a validação dos artefatos de inferência.

## Preparar o ambiente

Use Python 3.11 conforme `.python-version`. Em uma pasta extraída e local:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

A validação registrada nesta entrega foi feita em Python 3.10.11, com as
versões principais de ML e FastAPI declaradas no projeto. Uma instalação
limpa em Python 3.11 ainda precisa de confirmação. Evite ambientes virtuais
armazenados apenas sob demanda em pastas sincronizadas.

## Reproduzir o treinamento

Disponibilize a base autorizada em `data/raw/vin_share_Desafio_02.xlsx`, aba
`vin_share`, e siga a ordem de comandos em `docs/ia_ml_requisitos.md`.
Execute em cópia isolada: a pipeline grava modelos e relatórios.

## Executar a inferência local

É necessário obter o conjunto correspondente de artefatos confiáveis:

- `models/churn_pos_venda_rf_calibrated.joblib` e seu `.sha256`;
- `models/kmeans_segmentador_pos_venda.joblib` e seu `.sha256`;
- `models/cluster_segment_map.json` da mesma execução do segmentador.

Confira os checksums antes de carregar os modelos. Não substitua esses
arquivos por artefatos de execuções diferentes. O pacote atual não inclui
esses modelos: a verificação dos arquivos locais está pendente de acesso ao
conteúdo disponível sob demanda no macOS.

Configure `SECRET_KEY` e `ML_SERVICE_TOKEN` no ambiente local, conforme README,
e execute:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Envie o exemplo de features do README para `/predict`, autenticado pelo header
`X-ML-Service-Token`. `/health` confirma presença dos arquivos, não substitui
uma predição real para validar carregamento e compatibilidade.
