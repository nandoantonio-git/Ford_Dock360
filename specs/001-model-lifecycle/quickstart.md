# Quickstart: Ciclo de Vida do Modelo Ford VinGuard

Este guia valida a feature de ciclo de vida do modelo em modo acadêmico/demo. Ele assume que o dataset real e os artefatos base estão disponíveis localmente quando a demo completa for executada.

## 1. Pré-requisitos

- Estar na raiz do projeto: `~/Desktop/Ford_Dock 360`.
- Dataset real disponível em `data/raw/vin_share_Desafio_02.xlsx`.
- Ambiente Python com dependências do projeto instaladas.
- MLflow local configurável em `file:./mlruns`.
- Modelo champion existente em `models/` ou artefato equivalente documentado.

## 2. Verificações estáticas rápidas

```bash
pytest tests/test_leakage.py -v
pytest tests/test_model_lifecycle.py -v
python -m src.pipeline.feature_engineering_real --help
python -m src.pipeline.clustering_real --help
python -m src.pipeline.train_churn_real --help
python -m src.pipeline.holdout_gate --help
python -m src.pipeline.promotion_gate --help
python -m src.pipeline.drift_monitor --help
bash -n scripts/retrain_trigger.sh
```

Expected outcome:

- Testes passam.
- Todos os CLIs exibem ajuda.
- Script Bash não tem erro de sintaxe.

## 3. Gerar holdout temporal do gate

```bash
python -m src.pipeline.holdout_gate \
  --input data/processed/dataset_churn_pos_venda.csv \
  --output data/processed/holdout_temporal_gate.joblib \
  --data-corte-holdout 2024-01-31
```

Expected outcome:

- `data/processed/holdout_temporal_gate.joblib` criado.
- Metadados incluem colunas de features, target, corte e checksum.
- Nenhuma coluna de futuro/target entra no X.

## 4. Gerar candidato isolado

```bash
./scripts/retrain_trigger.sh 2024-01-31 \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo" \
  --scope "Gerar candidato isolado para gate acadêmico"
```

Expected outcome:

- Corte validado contra maturação de 18 meses.
- Autorização registrada.
- Artefatos candidatos em `models/candidate/`.
- Datasets candidatos em `data/processed/candidate/`.
- Nenhum arquivo de produção em `models/` é substituído.
- `src/pipeline/config.py` não é alterado.

## 5. Executar gate em modo relatório

```bash
python -m src.pipeline.promotion_gate \
  --candidate-dir models/candidate \
  --holdout-path data/processed/holdout_temporal_gate.joblib \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo" \
  --scope "Avaliar candidato para promoção" \
  --output-dir reports/model_lifecycle
```

Expected outcome:

- JSON criado em `reports/model_lifecycle/`.
- Resultado contém autorização, métricas, checks, checksums e decisão.
- `promoted` é falso porque `--promote` não foi informado.
- MLflow registra tags/métricas se disponível.

## 6. Promoção explícita quando aprovada

Somente executar em demo controlada quando a decisão report-only indicar aprovação técnica.

```bash
python -m src.pipeline.promotion_gate \
  --candidate-dir models/candidate \
  --holdout-path data/processed/holdout_temporal_gate.joblib \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo-promote" \
  --scope "Promover candidato aprovado para artefato champion" \
  --output-dir reports/model_lifecycle \
  --promote
```

Expected outcome:

- Artefatos só são copiados se todos os gates passarem.
- JSON registra checksums antes e depois.
- MLflow registra decisão `promoted`.

## 7. Rodar drift monitor

```bash
python -m src.pipeline.drift_monitor \
  --reference data/processed/dataset_churn_pos_venda.csv \
  --current data/processed/candidate/dataset_churn_pos_venda.csv \
  --output reports/drift_report.csv
```

Expected outcome:

- `reports/drift_report.csv` criado.
- Cada feature monitorada tem PSI e status `OK`, `OBSERVAR` ou `ALERTA`.

## 8. Demonstrar rollback documentado

Abrir `CHANGELOG_MODELOS.md` e apontar:

- último champion aprovado;
- URL/caminho do artefato;
- SHA256;
- autorização;
- procedimento para reverter variáveis de ambiente e reiniciar serviço.

Expected outcome:

- Rollback depende de artefato aprovado, não backup local informal.

## 9. Demonstração de consumidor

Abrir README ou documentação do contrato de consumo e mostrar:

- Java BFF é o único consumidor técnico da API ML.
- UI/consultor não chama ML diretamente.
- Consultor vê `perfil_previsto`, `risk_level`, `acao_recomendada` e prioridade.
- Probabilidade crua não é exposta ao consultor.

## 10. Evidência esperada para banca

Ao final da demo, mostrar:

1. `docs/politica_retreino.md`.
2. `reports/model_lifecycle/*.json`.
3. MLflow com tags/métricas do ciclo.
4. `reports/drift_report.csv`.
5. `CHANGELOG_MODELOS.md`.
6. README/documentação do consumidor.
