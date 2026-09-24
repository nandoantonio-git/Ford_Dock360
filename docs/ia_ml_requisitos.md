# Atendimento aos Requisitos de IA & Machine Learning — Sprint 3

## Fonte dos requisitos

Documento `Ford_V2.pdf`, Sprint 3 — Inteligência Artificial & Machine Learning.

## 1. Compreensão do problema

- Problema: prever risco de churn no pós-venda Ford.
- Objetivo: apoiar priorização e abordagem comercial no Dock 360.
- Tipo de problema:
  - classificação binária para churn (`churn` / `no_churn`);
  - clusterização não supervisionada para perfil comportamental.
- Evidências:
  - `README.md`;
  - `src/pipeline/feature_engineering_real.py`;
  - `docs/politica_retreino.md`.

## 2. Preparação dos dados

- Tratamento de datas, inconsistências e outliers em `src/pipeline/feature_engineering_real.py`.
- Criação de features por VIN usando apenas eventos até `DATA_CORTE`.
- Target de churn calculado pela janela futura de 18 meses após o corte.
- Separação explícita entre histórico permitido para features e janela futura usada para rótulo.
- Anti-leakage em `src/pipeline/leakage_checks.py`.
- Evidências de EDA:
  - `notebooks/01_eda_base1.ipynb`;
  - `notebooks/02_eda_base2.ipynb`;
  - `notebooks/00_pipeline_completo.ipynb`.

## 3. Desenvolvimento dos modelos

### Churn

O projeto compara modelos/configurações de churn via MLflow em `src/pipeline/mlflow_tracking.py`:

- `LogisticRegression_Balanced`;
- `RandomForest_Balanced`;
- `RandomForest_Calibrated`.

A execução dedicada é:

```bash
python3 -m src.pipeline.mlflow_tracking --only churn-comparison \
  --comparison-output reports/model_comparison_churn.csv
```

A saída esperada é a tabela `reports/model_comparison_churn.csv`, com métricas comparáveis e indicação do modelo selecionado.

### Perfil comportamental

O projeto usa K-Means para segmentação de perfil e compara diferentes valores de `k` em `src/pipeline/clustering_real.py` e `src/pipeline/mlflow_tracking.py`.

Métricas usadas:

- silhouette score;
- inertia.

## 4. Avaliação e comparação

Métricas de churn registradas na comparação:

- AUC-ROC;
- AUC-PR;
- precision;
- recall;
- F1.

AUC-PR é a métrica principal de comparação porque o caso de uso prioriza clientes em risco de abandono, onde a classe positiva é operacionalmente mais importante. AUC-ROC continua reportada para compatibilidade e leitura geral de separabilidade.

Métricas de clustering:

- silhouette score;
- inertia.

## 5. Conclusão

- Modelo selecionado: definido pelo melhor AUC-PR na comparação de churn, com AUC-ROC como desempate técnico.
- Justificativa: AUC-PR é mais adequada para priorização de clientes em risco de churn e para orientar ação comercial.
- Uso na aplicação:
  - o Java BFF é o consumidor técnico autorizado da API de ML;
  - o consultor da concessionária recebe nível de risco, perfil previsto, ação recomendada e prioridade operacional.
- Deploy:
  - API FastAPI;
  - artefatos versionados por checksum;
  - variáveis de ambiente para URLs/checksums dos modelos.
- Melhorias futuras:
  - gate champion/challenger completo;
  - evidência JSON + MLflow por ciclo de promoção;
  - monitoramento de drift por PSI;
  - rollback operacional baseado em versões aprovadas e checksums.

## Checklist resumido da rubrica

| Requisito IA/ML | Evidência no projeto | Status |
|---|---|---|
| Compreensão do problema | `README.md`, `docs/politica_retreino.md` | Atendido |
| Análise/preparação dos dados | notebooks e `feature_engineering_real.py` | Atendido |
| Tratamento de dados e criação de variáveis | `feature_engineering_real.py` | Atendido |
| Algoritmos/modelos | `train_churn_real.py`, `clustering_real.py`, `mlflow_tracking.py` | Atendido |
| Diferentes configurações/algoritmos | MLflow churn comparison + K-Means `k=2..8` | Atendido |
| Métricas adequadas | AUC-ROC, AUC-PR, precision, recall, F1, silhouette, inertia | Atendido |
| Comparação de resultados | `reports/model_comparison_churn.csv` e MLflow UI | Atendido quando a comparação for executada |
| Conclusão e justificativa | este documento, README e política de retreino | Atendido |
