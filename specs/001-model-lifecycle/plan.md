# Implementation Plan: Ciclo de Vida do Modelo Ford VinGuard

**Branch**: `main` | **Date**: 2026-09-23 | **Spec**: `specs/001-model-lifecycle/spec.md`
**Input**: Feature specification for model lifecycle, secure retraining, champion/challenger gate, drift, rollback, MLflow and JSON evidence.

## Summary

Implementar uma PoC acadêmica de ciclo de vida do modelo Ford VinGuard com retreino programado trimestral, maturação correta do rótulo de churn de 18 meses, artefatos candidatos isolados, gate champion/challenger seguro, evidência dupla em JSON e MLflow, monitoramento leve de drift e rollback por checksum aprovado.

O desenho mantém a filosofia do projeto: scripts simples, funções diretas, sem arquitetura corporativa pesada. Segurança é aplicada por menor privilégio, autorização explícita, fail-closed e rastreabilidade.

## Technical Context

**Language/Version**: Python 3 no ambiente local do projeto; Bash para orquestração simples.
**Primary Dependencies**: pandas, scikit-learn, joblib, MLflow, pytest, matplotlib.
**Storage**: arquivos locais em `data/processed/`, `models/`, `reports/`, `mlruns/`.
**Testing**: pytest para verificações unitárias/integração leve; scripts CLI com `--help` e execução controlada.
**Target Platform**: macOS/local acadêmico, com possibilidade de demo via terminal e MLflow local.
**Project Type**: pipeline de ML acadêmico com API/serviço já existente fora do escopo principal desta feature.
**Performance Goals**: demo em até 10 minutos; gate e evidências rápidos o suficiente para rodar com artefatos já gerados.
**Constraints**: não tocar modelos de produção sem `--promote`; não usar features pós-corte; não treinar com rótulo imaturo; não expor probabilidade crua à UI/consultor; não versionar dataset bruto ou modelos grandes.
**Scale/Scope**: dataset real Ford com centenas de milhares de ordens e cerca de 175 mil VINs; PoC de lifecycle, não MLOps de produção completo.

## Constitution Check

### Gate inicial

- **Escopo acadêmico**: PASS. A solução é demonstrável e limitada a PoC, sem prometer MLOps corporativo completo.
- **ML pós-venda**: PASS. O ciclo usa histórico até corte e target depois do corte.
- **Maturação do rótulo**: PASS. A elegibilidade do corte exige janela futura completa de 18 meses.
- **Pipeline sklearn**: PASS. O plano preserva preprocessing dentro de `sklearn.Pipeline`.
- **random_state/class_weight**: PASS. Mudanças em treino devem preservar `random_state=42` e `class_weight='balanced'`.
- **Métricas**: PASS. AUC-ROC continua reportada; AUC-PR decide gate.
- **Menor privilégio**: PASS. Trigger só gera candidato; gate é relatório por padrão.
- **Autorização explícita**: PASS. Ciclos exigem responsável, ticket e escopo.
- **Automação de segurança**: PASS. Leakage, maturação, holdout e checksums viram gates.
- **Rastreabilidade**: PASS. JSON e MLflow registram decisão.
- **Consumidores**: PASS. Java BFF é consumidor técnico; consultor recebe recomendação operacional.

## Project Structure

### Documentation artifacts

```text
specs/001-model-lifecycle/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── evidence-schema.json
│   ├── retrain-trigger-cli.md
│   ├── promotion-gate-cli.md
│   └── drift-monitor-cli.md
└── checklists/
    └── requirements.md
```

### Implementation paths planned

```text
docs/politica_retreino.md
CHANGELOG_MODELOS.md
scripts/retrain_trigger.sh
src/pipeline/leakage_checks.py
src/pipeline/holdout_gate.py
src/pipeline/promotion_gate.py
src/pipeline/drift_monitor.py
src/pipeline/mlflow_tracking.py
tests/test_model_lifecycle.py
reports/model_lifecycle/.gitkeep
```

## Phase 0: Research

Research completed in `research.md`.

Decisions resolved:

1. Maturação por corte, não por VIN individual.
2. Gate decide com AUC-PR e margem mínima de 0.005.
3. Holdout temporal fixo e versionado.
4. Gate fail-closed e report-only por padrão.
5. Evidência dupla: JSON versionável e MLflow visual.
6. Rollback por artefato aprovado com checksum.
7. Java BFF como consumidor técnico único.

## Phase 1: Design and Contracts

Design artifacts generated:

- `data-model.md`: entidades, campos, validações e estados.
- `contracts/evidence-schema.json`: contrato do JSON de evidência.
- `contracts/retrain-trigger-cli.md`: contrato CLI do trigger.
- `contracts/promotion-gate-cli.md`: contrato CLI do gate.
- `contracts/drift-monitor-cli.md`: contrato CLI do drift monitor.
- `quickstart.md`: roteiro de validação/demo.

## Implementation Strategy

### Step 1 — Documentation first

Criar `docs/politica_retreino.md` e `CHANGELOG_MODELOS.md` antes do código. Isso responde rapidamente ao professor e fixa escopo, rollback e consumidores.

### Step 2 — Refactor anti-leakage safely

Extrair `check_temporal_leakage` e `check_leakage` de `src/pipeline/train_churn_real.py` para `src/pipeline/leakage_checks.py`. Atualizar `train_churn_real.py`, `mlflow_tracking.py` e `tests/test_leakage.py` para reutilizar o módulo produtivo.

### Step 3 — Parameterize existing scripts without breaking defaults

Adicionar argumentos opcionais em:

- `feature_engineering_real.py`: `--data-corte`, `--janela-churn-meses`, `--output-dir`, `--fail-on-immature-window`.
- `clustering_real.py`: `--input-dir`, `--output-dir`.
- `train_churn_real.py`: `--input-dir`, `--output-dir`, métricas AUC-ROC e AUC-PR.

Sem argumentos, o comportamento atual deve continuar igual.

### Step 4 — Add temporal holdout artifact

Criar `holdout_gate.py` para gerar `data/processed/holdout_temporal_gate.joblib` com metadados e checksum. O gate não deve gerar holdout implicitamente; ausência de holdout é falha fechada.

### Step 5 — Add secure retrain trigger

Criar `scripts/retrain_trigger.sh` para validar autorização, cadência mínima, maturação por corte e gerar candidato isolado. O trigger não deve alterar `config.py` nem modelos de produção.

### Step 6 — Add promotion gate

Criar `promotion_gate.py` com decisão report-only por padrão. Com `--promote`, copiar artefatos apenas quando todos os gates passarem. Sempre gerar JSON quando os insumos mínimos permitirem registrar a decisão.

### Step 7 — Add MLflow lifecycle logging

Adicionar função em `mlflow_tracking.py` para registrar ciclo de lifecycle: tags, métricas e artefato JSON.

### Step 8 — Add PSI drift monitor

Criar `drift_monitor.py` com PSI para features-chave e relatório CSV.

### Step 9 — Validate and document demo

Rodar pytest, validações CLI, gerar evidências de demo e atualizar README com consumidor técnico/de negócio.

## Post-Design Constitution Check

- **Nenhuma violação nova identificada.**
- O desenho mantém scripts simples/funções, sem DDD/Clean Architecture.
- A promoção real continua protegida por autorização explícita.
- A documentação e contratos preservam escopo acadêmico e rastreabilidade.

## Risk Log

| Risk | Impact | Mitigation |
|---|---:|---|
| Scripts atuais têm caminhos fixos | Alto | Argparse com defaults iguais aos caminhos atuais |
| DATA_CORTE atual pode ser imaturo para 18 meses | Alto | Trigger/gate rejeitam corte imaturo; demo usa corte maduro |
| Holdout temporal ainda não existe | Alto | Criar `holdout_gate.py` antes do gate |
| MLflow indisponível na demo | Médio | JSON é evidência mínima independente |
| Modelos/dataset grandes no git | Médio | Não versionar dataset bruto/modelos grandes; versionar docs/evidências pequenas |
| Promoção acidental | Alto | `--promote` obrigatório e fail-closed |

## Verification Plan

Focused commands after implementation:

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

Demo path when data/model artifacts exist:

```bash
./scripts/retrain_trigger.sh 2024-01-31 \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo" \
  --scope "Gerar candidato isolado para gate acadêmico"

python -m src.pipeline.promotion_gate \
  --candidate-dir models/candidate \
  --holdout-path data/processed/holdout_temporal_gate.joblib \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo" \
  --scope "Avaliar candidato para promoção" \
  --output-dir reports/model_lifecycle

python -m src.pipeline.drift_monitor \
  --reference data/processed/dataset_churn_pos_venda.csv \
  --current data/processed/candidate/dataset_churn_pos_venda.csv \
  --output reports/drift_report.csv
```

## Next Phase

Run `/speckit-tasks` to generate dependency-ordered implementation tasks from this plan and the contracts.
