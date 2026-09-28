# Contract: `src.pipeline.promotion_gate`

## Purpose

Comparar modelo champion e challenger com segurança, gerar evidência de decisão e promover artefatos apenas com autorização explícita.

## Invocation: report-only mode

```bash
python -m src.pipeline.promotion_gate \
  --candidate-dir models/candidate \
  --holdout-path data/processed/holdout_temporal_gate.joblib \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo" \
  --scope "Avaliar candidato para promoção" \
  --output-dir reports/model_lifecycle
```

## Invocation: explicit promotion mode

```bash
python -m src.pipeline.promotion_gate \
  --candidate-dir models/candidate \
  --holdout-path data/processed/holdout_temporal_gate.joblib \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo-promote" \
  --scope "Promover candidato aprovado" \
  --output-dir reports/model_lifecycle \
  --promote
```

## Required options

- `--candidate-dir`: diretório contendo artefatos `.joblib` candidatos.
- `--holdout-path`: caminho do holdout temporal versionado.
- `--authorized-by`: responsável pela autorização.
- `--authorization-ticket`: identificador rastreável.
- `--scope`: escopo autorizado.
- `--output-dir`: diretório para evidência JSON.

## Optional options

- `--promote`: permite promoção efetiva quando todos os gates passam.
- `--auc-pr-margin`: margem mínima, default `0.005`.
- `--production-dir`: diretório champion, default `models/`.

## Required checks

- Autorização presente.
- Candidato existe.
- Champion existe.
- Holdout existe.
- Checksums calculáveis.
- Anti-leakage aprovado.
- Champion e challenger avaliados no mesmo holdout.
- AUC-PR challenger supera champion pela margem mínima.

## Decision rules

- Sem `--promote`, `promoted` deve ser falso mesmo quando `approved_for_promotion` for verdadeiro.
- Com `--promote`, copiar artefatos apenas se `approved_for_promotion` for verdadeiro.
- Qualquer falha crítica produz rejeição/fail-closed e não altera produção.
- Empate técnico mantém champion.

## Output

- Imprime JSON ou caminho do JSON gerado.
- Salva evidência em `reports/model_lifecycle/`.
- Registra no MLflow quando disponível.

## Evidence schema

A evidência deve seguir `contracts/evidence-schema.json`.
