# Contract: `src.pipeline.drift_monitor`

## Purpose

Comparar distribuição de features comportamentais entre uma base de referência e uma base atual/candidata usando Population Stability Index.

## Invocation

```bash
python -m src.pipeline.drift_monitor \
  --reference data/processed/dataset_churn_pos_venda.csv \
  --current data/processed/candidate/dataset_churn_pos_venda.csv \
  --output reports/drift_report.csv
```

## Required options

- `--reference`: CSV de referência, normalmente dataset usado pelo champion.
- `--current`: CSV atual/candidato.
- `--output`: caminho do CSV de relatório.

## Monitored features

Default inicial:

- `km_max_ate_corte`
- `meses_desde_ultimo_servico_ate_corte`
- `pct_agenda_ate_corte`

## PSI status rules

- `psi < 0.1`: `OK`
- `0.1 <= psi < 0.25`: `OBSERVAR`
- `psi >= 0.25`: `ALERTA`

## Output CSV columns

- `feature`
- `psi`
- `status`

## Failure behavior

Falhar fechado quando:

- arquivo de referência não existir;
- arquivo atual não existir;
- feature obrigatória não existir em qualquer base;
- PSI não puder ser calculado de forma confiável.

## Side effects allowed

- Criar diretório pai do relatório.
- Escrever `reports/drift_report.csv`.
- Imprimir tabela resumida.

## Side effects forbidden

- Promover modelo.
- Alterar dataset de treino.
- Alterar artefatos champion ou challenger.
