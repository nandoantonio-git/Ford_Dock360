# Data Model: Ciclo de Vida do Modelo Ford VinGuard

## Entity: RetrainingCycle

Representa uma execução autorizada de avaliação/retreino.

**Fields**:

- `cycle_id`: identificador único do ciclo.
- `created_at`: timestamp ISO-8601 da criação da evidência.
- `data_corte`: corte temporal avaliado.
- `label_window_months`: janela de churn, inicialmente 18.
- `max_service_date`: maior data de serviço observada no dataset usado.
- `maturity_ok`: boolean indicando se a janela futura está completa.
- `cadence_ok`: boolean indicando se a cadência mínima foi respeitada.
- `authorized_by`: responsável pela autorização.
- `authorization_ticket`: identificador de autorização.
- `authorization_scope`: escopo autorizado.
- `decision`: `approved_for_promotion`, `rejected`, `promoted`, ou `report_only`.
- `decision_reason`: justificativa legível da decisão.

**Validation rules**:

- `authorized_by`, `authorization_ticket` e `authorization_scope` são obrigatórios.
- `maturity_ok` deve ser verdadeiro para treinar/promover candidato.
- Ciclos sem autorização falham fechado.

## Entity: CutoffDate

Marco temporal que separa histórico permitido e janela futura de target.

**Fields**:

- `data_corte`: data selecionada.
- `fim_janela_churn`: `data_corte + label_window_months`.
- `data_max_observada`: maior `ServiceDate` disponível.
- `janela_futura_observavel`: boolean.

**Validation rules**:

- `fim_janela_churn <= data_max_observada` para corte maduro.
- Features usam apenas eventos `ServiceDate <= data_corte`.
- Target usa apenas eventos `data_corte < ServiceDate <= fim_janela_churn`.

## Entity: ModelArtifact

Artefato de modelo ou segmentador avaliado/promovido.

**Fields**:

- `name`: nome do arquivo.
- `path`: caminho local ou URL.
- `sha256`: checksum SHA256.
- `role`: `champion`, `challenger`, ou `rollback_candidate`.
- `model_type`: `churn` ou `segmentation`.
- `created_at`: timestamp quando disponível.

**Validation rules**:

- Artefatos promovidos precisam de checksum calculado e registrado.
- Rollback só pode usar artefato previamente aprovado.

## Entity: TemporalHoldout

Base fixa e versionada para comparação champion/challenger.

**Fields**:

- `path`: caminho do artefato holdout.
- `sha256`: checksum do arquivo.
- `feature_columns`: colunas de X.
- `target_column`: coluna y.
- `data_corte_holdout`: corte usado na composição.
- `created_at`: timestamp de geração.
- `n_rows`: número de VINs no holdout.

**Validation rules**:

- Deve existir antes do gate.
- Deve excluir colunas de futuro, target e pós-corte de X.
- Champion e challenger devem usar exatamente o mesmo holdout.

## Entity: PromotionGateResult

Resultado da comparação e das verificações críticas.

**Fields**:

- `cycle_id`
- `leakage_ok`
- `maturity_ok`
- `authorization_ok`
- `holdout_ok`
- `checksum_ok`
- `auc_pr_champion`
- `auc_pr_challenger`
- `auc_pr_delta`
- `auc_pr_margin_required`
- `approved_for_promotion`
- `promoted`
- `decision_reason`

**Validation rules**:

- `approved_for_promotion` só pode ser verdadeiro quando todos os checks críticos são verdadeiros.
- `promoted` só pode ser verdadeiro quando `approved_for_promotion` é verdadeiro e houve autorização explícita de promoção.
- Empate técnico mantém champion.

## Entity: LifecycleEvidence

Registro persistente do ciclo.

**Fields**:

- `cycle`: objeto `RetrainingCycle`.
- `gate`: objeto `PromotionGateResult`.
- `candidate_artifacts`: lista de `ModelArtifact`.
- `champion_artifacts_before`: lista de `ModelArtifact`.
- `champion_artifacts_after`: lista de `ModelArtifact`, quando aplicável.
- `mlflow`: metadados de experimento/run quando disponível.

**Validation rules**:

- Evidência JSON deve ser gerada para cada decisão do gate quando insumos mínimos existem.
- Evidência não deve incluir dados pessoais ou linhas brutas do dataset.

## Entity: DriftReport

Relatório de estabilidade das features.

**Fields**:

- `created_at`
- `reference_dataset`
- `current_dataset`
- `features`: lista de medições por feature.

**Feature measurement fields**:

- `feature`
- `psi`
- `status`: `OK`, `OBSERVAR`, ou `ALERTA`.

**Validation rules**:

- Features obrigatórias ausentes causam falha fechada.
- `psi < 0.1` é `OK`.
- `0.1 <= psi < 0.25` é `OBSERVAR`.
- `psi >= 0.25` é `ALERTA`.

## State Transitions

### Challenger lifecycle

```text
not_created -> candidate_generated -> gate_evaluated -> approved_for_promotion -> promoted
                                      └──────────────> rejected
                                      └──────────────> report_only
```

### Gate decision

```text
started -> validating_authorization -> validating_maturity -> validating_holdout -> validating_leakage -> evaluating_metrics -> writing_evidence -> completed
                                                                                                                   └────────────> failed_closed
```

### Rollback

```text
champion_active -> rollback_requested -> approved_previous_version_selected -> checksum_verified -> environment_repointed -> health_checked -> rollback_recorded
```
