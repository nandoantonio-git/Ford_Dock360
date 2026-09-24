# Tasks: Ciclo de Vida do Modelo Ford VinGuard

**Input**: Design artifacts from `specs/001-model-lifecycle/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`
**Generation mode**: manual fallback because `.specify/scripts/bash/setup-tasks.sh` is not present in this minimal Spec Kit bootstrap.

## Format: `[ID] [P?] [Story] Description with file path`

- `[P]` means task can run in parallel with other `[P]` tasks in the same phase when prerequisites are met.
- `[US1]`, `[US2]`, `[US3]`, `[US4]` map to user stories in `spec.md`.
- Tests are included because the spec/plan require automated safety checks and pytest validation.

## Phase 1: Setup

**Purpose**: Create acceptance-critical documentation and evidence locations before code changes.

- [X] T001 Create retraining policy document in `docs/politica_retreino.md`
- [X] T002 Add model changelog and rollback procedure in `CHANGELOG_MODELOS.md`
- [X] T003 [P] Create model lifecycle evidence directory placeholder in `reports/model_lifecycle/.gitkeep`
- [X] T004 [P] Add model lifecycle docs links to `README.md`
- [X] T005 [P] Add initial lifecycle test file skeleton in `tests/test_model_lifecycle.py`

---

## Phase 2: Foundational

**Purpose**: Shared safety utilities and compatibility refactors that block all user stories.

- [X] T006 Extract `check_temporal_leakage` and `check_leakage` into `src/pipeline/leakage_checks.py`
- [X] T007 Update churn training imports to use `src/pipeline/leakage_checks.py` in `src/pipeline/train_churn_real.py`
- [X] T008 Update MLflow tracking imports to use `src/pipeline/leakage_checks.py` in `src/pipeline/mlflow_tracking.py`
- [X] T009 Update leakage tests to import shared checks from `src/pipeline/leakage_checks.py` in `tests/test_leakage.py`
- [X] T010 Add unit tests for forbidden leakage columns and patterns in `tests/test_model_lifecycle.py`
- [X] T011 Add checksum helper functions for files and artifact manifests in `src/pipeline/model_lifecycle_utils.py`
- [X] T012 Add authorization validation helper for lifecycle commands in `src/pipeline/model_lifecycle_utils.py`
- [X] T013 Add maturity validation helper for cutoff windows in `src/pipeline/model_lifecycle_utils.py`
- [X] T014 Add tests for authorization, checksum and cutoff maturity helpers in `tests/test_model_lifecycle.py`
- [X] T015 Run foundational verification command `pytest tests/test_leakage.py tests/test_model_lifecycle.py -v`

---

## Phase 3: User Story 1 - Demonstrar retreino seguro e temporalmente válido (Priority: P1)

**Goal**: Generate isolated candidate artifacts only when the cutoff has a complete 18-month future label window and explicit authorization.

**Independent Test**: A mature cutoff with authorization can generate candidate outputs under candidate directories; an immature cutoff or missing authorization fails closed without touching production artifacts.

### Tests for User Story 1

- [X] T016 [P] [US1] Add tests for mature and immature cutoff behavior in `tests/test_model_lifecycle.py`
- [X] T017 [P] [US1] Add tests that missing `--authorized-by`, `--authorization-ticket`, or `--scope` fails closed in `tests/test_model_lifecycle.py`
- [X] T018 [P] [US1] Add tests that default feature engineering behavior remains backward compatible in `tests/test_model_lifecycle.py`

### Implementation for User Story 1

- [X] T019 [US1] Add argparse options `--data-corte`, `--janela-churn-meses`, `--output-dir`, and `--fail-on-immature-window` in `src/pipeline/feature_engineering_real.py`
- [X] T020 [US1] Update feature engineering output paths to respect `--output-dir` while preserving defaults in `src/pipeline/feature_engineering_real.py`
- [X] T021 [US1] Enforce fail-closed immature cutoff behavior when requested in `src/pipeline/feature_engineering_real.py`
- [X] T022 [US1] Add argparse options `--input-dir` and `--output-dir` to segmentation script in `src/pipeline/clustering_real.py`
- [X] T023 [US1] Add argparse options `--input-dir` and `--output-dir` to churn training script in `src/pipeline/train_churn_real.py`
- [X] T024 [US1] Add AUC-PR reporting while preserving AUC-ROC reporting in `src/pipeline/train_churn_real.py`
- [X] T025 [US1] Create secure retraining trigger script in `scripts/retrain_trigger.sh`
- [X] T026 [US1] Ensure trigger writes only to `data/processed/candidate/` and `models/candidate/` in `scripts/retrain_trigger.sh`
- [X] T027 [US1] Add usage/help and fail-closed argument handling to `scripts/retrain_trigger.sh`
- [X] T028 [US1] Validate User Story 1 with `bash -n scripts/retrain_trigger.sh` and `python -m src.pipeline.feature_engineering_real --help`

---

## Phase 4: User Story 2 - Decidir promoção champion/challenger com evidências (Priority: P1)

**Goal**: Compare champion and challenger on the same fixed temporal holdout and produce a secure report-only decision by default.

**Independent Test**: Running the gate without `--promote` creates evidence but never copies production artifacts; failures in holdout, leakage, checksum, authorization, or AUC-PR margin keep the champion.

### Tests for User Story 2

- [ ] T029 [P] [US2] Add tests for temporal holdout metadata and leakage-safe feature columns in `tests/test_model_lifecycle.py`
- [ ] T030 [P] [US2] Add tests for AUC-PR margin decision and champion retention on technical tie in `tests/test_model_lifecycle.py`
- [ ] T031 [P] [US2] Add tests that report-only gate never copies production artifacts in `tests/test_model_lifecycle.py`
- [ ] T032 [P] [US2] Add tests that `--promote` copies artifacts only after all critical checks pass in `tests/test_model_lifecycle.py`

### Implementation for User Story 2

- [ ] T033 [US2] Create temporal holdout generator CLI in `src/pipeline/holdout_gate.py`
- [ ] T034 [US2] Save holdout metadata, feature columns, target column, cutoff and checksum in `src/pipeline/holdout_gate.py`
- [ ] T035 [US2] Create promotion gate CLI argument parser in `src/pipeline/promotion_gate.py`
- [ ] T036 [US2] Implement candidate, champion, holdout and checksum validation in `src/pipeline/promotion_gate.py`
- [ ] T037 [US2] Implement anti-leakage validation through `src/pipeline/leakage_checks.py` in `src/pipeline/promotion_gate.py`
- [ ] T038 [US2] Implement AUC-PR champion/challenger scoring and margin decision in `src/pipeline/promotion_gate.py`
- [ ] T039 [US2] Implement report-only default behavior and explicit `--promote` artifact copy in `src/pipeline/promotion_gate.py`
- [ ] T040 [US2] Validate User Story 2 with `python -m src.pipeline.holdout_gate --help` and `python -m src.pipeline.promotion_gate --help`

---

## Phase 5: User Story 3 - Auditar histórico de ciclos e rollback (Priority: P1)

**Goal**: Persist every lifecycle decision in JSON and MLflow, with rollback based on approved checksummed versions.

**Independent Test**: After a gate run, JSON evidence and MLflow tags/metrics contain authorization, metrics, checksums, decision and reason; rollback documentation identifies a previous approved version.

### Tests for User Story 3

- [ ] T041 [P] [US3] Add tests that lifecycle evidence JSON contains required schema fields in `tests/test_model_lifecycle.py`
- [ ] T042 [P] [US3] Add tests that evidence JSON excludes raw VIN-level rows and sensitive dataset contents in `tests/test_model_lifecycle.py`
- [ ] T043 [P] [US3] Add tests for MLflow lifecycle logging with monkeypatched MLflow calls in `tests/test_model_lifecycle.py`

### Implementation for User Story 3

- [ ] T044 [US3] Implement lifecycle evidence builder matching `specs/001-model-lifecycle/contracts/evidence-schema.json` in `src/pipeline/promotion_gate.py`
- [ ] T045 [US3] Write gate evidence JSON to `reports/model_lifecycle/` in `src/pipeline/promotion_gate.py`
- [ ] T046 [US3] Add `log_model_lifecycle_cycle` function in `src/pipeline/mlflow_tracking.py`
- [ ] T047 [US3] Call MLflow lifecycle logging from `src/pipeline/promotion_gate.py`
- [ ] T048 [US3] Update rollback entries and instructions in `CHANGELOG_MODELOS.md`
- [ ] T049 [US3] Validate User Story 3 with `python -m src.pipeline.promotion_gate --help` and JSON evidence schema review in `reports/model_lifecycle/`

---

## Phase 6: User Story 4 - Explicar quem consome as predições (Priority: P2)

**Goal**: Document least-privilege prediction consumption: Java BFF calls ML; consultant sees only operational recommendations.

**Independent Test**: Documentation identifies the Java BFF as the only technical ML API consumer and confirms the UI/consultant never sees raw probability.

### Tests for User Story 4

- [ ] T050 [P] [US4] Add documentation assertion test for Java BFF and no raw probability exposure in `tests/test_model_lifecycle.py`

### Implementation for User Story 4

- [ ] T051 [US4] Document Java BFF as the only technical ML API consumer in `README.md`
- [ ] T052 [US4] Document consultant-facing outputs `perfil_previsto`, `risk_level`, `acao_recomendada`, and priority in `README.md`
- [ ] T053 [US4] Document least-privilege prediction flow in `docs/politica_retreino.md`
- [ ] T054 [US4] Validate User Story 4 by reviewing `README.md` and `docs/politica_retreino.md`

---

## Phase 7: Drift Monitoring and Demo Evidence

**Purpose**: Complete cross-cutting acceptance evidence for Sprint 3/4 demo.

- [ ] T055 [P] Create PSI drift monitor CLI in `src/pipeline/drift_monitor.py`
- [ ] T056 [P] Add tests for PSI thresholds `OK`, `OBSERVAR`, and `ALERTA` in `tests/test_model_lifecycle.py`
- [ ] T057 Add fail-closed missing-feature behavior to `src/pipeline/drift_monitor.py`
- [ ] T058 Add drift monitor validation command to `specs/001-model-lifecycle/quickstart.md`
- [ ] T059 Create or update demo evidence checklist in `docs/politica_retreino.md`
- [ ] T060 Validate drift monitor with `python -m src.pipeline.drift_monitor --help`

---

## Phase 8: Polish & Cross-Cutting Validation

**Purpose**: Final checks, cleanup and acceptance runbook.

- [ ] T061 Run focused lifecycle test suite with `pytest tests/test_model_lifecycle.py -v`
- [ ] T062 Run leakage regression tests with `pytest tests/test_leakage.py -v`
- [ ] T063 Run CLI help checks for all lifecycle modules listed in `specs/001-model-lifecycle/quickstart.md`
- [ ] T064 Run Bash syntax validation for `scripts/retrain_trigger.sh`
- [ ] T065 If dataset and champion artifacts exist, run candidate trigger demo command from `specs/001-model-lifecycle/quickstart.md`; otherwise record the missing prerequisite in `docs/politica_retreino.md`
- [ ] T066 If holdout and candidate artifacts exist, run promotion gate report-only demo from `specs/001-model-lifecycle/quickstart.md`; otherwise record the missing prerequisite in `docs/politica_retreino.md`
- [ ] T067 If reference and candidate datasets exist, run drift report demo from `specs/001-model-lifecycle/quickstart.md`; otherwise record the missing prerequisite in `docs/politica_retreino.md`
- [ ] T068 Review git diff to ensure no dataset, model artifact, `.DS_Store`, or unrelated notebook changes are staged in `.gitignore`, `data/`, `models/`, `.DS_Store`, `src/.DS_Store`, and `notebooks/00_pipeline_completo.ipynb`
- [ ] T069 Update `specs/001-model-lifecycle/quickstart.md` with final observed commands and expected outputs
- [ ] T070 Run final ad-hoc Spec Kit artifact validation for `.specify/feature.json` and `specs/001-model-lifecycle/tasks.md`

---

## Dependencies

### Phase dependencies

- Phase 1 Setup must complete before code implementation.
- Phase 2 Foundational must complete before US1, US2, US3 and US4 implementation.
- US1 must complete before the full US2 demo because US2 needs candidate artifacts.
- US2 must complete before the full US3 demo because US3 evidence is produced by the gate.
- US4 can run after Phase 2 and in parallel with US1/US2/US3 documentation work.
- Drift monitoring can run after candidate datasets exist, but its CLI and tests can be implemented after Phase 2.
- Polish runs last.

### User story dependency graph

```text
Phase 1 Setup
  -> Phase 2 Foundational
    -> US1 Retreino seguro
      -> US2 Gate champion/challenger
        -> US3 Evidência e rollback
    -> US4 Consumidores das predições
    -> Drift Monitoring
  -> Polish
```

## Parallel Execution Examples

### Setup parallel examples

```text
T003 can run in parallel with T004 and T005 after T001/T002 are understood.
```

### US1 parallel examples

```text
T016, T017 and T018 can be written in parallel because they touch distinct test scenarios in tests/test_model_lifecycle.py.
T022 and T023 can be implemented in parallel after T019-T021 because they touch different pipeline scripts.
```

### US2 parallel examples

```text
T029, T030, T031 and T032 can be drafted in parallel as independent tests.
T033-T034 and T035-T039 should be serialized by module, but holdout generation and gate CLI design can be split between agents after shared utility contracts are stable.
```

### US3 parallel examples

```text
T041, T042 and T043 can be written in parallel.
T046 can be implemented in parallel with T044-T045, then integrated by T047.
```

### US4 parallel examples

```text
T051 and T052 can be done together in README.md; T053 can be done in docs/politica_retreino.md in parallel if merge conflicts are coordinated.
```

## Implementation Strategy

### MVP first

MVP scope is US1 plus enough foundation to prove safe retraining:

1. T001-T015 foundational documentation and safety utilities.
2. T016-T028 User Story 1.
3. Validate with `bash -n scripts/retrain_trigger.sh` and feature engineering help/fail-closed tests.

### Incremental delivery

1. Deliver US1: safe candidate generation.
2. Deliver US2: report-only gate and AUC-PR decision.
3. Deliver US3: JSON + MLflow evidence and rollback.
4. Deliver US4: consumer documentation.
5. Deliver drift monitor and final demo evidence.

### Acceptance-critical evidence

The following artifacts are mandatory and must not be buried in generic documentation tasks:

- `docs/politica_retreino.md`
- `CHANGELOG_MODELOS.md`
- `reports/model_lifecycle/*.json` generated by the gate
- MLflow lifecycle run/tags/metrics when MLflow is available
- `reports/drift_report.csv` when reference/current datasets exist
- README section for Java BFF and consultant-facing outputs
