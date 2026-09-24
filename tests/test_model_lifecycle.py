"""Smoke tests for model lifecycle documentation and safety helpers."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.feature_engineering_real import build_snapshot
from src.pipeline.leakage_checks import check_leakage, check_temporal_leakage
from src.pipeline.mlflow_tracking import (
    _select_best_churn_candidate,
    _summarize_churn_metrics,
)
from src.pipeline.model_lifecycle_utils import (
    build_artifact_manifest,
    sha256_file,
    validate_authorization,
    validate_cutoff_maturity,
)

ROOT = Path(__file__).resolve().parents[1]


def test_model_lifecycle_docs_exist():
    assert (ROOT / "docs" / "politica_retreino.md").exists()
    assert (ROOT / "CHANGELOG_MODELOS.md").exists()
    assert (ROOT / "reports" / "model_lifecycle" / ".gitkeep").exists()


def test_model_lifecycle_docs_capture_security_principles():
    policy = (ROOT / "docs" / "politica_retreino.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG_MODELOS.md").read_text(encoding="utf-8")

    for expected in [
        "retreino programado trimestral",
        "Autorização explícita",
        "Falha segura",
        "AUC-PR",
        "Java BFF",
        "X-ML-Service-Token",
    ]:
        assert expected in policy

    for expected in ["SHA256", "rollback", "CHURN_MODEL_SHA256", "PERFIL_MODEL_SHA256"]:
        assert expected in changelog


def test_readme_links_model_lifecycle_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/politica_retreino.md" in readme
    assert "CHANGELOG_MODELOS.md" in readme


def test_leakage_checks_reject_forbidden_columns_and_patterns():
    df = pd.DataFrame({
        "churn_futuro_18m": [0, 1],
        "voltou_pos_corte": [False, True],
        "score_target": [0.1, 0.9],
        "ano_modelo": [2020, 2021],
    })

    with pytest.raises(ValueError, match="Colunas com leakage temporal em X"):
        check_leakage(df)

    with pytest.raises(ValueError, match="Colunas com leakage temporal em X"):
        check_temporal_leakage(df, ["ano_modelo", "score_futuro"])


def test_leakage_checks_accept_safe_features():
    df = pd.DataFrame({
        "ano_modelo": [2020, 2021],
        "modelo": ["KA", "RANGER"],
        "qtde_revisoes_ate_corte": [1, 3],
    })

    check_leakage(df)


def test_sha256_and_artifact_manifest(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"ford-vinguard")

    digest = sha256_file(artifact)
    assert len(digest) == 64
    assert digest == sha256_file(artifact)

    manifest = build_artifact_manifest([artifact], role="challenger", model_type="churn")
    assert manifest == [{
        "name": "model.joblib",
        "path": str(artifact),
        "sha256": digest,
        "role": "challenger",
        "model_type": "churn",
    }]


def test_validate_authorization_requires_explicit_fields():
    assert validate_authorization(" Fernando ", " Sprint3-demo ", " Avaliar candidato ") == {
        "authorized_by": "Fernando",
        "authorization_ticket": "Sprint3-demo",
        "authorization_scope": "Avaliar candidato",
    }

    with pytest.raises(ValueError, match="Autorizacao incompleta"):
        validate_authorization("Fernando", "", "Avaliar candidato")


def test_validate_cutoff_maturity_accepts_complete_label_window():
    result = validate_cutoff_maturity("2024-01-31", "2025-07-31", 18)

    assert result["data_corte"] == "2024-01-31"
    assert result["fim_janela_churn"] == "2025-07-31"
    assert result["data_max_observada"] == "2025-07-31"
    assert result["maturity_ok"] is True


def test_validate_cutoff_maturity_fails_closed_for_immature_window():
    with pytest.raises(ValueError, match="Janela futura de churn imatura"):
        validate_cutoff_maturity("2024-01-31", "2025-07-30", 18)


def _minimal_orders(max_future_date="07/31/2025"):
    return pd.DataFrame({
        "VIN_Hash": ["vin1", "vin1", "vin2", "vin2"],
        "MaintenanceID": [1, 2, 3, 4],
        "ServiceDate": ["01/15/2024", max_future_date, "01/20/2024", max_future_date],
        "ModelName": ["KA", "KA", "RANGER", "RANGER"],
        "ModelYear": [2020, 2020, 2021, 2021],
        "DealerCode": ["D1", "D1", "D2", "D2"],
        "KM": [10000, 15000, 20000, 25000],
        "SalesDate": ["01/01/2020", "01/01/2020", "01/01/2021", "01/01/2021"],
        "DeliveryDate": ["01/02/2020", "01/02/2020", "01/02/2021", "01/02/2021"],
        "WarrantyStartDate": ["01/02/2020", "01/02/2020", "01/02/2021", "01/02/2021"],
        "IsAgendaSchedule": [1, 1, 0, 0],
    })


def test_build_snapshot_accepts_mature_cutoff():
    snapshot = build_snapshot(
        _minimal_orders("07/31/2025"),
        "2024-01-31",
        18,
        fail_on_immature_window=True,
    )

    assert snapshot["janela_futura_observavel"].all()
    assert set(snapshot["data_corte"].astype(str)) == {"2024-01-31"}


def test_build_snapshot_fails_closed_for_immature_cutoff():
    with pytest.raises(ValueError, match="Janela futura de churn imatura"):
        build_snapshot(
            _minimal_orders("07/30/2025"),
            "2024-01-31",
            18,
            fail_on_immature_window=True,
        )


def test_pipeline_cli_help_exposes_candidate_options():
    commands = [
        [sys.executable, "-m", "src.pipeline.feature_engineering_real", "--help"],
        [sys.executable, "-m", "src.pipeline.clustering_real", "--help"],
        [sys.executable, "-m", "src.pipeline.train_churn_real", "--help"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        assert "--output-dir" in result.stdout

    fe_help = subprocess.run(
        [sys.executable, "-m", "src.pipeline.feature_engineering_real", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "--data-corte" in fe_help
    assert "--fail-on-immature-window" in fe_help


def test_retrain_trigger_requires_authorization():
    result = subprocess.run(
        ["bash", "scripts/retrain_trigger.sh", "2025-01-31"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "autorizacao explicita obrigatoria" in result.stderr


def test_select_best_churn_candidate_uses_auc_pr_then_auc_roc():
    rows = [
        {"model_name": "rf_a", "auc_pr": 0.30, "auc_roc": 0.80},
        {"model_name": "rf_b", "auc_pr": 0.35, "auc_roc": 0.78},
        {"model_name": "rf_c", "auc_pr": 0.35, "auc_roc": 0.82},
    ]

    best = _select_best_churn_candidate(rows)

    assert best["model_name"] == "rf_c"


def test_summarize_churn_metrics_returns_required_fields():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]

    metrics = _summarize_churn_metrics(y_true, y_score, threshold=0.5)

    assert set(metrics) == {"auc_roc", "auc_pr", "precision", "recall", "f1"}
    assert metrics["auc_roc"] == 1.0
    assert metrics["auc_pr"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_mlflow_tracking_help_exposes_churn_comparison():
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.mlflow_tracking", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "churn-comparison" in result.stdout
    assert "--comparison-output" in result.stdout


def test_mlflow_tracking_defines_churn_model_comparison_candidates():
    source = (ROOT / "src" / "pipeline" / "mlflow_tracking.py").read_text(encoding="utf-8")

    for expected in [
        "LogisticRegression_Balanced",
        "RandomForest_Balanced",
        "RandomForest_Calibrated",
        "auc_pr",
        "model_comparison_churn.csv",
    ]:
        assert expected in source
