import sys

import numpy as np
import pandas as pd
import pytest

from src.pipeline import train_churn_real as tc
from src.pipeline.config import TARGET_CHURN


class FakeCalibratedClassifier:
    def __init__(self, estimator=None, cv=None, method=None):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.was_fit = False

    def fit(self, x, y):
        self.was_fit = True
        return self

    def predict_proba(self, x):
        scores = np.full(len(x), 0.45)
        return np.column_stack([1 - scores, scores])


def _churn_dataset(n=20):
    rows = []
    for i in range(n):
        rows.append({
            "ano_modelo": 2020 + (i % 4),
            "qtde_revisoes_ate_corte": 1 + (i % 5),
            "meses_desde_ultimo_servico_ate_corte": float(i % 12),
            "meses_relacionamento_ate_corte": 12.0 + i,
            "n_dealers_usados_ate_corte": 1 + (i % 3),
            "km_max_ate_corte": 10000.0 + i * 1000,
            "pct_agenda_ate_corte": (i % 10) / 10,
            "intervalo_medio_revisoes_dias_ate_corte": 120.0 + i,
            "dias_ate_primeira_revisao": 90 + i,
            "idade_veiculo_meses_ate_corte": 24.0 + i,
            "modelo": "KA" if i % 2 == 0 else "RANGER",
            TARGET_CHURN: i % 2,
            "janela_futura_observavel": True,
        })
    return pd.DataFrame(rows)


def test_build_preprocessor_tem_transformers_esperados():
    preprocessor = tc._build_preprocessor()

    assert {name for name, _, _ in preprocessor.transformers} == {"numeric", "categorical"}
    assert preprocessor.transformers[0][2] == tc.FEATURES_NUMERIC
    assert preprocessor.transformers[1][2] == tc.FEATURES_CATEGORICAL


def test_load_data_rejeita_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        tc._load_data(str(tmp_path / "ausente.csv"))


def test_load_data_rejeita_coluna_obrigatoria_ausente(tmp_path):
    path = tmp_path / "dataset.csv"
    _churn_dataset().drop(columns=["modelo"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Colunas ausentes"):
        tc._load_data(str(path))


def test_load_data_retorna_x_sem_colunas_de_leakage(tmp_path):
    path = tmp_path / "dataset.csv"
    _churn_dataset().to_csv(path, index=False)

    x, y = tc._load_data(str(path))

    assert TARGET_CHURN not in x.columns
    assert "voltou_pos_corte" not in x.columns
    assert len(x) == len(y) == 20


def test_load_data_imprime_alerta_quando_janela_futura_incompleta(tmp_path, capsys):
    path = tmp_path / "dataset.csv"
    df = _churn_dataset()
    df.loc[0, "janela_futura_observavel"] = False
    df.to_csv(path, index=False)

    tc._load_data(str(path))

    assert "janela futura incompleta" in capsys.readouterr().out


def test_assert_metrics_not_suspicious_rejeita_auc_alto_demais():
    with pytest.raises(ValueError, match="suspeita de leakage"):
        tc.assert_metrics_not_suspicious(0.981)


def test_assert_metrics_not_suspicious_aceita_auc_plausivel():
    tc.assert_metrics_not_suspicious(0.85)


def test_train_churn_model_grava_modelo_e_checksum(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "dataset_churn_pos_venda.csv"
    _churn_dataset(60).to_csv(path, index=False)
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    monkeypatch.setattr(tc, "CalibratedClassifierCV", FakeCalibratedClassifier)
    monkeypatch.setattr(tc, "REPORT_PATH", str(reports_dir / "pr.png"))
    monkeypatch.setattr(tc, "CONFUSION_PATH", str(reports_dir / "cm.png"))
    monkeypatch.setattr(tc, "IMPORTANCE_PATH", str(reports_dir / "importance.csv"))
    monkeypatch.setattr(tc, "_save_feature_importance", lambda model: (reports_dir / "importance.csv").write_text("feature,importance\n"))

    model, auc = tc.train_churn_model(output_dir=str(models_dir), vins_path=str(path))

    assert isinstance(model, FakeCalibratedClassifier)
    assert model.was_fit is True
    assert auc == 0.5
    assert (models_dir / "churn_pos_venda_rf_calibrated.joblib").exists()
    assert (models_dir / "churn_pos_venda_rf_calibrated.sha256").exists()


def test_train_churn_model_usa_input_dir(monkeypatch, tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    (input_dir / "dataset_churn_pos_venda.csv").write_text("fake")
    models_dir = tmp_path / "models"
    called = {}

    def fake_load_data(path):
        called["path"] = path
        df = _churn_dataset(20)
        return df.drop(columns=[TARGET_CHURN]), df[TARGET_CHURN]

    monkeypatch.setattr(tc, "_load_data", fake_load_data)
    monkeypatch.setattr(tc, "CalibratedClassifierCV", FakeCalibratedClassifier)
    monkeypatch.setattr(tc, "_save_feature_importance", lambda model: None)

    tc.train_churn_model(input_dir=str(input_dir), output_dir=str(models_dir), vins_path="ignorado.csv")

    assert called["path"] == str(input_dir / "dataset_churn_pos_venda.csv")


def test_parse_args_train_churn(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_churn_real.py",
        "--input-dir", "processed",
        "--output-dir", "models-test",
        "--input-path", "dataset.csv",
    ])

    args = tc._parse_args()

    assert args.input_dir == "processed"
    assert args.output_dir == "models-test"
    assert args.input_path == "dataset.csv"


def test_save_feature_importance_grava_csv(monkeypatch, tmp_path):
    class FakePreprocessor:
        def get_feature_names_out(self):
            return ["numeric__km", "categorical__modelo_KA"]

    class FakeRf:
        feature_importances_ = [0.7, 0.3]

    estimator = type("Estimator", (), {"named_steps": {"preprocessor": FakePreprocessor(), "model": FakeRf()}})()
    calibrated = type("Calibrated", (), {"estimator": estimator})()
    model = type("Model", (), {"calibrated_classifiers_": [calibrated]})()
    output = tmp_path / "importance.csv"
    monkeypatch.setattr(tc, "IMPORTANCE_PATH", str(output))

    tc._save_feature_importance(model)

    saved = pd.read_csv(output)
    assert saved["feature"].tolist() == ["km", "modelo_KA"]
    assert saved["importance"].tolist() == [0.7, 0.3]
