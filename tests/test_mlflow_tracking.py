import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.pipeline import mlflow_tracking as mt
from src.pipeline.config import TARGET_CHURN


class FakeRun:
    def __init__(self, run_id="run-1"):
        self.info = type("Info", (), {"run_id": run_id})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMlflow:
    def __init__(self):
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.tags = []
        self.logged_models = []
        self.registered_models = []
        self.experiment = None
        self.tracking_uri = None
        self.sklearn = type("SklearnApi", (), {"log_model": self._log_model})()

    def _log_model(self, model, artifact_path):
        self.logged_models.append((model, artifact_path))

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        return FakeRun(run_name or "run")

    def set_tags(self, tags):
        self.tags.append(tags)

    def log_param(self, key, value):
        self.params.append((key, value))

    def log_metric(self, key, value):
        self.metrics.append((key, value))

    def log_artifact(self, path):
        self.artifacts.append(path)

    def register_model(self, model_uri, model_name):
        self.registered_models.append((model_uri, model_name))
        return type("ModelVersion", (), {"version": "1"})()


class FakeMlflowClient:
    def __init__(self):
        self.transitions = []

    def transition_model_version_stage(self, model_name, version, stage, archive_existing_versions=True):
        self.transitions.append((model_name, version, stage, archive_existing_versions))


class FakeCandidateModel:
    def __init__(self, quality):
        self.quality = quality
        self.was_fit = False

    def fit(self, x, y):
        self.was_fit = True
        return self

    def predict_proba(self, x):
        signal = (x["ano_modelo"].astype(int) % 2).to_numpy()
        if self.quality == "good":
            scores = np.where(signal == 1, 0.9, 0.1)
        else:
            scores = np.where(signal == 1, 0.1, 0.9)
        return np.column_stack([1 - scores, scores])


class FakeKMeansStep:
    inertia_ = 12.3


class FakeKMeansPipeline:
    named_steps = {"kmeans": FakeKMeansStep()}

    def __init__(self, k):
        self.k = k

    def fit_predict(self, x):
        return np.arange(len(x)) % self.k

    def __getitem__(self, item):
        return self

    def transform(self, x):
        return x.to_numpy(dtype=float)


def _churn_dataset_for_mlflow(n=40):
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
        })
    return pd.DataFrame(rows)


def test_git_value_retorna_vazio_em_erro(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(mt.subprocess, "run", fake_run)

    assert mt._git_value(["rev-parse", "HEAD"]) == ""


def test_git_value_retorna_stdout_sem_espacos(monkeypatch):
    completed = subprocess.CompletedProcess(args=["git"], returncode=0, stdout=" abc123\n", stderr="")
    monkeypatch.setattr(mt.subprocess, "run", lambda *args, **kwargs: completed)

    assert mt._git_value(["rev-parse", "HEAD"]) == "abc123"


def test_release_tags_usa_unreleased_quando_nao_ha_tag(monkeypatch):
    values = {
        ("rev-parse", "HEAD"): "abc123",
        ("tag", "--points-at", "HEAD"): "",
    }
    monkeypatch.setattr(mt, "_git_value", lambda args: values[tuple(args)])

    tags = mt._release_tags()

    assert tags["git_commit"] == "abc123"
    assert tags["git_tag"] == "unreleased"
    assert tags["release_version"] == "unreleased"
    assert tags["tracking_type"] == "pos_venda_real"


def test_set_release_tags_mescla_tags_extras(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mt, "_mlflow_modules", lambda: (fake_mlflow, object))
    monkeypatch.setattr(mt, "_release_tags", lambda: {"git_commit": "abc"})

    mt._set_release_tags({"pipeline_step": "teste"})

    assert fake_mlflow.tags == [{"git_commit": "abc", "pipeline_step": "teste"}]


def test_setup_mlflow_configura_tracking_uri(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mt, "_mlflow_modules", lambda: (fake_mlflow, object))

    client = mt.setup_mlflow("file:/tmp/mlruns-test")

    assert isinstance(client, object)
    assert fake_mlflow.tracking_uri == "file:/tmp/mlruns-test"


def test_select_best_churn_candidate_rejeita_lista_vazia():
    with pytest.raises(ValueError, match="Nenhum candidato"):
        mt._select_best_churn_candidate([])


def test_churn_model_candidates_tem_modelos_esperados():
    candidates = mt._churn_model_candidates()

    assert [candidate["model_name"] for candidate in candidates] == [
        "LogisticRegression_Balanced",
        "RandomForest_Balanced",
        "RandomForest_Calibrated",
    ]
    assert all("params" in candidate for candidate in candidates)
    assert candidates[0]["params"]["class_weight"] == "balanced"
    assert candidates[1]["params"]["class_weight"] == "balanced"


def test_register_churn_model_comparison_grava_csv_e_seleciona_melhor(monkeypatch, tmp_path):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mt, "_mlflow_modules", lambda: (fake_mlflow, FakeMlflowClient))
    monkeypatch.setattr(mt, "_release_tags", lambda: {"git_commit": "abc", "git_tag": "unreleased"})
    monkeypatch.setattr(mt, "_load_data", lambda: _churn_dataset_for_mlflow(40))
    monkeypatch.setattr(mt, "_churn_model_candidates", lambda: [
        {"model_name": "bad", "model_family": "fake", "estimator": FakeCandidateModel("bad"), "params": {"kind": "bad"}},
        {"model_name": "good", "model_family": "fake", "estimator": FakeCandidateModel("good"), "params": {"kind": "good"}},
    ])

    output = tmp_path / "model_comparison_churn.csv"
    comparison = mt.register_churn_model_comparison(output_path=str(output))

    assert output.exists()
    assert set(comparison["model_name"]) == {"bad", "good"}
    assert comparison.loc[comparison["model_name"] == "good", "selected"].item() is True
    assert comparison["selected"].sum() == 1
    assert fake_mlflow.experiment == "ford_classificacao_churn_comparison"
    assert fake_mlflow.artifacts == [str(output)]
    assert ("selected_model", "good") in fake_mlflow.params


def test_load_data_le_arquivo_padrao(monkeypatch, tmp_path):
    path = tmp_path / "dataset.csv"
    expected = _churn_dataset_for_mlflow(4)
    expected.to_csv(path, index=False)
    monkeypatch.setattr(mt, "VINS_PATH", str(path))

    loaded = mt._load_data()

    assert loaded.shape == expected.shape


def test_parse_args_mlflow(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mlflow_tracking.py", "--only", "churn-comparison", "--comparison-output", "out.csv"])

    args = mt._parse_args()

    assert args.only == "churn-comparison"
    assert args.comparison_output == "out.csv"


def test_register_churn_classifier_registra_modelo(monkeypatch):
    fake_mlflow = FakeMlflow()
    client = FakeMlflowClient()
    monkeypatch.setattr(mt, "_mlflow_modules", lambda: (fake_mlflow, lambda: client))
    monkeypatch.setattr(mt, "_release_tags", lambda: {"git_commit": "abc"})
    monkeypatch.setattr(mt, "_load_data", lambda: _churn_dataset_for_mlflow(40))

    class FakeCalibrated:
        def __init__(self, estimator=None, cv=None, method=None):
            self.estimator = estimator

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            scores = (x["ano_modelo"].astype(int) % 2).to_numpy() * 0.8 + 0.1
            return np.column_stack([1 - scores, scores])

    monkeypatch.setattr(mt, "CalibratedClassifierCV", FakeCalibrated)

    auc = mt.register_churn_classifier(model_name="modelo_teste")

    assert 0 <= auc <= 1
    assert fake_mlflow.logged_models
    assert fake_mlflow.registered_models[0][1] == "modelo_teste"
    assert client.transitions == [("modelo_teste", "1", "Production", True)]


def test_register_kmeans_experiment_registra_candidatos(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mt, "_mlflow_modules", lambda: (fake_mlflow, FakeMlflowClient))
    monkeypatch.setattr(mt, "_release_tags", lambda: {"git_commit": "abc"})
    monkeypatch.setattr(mt, "_load_data", lambda: _churn_dataset_for_mlflow(30))
    monkeypatch.setattr(mt, "_make_pipeline", lambda k: FakeKMeansPipeline(k))
    monkeypatch.setattr(mt, "silhouette_score", lambda x, labels, sample_size, random_state: float(len(set(labels))) / 10)

    rows = mt.register_kmeans_experiment()

    assert rows["k"].tolist() == list(range(2, 9))
    assert fake_mlflow.experiment == "ford_segmentacao_kmeans"
    assert ("selected_k", 8) in fake_mlflow.params
    assert ("selected_silhouette_score", 0.8) in fake_mlflow.metrics
