import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.models.schemas import PREDICT_FEATURES_EXAMPLE, PredictRequest
from src.api.services import predictor as predictor_module
from src.api.services.predictor import PredictorService


class FakeChurnModel:
    feature_names_in_ = list(PREDICT_FEATURES_EXAMPLE.keys())

    def __init__(self, probabilities=None, labels=None):
        self.probabilities = probabilities
        self.labels = labels

    def predict_proba(self, frame):
        assert list(frame.columns) == self.feature_names_in_
        return np.array(self.probabilities)


class FakePredictOnlyModel:
    feature_names_in_ = list(PREDICT_FEATURES_EXAMPLE.keys())

    def predict(self, frame):
        assert len(frame) == 2
        return np.array([0.35, 0.85])


class FakePerfilModel:
    def predict(self, frame):
        assert list(frame.columns) == predictor_module.CLUSTER_FEATURES
        return np.array([0, 1])


class BrokenPerfilModel:
    def predict(self, frame):
        raise RuntimeError("perfil quebrado")


class FakeEstimatorWrapper:
    estimator = type(
        "Estimator",
        (),
        {"named_steps": {"preprocessor": type("Preprocessor", (), {"feature_names_in_": ["modelo", "ano_modelo"]})()}},
    )()

    def predict_proba(self, frame):
        return np.array([[0.8, 0.2]])


@pytest.fixture
def service():
    svc = PredictorService()
    original = (
        svc.model_churn,
        svc.model_kmeans,
        dict(svc._segment_map),
        svc.feature_names,
        svc._perfil_loaded,
    )
    svc.model_churn = None
    svc.model_kmeans = None
    svc._segment_map = {}
    svc.feature_names = None
    svc._perfil_loaded = False
    try:
        yield svc
    finally:
        (
            svc.model_churn,
            svc.model_kmeans,
            svc._segment_map,
            svc.feature_names,
            svc._perfil_loaded,
        ) = original


def test_verify_checksum_aceita_sidecar_sha256_valido(tmp_path):
    model_path = tmp_path / "modelo.joblib"
    model_path.write_bytes(b"modelo-falso")
    model_path.with_suffix(".sha256").write_text(hashlib.sha256(b"modelo-falso").hexdigest())

    predictor_module._verify_checksum(model_path)


def test_verify_checksum_rejeita_sidecar_sha256_invalido(tmp_path):
    model_path = tmp_path / "modelo.joblib"
    model_path.write_bytes(b"modelo-falso")
    model_path.with_suffix(".sha256").write_text("0" * 64)

    with pytest.raises(HTTPException) as exc:
        predictor_module._verify_checksum(model_path)

    assert exc.value.status_code == 503
    assert "invalido" in exc.value.detail


def test_verify_checksum_rejeita_modelo_sem_sidecar_e_sem_fallback(tmp_path):
    model_path = tmp_path / "modelo-desconhecido.joblib"
    model_path.write_bytes(b"modelo-falso")

    with pytest.raises(HTTPException) as exc:
        predictor_module._verify_checksum(model_path)

    assert exc.value.status_code == 503
    assert "nao cadastrado" in exc.value.detail


def test_ensure_models_read_only_rejeita_diretorio_inexistente(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor_module, "MODELS_DIR", tmp_path / "models-ausente")

    with pytest.raises(HTTPException) as exc:
        predictor_module._ensure_models_read_only()

    assert exc.value.status_code == 503
    assert "modelos nao carregados" in exc.value.detail


def test_ensure_models_read_only_rejeita_falha_ao_remover_escrita(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "modelo.joblib"
    model_path.write_bytes(b"fake")

    monkeypatch.setattr(predictor_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predictor_module.Path, "chmod", lambda self, mode: (_ for _ in ()).throw(OSError("sem permissao")))

    with pytest.raises(HTTPException) as exc:
        predictor_module._ensure_models_read_only()

    assert exc.value.status_code == 503
    assert "read-only" in exc.value.detail


def test_load_churn_carrega_modelo_com_checksum_sidecar(monkeypatch, tmp_path, service):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / predictor_module.settings.CHURN_MODEL_FILENAME
    model_path.write_bytes(b"modelo-falso")
    model_path.with_suffix(".sha256").write_text(hashlib.sha256(b"modelo-falso").hexdigest())
    fake_model = FakeChurnModel(probabilities=[[0.7, 0.3]])

    monkeypatch.setattr(predictor_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predictor_module.joblib, "load", lambda path: fake_model)

    loaded = service._load_churn()

    assert loaded is fake_model
    assert service.feature_names == FakeChurnModel.feature_names_in_


def test_load_churn_retorna_503_quando_modelo_nao_existe(monkeypatch, tmp_path, service):
    monkeypatch.setattr(predictor_module, "MODELS_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        service._load_churn()

    assert exc.value.status_code == 503
    assert "modelo nao encontrado" in exc.value.detail


def test_load_perfil_carrega_modelo_e_mapa(monkeypatch, tmp_path, service):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / predictor_module.settings.PERFIL_MODEL_FILENAME
    model_path.write_bytes(b"perfil")
    model_path.with_suffix(".sha256").write_text(hashlib.sha256(b"perfil").hexdigest())
    model_path.with_name("cluster_segment_map.json").write_text(json.dumps({"0": "recorrente"}))
    fake_model = FakePerfilModel()

    monkeypatch.setattr(predictor_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predictor_module.joblib, "load", lambda path: fake_model)

    assert service._load_perfil() is fake_model
    assert service._segment_map == {"0": "recorrente"}
    assert service._perfil_loaded is True


def test_load_perfil_sem_modelo_marca_como_carregado(monkeypatch, tmp_path, service):
    monkeypatch.setattr(predictor_module, "MODELS_DIR", tmp_path)

    assert service._load_perfil() is None
    assert service._perfil_loaded is True


def test_load_perfil_sem_mapa_mantem_mapping_vazio(monkeypatch, tmp_path, service):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / predictor_module.settings.PERFIL_MODEL_FILENAME
    model_path.write_bytes(b"perfil")
    model_path.with_suffix(".sha256").write_text(hashlib.sha256(b"perfil").hexdigest())
    fake_model = FakePerfilModel()

    monkeypatch.setattr(predictor_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predictor_module.joblib, "load", lambda path: fake_model)

    assert service._load_perfil() is fake_model
    assert service._segment_map == {}


def test_build_batch_frame_rejeita_feature_obrigatoria_ausente(service):
    service.model_churn = FakeChurnModel(probabilities=[[0.7, 0.3]])
    service.feature_names = FakeChurnModel.feature_names_in_
    features = dict(PREDICT_FEATURES_EXAMPLE)
    features.pop("modelo")

    with pytest.raises(HTTPException) as exc:
        service._build_batch_frame([features])

    assert exc.value.status_code == 422
    assert exc.value.detail == {"missing_features": ["modelo"]}


def test_build_batch_frame_descobre_features_no_preprocessor(service):
    service.model_churn = FakeEstimatorWrapper()
    frame = service._build_batch_frame([{"modelo": "KA", "ano_modelo": 2020, "extra": "ignorado"}])

    assert list(frame.columns) == ["modelo", "ano_modelo"]
    assert service.feature_names == ["modelo", "ano_modelo"]


def test_predict_churn_batch_classifica_risco_baixo_medio_alto(service):
    service.model_churn = FakeChurnModel(
        probabilities=[[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]]
    )
    frame = pd.DataFrame([PREDICT_FEATURES_EXAMPLE] * 3)

    results = service._predict_churn_batch(frame)

    assert [(label.value, prob, risk.value) for label, prob, risk in results] == [
        ("no_churn", 0.1, "low"),
        ("no_churn", 0.45, "medium"),
        ("churn", 0.8, "high"),
    ]


def test_predict_churn_batch_aceita_modelo_sem_predict_proba(service):
    service.model_churn = FakePredictOnlyModel()
    frame = pd.DataFrame([PREDICT_FEATURES_EXAMPLE] * 2)

    results = service._predict_churn_batch(frame)

    assert [(label.value, prob, risk.value) for label, prob, risk in results] == [
        ("no_churn", 0.35, "low"),
        ("churn", 0.85, "high"),
    ]


def test_predict_perfil_retorna_none_quando_modelo_nao_esta_configurado(service):
    service._perfil_loaded = True
    service.model_kmeans = None
    frame = pd.DataFrame([PREDICT_FEATURES_EXAMPLE] * 2)

    assert service._predict_perfil_batch(frame) == [(None, None), (None, None)]


def test_predict_perfil_mapeia_clusters_para_segmentos(service):
    service._perfil_loaded = True
    service.model_kmeans = FakePerfilModel()
    service._segment_map = {"0": "recorrente", "1": "inativo"}
    frame = pd.DataFrame([PREDICT_FEATURES_EXAMPLE] * 2)

    assert service._predict_perfil_batch(frame) == [("recorrente", None), ("inativo", None)]


def test_predict_perfil_retorna_none_quando_modelo_falha(service):
    service._perfil_loaded = True
    service.model_kmeans = BrokenPerfilModel()
    frame = pd.DataFrame([PREDICT_FEATURES_EXAMPLE] * 2)

    assert service._predict_perfil_batch(frame) == [(None, None), (None, None)]


def test_predict_retorna_acao_recomendada_por_perfil(service):
    service.model_churn = FakeChurnModel(probabilities=[[0.8, 0.2]])
    service.feature_names = FakeChurnModel.feature_names_in_
    service.model_kmeans = FakePerfilModel()
    service._segment_map = {"0": "recorrente"}
    service._perfil_loaded = True

    response = service.predict(dict(PREDICT_FEATURES_EXAMPLE))

    assert response.prediction == "no_churn"
    assert response.churn_probability == 0.2
    assert response.risk_level == "low"
    assert response.perfil_previsto == "recorrente"
    assert response.acao_recomendada == predictor_module._ACOES["recorrente"]


def test_predict_batch_vazio_nao_chama_modelo(service):
    service.model_churn = FakeChurnModel(probabilities=[])

    assert service.predict_batch([]) == []


def test_predict_batch_preserva_reference_id_e_acao(service):
    service.model_churn = FakeChurnModel(probabilities=[[0.8, 0.2], [0.1, 0.9]])
    service.feature_names = FakeChurnModel.feature_names_in_
    service.model_kmeans = FakePerfilModel()
    service._segment_map = {"0": "recorrente", "1": "inativo"}
    service._perfil_loaded = True
    items = [
        PredictRequest(reference_id="vin-1", features=PREDICT_FEATURES_EXAMPLE),
        PredictRequest(reference_id="vin-2", features={**PREDICT_FEATURES_EXAMPLE, "modelo": "RANGER"}),
    ]

    responses = service.predict_batch(items)

    assert [response.reference_id for response in responses] == ["vin-1", "vin-2"]
    assert [response.prediction.value for response in responses] == ["no_churn", "churn"]
    assert [response.perfil_previsto for response in responses] == ["recorrente", "inativo"]
    assert all(response.acao_recomendada for response in responses)
