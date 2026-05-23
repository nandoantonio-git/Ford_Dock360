from unittest.mock import patch

from src.api.models.schemas import ChurnLabelEnum, PredictResponse, RiskLevelEnum

_FEATURES = {
    "ano_modelo": 2023,
    "qtde_revisoes_ate_corte": 2,
    "meses_desde_ultimo_servico_ate_corte": 8,
    "meses_relacionamento_ate_corte": 20,
    "n_dealers_usados_ate_corte": 1,
    "km_max_ate_corte": 28000,
    "pct_agenda_ate_corte": 0.5,
    "intervalo_medio_revisoes_dias_ate_corte": 180,
    "dias_ate_primeira_revisao": 120,
    "idade_veiculo_meses_ate_corte": 24,
    "modelo": "RANGER",
}

_MOCK_RESPONSE = PredictResponse(
    prediction=ChurnLabelEnum.no_churn,
    churn_probability=0.2,
    risk_level=RiskLevelEnum.low,
    perfil_previsto="recorrente",
    probabilidades_perfil={"recorrente": 0.9, "inativo": 0.1},
    acao_recomendada="Nenhuma acao ativa de recuperacao.",
)


def test_predict_sem_token_retorna_403(client):
    response = client.post("/predict", json={"features": _FEATURES})
    assert response.status_code == 403


def test_predict_token_invalido_retorna_401(client):
    response = client.post(
        "/predict",
        json={"features": _FEATURES},
        headers={"Authorization": "Bearer token-invalido"},
    )
    assert response.status_code == 401


def test_predict_viewer_insuficiente_retorna_403(client, viewer_headers):
    response = client.post("/predict", json={"features": _FEATURES}, headers=viewer_headers)
    assert response.status_code == 403


def test_predict_analyst_retorna_200(client, analyst_headers):
    with patch("src.api.routers.predict.predictor_service") as mock_svc:
        mock_svc.predict.return_value = _MOCK_RESPONSE
        response = client.post("/predict", json={"features": _FEATURES}, headers=analyst_headers)
    assert response.status_code == 200
    assert response.json()["prediction"] == "no_churn"


def test_predict_admin_retorna_200(client, admin_headers):
    with patch("src.api.routers.predict.predictor_service") as mock_svc:
        mock_svc.predict.return_value = _MOCK_RESPONSE
        response = client.post("/predict", json={"features": _FEATURES}, headers=admin_headers)
    assert response.status_code == 200


def test_predict_batch_analyst_retorna_200(client, analyst_headers):
    payload = {
        "items": [
            {"reference_id": "queue-1", "features": _FEATURES},
            {"reference_id": "queue-2", "features": {**_FEATURES, "ano_modelo": 2020}},
        ]
    }
    with patch("src.api.routers.predict.predictor_service") as mock_svc:
        mock_svc.predict_batch.return_value = [
            _MOCK_RESPONSE.model_copy(update={"reference_id": "queue-1"}),
            _MOCK_RESPONSE.model_copy(update={"reference_id": "queue-2"}),
        ]
        response = client.post("/predict-batch", json=payload, headers=analyst_headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["items"][0]["reference_id"] == "queue-1"
    assert body["items"][1]["reference_id"] == "queue-2"


def test_predict_batch_sem_reference_id_mantem_compatibilidade(client, analyst_headers):
    payload = {"items": [{"features": _FEATURES}]}
    with patch("src.api.routers.predict.predictor_service") as mock_svc:
        mock_svc.predict_batch.return_value = [_MOCK_RESPONSE]
        response = client.post("/predict/batch", json=payload, headers=analyst_headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["reference_id"] is None
