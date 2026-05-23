from enum import Enum
from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    viewer = "viewer"
    analyst = "analyst"
    admin = "admin"


class ChurnLabelEnum(str, Enum):
    churn = "churn"
    no_churn = "no_churn"


class RiskLevelEnum(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


PREDICT_FEATURES_EXAMPLE = {
    "ano_modelo": 2020,
    "qtde_revisoes_ate_corte": 2,
    "meses_desde_ultimo_servico_ate_corte": 14.2,
    "meses_relacionamento_ate_corte": 48.0,
    "n_dealers_usados_ate_corte": 1,
    "km_max_ate_corte": 48200,
    "pct_agenda_ate_corte": 0.65,
    "intervalo_medio_revisoes_dias_ate_corte": 220.0,
    "dias_ate_primeira_revisao": 180,
    "idade_veiculo_meses_ate_corte": 54.0,
    "modelo": "KA",
}


class PredictFeatures(BaseModel):
    ano_modelo: int = Field(..., json_schema_extra={"example": 2020})
    qtde_revisoes_ate_corte: int = Field(..., ge=0, json_schema_extra={"example": 2})
    meses_desde_ultimo_servico_ate_corte: float = Field(..., ge=0, json_schema_extra={"example": 14.2})
    meses_relacionamento_ate_corte: float = Field(..., ge=0, json_schema_extra={"example": 48.0})
    n_dealers_usados_ate_corte: int = Field(..., ge=0, json_schema_extra={"example": 1})
    km_max_ate_corte: float = Field(..., ge=0, json_schema_extra={"example": 48200})
    pct_agenda_ate_corte: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.65})
    intervalo_medio_revisoes_dias_ate_corte: float = Field(..., ge=0, json_schema_extra={"example": 220.0})
    dias_ate_primeira_revisao: int = Field(..., ge=0, json_schema_extra={"example": 180})
    idade_veiculo_meses_ate_corte: float = Field(..., ge=0, json_schema_extra={"example": 54.0})
    modelo: str = Field(..., min_length=1, json_schema_extra={"example": "KA"})


class PredictRequest(BaseModel):
    reference_id: str | None = Field(default=None, max_length=120)
    features: PredictFeatures

    model_config = {
        "json_schema_extra": {
            "example": {
                "reference_id": "queue-123",
                "features": PREDICT_FEATURES_EXAMPLE,
            }
        }
    }


class PredictResponse(BaseModel):
    reference_id: str | None = None
    prediction: ChurnLabelEnum
    churn_probability: float
    risk_level: RiskLevelEnum
    perfil_previsto: str | None = None
    probabilidades_perfil: dict | None = None
    acao_recomendada: str | None = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., max_length=500)

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "reference_id": "queue-123",
                        "features": PREDICT_FEATURES_EXAMPLE,
                    },
                    {
                        "reference_id": "queue-124",
                        "features": {
                            **PREDICT_FEATURES_EXAMPLE,
                            "ano_modelo": 2023,
                            "qtde_revisoes_ate_corte": 1,
                            "meses_desde_ultimo_servico_ate_corte": 6.0,
                            "meses_relacionamento_ate_corte": 18.0,
                            "km_max_ate_corte": 23400,
                            "idade_veiculo_meses_ate_corte": 18.0,
                            "modelo": "RANGER",
                        },
                    },
                ]
            }
        }
    }


class BatchPredictResponse(BaseModel):
    items: list[PredictResponse]


class DemoTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: RoleEnum
    expires_in_minutes: int
