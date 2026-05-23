import hashlib
import hmac
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.api.config import settings
from src.api.models.schemas import ChurnLabelEnum, PredictResponse, RiskLevelEnum

_logger = logging.getLogger(__name__)

CLUSTER_FEATURES = [
    "qtde_revisoes_ate_corte",
    "meses_desde_ultimo_servico_ate_corte",
    "meses_relacionamento_ate_corte",
    "n_dealers_usados_ate_corte",
    "km_max_ate_corte",
    "pct_agenda_ate_corte",
    "intervalo_medio_revisoes_dias_ate_corte",
]

_ACOES = {
    "baixo_engajamento": "Priorizar contato de retencao. Reforcar beneficios da rede autorizada e revisar vencimentos proximos.",
    "inativo": "Ativar Pulse Loop imediato. Oferecer diagnostico de retorno e incentivo para reagendamento na rede.",
    "multidealer": "Centralizar relacionamento. Confirmar concessionaria preferencial e padronizar proxima abordagem.",
    "recorrente": "Nenhuma acao ativa de recuperacao. Registrar acompanhamento preventivo para a proxima revisao.",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = _PROJECT_ROOT / settings.MODELS_DIR
# Fallback checksums for models that predate sidecar files (.sha256).
# The training pipeline writes a sidecar on each run, making this stale-proof.
_FALLBACK_SHA256 = {
    "churn_pos_venda_rf_calibrated.joblib": "c79d5e31e61e1e96718448b6179a0c3fab6b34ec3ca98f25cddd2d743a0fdb5f",
}


def _alert(message):
    print(f"ALERTA SEGURANCA: {message}")


def _ensure_models_read_only():
    if not MODELS_DIR.exists():
        raise HTTPException(status_code=503, detail="Servico indisponivel: modelos nao carregados")

    paths = [MODELS_DIR] + list(MODELS_DIR.glob("*.joblib"))
    for path in paths:
        mode = path.stat().st_mode
        if mode & 0o222:
            try:
                path.chmod(mode & ~0o222)
            except OSError as exc:
                _alert(f"falha ao remover permissao de escrita de {path}: {exc}")
                raise HTTPException(status_code=503, detail="Diretorio de modelos nao esta read-only") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    expected: str | None
    if sidecar.exists():
        expected = sidecar.read_text().strip()
    else:
        expected = _FALLBACK_SHA256.get(path.name)
    if expected is None:
        _alert(f"checksum SHA256 nao cadastrado para {path}")
        raise HTTPException(status_code=503, detail="Checksum SHA256 do modelo nao cadastrado")

    actual = _sha256(path)
    if not hmac.compare_digest(actual, expected):
        _alert(f"checksum SHA256 invalido para {path}")
        raise HTTPException(status_code=503, detail="Checksum SHA256 do modelo invalido")


class PredictorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_churn = None
            cls._instance.model_kmeans = None
            cls._instance._segment_map = {}
            cls._instance.feature_names = None
            cls._instance._perfil_loaded = False
        return cls._instance

    def _model_path(self, filename):
        return MODELS_DIR / filename

    def _load_churn(self):
        if self.model_churn is None:
            path = self._model_path(settings.CHURN_MODEL_FILENAME)
            if not path.exists():
                raise HTTPException(status_code=503, detail="Servico indisponivel: modelo nao encontrado")
            _ensure_models_read_only()
            _verify_checksum(path)
            self.model_churn = joblib.load(path)
            self.feature_names = list(getattr(self.model_churn, "feature_names_in_", []))
        return self.model_churn

    def _load_perfil(self):
        if not self._perfil_loaded:
            path = self._model_path(settings.PERFIL_MODEL_FILENAME)
            map_path = path.with_name("cluster_segment_map.json")
            if path.exists():
                _ensure_models_read_only()
                _verify_checksum(path)
                self.model_kmeans = joblib.load(path)
                if map_path.exists():
                    self._segment_map = json.loads(map_path.read_text())
                else:
                    _logger.warning(
                        "cluster_segment_map.json nao encontrado em %s — segmentos serao indices numericos",
                        map_path,
                    )
            else:
                _logger.warning(
                    "Modelo K-Means nao encontrado em %s — predicao de perfil desativada", path
                )
            self._perfil_loaded = True
        return self.model_kmeans

    def _build_frame(self, features):
        return self._build_batch_frame([features])

    def _build_batch_frame(self, features_list):
        self._load_churn()
        if not self.feature_names:
            estimator = getattr(self.model_churn, "estimator", None)
            preprocessor = getattr(estimator, "named_steps", {}).get("preprocessor")
            self.feature_names = list(getattr(preprocessor, "feature_names_in_", []))
        if self.feature_names:
            missing = sorted({c for features in features_list for c in self.feature_names if c not in features})
            if missing:
                raise HTTPException(status_code=422, detail={"missing_features": missing})
            features_list = [{c: features[c] for c in self.feature_names} for features in features_list]
        return pd.DataFrame(features_list)

    def _predict_churn(self, frame):
        return self._predict_churn_batch(frame)[0]

    def _predict_churn_batch(self, frame):
        model = self._load_churn()
        if hasattr(model, "predict_proba"):
            probs = [float(p[1]) for p in model.predict_proba(frame)]
        else:
            probs = [float(p) for p in model.predict(frame)]
        results = []
        for prob in probs:
            label = ChurnLabelEnum.churn if prob >= 0.5 else ChurnLabelEnum.no_churn
            risk = RiskLevelEnum.high if prob >= 0.70 else (RiskLevelEnum.medium if prob >= 0.40 else RiskLevelEnum.low)
            results.append((label, round(prob, 6), risk))
        return results

    def _predict_perfil(self, frame):
        return self._predict_perfil_batch(frame)[0]

    def _predict_perfil_batch(self, frame):
        model = self._load_perfil()
        if model is None:
            return [(None, None) for _ in range(len(frame))]
        try:
            x = pd.DataFrame([
                {c: row.get(c, np.nan) if isinstance(row, dict) else getattr(row, c, np.nan)
                 for c in CLUSTER_FEATURES}
                for row in (frame.to_dict("records") if isinstance(frame, pd.DataFrame) else frame)
            ])
            cluster_ids = model.predict(x)
            return [(self._segment_map.get(str(cid)), None) for cid in cluster_ids]
        except Exception as exc:
            _logger.warning("predict_perfil falhou: %s", exc)
            return [(None, None) for _ in range(len(frame))]

    def predict(self, features):
        frame = self._build_frame(features)
        label, prob_churn, risk = self._predict_churn(frame)
        perfil, _ = self._predict_perfil(frame)
        return PredictResponse(
            reference_id=None,
            prediction=label,
            churn_probability=prob_churn,
            risk_level=risk,
            perfil_previsto=perfil,
            probabilidades_perfil=None,
            acao_recomendada=_ACOES.get(perfil) if perfil else None,
        )

    def predict_batch(self, items):
        if not items:
            return []

        frame = self._build_batch_frame([item.features.model_dump() for item in items])
        churn_results = self._predict_churn_batch(frame)
        perfil_results = self._predict_perfil_batch(frame)

        responses = []
        for item, churn, perfil_result in zip(items, churn_results, perfil_results):
            label, prob_churn, risk = churn
            perfil, _ = perfil_result
            responses.append(PredictResponse(
                reference_id=item.reference_id,
                prediction=label,
                churn_probability=prob_churn,
                risk_level=risk,
                perfil_previsto=perfil,
                probabilidades_perfil=None,
                acao_recomendada=_ACOES.get(perfil) if perfil else None,
            ))
        return responses


predictor_service = PredictorService()
