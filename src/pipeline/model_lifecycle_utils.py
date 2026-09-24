"""Utilitarios simples para o ciclo de vida de modelos."""

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artifact_manifest(paths, role=None, model_type="other"):
    manifest = []
    for path in paths:
        path = Path(path)
        manifest.append({
            "name": path.name,
            "path": str(path),
            "sha256": sha256_file(path),
            "role": role,
            "model_type": model_type,
        })
    return manifest


def validate_authorization(authorized_by, authorization_ticket, scope):
    missing = []
    values = {
        "authorized_by": authorized_by,
        "authorization_ticket": authorization_ticket,
        "scope": scope,
    }
    for name, value in values.items():
        if value is None or not str(value).strip():
            missing.append(name)
    if missing:
        raise ValueError(f"Autorizacao incompleta: campos ausentes {missing}")
    return {
        "authorized_by": str(authorized_by).strip(),
        "authorization_ticket": str(authorization_ticket).strip(),
        "authorization_scope": str(scope).strip(),
    }


def validate_cutoff_maturity(data_corte, max_service_date, label_window_months=18):
    data_corte = pd.Timestamp(data_corte)
    max_service_date = pd.Timestamp(max_service_date)
    fim_janela = data_corte + pd.DateOffset(months=label_window_months)
    maturity_ok = bool(pd.notna(max_service_date) and max_service_date >= fim_janela)
    result = {
        "data_corte": data_corte.date().isoformat(),
        "fim_janela_churn": fim_janela.date().isoformat(),
        "data_max_observada": max_service_date.date().isoformat() if pd.notna(max_service_date) else None,
        "label_window_months": int(label_window_months),
        "maturity_ok": maturity_ok,
    }
    if not maturity_ok:
        raise ValueError(
            "Janela futura de churn imatura: "
            f"fim_janela={result['fim_janela_churn']} > "
            f"data_max_observada={result['data_max_observada']}"
        )
    return result
