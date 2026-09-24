"""Checks simples para evitar leakage temporal no X."""

from src.pipeline.config import LEAKAGE_COLUMNS, LEAKAGE_PATTERNS


def check_temporal_leakage(x, feature_cols=None):
    feature_cols = list(feature_cols or x.columns)
    found = []

    for col in feature_cols:
        if col in LEAKAGE_COLUMNS:
            found.append(col)
            continue
        if any(pattern in col.lower() for pattern in LEAKAGE_PATTERNS):
            found.append(col)

    if found:
        raise ValueError(f"Colunas com leakage temporal em X: {found}")


def check_leakage(x):
    check_temporal_leakage(x)
