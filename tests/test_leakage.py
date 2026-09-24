import pytest
import pandas as pd
from src.pipeline.config import LEAKAGE_COLUMNS, TARGET_CHURN
from src.pipeline.leakage_checks import check_leakage, check_temporal_leakage

def test_leakage_detection():
    for col in LEAKAGE_COLUMNS:
        df = pd.DataFrame({
            col: [1, 2, 3],
            "ano_modelo": [2020, 2021, 2022],
            "modelo": ["KA", "KA", "KA"]
        })
        with pytest.raises(ValueError, match="Colunas com leakage temporal em X"):
            check_leakage(df)

def test_no_leakage_passes():
    df = pd.DataFrame({
        "ano_modelo": [2020, 2021, 2022],
        "modelo": ["KA", "KA", "KA"],
        "qtde_revisoes_ate_corte": [1, 2, 3],
        "meses_desde_ultimo_servico_ate_corte": [3, 6, 9],
    })
    check_leakage(df)

def test_pattern_leakage_detection():
    df = pd.DataFrame({
        TARGET_CHURN: [0, 1, 1],
        "ano_modelo": [2020, 2021, 2022],
    })
    with pytest.raises(ValueError, match="Colunas com leakage temporal em X"):
        check_temporal_leakage(df)
