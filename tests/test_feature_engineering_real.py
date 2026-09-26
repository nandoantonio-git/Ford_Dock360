from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from src.pipeline import feature_engineering_real as fe
from src.pipeline.config import TARGET_CHURN
from src.pipeline.feature_engineering_real import build_snapshot, limpar_outliers, parsear_datas


def _orders_snapshot():
    return pd.DataFrame({
        "VIN_Hash": ["vin_retorna", "vin_retorna", "vin_churn", "vin_churn", "vin_remover"],
        "MaintenanceID": [1, 2, 3, 4, 5],
        "ServiceDate": ["01/15/2024", "07/31/2025", "01/20/2024", "08/01/2025", "01/10/2024"],
        "ModelName": ["KA", "KA", "RANGER", "RANGER", "MAVERICK"],
        "ModelYear": [2020, 2020, 2021, 2021, 2024],
        "DealerCode": ["D1", "D1", "D2", "D3", "D4"],
        "KM": [10000, 15000, 20000, 25000, 30000],
        "SalesDate": ["01/01/2020", "01/01/2020", "01/01/2021", "01/01/2021", "12/01/2025"],
        "DeliveryDate": ["01/02/2020", "01/02/2020", "01/02/2021", "01/02/2021", "12/02/2025"],
        "WarrantyStartDate": ["01/02/2020", "01/02/2020", "01/02/2021", "01/02/2021", "12/02/2025"],
        "IsAgendaSchedule": [1, 1, 0, 0, 1],
    })


def test_parsear_datas_cria_colunas_dt_com_formato_ford():
    df = pd.DataFrame({
        "ServiceDate": ["01/15/2024"],
        "SalesDate": ["01/01/2020"],
    })

    parsed = parsear_datas(df)

    assert str(parsed.loc[0, "ServiceDate_dt"].date()) == "2024-01-15"
    assert str(parsed.loc[0, "SalesDate_dt"].date()) == "2020-01-01"


def test_parsear_datas_usa_fallback_quando_formato_nao_e_ford():
    df = pd.DataFrame({"ServiceDate": ["2024-01-15", "2024-02-20"]})

    parsed = parsear_datas(df)

    assert parsed["ServiceDate_dt"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-15", "2024-02-20"]


def test_limpar_outliers_remove_km_negativo_e_absurdo():
    df = pd.DataFrame({"KM": [-1, 10000, 500001]})

    cleaned = limpar_outliers(df)

    assert np.isnan(cleaned.loc[0, "KM"])
    assert cleaned.loc[1, "KM"] == 10000
    assert np.isnan(cleaned.loc[2, "KM"])


def test_build_snapshot_calcula_churn_apenas_na_janela_futura():
    snapshot = build_snapshot(_orders_snapshot(), "2024-01-31", 18, fail_on_immature_window=True)

    rows = snapshot.set_index("VIN_Hash")

    assert set(rows.index) == {"vin_retorna", "vin_churn"}
    assert rows.loc["vin_retorna", TARGET_CHURN] == 0
    assert rows.loc["vin_churn", TARGET_CHURN] == 1
    assert rows.loc["vin_retorna", "qtde_revisoes_ate_corte"] == 1
    assert rows.loc["vin_churn", "qtde_revisoes_ate_corte"] == 1
    assert rows.loc["vin_retorna", "pct_agenda_ate_corte"] == 1
    assert rows.loc["vin_churn", "pct_agenda_ate_corte"] == 0
    assert rows["janela_futura_observavel"].all()


def test_build_snapshot_preenche_pct_agenda_nan_quando_coluna_ausente():
    df = _orders_snapshot().drop(columns=["IsAgendaSchedule"])

    snapshot = build_snapshot(df, "2024-01-31", 18, fail_on_immature_window=True)

    assert snapshot["pct_agenda_ate_corte"].isna().all()


def test_build_snapshot_corrige_primeira_revisao_antes_da_venda():
    df = _orders_snapshot()
    df.loc[df["VIN_Hash"] == "vin_churn", "SalesDate"] = "01/25/2024"

    snapshot = build_snapshot(df, "2024-01-31", 18, fail_on_immature_window=True)
    rows = snapshot.set_index("VIN_Hash")

    assert pd.isna(rows.loc["vin_churn", "dias_ate_primeira_revisao"])


def test_build_snapshot_falha_quando_janela_futura_imatura():
    df = _orders_snapshot()
    df.loc[df["ServiceDate"].isin(["07/31/2025", "08/01/2025"]), "ServiceDate"] = "07/30/2025"

    with pytest.raises(ValueError, match="Janela futura de churn imatura"):
        build_snapshot(df, "2024-01-31", 18, fail_on_immature_window=True)


def test_build_snapshot_falha_quando_nao_ha_historico_ate_corte():
    with pytest.raises(ValueError, match="Nenhum evento de servico"):
        build_snapshot(_orders_snapshot(), "2020-01-01", 18, fail_on_immature_window=False)


def test_feature_engineering_main_grava_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "carregar_ordens_servico", lambda input_path: _orders_snapshot())
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        input_path="fake.xlsx",
        data_corte="2024-01-31",
        janela_churn_meses=18,
        fail_on_immature_window=True,
    )

    snapshot = fe.main(args)

    assert not snapshot.empty
    assert (tmp_path / "snapshots_pos_venda.csv").exists()
    assert (tmp_path / "dataset_churn_pos_venda.csv").exists()


def test_carregar_ordens_servico_le_excel(monkeypatch, tmp_path):
    input_path = tmp_path / "ordens.xlsx"
    input_path.write_bytes(b"fake")
    expected = pd.DataFrame({"VIN_Hash": ["vin1"]})
    monkeypatch.setattr(fe.pd, "read_excel", lambda path, sheet_name: expected)

    loaded = fe.carregar_ordens_servico(str(input_path), sheet_name="vin_share")

    assert loaded is expected


def test_carregar_ordens_servico_rejeita_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset nao encontrado"):
        fe.carregar_ordens_servico(str(tmp_path / "ausente.xlsx"))


def test_parse_args_feature_engineering(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "feature_engineering_real.py",
        "--data-corte", "2024-01-31",
        "--janela-churn-meses", "12",
        "--output-dir", "out",
        "--input-path", "in.xlsx",
        "--fail-on-immature-window",
    ])

    args = fe._parse_args()

    assert args.data_corte == "2024-01-31"
    assert args.janela_churn_meses == 12
    assert args.output_dir == "out"
    assert args.input_path == "in.xlsx"
    assert args.fail_on_immature_window is True
