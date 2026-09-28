import json
import zipfile

import pandas as pd
import pytest

from src.pipeline import select_mock_vins_pca as sv


def _snapshot_rows():
    grupos = [
        ("fiel", 4, 4, 1, 0.9, 20000),
        ("recente", 1, 3, 1, 0.6, 15000),
        ("atraso", 2, 9, 1, 0.6, 30000),
        ("inativo", 2, 18, 1, 0.6, 45000),
        ("abandono", 1, 14, 1, 0.6, 12000),
        ("extremo", 2, 5, 4, 0.2, 800000),
        ("fiel2", 5, 2, 1, 0.8, 25000),
        ("recente2", 2, 5, 1, 0.7, 18000),
        ("atraso2", 3, 10, 2, 0.5, 32000),
        ("inativo2", 3, 20, 2, 0.5, 52000),
        ("abandono2", 1, 16, 1, 0.5, 14000),
        ("extremo2", 2, 4, 3, 0.3, 70000),
    ]
    rows = []
    for i, (_, revisoes, meses, dealers, agenda, km) in enumerate(grupos):
        rows.append({
            sv.VIN_COL: f"hash_{i:03d}",
            sv.MODEL_COL: "KA" if i % 2 == 0 else "RANGER",
            sv.YEAR_COL: 2020 + (i % 3),
            "sales_date": "2020-01-01",
            "delivery_date": "2020-01-02",
            "dealer_code_ate_corte": f"D{i % 4}",
            "data_corte": "2024-10-31",
            "qtde_revisoes_ate_corte": revisoes,
            "meses_desde_ultimo_servico_ate_corte": meses,
            "meses_relacionamento_ate_corte": 24 + i,
            "n_dealers_usados_ate_corte": dealers,
            "km_max_ate_corte": km,
            "pct_agenda_ate_corte": agenda,
            "intervalo_medio_revisoes_dias_ate_corte": 120 + i,
            "dias_ate_primeira_revisao": 90 + i,
            "idade_veiculo_meses_ate_corte": 36 + i,
            "churn_futuro_18m": i % 2,
        })
    return pd.DataFrame(rows)


def test_quota_by_sqrt_distribui_total_com_minimos():
    values = pd.Series(["KA"] * 9 + ["RANGER"] * 4 + ["MAVERICK"] * 1)

    quotas = sv._quota_by_sqrt(values, total=6, min_count=2, min_quota=1)

    assert sum(quotas.values()) == 6
    assert set(quotas) == {"KA", "RANGER"}
    assert all(value >= 1 for value in quotas.values())


def test_allocate_with_caps_respeita_limite_total_e_caps():
    allocation = sv._allocate_with_caps(
        weights={"KA": 10, "RANGER": 1},
        caps={"KA": 2, "RANGER": 5},
        quota=5,
    )

    assert sum(allocation.values()) == 5
    assert allocation["KA"] <= 2
    assert allocation["RANGER"] <= 5


def test_load_snapshots_filtra_invalidos_e_valida_features(monkeypatch, tmp_path):
    path = tmp_path / "snapshots.csv"
    df = _snapshot_rows()
    invalid = df.iloc[[0]].copy()
    invalid[sv.VIN_COL] = None
    pd.concat([df, invalid], ignore_index=True).to_csv(path, index=False)
    monkeypatch.setattr(sv, "SNAPSHOTS_PATH", path)

    loaded = sv._load_snapshots()

    assert len(loaded) == len(df) - 1  # linha de km 800000 tambem e filtrada
    assert loaded[sv.VIN_COL].notna().all()
    assert loaded["km_max_ate_corte"].le(500000).all()


def test_build_group_masks_identifica_grupos_comportamentais():
    df = _snapshot_rows()
    masks = sv._build_group_masks(df)

    assert masks["fiel_recorrente"].any()
    assert masks["recente_baixo_risco"].any()
    assert masks["atraso_moderado_medio_risco"].any()
    assert masks["alto_risco_inativo"].any()
    assert masks["abandono_provavel_1_servico_antigo"].any()
    assert masks["extremos_uteis"].any()


def test_select_nearest_to_centroids_retorna_todos_quando_quota_cobre_subset():
    df = _snapshot_rows().head(2)
    picks = sv._select_nearest_to_centroids(
        df,
        quota=2,
        group_name="teste",
        selected_vins=set(),
        model_counts={},
        year_counts={},
        model_caps={},
        year_caps={},
    )

    assert len(picks) == 2
    assert all(pick["selection_distance"] == 0.0 for pick in picks)
    assert all(pick["grupo_selecao"] == "teste" for pick in picks)


def test_build_feature_snapshot_imputa_nulos():
    full_df = _snapshot_rows()
    selected = full_df.head(2).copy()
    selected.insert(0, "vin_simulado", ["VIN-0001", "VIN-0002"])
    selected.loc[selected.index[0], "km_max_ate_corte"] = pd.NA
    selected["grupo_selecao"] = "teste"

    out = sv._build_feature_snapshot(selected, full_df)

    assert bool(out.loc[0, "km_max_ate_corte_foi_imputado"]) is True
    assert pd.notna(out.loc[0, "km_max_ate_corte"])
    assert out["ano_modelo"].dtype.kind in {"i", "u"}


def test_xlsx_helpers_lidam_com_colunas_e_aba_ausente(tmp_path):
    assert sv._xlsx_col_idx("A1") == 0
    assert sv._xlsx_col_idx("AA10") == 26
    xlsx = tmp_path / "empty.xlsx"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr("xl/workbook.xml", """<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets></sheets></workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", """<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'></Relationships>""")
    with zipfile.ZipFile(xlsx) as zf:
        with pytest.raises(ValueError, match="Aba nao encontrada"):
            sv._sheet_xml_path(zf, "vin_share")


def test_main_seleciona_vins_e_grava_outputs(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    snapshots_path = processed / "snapshots_pos_venda.csv"
    raw_xlsx = tmp_path / "raw.xlsx"
    raw_xlsx.write_bytes(b"fake")
    _snapshot_rows().to_csv(snapshots_path, index=False)

    monkeypatch.setattr(sv, "SNAPSHOTS_PATH", snapshots_path)
    monkeypatch.setattr(sv, "RAW_XLSX_PATH", raw_xlsx)
    monkeypatch.setattr(sv, "TOTAL_VINS", 6)
    monkeypatch.setattr(sv, "QUOTAS", {
        "fiel_recorrente": 1,
        "recente_baixo_risco": 1,
        "atraso_moderado_medio_risco": 1,
        "alto_risco_inativo": 1,
        "abandono_provavel_1_servico_antigo": 1,
        "extremos_uteis": 1,
    })
    monkeypatch.setattr(sv, "OUTPUT_SELECTED", processed / "selected.csv")
    monkeypatch.setattr(sv, "OUTPUT_FEATURES", processed / "features.csv")
    monkeypatch.setattr(sv, "OUTPUT_SERVICES", processed / "services.csv")
    monkeypatch.setattr(sv, "OUTPUT_VEHICLES", processed / "vehicles.csv")
    monkeypatch.setattr(sv, "OUTPUT_REPORT", processed / "report.json")
    monkeypatch.setattr(sv, "_export_services_for_vins", lambda vins: len(list(vins)) * 2)

    sv.main()

    selected = pd.read_csv(processed / "selected.csv")
    features = pd.read_csv(processed / "features.csv")
    vehicles = pd.read_csv(processed / "vehicles.csv")
    report = json.loads((processed / "report.json").read_text())

    assert len(selected) == 6
    assert len(features) == 6
    assert len(vehicles) == 6
    assert report["total_vins_selecionados"] == 6
    assert report["service_rows_exported"] == 12


def test_ensure_inputs_rejeita_arquivos_ausentes(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "SNAPSHOTS_PATH", tmp_path / "snapshots.csv")
    monkeypatch.setattr(sv, "RAW_XLSX_PATH", tmp_path / "raw.xlsx")

    with pytest.raises(FileNotFoundError, match="feature_engineering_real"):
        sv._ensure_inputs()

    (tmp_path / "snapshots.csv").write_text("x")
    with pytest.raises(FileNotFoundError, match="raw.xlsx"):
        sv._ensure_inputs()


def test_load_snapshots_rejeita_feature_numerica_ausente(monkeypatch, tmp_path):
    path = tmp_path / "snapshots.csv"
    _snapshot_rows().drop(columns=[sv.FEATURES_NUMERICAS[0]]).to_csv(path, index=False)
    monkeypatch.setattr(sv, "SNAPSHOTS_PATH", path)

    with pytest.raises(ValueError, match="Feature numerica ausente"):
        sv._load_snapshots()


def test_allocate_helpers_retornam_vazio_sem_capacidade():
    assert sv._allocate_with_caps({"KA": 1}, {"KA": 0}, 3) == {}
    assert sv._allocate_with_caps({}, {"KA": 1}, 3) == {}
    assert sv._allocate_strata_quota(pd.Series([], dtype=object), 3, {}, {}) == {}


def test_select_vins_rejeita_total_final_incompleto(monkeypatch):
    df = _snapshot_rows().head(2)
    monkeypatch.setattr(sv, "TOTAL_VINS", 3)
    monkeypatch.setattr(sv, "QUOTAS", {key: 0 for key in sv.QUOTAS})

    with pytest.raises(ValueError, match="Selecao final deveria"):
        sv.select_vins(df)


def test_cell_value_variantes():
    import xml.etree.ElementTree as ET
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    assert sv._cell_value(ET.fromstring(f"<c xmlns='{ns[1:-1]}' t='s'><v>0</v></c>"), ["VIN_Hash"]) == "VIN_Hash"
    assert sv._cell_value(ET.fromstring(f"<c xmlns='{ns[1:-1]}' t='b'><v>1</v></c>"), []) == "TRUE"
    assert sv._cell_value(ET.fromstring(f"<c xmlns='{ns[1:-1]}' t='inlineStr'><is><t>texto</t></is></c>"), []) == "texto"
    assert sv._cell_value(ET.fromstring(f"<c xmlns='{ns[1:-1]}'><v>123</v></c>"), []) == "123"


def test_export_services_for_vins_filtra_xlsx(monkeypatch, tmp_path):
    xlsx = tmp_path / "raw.xlsx"
    output = tmp_path / "services.csv"
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", f"""
<sst xmlns='{main_ns}'>
  <si><t>VIN_Hash</t></si><si><t>ServiceDate</t></si><si><t>vin1</t></si><si><t>vin2</t></si><si><t>01/01/2024</t></si>
</sst>""")
        zf.writestr("xl/workbook.xml", f"""
<workbook xmlns='{main_ns}' xmlns:r='{rel_ns}'><sheets><sheet name='vin_share' r:id='rId1'/></sheets></workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", f"""
<Relationships xmlns='{pkg_ns}'><Relationship Id='rId1' Target='worksheets/sheet1.xml'/></Relationships>""")
        zf.writestr("xl/worksheets/sheet1.xml", f"""
<worksheet xmlns='{main_ns}'><sheetData>
  <row r='1'><c r='A1' t='s'><v>0</v></c><c r='B1' t='s'><v>1</v></c></row>
  <row r='2'><c r='A2' t='s'><v>2</v></c><c r='B2' t='s'><v>4</v></c></row>
  <row r='3'><c r='A3' t='s'><v>3</v></c><c r='B3' t='s'><v>4</v></c></row>
</sheetData></worksheet>""")

    monkeypatch.setattr(sv, "RAW_XLSX_PATH", xlsx)
    monkeypatch.setattr(sv, "OUTPUT_SERVICES", output)

    rows = sv._export_services_for_vins(["vin2"])

    assert rows == 1
    assert output.read_text().splitlines() == ["VIN_Hash,ServiceDate", "vin2,01/01/2024"]


def test_sheet_xml_path_rejeita_relacionamento_ausente(tmp_path):
    xlsx = tmp_path / "missing_rel.xlsx"
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr("xl/workbook.xml", f"<workbook xmlns='{main_ns}' xmlns:r='{rel_ns}'><sheets><sheet name='vin_share' r:id='rId1'/></sheets></workbook>")
        zf.writestr("xl/_rels/workbook.xml.rels", f"<Relationships xmlns='{pkg_ns}'></Relationships>")

    with zipfile.ZipFile(xlsx) as zf:
        with pytest.raises(ValueError, match="Relacionamento da aba"):
            sv._sheet_xml_path(zf, "vin_share")


def test_select_nearest_to_centroids_retorna_vazio_para_subset_vazio():
    picks = sv._select_nearest_to_centroids(
        _snapshot_rows().iloc[0:0],
        quota=1,
        group_name="vazio",
        selected_vins=set(),
        model_counts={},
        year_counts={},
        model_caps={},
        year_caps={},
    )

    assert picks == []
