import pandas as pd
import pytest

from src.pipeline import generate_clientes_seed as gc


def _vehicle_rows(n=500):
    grupos = list(gc.RISK_BY_GROUP)
    rows = []
    for i in range(n):
        rows.append({
            "vin_simulado": f"VIN-{i + 1:04d}",
            "VIN_Hash": f"hash_{i + 1:04d}",
            "modelo": "KA" if i % 2 == 0 else "RANGER",
            "ano_modelo": 2020 + (i % 5),
            "km_max_ate_corte": 10000 + i,
            "sales_date": "2020-01-01",
            "delivery_date": "2020-01-02",
            "grupo_selecao": grupos[i % len(grupos)],
        })
    return pd.DataFrame(rows)


def _feature_rows(n=500):
    grupos = list(gc.RISK_BY_GROUP)
    rows = []
    for i in range(n):
        rows.append({
            "VIN_Hash": f"hash_{i + 1:04d}",
            "vin_simulado": f"VIN-{i + 1:04d}",
            "data_corte": "2024-10-31",
            "meses_desde_ultimo_servico_ate_corte": float(i % 18),
            "idade_veiculo_meses_ate_corte": float(12 + i % 60),
            "grupo_selecao": grupos[i % len(grupos)],
        })
    return pd.DataFrame(rows)


def test_slug_e_hash_token_sao_deterministicos():
    assert gc._slug("João da Silva!") == "jo.o.da.silva"
    assert gc._hash_token("vin", 123) == gc._hash_token("vin", 123)
    assert len(gc._hash_token("vin", 123, length=12)) == 12


def test_status_garantia_ultima_revisao_e_dias_sem_servico():
    row = pd.Series({
        "idade_veiculo_meses_ate_corte": 24,
        "data_corte": "2024-10-31",
        "meses_desde_ultimo_servico_ate_corte": 2,
    })

    assert gc._status_garantia(row) == "Garantia ativa"
    assert gc._dias_sem_servico(row) == 61
    assert gc._ultima_revisao(row) == "2024-08-31"

    row["idade_veiculo_meses_ate_corte"] = 48
    row["meses_desde_ultimo_servico_ate_corte"] = None
    assert gc._status_garantia(row) == "Garantia expirada"
    assert gc._dias_sem_servico(row) == 0
    assert gc._ultima_revisao(row) == "2024-10-31"


def test_assign_client_groups_valida_configuracao_inconsistente(monkeypatch):
    monkeypatch.setattr(gc, "TARGET_CLIENTS", 2)
    monkeypatch.setattr(gc, "CLIENT_GROUP_SIZES", [1])

    with pytest.raises(ValueError, match="Configuracao de grupos"):
        gc._assign_client_groups(pd.DataFrame({"VIN_Hash": ["vin1", "vin2"]}))


def test_build_clientes_gera_dados_sinteticos_unicos():
    group_a = pd.DataFrame({"VIN_Hash": ["vin1"], "grupo_selecao": ["fiel_recorrente"]})
    group_b = pd.DataFrame({"VIN_Hash": ["vin2", "vin3"], "grupo_selecao": ["alto_risco_inativo", "recente_baixo_risco"]})

    clientes = gc._build_clientes([group_a, group_b])

    assert len(clientes) == 2
    assert clientes["email"].str.endswith(".invalid").all()
    assert clientes["email"].is_unique
    assert clientes["documento"].is_unique
    assert clientes.loc[0, "nivel_risco"] == "BAIXO"
    assert clientes.loc[1, "nivel_risco"] == "ALTO"


def test_build_veiculos_valida_total_e_vin_unico(monkeypatch):
    monkeypatch.setattr(gc, "TARGET_CLIENTS", 1)
    groups = [_vehicle_rows(1).assign(
        data_corte="2024-10-31",
        meses_desde_ultimo_servico_ate_corte=1,
        idade_veiculo_meses_ate_corte=24,
    )]

    with pytest.raises(ValueError, match="Esperado 500 veiculos"):
        gc._build_veiculos(groups)


def test_main_gera_clientes_e_veiculos_seed(monkeypatch, tmp_path):
    input_vehicles = tmp_path / "veiculos_seed.csv"
    input_features = tmp_path / "vin_share_feature_snapshots_selected_500.csv"
    output_clientes = tmp_path / "clientes_seed.csv"
    output_veiculos = tmp_path / "veiculos_seed_com_clientes.csv"
    _vehicle_rows().to_csv(input_vehicles, index=False)
    _feature_rows().to_csv(input_features, index=False)

    monkeypatch.setattr(gc, "INPUT_VEHICLES", input_vehicles)
    monkeypatch.setattr(gc, "INPUT_FEATURES", input_features)
    monkeypatch.setattr(gc, "OUTPUT_CLIENTES", output_clientes)
    monkeypatch.setattr(gc, "OUTPUT_VEHICLES_CLIENTES", output_veiculos)

    clientes, veiculos = gc.main()

    assert len(clientes) == 340
    assert len(veiculos) == 500
    assert clientes["email"].is_unique
    assert clientes["documento"].is_unique
    assert veiculos["vin_hash"].is_unique
    assert veiculos["cliente_id"].between(1, 340).all()
    assert output_clientes.exists()
    assert output_veiculos.exists()


def test_load_inputs_rejeita_total_diferente_de_500(monkeypatch, tmp_path):
    input_vehicles = tmp_path / "veiculos_seed.csv"
    input_features = tmp_path / "vin_share_feature_snapshots_selected_500.csv"
    _vehicle_rows(2).to_csv(input_vehicles, index=False)
    _feature_rows(2).to_csv(input_features, index=False)
    monkeypatch.setattr(gc, "INPUT_VEHICLES", input_vehicles)
    monkeypatch.setattr(gc, "INPUT_FEATURES", input_features)

    with pytest.raises(ValueError, match="Esperado 500 veiculos"):
        gc._load_inputs()


def test_load_inputs_rejeita_arquivos_ausentes(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "INPUT_VEHICLES", tmp_path / "veiculos.csv")
    monkeypatch.setattr(gc, "INPUT_FEATURES", tmp_path / "features.csv")

    with pytest.raises(FileNotFoundError, match="veiculos.csv"):
        gc._load_inputs()

    (tmp_path / "veiculos.csv").write_text("x")
    with pytest.raises(FileNotFoundError, match="features.csv"):
        gc._load_inputs()


def test_load_inputs_rejeita_colunas_ausentes(monkeypatch, tmp_path):
    input_vehicles = tmp_path / "veiculos_seed.csv"
    input_features = tmp_path / "features.csv"
    _vehicle_rows().drop(columns=["modelo"]).to_csv(input_vehicles, index=False)
    _feature_rows().to_csv(input_features, index=False)
    monkeypatch.setattr(gc, "INPUT_VEHICLES", input_vehicles)
    monkeypatch.setattr(gc, "INPUT_FEATURES", input_features)

    with pytest.raises(ValueError, match="Colunas ausentes"):
        gc._load_inputs()

    _vehicle_rows().to_csv(input_vehicles, index=False)
    _feature_rows().drop(columns=["data_corte"]).to_csv(input_features, index=False)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        gc._load_inputs()


def test_load_inputs_rejeita_vin_duplicado(monkeypatch, tmp_path):
    input_vehicles = tmp_path / "veiculos_seed.csv"
    input_features = tmp_path / "features.csv"
    vehicles = _vehicle_rows()
    vehicles.loc[1, "VIN_Hash"] = vehicles.loc[0, "VIN_Hash"]
    vehicles.to_csv(input_vehicles, index=False)
    _feature_rows().to_csv(input_features, index=False)
    monkeypatch.setattr(gc, "INPUT_VEHICLES", input_vehicles)
    monkeypatch.setattr(gc, "INPUT_FEATURES", input_features)

    with pytest.raises(ValueError, match="VIN_Hash deve ser unico"):
        gc._load_inputs()


def test_build_clientes_rejeita_documentos_duplicados(monkeypatch):
    groups = [
        pd.DataFrame({"VIN_Hash": ["vin1"], "grupo_selecao": ["fiel_recorrente"]}),
        pd.DataFrame({"VIN_Hash": ["vin2"], "grupo_selecao": ["recente_baixo_risco"]}),
    ]
    monkeypatch.setattr(gc, "_hash_token", lambda *parts, length=10: "REPETIDO")

    with pytest.raises(ValueError, match="Documentos sinteticos duplicados"):
        gc._build_clientes(groups)


def test_build_veiculos_rejeita_vin_duplicado(monkeypatch):
    vehicles = _vehicle_rows().assign(
        data_corte="2024-10-31",
        meses_desde_ultimo_servico_ate_corte=1,
        idade_veiculo_meses_ate_corte=24,
    )
    vehicles.loc[1, "VIN_Hash"] = vehicles.loc[0, "VIN_Hash"]
    groups = [vehicles]

    with pytest.raises(ValueError, match="vin_hash deve ser unico"):
        gc._build_veiculos(groups)
