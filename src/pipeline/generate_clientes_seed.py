"""Gera clientes sinteticos realistas para os 500 VINs mock.

Entrada:
  - data/processed/veiculos_seed.csv
  - data/processed/vin_share_feature_snapshots_selected_500.csv

Saida:
  - data/processed/clientes_seed.csv
  - data/processed/veiculos_seed_com_clientes.csv

Os dados gerados sao artificiais: emails usam o dominio reservado .invalid,
documentos sao identificadores sinteticos e telefones seguem faixa mock.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config import RANDOM_STATE


INPUT_VEHICLES = Path("data/processed/veiculos_seed.csv")
INPUT_FEATURES = Path("data/processed/vin_share_feature_snapshots_selected_500.csv")
OUTPUT_CLIENTES = Path("data/processed/clientes_seed.csv")
OUTPUT_VEHICLES_CLIENTES = Path("data/processed/veiculos_seed_com_clientes.csv")

FIXED_TIMESTAMP = "2026-05-23 00:00:00"
TARGET_CLIENTS = 340
CLIENT_GROUP_SIZES = [1] * 205 + [2] * 110 + [3] * 25

FIRST_NAMES = [
    "Ana", "Beatriz", "Camila", "Carolina", "Daniela", "Fernanda",
    "Gabriela", "Helena", "Isabela", "Juliana", "Larissa", "Mariana",
    "Natalia", "Patricia", "Renata", "Sofia", "Aline", "Bianca",
    "Bruna", "Claudia", "Eduardo", "Felipe", "Gabriel", "Gustavo",
    "Henrique", "Joao", "Lucas", "Marcelo", "Marcos", "Matheus",
    "Paulo", "Pedro", "Rafael", "Ricardo", "Rodrigo", "Thiago",
]

LAST_NAMES = [
    "Almeida", "Barbosa", "Cardoso", "Carvalho", "Costa", "Dias",
    "Fernandes", "Ferreira", "Gomes", "Lima", "Martins", "Mendes",
    "Moreira", "Nascimento", "Oliveira", "Pereira", "Ribeiro",
    "Rocha", "Santana", "Santos", "Silva", "Soares", "Souza",
    "Teixeira", "Vieira",
]

EMAIL_DOMAINS = [
    "gmail.com.invalid",
    "outlook.com.invalid",
    "hotmail.com.invalid",
    "yahoo.com.invalid",
]

RISK_BY_GROUP = {
    "fiel_recorrente": "BAIXO",
    "recente_baixo_risco": "BAIXO",
    "atraso_moderado_medio_risco": "MEDIO",
    "alto_risco_inativo": "ALTO",
    "abandono_provavel_1_servico_antigo": "ALTO",
    "extremos_uteis": "MEDIO",
}

RISK_ORDER = {
    "BAIXO": 0,
    "MEDIO": 1,
    "ALTO": 2,
    "CRITICO": 3,
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".")


def _hash_token(*parts: object, length: int = 10) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length].upper()


def _load_inputs() -> pd.DataFrame:
    if not INPUT_VEHICLES.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_VEHICLES}")
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_FEATURES}")

    vehicles = pd.read_csv(INPUT_VEHICLES, dtype={"VIN_Hash": "string", "vin_simulado": "string"})
    features = pd.read_csv(
        INPUT_FEATURES,
        dtype={"VIN_Hash": "string", "vin_simulado": "string", "grupo_selecao": "string"},
    )

    required_vehicle_cols = {
        "vin_simulado",
        "VIN_Hash",
        "modelo",
        "ano_modelo",
        "km_max_ate_corte",
        "sales_date",
        "delivery_date",
        "grupo_selecao",
    }
    missing = sorted(required_vehicle_cols - set(vehicles.columns))
    if missing:
        raise ValueError(f"Colunas ausentes em {INPUT_VEHICLES}: {missing}")

    feature_cols = {
        "VIN_Hash",
        "data_corte",
        "meses_desde_ultimo_servico_ate_corte",
        "idade_veiculo_meses_ate_corte",
        "grupo_selecao",
    }
    missing_features = sorted(feature_cols - set(features.columns))
    if missing_features:
        raise ValueError(f"Colunas ausentes em {INPUT_FEATURES}: {missing_features}")

    merged = vehicles.merge(
        features[[
            "VIN_Hash",
            "data_corte",
            "meses_desde_ultimo_servico_ate_corte",
            "idade_veiculo_meses_ate_corte",
            "grupo_selecao",
        ]],
        on="VIN_Hash",
        how="left",
        suffixes=("", "_feature"),
    )
    merged["grupo_selecao"] = merged["grupo_selecao"].fillna(merged["grupo_selecao_feature"])
    merged = merged.drop(columns=[col for col in ["grupo_selecao_feature"] if col in merged.columns])

    if len(merged) != 500:
        raise ValueError(f"Esperado 500 veiculos, recebido {len(merged)}")
    if merged["VIN_Hash"].nunique() != 500:
        raise ValueError("VIN_Hash deve ser unico nos 500 veiculos")

    return merged.sort_values(["grupo_selecao", "modelo", "ano_modelo", "VIN_Hash"]).reset_index(drop=True)


def _assign_client_groups(vehicles: pd.DataFrame) -> list[pd.DataFrame]:
    if len(CLIENT_GROUP_SIZES) != TARGET_CLIENTS or sum(CLIENT_GROUP_SIZES) != len(vehicles):
        raise ValueError("Configuracao de grupos de clientes inconsistente")

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_sizes = CLIENT_GROUP_SIZES.copy()
    rng.shuffle(shuffled_sizes)

    shuffled = vehicles.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    groups = []
    start = 0
    for size in shuffled_sizes:
        groups.append(shuffled.iloc[start:start + size].copy())
        start += size
    return groups


def _dominant_segment(group: pd.DataFrame) -> str:
    counts = Counter(group["grupo_selecao"].fillna("sem_segmento"))
    return counts.most_common(1)[0][0]


def _max_risk(group: pd.DataFrame) -> str:
    risks = [RISK_BY_GROUP.get(value, "MEDIO") for value in group["grupo_selecao"].fillna("sem_segmento")]
    return max(risks, key=lambda risk: RISK_ORDER[risk])


def _synthetic_name(client_id: int) -> tuple[str, str, str]:
    first = FIRST_NAMES[(client_id * 7 + RANDOM_STATE) % len(FIRST_NAMES)]
    last = LAST_NAMES[(client_id * 11 + RANDOM_STATE) % len(LAST_NAMES)]
    return first, last, f"{first} {last}"


def _build_clientes(groups: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for client_id, group in enumerate(groups, start=1):
        first, last, name = _synthetic_name(client_id)
        domain = EMAIL_DOMAINS[client_id % len(EMAIL_DOMAINS)]
        email = f"{_slug(first)}.{_slug(last)}.{client_id:03d}@{domain}"
        documento = f"DOC_{_hash_token(client_id, name, group['VIN_Hash'].iloc[0])}"
        telefone = f"+55119{client_id:08d}"[:14]

        rows.append({
            "id": client_id,
            "nome": name,
            "email": email,
            "telefone": telefone,
            "documento": documento,
            "segmento": _dominant_segment(group),
            "nivel_risco": _max_risk(group),
            "ativo": "true",
            "criado_em": FIXED_TIMESTAMP,
            "atualizado_em": FIXED_TIMESTAMP,
            "excluido_em": "",
        })

    clientes = pd.DataFrame(rows)
    if clientes["email"].nunique() != len(clientes):
        raise ValueError("Emails sinteticos duplicados")
    if clientes["documento"].nunique() != len(clientes):
        raise ValueError("Documentos sinteticos duplicados")
    return clientes


def _status_garantia(row: pd.Series) -> str:
    idade_meses = row.get("idade_veiculo_meses_ate_corte")
    if pd.notna(idade_meses) and float(idade_meses) <= 36:
        return "Garantia ativa"
    return "Garantia expirada"


def _ultima_revisao(row: pd.Series) -> str:
    data_corte = pd.to_datetime(row["data_corte"], errors="coerce")
    meses = pd.to_numeric(row["meses_desde_ultimo_servico_ate_corte"], errors="coerce")
    if pd.isna(data_corte) or pd.isna(meses):
        return "2024-10-31"
    dias = int(max(0, round(float(meses) * 30.44)))
    return (data_corte - pd.Timedelta(days=dias)).date().isoformat()


def _dias_sem_servico(row: pd.Series) -> int:
    meses = pd.to_numeric(row["meses_desde_ultimo_servico_ate_corte"], errors="coerce")
    if pd.isna(meses):
        return 0
    return int(max(0, round(float(meses) * 30.44)))


def _build_veiculos(groups: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    vehicle_id = 1
    for client_id, group in enumerate(groups, start=1):
        for _, row in group.iterrows():
            km = pd.to_numeric(row["km_max_ate_corte"], errors="coerce")
            rows.append({
                "id": vehicle_id,
                "cliente_id": client_id,
                "modelo": row["modelo"],
                "ano": int(row["ano_modelo"]),
                "vin_simulado": row["vin_simulado"],
                "km_atual": int(0 if pd.isna(km) else max(0, round(float(km)))),
                "ultima_revisao": _ultima_revisao(row),
                "dias_sem_servico": _dias_sem_servico(row),
                "status_garantia": _status_garantia(row),
                "vin_hash": row["VIN_Hash"],
                "criado_em": FIXED_TIMESTAMP,
                "atualizado_em": FIXED_TIMESTAMP,
            })
            vehicle_id += 1

    veiculos = pd.DataFrame(rows)
    if len(veiculos) != 500:
        raise ValueError(f"Esperado 500 veiculos com cliente, recebido {len(veiculos)}")
    if veiculos["vin_hash"].nunique() != 500:
        raise ValueError("vin_hash deve ser unico em veiculos_seed_com_clientes")
    if veiculos["cliente_id"].isna().any():
        raise ValueError("Todos os veiculos devem ter cliente_id")
    return veiculos.sort_values("id").reset_index(drop=True)


def _print_summary(clientes: pd.DataFrame, veiculos: pd.DataFrame) -> None:
    counts = veiculos.groupby("cliente_id").size().value_counts().sort_index().to_dict()
    print(f"Clientes gerados: {len(clientes)}")
    print(f"Veiculos vinculados: {len(veiculos)}")
    print(f"Distribuicao de veiculos por cliente: {counts}")
    print(f"Salvo: {OUTPUT_CLIENTES}")
    print(f"Salvo: {OUTPUT_VEHICLES_CLIENTES}")


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    vehicles = _load_inputs()
    groups = _assign_client_groups(vehicles)
    clientes = _build_clientes(groups)
    veiculos = _build_veiculos(groups)

    OUTPUT_CLIENTES.parent.mkdir(parents=True, exist_ok=True)
    clientes.to_csv(OUTPUT_CLIENTES, index=False)
    veiculos.to_csv(OUTPUT_VEHICLES_CLIENTES, index=False)

    _print_summary(clientes, veiculos)
    return clientes, veiculos


if __name__ == "__main__":
    main()
