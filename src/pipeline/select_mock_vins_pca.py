"""Seleciona 500 VINs para base operacional mock.

Usa cotas comportamentais definidas por regra e PCA + MiniBatchKMeans dentro
de cada grupo para escolher VINs proximos dos centroides.
"""

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import iterparse, parse
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import RANDOM_STATE, SNAPSHOT_FEATURES_NUMERIC


SNAPSHOTS_PATH = Path("data/processed/snapshots_pos_venda.csv")
RAW_XLSX_PATH = Path("data/raw/vin_share_Desafio_02.xlsx")
RAW_SHEET_NAME = "vin_share"

OUTPUT_SELECTED = Path("data/processed/selected_vin_hashes_500.csv")
OUTPUT_FEATURES = Path("data/processed/vin_share_feature_snapshots_selected_500.csv")
OUTPUT_SERVICES = Path("data/processed/vin_share_servicos_seed.csv")
OUTPUT_VEHICLES = Path("data/processed/veiculos_seed.csv")
OUTPUT_REPORT = Path("data/processed/selected_vin_distribution_report.json")

TOTAL_VINS = 500
QUOTAS = {
    "fiel_recorrente": 100,
    "recente_baixo_risco": 70,
    "atraso_moderado_medio_risco": 90,
    "alto_risco_inativo": 110,
    "abandono_provavel_1_servico_antigo": 80,
    "extremos_uteis": 50,
}

SELECTION_ORDER = [
    "abandono_provavel_1_servico_antigo",
    "fiel_recorrente",
    "recente_baixo_risco",
    "atraso_moderado_medio_risco",
    "alto_risco_inativo",
    "extremos_uteis",
]

MODEL_COL = "modelo"
YEAR_COL = "ano_modelo"
VIN_COL = "VIN_Hash"
FEATURES_NUMERICAS = SNAPSHOT_FEATURES_NUMERIC


def _ensure_inputs():
    if not SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {SNAPSHOTS_PATH}. "
            "Rode antes: python -m src.pipeline.feature_engineering_real"
        )
    if not RAW_XLSX_PATH.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {RAW_XLSX_PATH}")


def _load_snapshots():
    df = pd.read_csv(SNAPSHOTS_PATH, dtype={VIN_COL: "string", MODEL_COL: "string"})
    for col in FEATURES_NUMERICAS:
        if col not in df.columns:
            raise ValueError(f"Feature numerica ausente no snapshot: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[MODEL_COL] = df[MODEL_COL].astype("string").str.strip()
    df[VIN_COL] = df[VIN_COL].astype("string").str.strip()

    eligible = df[
        df[VIN_COL].notna()
        & df[MODEL_COL].notna()
        & df[YEAR_COL].notna()
        & df["sales_date"].notna()
    ].copy()

    eligible = eligible[eligible["km_max_ate_corte"].isna() | (eligible["km_max_ate_corte"] <= 500_000)]
    eligible = eligible[eligible["meses_desde_ultimo_servico_ate_corte"].ge(0)]
    eligible = eligible[eligible["idade_veiculo_meses_ate_corte"].ge(0)]
    eligible = eligible.drop_duplicates(subset=[VIN_COL], keep="first")
    eligible[YEAR_COL] = eligible[YEAR_COL].round().astype(int)
    return eligible.reset_index(drop=True)


def _build_group_masks(df):
    high_km = df["km_max_ate_corte"] >= df["km_max_ate_corte"].quantile(0.95)
    multi_dealer = df["n_dealers_usados_ate_corte"] >= 3
    low_agenda = df["pct_agenda_ate_corte"].fillna(1) < 0.5

    return {
        "fiel_recorrente": (
            (df["qtde_revisoes_ate_corte"] >= 3)
            & (df["meses_desde_ultimo_servico_ate_corte"] <= 8)
            & (df["n_dealers_usados_ate_corte"] <= 2)
            & (df["pct_agenda_ate_corte"].fillna(0) >= 0.7)
        ),
        "recente_baixo_risco": (
            (df["meses_desde_ultimo_servico_ate_corte"] <= 6)
            & (df["qtde_revisoes_ate_corte"] >= 1)
        ),
        "atraso_moderado_medio_risco": (
            (df["qtde_revisoes_ate_corte"] >= 2)
            & (df["meses_desde_ultimo_servico_ate_corte"] > 6)
            & (df["meses_desde_ultimo_servico_ate_corte"] <= 12)
        ),
        "alto_risco_inativo": (
            (df["qtde_revisoes_ate_corte"] >= 2)
            & (df["meses_desde_ultimo_servico_ate_corte"] > 12)
        ),
        "abandono_provavel_1_servico_antigo": (
            (df["qtde_revisoes_ate_corte"] == 1)
            & (df["meses_desde_ultimo_servico_ate_corte"] > 8)
        ),
        "extremos_uteis": high_km | multi_dealer | low_agenda,
    }


def _quota_by_sqrt(values, total, min_count=5, min_quota=2):
    counts = values.dropna().value_counts()
    counts = counts[counts >= min_count]
    if counts.empty:
        return {}

    weights = np.sqrt(counts.astype(float))
    raw = weights / weights.sum() * total
    quotas = raw.round().astype(int)

    for key in quotas.index:
        quotas.loc[key] = max(min_quota, int(quotas.loc[key]))

    while int(quotas.sum()) > total:
        key = (quotas - raw).idxmax()
        if quotas.loc[key] > 1:
            quotas.loc[key] -= 1
        else:
            break

    while int(quotas.sum()) < total:
        key = (raw - quotas).idxmax()
        quotas.loc[key] += 1

    return quotas.astype(int).to_dict()


def _build_caps(df):
    model_targets = _quota_by_sqrt(df[MODEL_COL], TOTAL_VINS, min_count=5, min_quota=2)
    year_targets = _quota_by_sqrt(df[YEAR_COL], TOTAL_VINS, min_count=5, min_quota=2)

    model_caps = {
        key: max(target + 4, int(math.ceil(target * 1.15)))
        for key, target in model_targets.items()
    }
    year_caps = {
        key: max(target + 8, int(math.ceil(target * 1.12)))
        for key, target in year_targets.items()
    }
    return model_targets, year_targets, model_caps, year_caps


def _allocate_with_caps(weights, caps, quota):
    quota = min(int(quota), int(sum(caps.values())))
    if quota <= 0:
        return {}

    weights = {key: float(value) for key, value in weights.items() if caps.get(key, 0) > 0 and value > 0}
    if not weights:
        return {}

    total_weight = sum(weights.values())
    raw = {key: weights[key] / total_weight * quota for key in weights}
    allocation = {
        key: min(int(math.floor(raw[key])), int(caps[key]))
        for key in weights
    }

    while sum(allocation.values()) < quota:
        candidates = [key for key in weights if allocation[key] < caps[key]]
        if not candidates:
            break
        key = max(candidates, key=lambda item: (raw[item] - allocation[item], weights[item], str(item)))
        allocation[key] += 1

    return {key: value for key, value in allocation.items() if value > 0}


def _allocate_strata_quota(values, quota, targets, current_counts):
    available = values.dropna().value_counts().to_dict()
    if not available:
        return {}

    remaining_target = {
        key: max(int(targets.get(key, 0)) - int(current_counts[key]), 0)
        for key in available
    }
    target_capacity = sum(min(int(available[key]), int(remaining_target[key])) for key in available)

    if target_capacity >= quota:
        caps = {key: min(int(available[key]), int(remaining_target[key])) for key in available}
    else:
        caps = {key: int(available[key]) for key in available}

    weights = {
        key: math.sqrt(float(available[key])) * max(float(remaining_target[key]), 1.0)
        for key in available
    }
    return _allocate_with_caps(weights, caps, quota)


def _under_caps(row, model_counts, year_counts, model_caps, year_caps):
    model = row[MODEL_COL]
    year = int(row[YEAR_COL])
    model_cap = model_caps.get(model, 4)
    year_cap = year_caps.get(year, TOTAL_VINS)
    return model_counts[model] < model_cap and year_counts[year] < year_cap


def _fit_pca_space(subset):
    x_num = subset[FEATURES_NUMERICAS].copy()
    used_features = [col for col in FEATURES_NUMERICAS if not x_num[col].isna().all()]
    x_num = x_num[used_features]
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90, random_state=RANDOM_STATE)),
    ])
    return pipeline.fit_transform(x_num), pipeline


def _select_nearest_to_centroids(
    subset,
    quota,
    group_name,
    selected_vins,
    model_counts,
    year_counts,
    model_caps,
    year_caps,
):
    subset = subset[~subset[VIN_COL].isin(selected_vins)].copy()
    if subset.empty:
        return []

    subset = subset.sort_values([MODEL_COL, YEAR_COL, VIN_COL]).reset_index(drop=True)
    quota = min(quota, len(subset))

    if quota == len(subset):
        picks = []
        for local_idx, row in subset.iterrows():
            picks.append({
                "row": row.copy(),
                "local_idx": int(local_idx),
                "grupo_selecao": group_name,
                "selection_cluster": int(local_idx),
                "selection_distance": 0.0,
                "pca_components": 0,
                "selection_cap_relaxed": False,
            })
        return picks

    x_pca, pipeline = _fit_pca_space(subset)
    kmeans = MiniBatchKMeans(
        n_clusters=quota,
        random_state=RANDOM_STATE,
        n_init=10,
        batch_size=min(2048, max(100, len(subset))),
    )
    labels = kmeans.fit_predict(x_pca)

    picks = []
    picked_local = set()

    for cluster_id in range(quota):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue

        distances = np.linalg.norm(x_pca[idxs] - kmeans.cluster_centers_[cluster_id], axis=1)
        ordered = idxs[np.argsort(distances)]

        chosen = None
        cap_relaxed = False
        for local_idx in ordered:
            row = subset.iloc[local_idx]
            if row[VIN_COL] in selected_vins or int(local_idx) in picked_local:
                continue
            if _under_caps(row, model_counts, year_counts, model_caps, year_caps):
                chosen = int(local_idx)
                break

        if chosen is None:
            for local_idx in ordered:
                row = subset.iloc[local_idx]
                if row[VIN_COL] not in selected_vins and int(local_idx) not in picked_local:
                    chosen = int(local_idx)
                    cap_relaxed = True
                    break

        if chosen is not None:
            dist = float(np.linalg.norm(x_pca[chosen] - kmeans.cluster_centers_[cluster_id]))
            picked_local.add(chosen)
            picks.append({
                "row": subset.iloc[chosen].copy(),
                "local_idx": chosen,
                "grupo_selecao": group_name,
                "selection_cluster": int(cluster_id),
                "selection_distance": dist,
                "pca_components": int(pipeline.named_steps["pca"].n_components_),
                "selection_cap_relaxed": cap_relaxed,
            })

    if len(picks) < quota:
        selected_local = {p["local_idx"] for p in picks}
        centroid = x_pca.mean(axis=0)
        all_distances = np.linalg.norm(x_pca - centroid, axis=1)
        for local_idx in np.argsort(all_distances):
            if len(picks) >= quota:
                break
            row = subset.iloc[local_idx]
            if row[VIN_COL] in selected_vins or int(local_idx) in selected_local:
                continue
            selected_local.add(int(local_idx))
            picks.append({
                "row": subset.iloc[local_idx].copy(),
                "local_idx": int(local_idx),
                "grupo_selecao": group_name,
                "selection_cluster": -1,
                "selection_distance": float(all_distances[local_idx]),
                "pca_components": int(pipeline.named_steps["pca"].n_components_),
                "selection_cap_relaxed": True,
            })

    return picks


def _append_picks(selected_rows, picks, selected_vins, model_counts, year_counts):
    for pick in picks:
        row = pick["row"].copy()
        row["grupo_selecao"] = pick["grupo_selecao"]
        row["selection_cluster"] = pick["selection_cluster"]
        row["selection_distance"] = round(pick["selection_distance"], 6)
        row["pca_components"] = pick["pca_components"]
        row["selection_cap_relaxed"] = bool(pick["selection_cap_relaxed"])

        selected_rows.append(row)
        vin = row[VIN_COL]
        selected_vins.add(vin)
        model_counts[row[MODEL_COL]] += 1
        year_counts[int(row[YEAR_COL])] += 1


def select_vins(df):
    group_masks = _build_group_masks(df)
    model_targets, year_targets, model_caps, year_caps = _build_caps(df)

    selected_rows = []
    selected_vins = set()
    model_counts = Counter()
    year_counts = Counter()

    group_pool_counts = {group: int(mask.sum()) for group, mask in group_masks.items()}
    group_pick_details = {}

    for group in SELECTION_ORDER:
        quota = QUOTAS[group]
        subset = df[group_masks[group]].copy()
        selected_before_group = len(selected_rows)
        subset = subset[~subset[VIN_COL].isin(selected_vins)].copy()
        model_quotas = _allocate_strata_quota(subset[MODEL_COL], quota, model_targets, model_counts)

        for model, model_quota in model_quotas.items():
            selected_before_model = len(selected_rows)
            model_subset = subset[subset[MODEL_COL] == model].copy()
            year_quotas = _allocate_strata_quota(model_subset[YEAR_COL], model_quota, year_targets, year_counts)

            for year, year_quota in year_quotas.items():
                year_subset = model_subset[model_subset[YEAR_COL] == year].copy()
                picks = _select_nearest_to_centroids(
                    year_subset,
                    year_quota,
                    group,
                    selected_vins,
                    model_counts,
                    year_counts,
                    model_caps,
                    year_caps,
                )
                _append_picks(selected_rows, picks, selected_vins, model_counts, year_counts)

            model_missing = model_quota - (len(selected_rows) - selected_before_model)
            if model_missing > 0:
                model_remainder = model_subset[~model_subset[VIN_COL].isin(selected_vins)].copy()
                picks = _select_nearest_to_centroids(
                    model_remainder,
                    model_missing,
                    group,
                    selected_vins,
                    model_counts,
                    year_counts,
                    model_caps,
                    year_caps,
                )
                _append_picks(selected_rows, picks, selected_vins, model_counts, year_counts)

        group_missing = quota - (len(selected_rows) - selected_before_group)
        if group_missing > 0:
            group_remainder = subset[~subset[VIN_COL].isin(selected_vins)].copy()
            picks = _select_nearest_to_centroids(
                group_remainder,
                group_missing,
                group,
                selected_vins,
                model_counts,
                year_counts,
                model_caps,
                year_caps,
            )
            _append_picks(selected_rows, picks, selected_vins, model_counts, year_counts)

        group_pick_details[group] = {
            "quota": quota,
            "pool": group_pool_counts[group],
            "selected": len(selected_rows) - selected_before_group,
        }

    if len(selected_rows) < TOTAL_VINS:
        missing = TOTAL_VINS - len(selected_rows)
        remainder = df[~df[VIN_COL].isin(selected_vins)].copy()
        picks = _select_nearest_to_centroids(
            remainder,
            missing,
            "complemento_balanceado",
            selected_vins,
            model_counts,
            year_counts,
            model_caps,
            year_caps,
        )
        _append_picks(selected_rows, picks, selected_vins, model_counts, year_counts)
        group_pick_details["complemento_balanceado"] = {
            "quota": missing,
            "pool": int(len(remainder)),
            "selected": len(picks),
        }

    selected = pd.DataFrame(selected_rows)
    selected = selected.drop_duplicates(subset=[VIN_COL], keep="first")
    if len(selected) != TOTAL_VINS:
        raise ValueError(f"Selecao final deveria ter {TOTAL_VINS} VINs, mas teve {len(selected)}")

    selected = selected.sort_values(["grupo_selecao", MODEL_COL, YEAR_COL, VIN_COL]).reset_index(drop=True)
    selected.insert(0, "vin_simulado", [f"VIN-{i:04d}" for i in range(1, len(selected) + 1)])
    return selected, group_pool_counts, group_pick_details, model_targets, year_targets, model_caps, year_caps


def _build_feature_snapshot(selected, full_df):
    medians = full_df[FEATURES_NUMERICAS].median(numeric_only=True)
    out_cols = [
        "vin_simulado",
        VIN_COL,
        "data_corte",
        "grupo_selecao",
        *FEATURES_NUMERICAS,
        MODEL_COL,
    ]
    out = selected[out_cols].copy()

    for col in FEATURES_NUMERICAS:
        out[col + "_foi_imputado"] = out[col].isna()
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians[col])

    out[YEAR_COL] = out[YEAR_COL].round().astype(int)
    out["qtde_revisoes_ate_corte"] = out["qtde_revisoes_ate_corte"].round().astype(int)
    out["n_dealers_usados_ate_corte"] = out["n_dealers_usados_ate_corte"].round().astype(int)
    out["dias_ate_primeira_revisao"] = out["dias_ate_primeira_revisao"].round().astype(int)
    return out


def _build_vehicles(selected):
    cols = [
        "vin_simulado",
        VIN_COL,
        MODEL_COL,
        YEAR_COL,
        "dealer_code_ate_corte",
        "km_max_ate_corte",
        "sales_date",
        "delivery_date",
        "grupo_selecao",
    ]
    return selected[cols].copy()


def _xlsx_col_idx(cell_ref):
    match = re.match(r"([A-Z]+)", cell_ref)
    letters = match.group(1)
    idx = 0
    for char in letters:
        idx = idx * 26 + ord(char) - 64
    return idx - 1


def _load_shared_strings(zip_file):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings = []
    with zip_file.open("xl/sharedStrings.xml") as handle:
        for _, elem in iterparse(handle, events=("end",)):
            if elem.tag == ns + "si":
                strings.append("".join(t.text or "" for t in elem.iter(ns + "t")))
                elem.clear()
    return strings


def _cell_value(cell, shared_strings):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t")
    value = cell.find(ns + "v")
    if cell_type == "s":
        return shared_strings[int(value.text)] if value is not None and value.text is not None else ""
    if cell_type == "b":
        return "TRUE" if value is not None and value.text == "1" else "FALSE"
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(ns + "t"))
    return value.text if value is not None and value.text is not None else ""


def _sheet_xml_path(zip_file, sheet_name):
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    rels_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    workbook = parse(zip_file.open("xl/workbook.xml"))
    rel_id = None
    for sheet in workbook.iter(main_ns + "sheet"):
        if sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib.get(rel_ns + "id")
            break
    if rel_id is None:
        raise ValueError(f"Aba nao encontrada no XLSX: {sheet_name}")

    rels = parse(zip_file.open("xl/_rels/workbook.xml.rels"))
    for rel in rels.iter(rels_ns + "Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"

    raise ValueError(f"Relacionamento da aba nao encontrado: {sheet_name}")


def _export_services_for_vins(selected_vins):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    selected_vins = set(selected_vins)
    rows_written = 0

    with ZipFile(RAW_XLSX_PATH) as zip_file:
        shared_strings = _load_shared_strings(zip_file)
        sheet_path = _sheet_xml_path(zip_file, RAW_SHEET_NAME)

        with zip_file.open(sheet_path) as sheet, OUTPUT_SERVICES.open("w", newline="", encoding="utf-8") as output:
            writer = None
            headers = None
            vin_idx = None

            for _, row in iterparse(sheet, events=("end",)):
                if row.tag != ns + "row":
                    continue

                values = []
                max_idx = -1
                for cell in row.iter(ns + "c"):
                    idx = _xlsx_col_idx(cell.attrib.get("r", "A"))
                    if idx > max_idx:
                        max_idx = idx
                    values.append((idx, _cell_value(cell, shared_strings)))

                if headers is None:
                    arr = [""] * (max_idx + 1)
                    for idx, value in values:
                        arr[idx] = value
                    headers = arr
                    vin_idx = headers.index(VIN_COL)
                    writer = csv.writer(output)
                    writer.writerow(headers)
                    row.clear()
                    continue

                arr = [""] * len(headers)
                for idx, value in values:
                    if idx < len(arr):
                        arr[idx] = value

                if arr[vin_idx] in selected_vins:
                    writer.writerow(arr)
                    rows_written += 1

                row.clear()

    return rows_written


def _write_report(
    df,
    selected,
    group_pool_counts,
    group_pick_details,
    model_targets,
    year_targets,
    model_caps,
    year_caps,
    service_rows,
):
    feature_nulls_before = selected[FEATURES_NUMERICAS].isna().sum().astype(int).to_dict()
    report = {
        "random_state": RANDOM_STATE,
        "total_vins_entrada_snapshot": int(len(df)),
        "total_vins_selecionados": int(len(selected)),
        "quotas_configuradas": QUOTAS,
        "pool_por_grupo": group_pool_counts,
        "selecao_por_grupo": group_pick_details,
        "selecionados_por_grupo": selected["grupo_selecao"].value_counts().sort_index().astype(int).to_dict(),
        "selecionados_por_modelo": selected[MODEL_COL].value_counts().astype(int).to_dict(),
        "selecionados_por_ano_modelo": selected[YEAR_COL].value_counts().sort_index().astype(int).to_dict(),
        "selecionados_por_churn_futuro_18m": (
            selected["churn_futuro_18m"].value_counts(dropna=False).astype(int).to_dict()
            if "churn_futuro_18m" in selected.columns else {}
        ),
        "model_targets_sqrt": {str(k): int(v) for k, v in model_targets.items()},
        "year_targets_sqrt": {str(k): int(v) for k, v in year_targets.items()},
        "model_caps_usados": {str(k): int(v) for k, v in model_caps.items()},
        "year_caps_usados": {str(k): int(v) for k, v in year_caps.items()},
        "feature_nulls_before_feature_snapshot_imputation": feature_nulls_before,
        "service_rows_exported": int(service_rows),
        "outputs": {
            "selected": str(OUTPUT_SELECTED),
            "feature_snapshots": str(OUTPUT_FEATURES),
            "services": str(OUTPUT_SERVICES),
            "vehicles": str(OUTPUT_VEHICLES),
            "report": str(OUTPUT_REPORT),
        },
    }
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    os.makedirs("data/processed", exist_ok=True)
    _ensure_inputs()

    df = _load_snapshots()
    selected, group_pool_counts, group_pick_details, model_targets, year_targets, model_caps, year_caps = select_vins(df)

    selected.to_csv(OUTPUT_SELECTED, index=False)
    _build_feature_snapshot(selected, df).to_csv(OUTPUT_FEATURES, index=False)
    _build_vehicles(selected).to_csv(OUTPUT_VEHICLES, index=False)
    service_rows = _export_services_for_vins(selected[VIN_COL])

    _write_report(
        df,
        selected,
        group_pool_counts,
        group_pick_details,
        model_targets,
        year_targets,
        model_caps,
        year_caps,
        service_rows,
    )

    print(f"VINs selecionados: {len(selected):,}")
    print(f"Servicos exportados: {service_rows:,}")
    print(f"Salvo: {OUTPUT_SELECTED}")
    print(f"Salvo: {OUTPUT_FEATURES}")
    print(f"Salvo: {OUTPUT_SERVICES}")
    print(f"Salvo: {OUTPUT_VEHICLES}")
    print(f"Salvo: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
