import json
import sys

import pandas as pd
import pytest

from src.pipeline import clustering_real as cr
from src.pipeline.clustering_real import CLUSTER_FEATURES, _interpretar_clusters, _make_pipeline


def _snapshot_clustering_df():
    base = [
        [5, 2, 40, 1, 50000, 0.8, 120],
        [1, 30, 10, 1, 12000, 0.1, 300],
        [3, 8, 24, 4, 35000, 0.5, 180],
        [1, 12, 12, 1, 8000, 0.2, 240],
        [6, 1, 50, 1, 60000, 0.9, 100],
        [1, 35, 8, 1, 9000, 0.0, 320],
        [2, 7, 20, 3, 30000, 0.4, 200],
        [1, 15, 14, 1, 10000, 0.3, 260],
    ]
    rows = []
    for i, values in enumerate(base):
        row = {"VIN_Hash": f"vin_{i}"}
        row.update(dict(zip(cr.CLUSTER_FEATURES, values)))
        rows.append(row)
    return pd.DataFrame(rows)


def test_make_pipeline_usa_kmeans_deterministico():
    pipeline = _make_pipeline(4)

    assert list(pipeline.named_steps) == ["imputer", "scaler", "kmeans"]
    assert pipeline.named_steps["kmeans"].n_clusters == 4
    assert pipeline.named_steps["kmeans"].random_state == 42
    assert pipeline.named_steps["kmeans"].n_init == 10


def test_interpretar_clusters_mapeia_segmentos_esperados():
    df = pd.DataFrame([
        [5, 2, 40, 1, 50000, 0.8, 120],
        [1, 30, 10, 1, 12000, 0.1, 300],
        [3, 8, 24, 4, 35000, 0.5, 180],
        [1, 12, 12, 1, 8000, 0.2, 240],
    ], columns=CLUSTER_FEATURES)
    labels = [0, 1, 2, 3]

    mapping = _interpretar_clusters(df, labels)

    assert set(mapping.values()) == {"recorrente", "inativo", "multidealer", "baixo_engajamento"}
    assert mapping[0] == "recorrente"
    assert mapping[1] == "inativo"
    assert mapping[2] == "multidealer"


def test_run_clustering_grava_labels_modelo_checksum_e_mapa(monkeypatch, tmp_path):
    input_path = tmp_path / "snapshots_pos_venda.csv"
    _snapshot_clustering_df().to_csv(input_path, index=False)
    models_dir = tmp_path / "models"

    monkeypatch.setattr(cr, "_plot_elbow_silhouette", lambda x: None)
    monkeypatch.setattr(cr, "_plot_pca_clusters", lambda x, labels, pipeline, mapping: None)

    labels, pipeline = cr.run_clustering(
        input_dir=str(tmp_path),
        output_dir=str(models_dir),
        n_clusters=4,
    )

    assert len(labels) == 8
    assert set(labels["segmento_pos_venda"]) == {"recorrente", "inativo", "multidealer", "baixo_engajamento"}
    assert (tmp_path / "segmentos_pos_venda.csv").exists()
    assert (models_dir / "kmeans_segmentador_pos_venda.joblib").exists()
    assert (models_dir / "kmeans_segmentador_pos_venda.sha256").exists()
    assert json.loads((models_dir / "cluster_segment_map.json").read_text())
    assert pipeline.feature_names_in_snapshot_ == cr.CLUSTER_FEATURES


def test_run_clustering_falha_com_coluna_ausente(tmp_path):
    input_path = tmp_path / "snapshots_pos_venda.csv"
    df = _snapshot_clustering_df().drop(columns=[cr.CLUSTER_FEATURES[0]])
    df.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Colunas de clustering ausentes"):
        cr.run_clustering(input_dir=str(tmp_path), output_dir=str(tmp_path / "models"))


def test_run_clustering_falha_quando_snapshot_nao_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="Rode antes"):
        cr.run_clustering(input_path=str(tmp_path / "ausente.csv"), output_dir=str(tmp_path / "models"))


def test_parse_args_expoe_parametros_de_clustering(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "clustering_real.py",
        "--input-dir", "data/processed",
        "--output-dir", "models-test",
        "--n-clusters", "5",
    ])

    args = cr._parse_args()

    assert args.input_dir == "data/processed"
    assert args.output_dir == "models-test"
    assert args.n_clusters == 5


def test_plot_functions_salvam_arquivos(monkeypatch, tmp_path):
    df = pd.concat([_snapshot_clustering_df(), _snapshot_clustering_df()], ignore_index=True)
    df["VIN_Hash"] = [f"vin_plot_{i}" for i in range(len(df))]
    df["km_max_ate_corte"] = df["km_max_ate_corte"] + df.index
    x = df[cr.CLUSTER_FEATURES]
    pipeline = cr._make_pipeline(4)
    labels = pipeline.fit_predict(x)
    mapping = cr._interpretar_clusters(df, labels)
    elbow_path = tmp_path / "elbow.png"
    pca_path = tmp_path / "pca.png"

    monkeypatch.setattr(cr, "ELBOW_PATH", str(elbow_path))
    monkeypatch.setattr(cr, "PCA_PATH", str(pca_path))

    cr._plot_elbow_silhouette(x)
    cr._plot_pca_clusters(x, labels, pipeline, mapping)

    assert elbow_path.exists()
    assert pca_path.exists()
