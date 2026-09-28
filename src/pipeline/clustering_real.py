"""
Segmentacao comportamental pos-venda por K-Means.

Usa somente features de servico calculadas ate a data de corte. Os nomes dos
segmentos sao neutros e devem ser validados com negocio antes de uso comercial.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import CLUSTER_FEATURES_POS_VENDA, N_CLUSTERS, RANDOM_STATE


INPUT_PATH = "data/processed/snapshots_pos_venda.csv"
OUTPUT_LABELS = "data/processed/segmentos_pos_venda.csv"
MODEL_PATH = "models/kmeans_segmentador_pos_venda.joblib"
SEGMENT_MAP_PATH = "models/cluster_segment_map.json"
ELBOW_PATH = "reports/elbow_silhouette_pos_venda.png"
PCA_PATH = "reports/clusters_pca_pos_venda.png"
CLUSTER_FEATURES = CLUSTER_FEATURES_POS_VENDA


def _make_pipeline(n_clusters):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)),
    ])


def _interpretar_clusters(df, labels):
    """
    Mapeia cluster -> segmento neutro com base nos centroides.
      - recorrente: muitas revisoes e baixo tempo desde ultimo servico
      - inativo: maior tempo desde ultimo servico ate a data de corte
      - multidealer: mais dealers diferentes
      - baixo_engajamento: restante, geralmente menor intensidade de uso
    """
    df_c = df[CLUSTER_FEATURES].copy()
    df_c["cluster"] = labels
    centroides = df_c.groupby("cluster").mean()

    print("\n=== Centroides dos clusters (medias) ===")
    print(centroides.round(2))

    mapping = {}
    used = set()

    cluster_inativo = centroides["meses_desde_ultimo_servico_ate_corte"].idxmax()
    mapping[cluster_inativo] = "inativo"
    used.add("inativo")

    restantes = centroides.drop(cluster_inativo)
    cluster_recorrente = restantes["qtde_revisoes_ate_corte"].idxmax()
    mapping[cluster_recorrente] = "recorrente"
    used.add("recorrente")

    restantes = restantes.drop(cluster_recorrente)
    cluster_multidealer = restantes["n_dealers_usados_ate_corte"].idxmax()
    mapping[cluster_multidealer] = "multidealer"
    used.add("multidealer")

    cluster_baixo_engajamento = restantes.drop(cluster_multidealer).index[0]
    mapping[cluster_baixo_engajamento] = "baixo_engajamento"

    print("\n=== Mapeamento cluster -> segmento_pos_venda ===")
    for cluster, segmento in sorted(mapping.items()):
        print(f"  Cluster {cluster}: {segmento}")

    return mapping


def _plot_elbow_silhouette(x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inertias = []
    silhouettes = []
    ks = range(2, 9)

    for k in ks:
        pipeline = _make_pipeline(k)
        labels_k = pipeline.fit_predict(x)
        x_scaled = pipeline[:-1].transform(x)
        inertias.append(pipeline.named_steps["kmeans"].inertia_)
        # Sample para silhouette nao demorar
        sample_size = min(10000, len(x))
        sil = silhouette_score(x_scaled, labels_k, sample_size=sample_size, random_state=RANDOM_STATE)
        silhouettes.append(sil)
        print(f"  k={k}: inertia={inertias[-1]:.0f}, silhouette={sil:.4f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(list(ks), inertias, marker="o", color="#003478")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(list(ks), silhouettes, marker="o", color="#003478")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette por k")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {ELBOW_PATH}")


def _plot_pca_clusters(x, labels, pipeline, mapping):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_scaled = pipeline[:-1].transform(x)

    # Sample para visualizacao
    sample_size = min(50000, len(x_scaled))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(x_scaled), sample_size, replace=False)
    x_sample = x_scaled[sample_idx]
    labels_sample = labels[sample_idx]

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(x_sample)

    cores = {
        "recorrente": "#003478",
        "multidealer": "#FF7F0E",
        "inativo": "#D62728",
        "baixo_engajamento": "#6FA8C9",
    }

    plt.figure(figsize=(10, 7))
    for cluster_id, segmento in mapping.items():
        idx = labels_sample == cluster_id
        plt.scatter(
            components[idx, 0],
            components[idx, 1],
            c=cores.get(segmento, "gray"),
            label=segmento,
            alpha=0.4,
            s=8,
        )
    plt.legend(title="Segmento")
    plt.title("Segmentos Pos-Venda (PCA 2D)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variancia)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variancia)")
    plt.tight_layout()
    plt.savefig(PCA_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {PCA_PATH}")


def run_clustering(input_path=INPUT_PATH, n_clusters=N_CLUSTERS, input_dir=None, output_dir="models"):
    if input_dir:
        input_path = os.path.join(input_dir, "snapshots_pos_venda.csv")
        output_labels = os.path.join(input_dir, "segmentos_pos_venda.csv")
    else:
        output_labels = OUTPUT_LABELS

    os.makedirs(os.path.dirname(output_labels), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = os.path.join(output_dir, "kmeans_segmentador_pos_venda.joblib")
    segment_map_path = os.path.join(output_dir, "cluster_segment_map.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Rode antes: python -m src.pipeline.feature_engineering_real")

    df = pd.read_csv(input_path)
    print(f"Carregado: {df.shape[0]:,} VINs")

    missing = [col for col in CLUSTER_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas de clustering ausentes: {missing}")

    x = df[CLUSTER_FEATURES]

    print("\n=== Selecao de k via Elbow + Silhouette ===")
    _plot_elbow_silhouette(x)

    print(f"\n=== Treinando segmentador K-Means pos-venda com k={n_clusters} ===")
    pipeline = _make_pipeline(n_clusters)
    labels = pipeline.fit_predict(x)

    mapping = _interpretar_clusters(df, labels)
    pipeline.segment_mapping_ = mapping
    pipeline.feature_names_in_snapshot_ = CLUSTER_FEATURES

    df_labels = pd.DataFrame({
        "VIN_Hash": df["VIN_Hash"],
        "cluster_raw": labels,
        "segmento_pos_venda": pd.Series(labels).map(mapping),
    })
    df_labels.to_csv(output_labels, index=False)
    print(f"\nSalvo: {output_labels}")
    joblib.dump(pipeline, model_path, compress=3)
    Path(model_path).with_suffix(".sha256").write_text(
        hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    )
    print(f"Salvo: {model_path}")

    segment_map = {str(k): v for k, v in mapping.items()}
    with open(segment_map_path, "w") as f:
        json.dump(segment_map, f, indent=2)
    print(f"Mapeamento salvo: {segment_map_path}")

    print("\n=== Distribuicao final ===")
    print(df_labels["segmento_pos_venda"].value_counts(normalize=True).round(3))

    _plot_pca_clusters(x, labels, pipeline, mapping)

    return df_labels, pipeline


def _parse_args():
    parser = argparse.ArgumentParser(description="Treina segmentador K-Means pos-venda.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--input-path", default=INPUT_PATH)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_clustering(
        input_path=args.input_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
    )
