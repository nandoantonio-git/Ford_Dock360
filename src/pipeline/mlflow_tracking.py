"""MLflow Tracking para experimentos pos-venda com snapshots por VIN."""

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.pipeline.config import (
    CLUSTER_FEATURES_POS_VENDA,
    RANDOM_STATE,
    SNAPSHOT_FEATURES_CATEGORICAL,
    SNAPSHOT_FEATURES_NUMERIC,
    TARGET_CHURN,
    TEST_SIZE,
)
from src.pipeline.clustering_real import _make_pipeline
from src.pipeline.train_churn_real import _build_preprocessor as build_preprocessor_churn
from src.pipeline.leakage_checks import check_temporal_leakage

VINS_PATH = "data/processed/dataset_churn_pos_venda.csv"
TRACKING_URI = "file:./mlruns"


def _mlflow_modules():
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    return mlflow, MlflowClient


def _git_value(args):
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    return result.stdout.strip()


def _release_tags():
    commit = _git_value(["rev-parse", "HEAD"])
    tag = _git_value(["tag", "--points-at", "HEAD"]).splitlines()
    tag = tag[0] if tag else "unreleased"
    return {
        "git_commit": commit,
        "git_tag": tag,
        "release_version": tag,
        "tracking_type": "pos_venda_real",
    }


def _set_release_tags(extra=None):
    mlflow, _ = _mlflow_modules()
    tags = _release_tags()
    if extra:
        tags.update(extra)
    mlflow.set_tags(tags)


def setup_mlflow(tracking_uri=TRACKING_URI):
    mlflow, MlflowClient = _mlflow_modules()
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _load_data():
    df = pd.read_csv(VINS_PATH)
    return df


def _summarize_churn_metrics(y_true, y_score, threshold=0.5):
    y_pred = (pd.Series(y_score) >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _select_best_churn_candidate(rows):
    if not rows:
        raise ValueError("Nenhum candidato de churn para selecionar")
    return max(rows, key=lambda row: (row["auc_pr"], row["auc_roc"]))


def _churn_model_candidates():
    return [
        {
            "model_name": "LogisticRegression_Balanced",
            "model_family": "linear",
            "estimator": Pipeline([
                ("preprocessor", build_preprocessor_churn()),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "params": {"class_weight": "balanced", "max_iter": 1000},
        },
        {
            "model_name": "RandomForest_Balanced",
            "model_family": "tree_ensemble",
            "estimator": Pipeline([
                ("preprocessor", build_preprocessor_churn()),
                ("model", RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            "params": {"n_estimators": 50, "max_depth": 8, "class_weight": "balanced"},
        },
        {
            "model_name": "RandomForest_Calibrated",
            "model_family": "tree_ensemble_calibrated",
            "estimator": CalibratedClassifierCV(
                estimator=Pipeline([
                    ("preprocessor", build_preprocessor_churn()),
                    ("model", RandomForestClassifier(
                        n_estimators=50,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )),
                ]),
                cv=5,
                method="isotonic",
            ),
            "params": {
                "n_estimators": 50,
                "max_depth": 8,
                "class_weight": "balanced",
                "calibration": "isotonic",
                "cv": 5,
            },
        },
    ]


def register_kmeans_experiment(experiment_name="ford_segmentacao_kmeans"):
    mlflow, _ = _mlflow_modules()
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = _load_data()
    x = df[CLUSTER_FEATURES_POS_VENDA]

    candidate_rows = []
    for k in range(2, 9):
        pipeline = _make_pipeline(k)
        labels = pipeline.fit_predict(x)
        x_scaled = pipeline[:-1].transform(x)
        sample_size = min(10000, len(x))
        candidate_rows.append({
            "k": k,
            "silhouette_score": silhouette_score(
                x_scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE
            ),
            "inertia": pipeline.named_steps["kmeans"].inertia_,
        })
    best = max(candidate_rows, key=lambda row: row["silhouette_score"])

    for row in candidate_rows:
        with mlflow.start_run(run_name=f"kmeans_k_{row['k']}"):
            _set_release_tags({"pipeline_step": "kmeans_candidate"})
            mlflow.log_param("k", row["k"])
            mlflow.log_metric("silhouette_score", row["silhouette_score"])
            mlflow.log_metric("inertia", row["inertia"])

    with mlflow.start_run(run_name="kmeans_selected_k"):
        _set_release_tags({"pipeline_step": "kmeans_selected"})
        mlflow.log_param("selected_k", best["k"])
        mlflow.log_metric("selected_silhouette_score", best["silhouette_score"])
        mlflow.log_metric("selected_inertia", best["inertia"])

    return pd.DataFrame(candidate_rows)


def register_churn_classifier(experiment_name="ford_classificacao_churn", model_name="ford_churn_classifier"):
    mlflow, _ = _mlflow_modules()
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = _load_data()
    feature_cols = SNAPSHOT_FEATURES_NUMERIC + SNAPSHOT_FEATURES_CATEGORICAL
    x = df[feature_cols]
    y = df[TARGET_CHURN]
    check_temporal_leakage(x, feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor_churn()
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])
    model = CalibratedClassifierCV(estimator=pipeline, cv=5, method="isotonic")

    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)

    with mlflow.start_run(run_name="RandomForest_Calibrated") as run:
        _set_release_tags({"pipeline_step": "churn_classifier", "target": TARGET_CHURN})
        mlflow.log_param("model", "RandomForest_Calibrated")
        mlflow.log_metric("auc_roc", auc)
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(model_name, mv.version, "Production", archive_existing_versions=True)

    return auc


def register_churn_model_comparison(
    experiment_name="ford_classificacao_churn_comparison",
    output_path="reports/model_comparison_churn.csv",
):
    mlflow, _ = _mlflow_modules()
    setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = _load_data()
    feature_cols = SNAPSHOT_FEATURES_NUMERIC + SNAPSHOT_FEATURES_CATEGORICAL
    x = df[feature_cols]
    y = df[TARGET_CHURN]
    check_temporal_leakage(x, feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    for candidate in _churn_model_candidates():
        model = candidate["estimator"]
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_test)[:, 1]
        metrics = _summarize_churn_metrics(y_test, y_score)
        row = {
            "model_name": candidate["model_name"],
            "model_family": candidate["model_family"],
            **metrics,
            **{f"param_{k}": v for k, v in candidate["params"].items()},
        }
        rows.append(row)

        with mlflow.start_run(run_name=candidate["model_name"]):
            _set_release_tags({
                "pipeline_step": "churn_model_comparison",
                "target": TARGET_CHURN,
                "selection_metric": "auc_pr",
            })
            mlflow.log_param("model_name", candidate["model_name"])
            mlflow.log_param("model_family", candidate["model_family"])
            for key, value in candidate["params"].items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

    comparison = pd.DataFrame(rows).sort_values(["auc_pr", "auc_roc"], ascending=False)
    best = _select_best_churn_candidate(comparison.to_dict("records"))
    comparison["selected"] = comparison["model_name"] == best["model_name"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    with mlflow.start_run(run_name="churn_model_selected"):
        _set_release_tags({
            "pipeline_step": "churn_model_selected",
            "target": TARGET_CHURN,
            "selection_metric": "auc_pr",
        })
        mlflow.log_param("selected_model", best["model_name"])
        mlflow.log_metric("selected_auc_pr", best["auc_pr"])
        mlflow.log_metric("selected_auc_roc", best["auc_roc"])
        mlflow.log_artifact(str(output_path))

    return comparison


def _parse_args():
    parser = argparse.ArgumentParser(description="Registra experimentos MLflow do pipeline Ford VinGuard.")
    parser.add_argument(
        "--only",
        choices=["all", "kmeans", "churn", "churn-comparison"],
        default="all",
    )
    parser.add_argument("--comparison-output", default="reports/model_comparison_churn.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.only in {"all", "kmeans"}:
        register_kmeans_experiment()
    if args.only in {"all", "churn"}:
        register_churn_classifier()
    if args.only in {"all", "churn-comparison"}:
        register_churn_model_comparison(output_path=args.comparison_output)
