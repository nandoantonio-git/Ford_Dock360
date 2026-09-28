"""
Classificador de risco de abandono no pos-venda.

Treina churn_futuro_18m usando features comportamentais calculadas somente
ate DATA_CORTE. O target vem exclusivamente de eventos posteriores a DATA_CORTE.
"""

import argparse
import hashlib
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.config import (
    N_ESTIMATORS,
    RANDOM_STATE,
    SNAPSHOT_FEATURES_CATEGORICAL,
    SNAPSHOT_FEATURES_NUMERIC,
    TARGET_CHURN,
    TEST_SIZE,
)
from src.pipeline.leakage_checks import check_leakage, check_temporal_leakage


VINS_PATH = "data/processed/dataset_churn_pos_venda.csv"
MODEL_PATH = "models/churn_pos_venda_rf_calibrated.joblib"
REPORT_PATH = "reports/precision_recall_churn_pos_venda.png"
CONFUSION_PATH = "reports/confusion_matrix_churn_pos_venda.png"
IMPORTANCE_PATH = "reports/feature_importance_churn_pos_venda.csv"

FEATURES_NUMERIC = SNAPSHOT_FEATURES_NUMERIC
FEATURES_CATEGORICAL = SNAPSHOT_FEATURES_CATEGORICAL



def _build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, FEATURES_NUMERIC),
        ("categorical", categorical_pipeline, FEATURES_CATEGORICAL),
    ], remainder="drop")


def _load_data(vins_path=VINS_PATH):
    if not os.path.exists(vins_path):
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {vins_path}\n"
            f"Rode antes: python -m src.pipeline.feature_engineering_real"
        )

    df = pd.read_csv(vins_path)

    feature_cols = FEATURES_NUMERIC + FEATURES_CATEGORICAL
    missing = [col for col in feature_cols + [TARGET_CHURN] if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset de churn pos-venda: {missing}")

    x = df[feature_cols].copy()
    y = df[TARGET_CHURN].astype(int)

    check_temporal_leakage(x, feature_cols)

    print(f"Dataset: {len(x):,} VINs")
    if "janela_futura_observavel" in df.columns and not bool(df["janela_futura_observavel"].all()):
        print(
            "ALERTA: dataset contem target com janela futura incompleta. "
            "Prefira uma DATA_CORTE mais antiga para avaliacao final."
        )
    print(f"Features pos-venda: {feature_cols}")
    print(f"Distribuicao de {TARGET_CHURN}:\n{y.value_counts(normalize=True).round(3)}")
    return x, y


def assert_metrics_not_suspicious(auc):
    if auc > 0.98:
        raise ValueError(
            f"AUC-ROC = {auc:.4f} acima de 0.98 — suspeita de leakage. "
            f"Auditar features do X de treino."
        )


def _save_feature_importance(model):
    calibrated = model.calibrated_classifiers_[0]
    estimator = getattr(calibrated, "estimator", None)
    if estimator is None:
        estimator = getattr(calibrated, "base_estimator")
    preprocessor = estimator.named_steps["preprocessor"]
    rf = estimator.named_steps["model"]
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    importance = pd.DataFrame({
        "feature": names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Salvo: {IMPORTANCE_PATH}")


def train_churn_model(input_dir=None, output_dir="models", vins_path=VINS_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if input_dir:
        vins_path = os.path.join(input_dir, "dataset_churn_pos_venda.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    model_path = os.path.join(output_dir, "churn_pos_venda_rf_calibrated.joblib")

    x, y = _load_data(vins_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _build_preprocessor()
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])

    print("\n=== Treinando RandomForest com calibracao isotonica (cv=5) ===")
    print("Esse passo treina o modelo 5 vezes e pode demorar alguns minutos...")

    model = CalibratedClassifierCV(estimator=pipeline, cv=5, method="isotonic")
    model.fit(x_train, y_train)

    # Avaliacao
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_score)
    auc_pr = average_precision_score(y_test, y_score)
    print("\n=== Avaliacao holdout aleatoria estratificada ===")
    print("Nota: para avaliacao final, prefira multiplos snapshots e split temporal.")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['no_churn', 'churn'])}")

    # Anti-leakage check
    assert_metrics_not_suspicious(auc)

    # Curva precision-recall
    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title(f"Precision-Recall - Churn Pos-Venda (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSalvo: {REPORT_PATH}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no_churn", "churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de Confusao - Churn Pos-Venda")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {CONFUSION_PATH}")

    _save_feature_importance(model)

    # Salvar modelo
    joblib.dump(model, model_path, compress=3)
    Path(model_path).with_suffix(".sha256").write_text(
        hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    )
    print(f"Salvo: {model_path}")

    return model, auc


def _parse_args():
    parser = argparse.ArgumentParser(description="Treina classificador de churn pos-venda.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--input-path", default=VINS_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_churn_model(input_dir=args.input_dir, output_dir=args.output_dir, vins_path=args.input_path)
