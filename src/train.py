"""Train, compare and persist the placement prediction model.

Run:  python -m src.train
"""
import json
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TARGET


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.path.insert(0, str(DATA_PATH.parent))
        from generate_dataset import generate  # noqa: WPS433

        generate().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def build_candidates() -> dict:
    return {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
            ]
        ),
    }


def train() -> dict:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv_scores = {}
    candidates = build_candidates()
    for name, pipe in candidates.items():
        cv_scores[name] = float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean())

    best_name = max(cv_scores, key=cv_scores.get)
    best = candidates[best_name].fit(X_train, y_train)

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "best_model": best_name,
        "cv_f1": cv_scores,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
