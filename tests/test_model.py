import joblib
import pandas as pd

from src.config import DATA_PATH, FEATURES, MODEL_PATH, TARGET


def test_dataset_schema():
    df = pd.read_csv(DATA_PATH)
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert len(df) >= 1000
    assert df.isnull().sum().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}


def test_model_artifact_predicts():
    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(DATA_PATH)[FEATURES].head(10)
    preds = model.predict(X)
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})


def test_model_accuracy_threshold():
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    acc = accuracy_score(df[TARGET], model.predict(df[FEATURES]))
    assert acc > 0.8
