import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.split import split_data
from src.features.preprocessing import FEATURES, build_preprocessor

RAW = "data/raw/customer_churn_historical.csv"


def _df():
    return pd.read_csv(RAW)


def test_split_is_reproducible_and_disjoint():
    df = _df()
    a_tr, a_te = split_data(df, "Churn", 0.2, 42)
    b_tr, b_te = split_data(df, "Churn", 0.2, 42)
    assert a_tr.index.equals(b_tr.index) and a_te.index.equals(b_te.index)
    assert set(a_tr.customerID).isdisjoint(a_te.customerID)
    assert abs(len(a_te) / len(df) - 0.2) < 0.01
    assert abs((a_tr.Churn == "Yes").mean() - (a_te.Churn == "Yes").mean()) < 0.01


def test_customer_id_is_not_a_feature():
    assert "customerID" not in FEATURES and "Churn" not in FEATURES


def test_pipeline_handles_missing_and_unseen_categories():
    df = _df()
    X, y = df[FEATURES], (df.Churn == "Yes").astype(int)
    pipe = Pipeline([("pre", build_preprocessor()), ("m", LogisticRegression(max_iter=1000))]).fit(X, y)
    row = X.head(2).copy()
    row.loc[row.index[0], "TotalCharges"] = np.nan
    row.loc[row.index[1], "Contract"] = "Categoria nueva"
    proba = pipe.predict_proba(row)[:, 1]
    assert proba.shape == (2,) and ((proba >= 0) & (proba <= 1)).all()
