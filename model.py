"""Data loading + Linear Regression logic used by the Streamlit app."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
TARGET = "Profit"


def load_data(source="data/startups.csv"):
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def train_model(df, test_size=0.2, seed=42):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    return {
        "model": model,
        "X_test": X_te, "y_test": y_te, "pred": pred,
        "r2": r2_score(y_te, pred),
        "mae": mean_absolute_error(y_te, pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "intercept": float(model.intercept_),
        "coefs": dict(zip(FEATURES, map(float, model.coef_))),
        "n_train": len(X_tr), "n_test": len(X_te),
    }


def predict_profit(model, rd, admin, mkt):
    row = pd.DataFrame([[rd, admin, mkt]], columns=FEATURES)
    return float(model.predict(row)[0])
