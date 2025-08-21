import argparse, json, joblib, pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.feature_engineering import make_features, CALENDAR_FEATURES

NUMERIC_FEATURES = ["on_promo","price","lag_1","lag_7","lag_14","rollmean_7","rollmean_14"] + CALENDAR_FEATURES

def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_id","sku_id","date"])
    return df

def build_dataset(df):
    df_feat = make_features(df)
    df_feat = df_feat.dropna(subset=["lag_1","lag_7","lag_14","rollmean_7","rollmean_14"])  # avoid leakage
    X = df_feat[["store_id","sku_id"] + NUMERIC_FEATURES].copy()
    # encode ids
    X["store_id"] = X["store_id"].astype("category").cat.codes
    X["sku_id"] = X["sku_id"].astype("category").cat.codes
    y = df_feat["sales"]
    return X, y, df_feat

def main(args):
    df = load_data(args.data_path)
    X, y, df_feat = build_dataset(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.3f}")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    # Persist encodings meta (unique categories) and feature list
    meta = {
        "store_cats": df_feat["store_id"].astype("category").cat.categories.tolist(),
        "sku_cats": df_feat["sku_id"].astype("category").cat.categories.tolist(),
        "features": ["store_id","sku_id"] + NUMERIC_FEATURES
    }
    with open(out_dir / "features.json","w") as f:
        json.dump(meta, f)
    print("Saved model to models/model.joblib and features to models/features.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sample_sales.csv")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon (for API usage)")
    main(parser.parse_args())
