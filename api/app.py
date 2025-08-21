from flask import Flask, request, jsonify
import pandas as pd, numpy as np, joblib, json
from datetime import datetime, timedelta
from utils.feature_engineering import make_features

app = Flask(__name__)

# Load artifacts once
model = joblib.load("models/model.joblib")
with open("models/features.json") as f:
    META = json.load(f)

def encode_features(df_row):
    # Expect df_row has all feature columns
    store_cats = pd.Series(META["store_cats"], dtype="category")
    sku_cats = pd.Series(META["sku_cats"], dtype="category")
    X = df_row[["store_id","sku_id"] + [c for c in META["features"] if c not in ["store_id","sku_id"]]].copy()
    X["store_id"] = pd.Categorical(X["store_id"], categories=store_cats).codes
    X["sku_id"] = pd.Categorical(X["sku_id"], categories=sku_cats).codes
    X = X.fillna(method="ffill").fillna(method="bfill")
    return X

@app.route("/health", methods=["GET"])
def health():
    return {"status":"ok"}

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    # Required: history (list of rows), future_days
    # Each history row must include: date, store_id, sku_id, on_promo, price, sales
    history = pd.DataFrame(payload["history"])
    history["date"] = pd.to_datetime(history["date"])
    future_days = int(payload.get("future_days", 14))

    hist = history.sort_values("date").copy()
    forecasts = []

    for _ in range(future_days):
        last_row = hist.iloc[-1]
        next_date = last_row["date"] + pd.Timedelta(days=1)
        fut_row = pd.DataFrame([{
            "date": next_date,
            "store_id": last_row["store_id"],
            "sku_id": last_row["sku_id"],
            "on_promo": 0,
            "price": last_row["price"],
            "sales": np.nan
        }])

        df_feat = pd.concat([hist, fut_row], ignore_index=True).sort_values("date")
        df_feat = make_features(df_feat)
        row = df_feat.iloc[[-1]].copy()
        X = encode_features(row)
        yhat = float(model.predict(X)[0])

        fut_row.loc[:, "sales"] = yhat
        hist = pd.concat([hist, fut_row], ignore_index=True)
        forecasts.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "store_id": int(last_row["store_id"]),
            "sku_id": str(last_row["sku_id"]),
            "forecast": yhat
        })

    return jsonify({"forecasts": forecasts})

if __name__ == "__main__":
    # Run: python api/app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
