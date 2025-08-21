import argparse, json, joblib, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.feature_engineering import make_features, CALENDAR_FEATURES

# --- Load trained artifacts ---
def load_artifacts():
    model = joblib.load("models/model.joblib")
    with open("models/features.json") as f:
        meta = json.load(f)
    return model, meta

# --- Load historical sales for one SKU/Store ---
def load_history(data_path, store_id, sku_id):
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].copy()
    df = df.sort_values("date")
    return df

# --- Generate future calendar rows ---
def future_calendar(df, days):
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="D")
    last_price = df.iloc[-1]["price"]
    fut = pd.DataFrame({
        "date": future_dates,
        "store_id": df.iloc[-1]["store_id"],
        "sku_id": df.iloc[-1]["sku_id"],
        "on_promo": 0,
        "price": last_price,
        "sales": np.nan
    })
    return fut

# --- Iterative forecasting (using previous predictions as lags) ---
def iterative_forecast(model, meta, history, future_days):
    hist = history.copy()
    forecasts = []

    store_cats = pd.Series(meta["store_cats"], dtype="category")
    sku_cats = pd.Series(meta["sku_cats"], dtype="category")
    feat_cols = meta["features"]

    for _ in range(future_days):
        fut_row = future_calendar(hist, 1)
        df_feat = pd.concat([hist, fut_row], ignore_index=True)
        df_feat = df_feat.sort_values("date")
        df_feat = make_features(df_feat)
        row = df_feat.iloc[[-1]].copy()

        X = row[["store_id","sku_id"] + [c for c in feat_cols if c not in ["store_id","sku_id"]]].copy()
        X["store_id"] = pd.Categorical(X["store_id"], categories=store_cats).codes
        X["sku_id"] = pd.Categorical(X["sku_id"], categories=sku_cats).codes
        X = X.ffill().bfill()
        yhat = model.predict(X)[0]

        fut_row.loc[:, "sales"] = yhat
        hist = pd.concat([hist, fut_row], ignore_index=True)
        forecasts.append(hist.iloc[-1][["date","store_id","sku_id","sales"]])

    return pd.DataFrame(forecasts).rename(columns={"sales":"forecast"})

# --- Plot forecast (Matplotlib) ---
def plot_forecast(history, forecast_df, store_id, sku_id):
    plt.figure(figsize=(12,6))
    plt.plot(history["date"], history["sales"], label="History", color="blue")
    plt.plot(forecast_df["date"], forecast_df["forecast"], marker="o", label="Forecast", color="orange")
    plt.plot(forecast_df["date"], forecast_df["optimal_stock"], linestyle="--", label="Optimal Stock", color="green")
    plt.title(f"Forecast for Store {store_id}, SKU {sku_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = Path("models") / f"forecast_plot_{store_id}_{sku_id}.png"
    plt.savefig(out_path)
    print(f"Saved forecast plot to {out_path}")
    plt.show()

# --- Optional interactive plot (Plotly) ---
def plotly_forecast(history, forecast_df, store_id, sku_id):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["sales"], mode='lines+markers', name='History'))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["optimal_stock"], mode='lines', name='Optimal Stock', line=dict(dash='dash')))
    fig.update_layout(title=f"Forecast for Store {store_id}, SKU {sku_id}",
                      xaxis_title='Date', yaxis_title='Sales', template='plotly_white')
    fig.show()

# --- Main function ---
def main(args):
    model, meta = load_artifacts()
    hist = load_history(args.data_path, args.store_id, args.sku_id)
    fc = iterative_forecast(model, meta, hist, args.future_days)

    # Optimal stock with 10% buffer
    fc["optimal_stock"] = (fc["forecast"] * 1.1).round()

    print(fc.to_string(index=False))

    # Save CSV
    out_path = Path("models") / f"forecast_{args.store_id}_{args.sku_id}.csv"
    fc.to_csv(out_path, index=False)
    print(f"Saved forecast CSV to {out_path}")

    # Plot
    plot_forecast(hist, fc, args.store_id, args.sku_id)
    # Optional interactive
    plotly_forecast(hist, fc, args.store_id, args.sku_id)

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sample_sales.csv")
    parser.add_argument("--future_days", type=int, default=14)
    parser.add_argument("--store_id", type=int, default=101)
    parser.add_argument("--sku_id", type=str, default="A100")
    main(parser.parse_args())
