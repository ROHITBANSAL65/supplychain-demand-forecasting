import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
from utils.feature_engineering import make_features

# --- Load artifacts once ---
@st.cache_data
def load_artifacts():
    model = joblib.load("models/model.joblib")
    with open("models/features.json") as f:
        meta = json.load(f)
    return model, meta

# --- Load sales history ---
@st.cache_data
def load_history(data_path, store_id, sku_id):
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].copy()
    df = df.sort_values("date")
    return df

# --- Generate future rows ---
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

# --- Iterative forecast ---
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

# --- Streamlit app ---
def main():
    st.title("📈 Supply Chain Demand Forecasting")
    
    model, meta = load_artifacts()
    
    # Sidebar controls
    data_path = st.sidebar.text_input("Sales CSV Path", value="data/sample_sales.csv")
    store_id = st.sidebar.number_input("Store ID", value=101, step=1)
    sku_id = st.sidebar.text_input("SKU ID", value="A100")
    future_days = st.sidebar.number_input("Forecast Horizon (days)", value=14, step=1)
    
    if st.sidebar.button("Run Forecast"):
        hist = load_history(data_path, store_id, sku_id)
        fc = iterative_forecast(model, meta, hist, future_days)
        fc["optimal_stock"] = (fc["forecast"] * 1.1).round()

        st.subheader("Forecast Table")
        st.dataframe(fc)

        # Matplotlib plot
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist["date"], hist["sales"], label="History", color="blue")
        ax.plot(fc["date"], fc["forecast"], marker="o", color="orange", label="Forecast")
        ax.plot(fc["date"], fc["optimal_stock"], linestyle="--", color="green", label="Optimal Stock")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title(f"Forecast for Store {store_id}, SKU {sku_id}")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Plotly interactive
        fig2 = px.line(fc, x="date", y=["forecast","optimal_stock"], markers=True,
                       title=f"Interactive Forecast for Store {store_id}, SKU {sku_id}")
        st.plotly_chart(fig2)

        # Optionally download CSV
        csv_path = Path(f"models/forecast_{store_id}_{sku_id}.csv")
        fc.to_csv(csv_path, index=False)
        st.success(f"Forecast saved to {csv_path}")

if __name__ == "__main__":
    main()
