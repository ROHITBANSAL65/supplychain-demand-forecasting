import pandas as pd
import numpy as np

CALENDAR_FEATURES = ["dayofweek","weekofyear","month","year","is_weekend"]

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["dayofweek"] = d["date"].dt.dayofweek
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    return d

def add_lag_features(df: pd.DataFrame, lags=(1,7,14), group_cols=("store_id","sku_id")) -> pd.DataFrame:
    d = df.sort_values(["store_id","sku_id","date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    for lag in lags:
        d[f"lag_{lag}"] = d.groupby(list(group_cols))["sales"].shift(lag)
    return d

def add_roll_features(df: pd.DataFrame, windows=(7,14), group_cols=("store_id","sku_id")) -> pd.DataFrame:
    d = df.sort_values(["store_id","sku_id","date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    for w in windows:
        d[f"rollmean_{w}"] = d.groupby(list(group_cols))["sales"].shift(1).rolling(w).mean()
    return d

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_calendar_features(df)
    d = add_lag_features(d)
    d = add_roll_features(d)
    return d
