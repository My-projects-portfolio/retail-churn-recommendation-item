from __future__ import annotations
import pandas as pd
from typing import Literal

def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "CustomerID" not in df.columns:
        raise ValueError("Expected column 'CustomerID' in dataset.")
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df

def _clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    req = {"InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    out = df.copy()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out = out.dropna(subset=["InvoiceDate"])

    # remove cancellations/returns & impossible rows
    out = out[~out["InvoiceNo"].astype(str).str.startswith("C")]
    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]

    if "TotalPrice" not in out.columns:
        out["TotalPrice"] = out["Quantity"] * out["UnitPrice"]

    return out

def _risk_labeler(
    df_cust: pd.DataFrame,
    latency_metric: Literal["median","mean","p75"] = "median",
    monetary_metric: Literal["median","mean","p25"] = "median",
) -> pd.Series:
    if latency_metric == "median":
        lat_thr = df_cust["days_since_last_purchase"].median()
    elif latency_metric == "mean":
        lat_thr = df_cust["days_since_last_purchase"].mean()
    elif latency_metric == "p75":
        lat_thr = df_cust["days_since_last_purchase"].quantile(0.75)
    else:
        raise ValueError("Unsupported latency_metric")

    if monetary_metric == "median":
        mon_thr = df_cust["monetary"].median()
    elif monetary_metric == "mean":
        mon_thr = df_cust["monetary"].mean()
    elif monetary_metric == "p25":
        mon_thr = df_cust["monetary"].quantile(0.25)
    else:
        raise ValueError("Unsupported monetary_metric")

    def label(row):
        late = row["days_since_last_purchase"] > lat_thr
        lowm = row["monetary"] < mon_thr
        if late and lowm:
            return "High Risk"
        elif late:
            return "Medium Risk"
        else:
            return "Low Risk"

    return df_cust.apply(label, axis=1)

def prepare_churn_features(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = _normalize_ids(df)
    df = _clean_transactions(df)

    reference_date = df["InvoiceDate"].max()

    cust = (
        df.groupby("CustomerID")
          .agg(
              last_purchase=("InvoiceDate", "max"),
              frequency=("InvoiceNo", "nunique"),
              monetary=("TotalPrice", "sum"),
          )
          .reset_index()
    )
    cust["days_since_last_purchase"] = (reference_date - cust["last_purchase"]).dt.days.astype(int)
    cust = cust.drop(columns=["last_purchase"])
    cust = cust[["CustomerID", "days_since_last_purchase", "frequency", "monetary"]]
    cust["risk_level"] = _risk_labeler(cust, latency_metric="median", monetary_metric="median")
    return cust
