"""ML logic: RFM + KMeans segmentation, churn model, forecasting, inventory, drift."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------- RFM + segmentation ----------
@st.cache_data
def rfm(df: pd.DataFrame) -> pd.DataFrame:
    snap = df.InvoiceDate.max() + pd.Timedelta(days=1)
    return df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snap - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()


@st.cache_data
def segment(df: pd.DataFrame, k: int = 4):
    r = rfm(df)
    X = StandardScaler().fit_transform(np.log1p(r[["Recency", "Frequency", "Monetary"]]))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    r["Cluster"] = km.labels_
    order = r.groupby("Cluster").Monetary.mean().sort_values().index
    names = ["At Risk", "Potential", "Loyal", "Champions", "VIP", "New"][:k]
    r["Segment"] = r.Cluster.map(dict(zip(order, names)))
    sil = silhouette_score(X, km.labels_) if len(r) > k else float("nan")
    pcs = PCA(2, random_state=42).fit_transform(X)
    r["PCA1"], r["PCA2"] = pcs[:, 0], pcs[:, 1]
    return r, sil


# ---------- Churn ----------
FEATS = ["Recency", "Frequency", "Monetary", "AvgLine", "Tenure", "Orders90"]


@st.cache_data
def churn_model(df: pd.DataFrame, horizon: int = 90):
    """Features from the past, label = no purchase in the last `horizon` days (no leakage)."""
    cutoff = df.InvoiceDate.max() - pd.Timedelta(days=horizon)
    past, fut = df[df.InvoiceDate <= cutoff], df[df.InvoiceDate > cutoff]
    f = past.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique"), Monetary=("Revenue", "sum"),
        AvgLine=("Revenue", "mean"),
        Tenure=("InvoiceDate", lambda x: (cutoff - x.min()).days)).reset_index()
    recent = past[past.InvoiceDate > cutoff - pd.Timedelta(days=90)].groupby("CustomerID").Invoice.nunique()
    f["Orders90"] = f.CustomerID.map(recent).fillna(0)
    f["Churned"] = (~f.CustomerID.isin(fut.CustomerID)).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(f[FEATS], f.Churned, test_size=0.25,
                                          stratify=f.Churned, random_state=42)
    rf = RandomForestClassifier(300, max_depth=8, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    f["ChurnProbability"] = rf.predict_proba(f[FEATS])[:, 1].round(2)
    f["TopReason"] = np.select(
        [f.Orders90 == 0, f.Recency > 180, f.Frequency <= 2, f.Monetary < f.Monetary.median()],
        ["No Recent Orders", "High Recency", "Low Frequency", "Low Spend"],
        default="Declining Activity")
    imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()
    return f, float(auc), imp


# ---------- Forecasting (trend + weekday seasonality; swap in Prophet if you like) ----------
def forecast(series: pd.Series, horizon: int = 30, holdout: int = 30, window: int = 150):
    s = series.asfreq("D").fillna(0).iloc[-window:]
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]

    def fit(tr):
        t = np.arange(len(tr))
        coef = np.polyfit(t, tr.values, 1)
        resid = tr.values - np.polyval(coef, t)
        dow = pd.Series(resid).groupby(tr.index.dayofweek).mean()
        fitted = np.polyval(coef, t) + dow.reindex(tr.index.dayofweek).values
        sigma = (tr.values - fitted).std()
        return coef, dow, sigma

    def predict(model, tr_len, idx):
        coef, dow, _ = model
        t = np.arange(tr_len, tr_len + len(idx))
        return np.clip(np.polyval(coef, t) + dow.reindex(idx.dayofweek).values, 0, None)

    m = fit(train)
    yhat_test = predict(m, len(train), test.index)
    nz = test.values > 0
    mape = float(np.mean(np.abs(test.values[nz] - yhat_test[nz]) / test.values[nz]) * 100)

    full = fit(s)
    fidx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)
    yhat_f = predict(full, len(s), fidx)
    sig = full[2]

    hist = pd.DataFrame({"ds": s.index, "actual": s.values})
    fc = pd.DataFrame({"ds": fidx, "yhat": yhat_f})
    out = pd.concat([hist, fc], ignore_index=True)
    out["yhat"] = out.yhat.round(2)
    out["yhat_upper"] = out.yhat + 1.28 * sig
    out["yhat_lower"] = (out.yhat - 1.28 * sig).clip(lower=0)
    return out, mape


# ---------- Inventory (EOQ / safety stock / reorder point) ----------
@st.cache_data
def inventory(df: pd.DataFrame, lead_time: int = 7, z: float = 1.65,
              order_cost: float = 50, hold_rate: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    last = df[df.InvoiceDate > df.InvoiceDate.max() - pd.Timedelta(days=180)]
    piv = last.pivot_table(index="StockCode", columns=last.InvoiceDate.dt.normalize(),
                           values="Quantity", aggfunc="sum", fill_value=0)
    inv = pd.DataFrame({"AvgDaily": piv.mean(axis=1), "StdDaily": piv.std(axis=1)}).reset_index()
    meta = df.drop_duplicates("StockCode")[["StockCode", "ProductCategory", "Price"]]
    inv = inv.merge(meta, on="StockCode")
    inv["SafetyStock"] = (z * inv.StdDaily * np.sqrt(lead_time)).round()
    inv["ReorderPoint"] = (inv.AvgDaily * lead_time + inv.SafetyStock).round()
    hold = (inv.Price * hold_rate).clip(lower=0.01)
    inv["EOQ"] = np.sqrt(2 * inv.AvgDaily * 365 * order_cost / hold).round()
    inv["StockOnHand"] = (inv.ReorderPoint * rng.uniform(0, 3.5, len(inv))).round()
    inv["CoverDays"] = (inv.StockOnHand / inv.AvgDaily.replace(0, np.nan)).fillna(0).round(1)
    inv["Status"] = np.select(
        [inv.StockOnHand <= 0, inv.StockOnHand < inv.ReorderPoint,
         inv.StockOnHand > inv.ReorderPoint + inv.EOQ],
        ["Stockout", "Reorder", "Overstock"], default="Healthy")
    inv["HoldingCost"] = (inv.StockOnHand * hold).round(2)
    inv["RiskScore"] = (100 * (1 - (inv.StockOnHand / inv.ReorderPoint.replace(0, np.nan)).clip(0, 1)
                               .fillna(1))).round(1)
    return inv


# ---------- Drift (PSI) ----------
def psi(ref, cur, bins=10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    b = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((b - a) * np.log(b / a)))
