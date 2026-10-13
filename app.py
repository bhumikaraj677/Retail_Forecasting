from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------------
st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retail_sales_dataset.csv"
TEST_FRACTION = 0.20
RANDOM_STATE = 42

FEATURES = [
    "Day", "Month", "Year", "Day_of_Week", "Day_of_Year", "Is_Weekend",
    "Weekday_sin", "Weekday_cos", "Lag_1", "Lag_7", "Lag_14",
    "Rolling_7", "Rolling_14", "Rolling_30",
]


# ------------------------------------------------------------
# DATA FUNCTIONS
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors="coerce")
    df = df.dropna(subset=["Date", "Total Amount"]).copy()
    df["Product Category"] = (
        df["Product Category"].fillna("Unknown").astype(str).str.strip()
    )
    return df.sort_values("Date").reset_index(drop=True)


def build_daily(df):
    recorded = df.groupby("Date")["Total Amount"].sum().sort_index()
    all_dates = pd.date_range(recorded.index.min(), recorded.index.max(), freq="D")
    daily = recorded.reindex(all_dates, fill_value=0.0)
    daily.index.name = "Date"
    return daily.reset_index()


def build_features(daily_df):
    f = daily_df.copy()
    f["Day"] = f["Date"].dt.day
    f["Month"] = f["Date"].dt.month
    f["Year"] = f["Date"].dt.year
    f["Day_of_Week"] = f["Date"].dt.dayofweek
    f["Day_of_Year"] = f["Date"].dt.dayofyear
    f["Is_Weekend"] = f["Day_of_Week"].isin([5, 6]).astype(int)
    f["Weekday_sin"] = np.sin(2 * np.pi * f["Day_of_Week"] / 7)
    f["Weekday_cos"] = np.cos(2 * np.pi * f["Day_of_Week"] / 7)
    f["Lag_1"] = f["Total Amount"].shift(1)
    f["Lag_7"] = f["Total Amount"].shift(7)
    f["Lag_14"] = f["Total Amount"].shift(14)
    for w in (7, 14, 30):
        f[f"Rolling_{w}"] = (
            f["Total Amount"].shift(1).rolling(window=w, min_periods=w).mean()
        )
    return f.dropna(subset=FEATURES).reset_index(drop=True)


def get_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_features=0.8, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_features=0.8, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2,
            loss="huber", random_state=RANDOM_STATE,
        ),
    }


@st.cache_data(show_spinner="Models train ho rahe hain...")
def run_pipeline(forecast_days):
    df = load_data()
    daily_df = build_daily(df)
    fdf = build_features(daily_df)

    X, y = fdf[FEATURES], fdf["Total Amount"]
    split = int(len(fdf) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_dates = fdf["Date"].iloc[split:]

    models = get_models()
    preds = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        preds[name] = np.maximum(model.predict(X_test), 0)

    all_preds = {
        **preds,
        "Previous Day Baseline": X_test["Lag_1"].to_numpy(),
        "7-Day Average Baseline": X_test["Rolling_7"].to_numpy(),
        "Same Weekday Baseline": X_test["Lag_7"].to_numpy(),
    }
    rows = []
    for name, p in all_preds.items():
        p = np.maximum(np.asarray(p, dtype=float), 0)
        rows.append({
            "Model": name,
            "MAE": round(float(mean_absolute_error(y_test, p)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p))), 2),
            "R2": round(float(r2_score(y_test, p)), 4),
        })
    metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    best_name = min(preds, key=lambda n: mean_absolute_error(y_test, preds[n]))
    best_pred = preds[best_name]

    test_results = pd.DataFrame({
        "Date": test_dates.to_numpy(),
        "Actual Sales": y_test.to_numpy(),
        "Predicted Sales": np.round(best_pred, 2),
    })

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": trained[best_name].feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Refit best model on all data, then forecast recursively.
    final_model = get_models()[best_name]
    final_model.fit(X, y)

    history = daily_df["Total Amount"].astype(float).tolist()
    last_date = daily_df["Date"].max()
    future = []
    for step in range(1, forecast_days + 1):
        d = last_date + pd.Timedelta(days=step)
        wd = d.dayofweek
        row = pd.DataFrame([{
            "Day": d.day, "Month": d.month, "Year": d.year,
            "Day_of_Week": wd, "Day_of_Year": d.dayofyear,
            "Is_Weekend": int(wd in [5, 6]),
            "Weekday_sin": np.sin(2 * np.pi * wd / 7),
            "Weekday_cos": np.cos(2 * np.pi * wd / 7),
            "Lag_1": history[-1], "Lag_7": history[-7], "Lag_14": history[-14],
            "Rolling_7": float(np.mean(history[-7:])),
            "Rolling_14": float(np.mean(history[-14:])),
            "Rolling_30": float(np.mean(history[-30:])),
        }])
        p = max(0.0, float(final_model.predict(row[FEATURES])[0]))
        future.append({"Date": d, "Predicted Sales": round(p, 2)})
        history.append(p)

    return {
        "df": df,
        "daily_df": daily_df,
        "metrics_df": metrics_df,
        "best_name": best_name,
        "test_results": test_results,
        "importance_df": importance_df,
        "future_df": pd.DataFrame(future),
    }


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("📈 Retail Sales Forecasting using Machine Learning")
st.caption("Daily sales data se next days ki sales ka forecast.")

if not DATA_FILE.exists():
    st.error("retail_sales_dataset.csv nahi mili. Isse app.py ke saath same folder mein rakho.")
    st.stop()

forecast_days = st.sidebar.slider("Forecast days", min_value=3, max_value=30, value=7)
st.sidebar.info("Missing dates ko zero-sales days maana gaya hai.")

res = run_pipeline(forecast_days)
df, daily_df = res["df"], res["daily_df"]

best_row = res["metrics_df"][res["metrics_df"]["Model"] == res["best_name"]].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", res["best_name"])
c2.metric("MAE", f"{best_row['MAE']:.2f}")
c3.metric("RMSE", f"{best_row['RMSE']:.2f}")
c4.metric("R²", f"{best_row['R2']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "🤖 Model Comparison", "🔮 Forecast", "ℹ️ About"]
)

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        monthly = df.set_index("Date")["Total Amount"].resample("MS").sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set_title("Monthly Sales Trend")
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        cat = df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        cat.plot(kind="bar", ax=ax)
        ax.set_title("Sales by Product Category")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily_df["Date"], daily_df["Total Amount"], linewidth=1.2)
    ax.set_title("Daily Sales Trend")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Model vs Baseline (lower MAE = better)")
    st.dataframe(res["metrics_df"], use_container_width=True)

    tr = res["test_results"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(tr["Date"], tr["Actual Sales"], label="Actual", linewidth=2)
    ax.plot(tr["Date"], tr["Predicted Sales"], label=f"{res['best_name']} Prediction",
            linestyle="--", linewidth=2)
    ax.set_title("Actual vs Predicted Daily Sales (Test Period)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    imp = res["importance_df"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(imp["Feature"], imp["Importance"])
    ax.set_title(f"Feature Importance: {res['best_name']}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader(f"Next {forecast_days} days forecast")
    fut = res["future_df"]

    fig, ax = plt.subplots(figsize=(12, 4))
    tail = daily_df.tail(30)
    ax.plot(tail["Date"], tail["Total Amount"], marker="o", markersize=3,
            label="Historical (last 30 days)")
    ax.plot(fut["Date"], fut["Predicted Sales"], marker="o", linestyle="--", label="Forecast")
    ax.set_title("Historical Sales and Forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    show = fut.copy()
    show["Date"] = show["Date"].dt.strftime("%d %b %Y")
    st.dataframe(show, use_container_width=True)
    st.download_button(
        "Forecast CSV download karo",
        fut.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv",
    )

with tab4:
    st.markdown(
        """
**Project:** Retail Sales Forecasting using Machine Learning

**Method:**
- Transactions ko daily sales mein convert kiya gaya
- Lag aur rolling-average features banaye gaye
- Random Forest, Extra Trees aur Gradient Boosting ko train karke baselines se compare kiya
- Sabse kam MAE wale model se recursive forecast banaya gaya

**Note:** Forecast sirf ek estimate hai. Dates jinme koi transaction nahi hai, unhe zero-sales maana gaya hai.
        """
    )