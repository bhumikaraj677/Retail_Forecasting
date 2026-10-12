
# ============================================================
# RETAIL SALES FORECASTING USING MACHINE LEARNING
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

# Save charts without requiring a graphical interface.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retail_sales_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# "zero" means dates with no recorded transactions are treated
# as zero-sales days. Use this only if that assumption is
# reasonable for your dataset.
MISSING_DATE_POLICY = "zero"

FORECAST_DAYS = 7
TEST_FRACTION = 0.20
RANDOM_STATE = 42

print("=" * 70)
print("RETAIL SALES FORECASTING PROJECT")
print("=" * 70)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n"
        "Keep retail_sales_dataset.csv in the same folder as main.py."
    )

df = pd.read_csv(DATA_FILE)

required_columns = [
    "Date",
    "Total Amount",
    "Product Category"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Columns found: {df.columns.tolist()}"
    )

print("\nFIRST 5 ROWS")
print(df.head().to_string(index=False))

print("\nORIGINAL DATASET SHAPE:", df.shape)
print("\nCOLUMN NAMES:", df.columns.tolist())

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum().to_string())

print("\nEXACT DUPLICATE ROWS:", int(df.duplicated().sum()))


# ============================================================
# 3. CLEAN DATA
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
).dt.normalize()

df["Total Amount"] = pd.to_numeric(
    df["Total Amount"],
    errors="coerce"
)

# Remove rows that have invalid dates or sales amounts.
invalid_mask = (
    df["Date"].isna()
    | df["Total Amount"].isna()
    | ~np.isfinite(df["Total Amount"])
)

invalid_rows_removed = int(invalid_mask.sum())
df = df.loc[~invalid_mask].copy()

# Remove rows with missing category labels from category analysis
# by labelling them "Unknown". This does not remove their sales.
df["Product Category"] = (
    df["Product Category"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .replace("", "Unknown")
)

df = df.sort_values("Date").reset_index(drop=True)

if df.empty:
    raise ValueError(
        "No valid data remains after cleaning. "
        "Check the Date and Total Amount columns."
    )

if (df["Total Amount"] < 0).any():
    print(
        "\nWARNING: Negative sales amounts exist. "
        "They may represent returns or corrections; "
        "review them before interpreting the forecast."
    )

print("\nINVALID ROWS REMOVED:", invalid_rows_removed)
print("DATASET SHAPE AFTER CLEANING:", df.shape)

print("\nMISSING VALUES AFTER CLEANING")
print(df[required_columns].isnull().sum().to_string())

first_date = df["Date"].min()
last_date = df["Date"].max()

print("\nTRANSACTION DATE RANGE")
print(first_date.date(), "to", last_date.date())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# Monthly totals from the original transaction records.
monthly_sales = (
    df.set_index("Date")["Total Amount"]
    .resample("MS")
    .sum()
    .rename("Total Amount")
    .reset_index()
)

print("\nMONTHLY SALES")
print(monthly_sales.to_string(index=False))

monthly_sales.to_csv(
    OUTPUT_DIR / "monthly_sales.csv",
    index=False
)

plt.figure(figsize=(12, 5))
plt.plot(
    monthly_sales["Date"],
    monthly_sales["Total Amount"],
    marker="o",
    linewidth=2
)
plt.title("Monthly Retail Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "monthly_sales.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# Sales by product category.
category_sales = (
    df.groupby("Product Category")["Total Amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSALES BY PRODUCT CATEGORY")
print(category_sales.to_string())

category_sales.rename("Total Sales").to_csv(
    OUTPUT_DIR / "category_sales.csv"
)

plt.figure(figsize=(9, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "category_sales.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 5. BUILD A CONTINUOUS DAILY SALES SERIES
# ============================================================

# Sum all transactions occurring on each recorded date.
recorded_daily_sales = (
    df.groupby("Date")["Total Amount"]
    .sum()
    .sort_index()
)

# Create every calendar date between the first and last date.
all_dates = pd.date_range(
    start=recorded_daily_sales.index.min(),
    end=recorded_daily_sales.index.max(),
    freq="D"
)

missing_dates = all_dates.difference(
    recorded_daily_sales.index
)

print("\nDAILY DATE COMPLETENESS")
print("Calendar days:", len(all_dates))
print("Dates with recorded transactions:", len(recorded_daily_sales))
print("Dates without transactions:", len(missing_dates))

if len(missing_dates) > 0:
    print(
        "\nIMPORTANT DATA ASSUMPTION:\n"
        "The current configuration treats dates without recorded "
        "transactions as zero-sales days. The CSV alone cannot "
        "prove that sales were genuinely zero on those dates."
    )

if MISSING_DATE_POLICY != "zero":
    raise ValueError(
        "Unsupported MISSING_DATE_POLICY. "
        "This version currently supports 'zero' only. "
        "Do not change this setting unless the daily-series logic "
        "is also updated."
    )

daily_sales = recorded_daily_sales.reindex(
    all_dates,
    fill_value=0.0
)

daily_sales.index.name = "Date"
daily_sales.name = "Total Amount"

daily_df = daily_sales.reset_index()

print("\nFIRST 10 DAYS OF DAILY SALES")
print(daily_df.head(10).to_string(index=False))

print("\nDAILY SALES SUMMARY")
print(f"Daily observations: {len(daily_df)}")
print(f"Average daily sales: {daily_df['Total Amount'].mean():.2f}")
print(f"Minimum daily sales: {daily_df['Total Amount'].min():.2f}")
print(f"Maximum daily sales: {daily_df['Total Amount'].max():.2f}")
print(f"Daily sales standard deviation: {daily_df['Total Amount'].std():.2f}")

daily_df.to_csv(
    OUTPUT_DIR / "daily_sales.csv",
    index=False
)

plt.figure(figsize=(12, 5))
plt.plot(
    daily_df["Date"],
    daily_df["Total Amount"],
    linewidth=1.5
)
plt.title("Daily Retail Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "daily_sales.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

forecast_df = daily_df.copy()

forecast_df["Day"] = forecast_df["Date"].dt.day
forecast_df["Month"] = forecast_df["Date"].dt.month
forecast_df["Year"] = forecast_df["Date"].dt.year
forecast_df["Day_of_Week"] = forecast_df["Date"].dt.dayofweek
forecast_df["Day_of_Year"] = forecast_df["Date"].dt.dayofyear

forecast_df["Is_Weekend"] = (
    forecast_df["Day_of_Week"].isin([5, 6]).astype(int)
)

# Cyclical encoding helps represent the repeating weekly pattern.
forecast_df["Weekday_sin"] = np.sin(
    2 * np.pi * forecast_df["Day_of_Week"] / 7
)
forecast_df["Weekday_cos"] = np.cos(
    2 * np.pi * forecast_df["Day_of_Week"] / 7
)

# Historical sales features. Every feature uses only earlier dates.
forecast_df["Lag_1"] = forecast_df["Total Amount"].shift(1)
forecast_df["Lag_7"] = forecast_df["Total Amount"].shift(7)
forecast_df["Lag_14"] = forecast_df["Total Amount"].shift(14)

forecast_df["Rolling_7"] = (
    forecast_df["Total Amount"]
    .shift(1)
    .rolling(window=7, min_periods=7)
    .mean()
)

forecast_df["Rolling_14"] = (
    forecast_df["Total Amount"]
    .shift(1)
    .rolling(window=14, min_periods=14)
    .mean()
)

forecast_df["Rolling_30"] = (
    forecast_df["Total Amount"]
    .shift(1)
    .rolling(window=30, min_periods=30)
    .mean()
)

features = [
    "Day",
    "Month",
    "Year",
    "Day_of_Week",
    "Day_of_Year",
    "Is_Weekend",
    "Weekday_sin",
    "Weekday_cos",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_7",
    "Rolling_14",
    "Rolling_30"
]

# The first 30 dates cannot have all required historical features.
forecast_df = (
    forecast_df.dropna(subset=features)
    .reset_index(drop=True)
)

if len(forecast_df) < 60:
    raise ValueError(
        f"Only {len(forecast_df)} usable rows remain after "
        "feature engineering. At least 60 are required."
    )

print("\nFORECASTING DATASET SHAPE:", forecast_df.shape)
print("\nFEATURES USED")
print(features)

print("\nFIRST 5 FORECASTING ROWS")
print(forecast_df.head().to_string(index=False))


# ============================================================
# 7. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

# Do not shuffle time-series observations.
X = forecast_df[features]
y = forecast_df["Total Amount"]

split_index = int(len(forecast_df) * (1 - TEST_FRACTION))

if split_index < 30 or len(forecast_df) - split_index < 10:
    raise ValueError(
        "Not enough observations for the chronological train-test split."
    )

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()
y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

test_dates = forecast_df["Date"].iloc[split_index:].copy()

print("\nTRAINING ROWS:", len(X_train))
print("TESTING ROWS:", len(X_test))

print(
    "TRAINING DATE RANGE:",
    forecast_df["Date"].iloc[0].date(),
    "to",
    forecast_df["Date"].iloc[split_index - 1].date()
)

print(
    "TEST DATE RANGE:",
    test_dates.iloc[0].date(),
    "to",
    test_dates.iloc[-1].date()
)


# ============================================================
# 8. TRAIN AND COMPARE MACHINE-LEARNING MODELS
# ============================================================

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_features=0.8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE
    )
}

model_predictions = {}
trained_models = {}

for model_name, candidate_model in models.items():
    print(f"\nTraining {model_name}...")

    candidate_model.fit(X_train, y_train)

    predictions = candidate_model.predict(X_test)

    # Sales cannot be negative, so clamp negative predictions to zero.
    predictions = np.maximum(predictions, 0)

    trained_models[model_name] = candidate_model
    model_predictions[model_name] = predictions

    print(f"{model_name} training completed.")


# ============================================================
# 9. COMPARE MODELS AGAINST BASELINES
# ============================================================

# These baselines use the actual previous-day/week sales values
# available at each test date. Therefore, this is a one-day-ahead
# evaluation, not a seven-day recursive evaluation.
baseline_predictions = {
    "Previous Day Baseline": X_test["Lag_1"].to_numpy(),
    "7-Day Average Baseline": X_test["Rolling_7"].to_numpy(),
    "Same Weekday Baseline": X_test["Lag_7"].to_numpy()
}

all_predictions = {
    **model_predictions,
    **baseline_predictions
}

metrics_rows = []

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

for model_name, predictions in all_predictions.items():
    predictions = np.maximum(
        np.asarray(predictions, dtype=float),
        0
    )

    mae_value = mean_absolute_error(y_test, predictions)

    rmse_value = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2_value = r2_score(y_test, predictions)

    metrics_rows.append({
        "Model": model_name,
        "MAE": round(float(mae_value), 2),
        "RMSE": round(float(rmse_value), 2),
        "R2": round(float(r2_value), 4)
    })

    print(f"\nModel: {model_name}")
    print(f"MAE:  {mae_value:.2f}")
    print(f"RMSE: {rmse_value:.2f}")
    print(f"R2:   {r2_value:.4f}")

metrics_df = pd.DataFrame(metrics_rows).sort_values(
    by="MAE",
    ascending=True
)

metrics_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# Select the best machine-learning model by test MAE.
# The test results are for comparison; a separate future period
# would be needed for a fully independent final performance estimate.
best_ml_name = min(
    model_predictions,
    key=lambda name: mean_absolute_error(
        y_test,
        model_predictions[name]
    )
)

best_model = trained_models[best_ml_name]
y_pred = model_predictions[best_ml_name]

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("BEST MACHINE-LEARNING MODEL ON THE TEST PERIOD")
print("=" * 70)
print("Model:", best_ml_name)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")


# ============================================================
# 10. SAVE TEST PREDICTIONS
# ============================================================

test_results = pd.DataFrame({
    "Date": test_dates.to_numpy(),
    "Actual Sales": y_test.to_numpy(),
    "Predicted Sales": np.round(y_pred, 2)
})

test_results["Prediction Error"] = (
    test_results["Actual Sales"]
    - test_results["Predicted Sales"]
)

test_results["Absolute Error"] = (
    test_results["Prediction Error"].abs()
)

test_results.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

print("\nFIRST 10 TEST PREDICTIONS")
print(test_results.head(10).to_string(index=False))


# ============================================================
# 11. PLOT ACTUAL VS PREDICTED SALES
# ============================================================

plt.figure(figsize=(13, 5))

plt.plot(
    test_dates,
    y_test.to_numpy(),
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    test_dates,
    y_pred,
    label=f"{best_ml_name} Prediction",
    linestyle="--",
    linewidth=2
)

plt.title("Actual vs Predicted Daily Retail Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_performance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    print("\nFEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    plt.title(f"Feature Importance: {best_ml_name}")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()
else:
    importance_df = pd.DataFrame(
        columns=["Feature", "Importance"]
    )


# ============================================================
# 13. REFIT SELECTED MODEL ON ALL USABLE HISTORY
# ============================================================

# The holdout test has already been used for evaluation.
# Refit the selected model on all available labelled rows before
# generating the future forecast.
final_model = models[best_ml_name]

final_model.fit(X, y)

print(f"\nFinal forecasting model refitted: {best_ml_name}")


# ============================================================
# 14. FORECAST THE NEXT 7 DAYS
# ============================================================

history = daily_df["Total Amount"].astype(float).tolist()
last_date = daily_df["Date"].max()

if len(history) < 30:
    raise ValueError(
        "At least 30 daily observations are required for forecasting."
    )

future_predictions = []

for step in range(1, FORECAST_DAYS + 1):
    future_date = last_date + pd.Timedelta(days=step)

    weekday = future_date.dayofweek

    future_row = pd.DataFrame([{
        "Day": future_date.day,
        "Month": future_date.month,
        "Year": future_date.year,
        "Day_of_Week": weekday,
        "Day_of_Year": future_date.dayofyear,
        "Is_Weekend": int(weekday in [5, 6]),
        "Weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "Weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "Lag_1": history[-1],
        "Lag_7": history[-7],
        "Lag_14": history[-14],
        "Rolling_7": float(np.mean(history[-7:])),
        "Rolling_14": float(np.mean(history[-14:])),
        "Rolling_30": float(np.mean(history[-30:]))
    }])

    prediction = float(
        final_model.predict(future_row[features])[0]
    )

    prediction = max(0.0, prediction)

    future_predictions.append({
        "Date": future_date,
        "Predicted Sales": round(prediction, 2)
    })

    # Recursive forecast: each prediction becomes history for
    # later forecast dates.
    history.append(prediction)

future_df = pd.DataFrame(future_predictions)

future_df.to_csv(
    OUTPUT_DIR / "next_7_days_forecast.csv",
    index=False
)

print("\n" + "=" * 70)
print("NEXT 7 DAYS SALES FORECAST")
print("=" * 70)
print(future_df.to_string(index=False))


# ============================================================
# 15. PLOT HISTORICAL SALES AND FORECAST
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    daily_df["Date"].tail(30),
    daily_df["Total Amount"].tail(30),
    label="Historical Sales (Last 30 Days)",
    marker="o",
    markersize=3
)

plt.plot(
    future_df["Date"],
    future_df["Predicted Sales"],
    label="Next 7 Days Forecast",
    marker="o",
    linestyle="--"
)

plt.title("Retail Sales: Historical Data and 7-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sales_forecast.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 16. SAVE PROJECT SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "project_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("RETAIL SALES FORECASTING PROJECT SUMMARY\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Valid transaction rows: {len(df)}\n")
    file.write(f"Invalid rows removed: {invalid_rows_removed}\n")
    file.write(f"First transaction date: {first_date.date()}\n")
    file.write(f"Last transaction date: {last_date.date()}\n")
    file.write(f"Calendar days: {len(daily_df)}\n")
    file.write(f"Dates without transactions: {len(missing_dates)}\n")
    file.write(
        f"Missing-date policy: {MISSING_DATE_POLICY}\n"
    )
    file.write(
        "Caution: zero-sales interpretation for missing dates "
        "has not been independently verified.\n\n"
    )

    file.write(f"Usable forecasting rows: {len(forecast_df)}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n\n")

    file.write(f"Selected ML model: {best_ml_name}\n")
    file.write(f"Selected model MAE: {mae:.2f}\n")
    file.write(f"Selected model RMSE: {rmse:.2f}\n")
    file.write(f"Selected model R2: {r2:.4f}\n\n")

    file.write("MODEL COMPARISON\n")
    file.write(metrics_df.to_string(index=False))
    file.write("\n\nNEXT 7 DAYS FORECAST\n")
    file.write(future_df.to_string(index=False))
    file.write("\n")


# ============================================================
# 17. FINAL PROJECT STATUS
# ============================================================

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSelected ML model:", best_ml_name)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

print("\nReports and charts saved in:")
print(OUTPUT_DIR.resolve())

files_created = [
    "monthly_sales.png",
    "category_sales.png",
    "daily_sales.png",
    "model_performance.png",
    "sales_forecast.png",
    "monthly_sales.csv",
    "category_sales.csv",
    "daily_sales.csv",
    "model_comparison.csv",
    "test_predictions.csv",
    "next_7_days_forecast.csv",
    "project_summary.txt"
]

if not importance_df.empty:
    files_created.append("feature_importance.png")
    files_created.append("feature_importance.csv")

print("\nOUTPUT FILES")
for filename in files_created:
    print("-", filename)

print("\nPROJECT FINISHED")
