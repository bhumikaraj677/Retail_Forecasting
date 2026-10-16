
"""Load and prepare the project's real retail sales dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "retail_sales_dataset.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the original CSV and map its columns to dashboard field names."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Place retail_sales_dataset.csv in the project root folder."
        )

    raw = pd.read_csv(DATA_FILE)

    required = [
        "Transaction ID",
        "Date",
        "Customer ID",
        "Product Category",
        "Quantity",
        "Price per Unit",
        "Total Amount",
    ]
    missing = [col for col in required if col not in raw.columns]

    if missing:
        raise ValueError(
            "The retail CSV is missing required columns: "
            + ", ".join(missing)
        )

    df = raw.copy()

    # Map existing columns; do not invent country or stock-level data.
    df["Invoice"] = df["Transaction ID"].astype(str)
    df["CustomerID"] = df["Customer ID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["Date"], errors="coerce")
    df["ProductCategory"] = (
        df["Product Category"].fillna("Unknown").astype(str)
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price per Unit"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Total Amount"], errors="coerce")

    # The source contains categories, not individual product descriptions.
    df["Description"] = "Category: " + df["ProductCategory"]

    df = df.dropna(
        subset=[
            "InvoiceDate",
            "CustomerID",
            "Quantity",
            "Price",
            "Revenue",
        ]
    )

    df = df[
        (df["CustomerID"].str.strip() != "")
        & (df["Quantity"] >= 0)
        & (df["Revenue"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid sales rows remain after data cleaning.")

    return df.sort_values("InvoiceDate").reset_index(drop=True)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Provide date and category filters for dashboard pages."""

    st.sidebar.markdown("### Filters")

    dmin = df["InvoiceDate"].min().date()
    dmax = df["InvoiceDate"].max().date()

    date_range = st.sidebar.date_input(
        "Sales date range",
        value=(dmin, dmax),
        min_value=dmin,
        max_value=dmax,
    )

    categories = st.sidebar.multiselect(
        "Product category",
        options=sorted(df["ProductCategory"].unique()),
    )

    out = df.copy()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        out = out[
            (out["InvoiceDate"].dt.date >= start_date)
            & (out["InvoiceDate"].dt.date <= end_date)
        ]

    if categories:
        out = out[out["ProductCategory"].isin(categories)]

    if out.empty:
        st.warning("No sales data matches the selected filters.")
        st.stop()

    return out

