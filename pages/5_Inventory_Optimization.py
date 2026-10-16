import streamlit as st

from utils.data import load_data
from utils.ui import page

page(
    "Inventory Optimization",
    "Review inventory data availability and identify what is needed for stock optimization.",
    "📦",
)

df = load_data()

st.subheader("Inventory Data Availability")

st.info(
    "Inventory optimization cannot be calculated reliably from the current "
    "retail sales dataset because it does not contain product SKU codes, "
    "stock-on-hand quantities, supplier lead times, or recorded stockouts."
)

col1, col2, col3 = st.columns(3)

col1.metric("Sales Records", f"{len(df):,}")
col2.metric("Product Categories", f"{df['ProductCategory'].nunique():,}")
col3.metric("Inventory Levels", "Not available")

st.subheader("Available Sales Information")

category_sales = (
    df.groupby("ProductCategory", as_index=False)
    .agg(
        Units_Sold=("Quantity", "sum"),
        Total_Sales=("Revenue", "sum"),
    )
    .sort_values("Units_Sold", ascending=False)
)

st.dataframe(
    category_sales,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Units sold and sales revenue describe historical demand. They do not "
    "represent current inventory or available stock."
)

st.subheader("Data Needed for Inventory Optimization")

st.markdown(
    """
    To calculate reorder points and safety stock, add a dataset containing:

    - **Product SKU:** a unique identifier for each product
    - **Stock on Hand:** the current available quantity
    - **Supplier Lead Time:** the number of days required for replenishment
    - **Historical Product-Level Sales:** sales linked to each SKU
    - **Stockout Records:** dates when products were unavailable
    """
)

