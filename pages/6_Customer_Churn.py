import streamlit as st
import plotly.express as px

from utils.data import load_data
from utils.ui import page, style

page(
    "Customer Churn Analysis",
    "Review customer purchasing activity and assess whether churn analysis is supported by the available data.",
    "👥",
)

df = load_data()

st.subheader("Customer Data Overview")

customer_summary = (
    df.groupby("CustomerID", as_index=False)
    .agg(
        Transactions=("Invoice", "nunique"),
        Total_Sales=("Revenue", "sum"),
        Last_Purchase=("InvoiceDate", "max"),
    )
)

total_customers = customer_summary["CustomerID"].nunique()
repeat_customers = int((customer_summary["Transactions"] > 1).sum())
single_purchase_customers = int((customer_summary["Transactions"] == 1).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Unique Customers", f"{total_customers:,}")
col2.metric("Repeat Customers", f"{repeat_customers:,}")
col3.metric("Single-Purchase Customers", f"{single_purchase_customers:,}")

st.warning(
    "A reliable churn model cannot currently be evaluated from this dataset. "
    "Each customer has only one recorded transaction, so repeat-purchase "
    "behavior and verified churn labels are unavailable."
)

st.subheader("Customer Purchase Summary")

st.dataframe(
    customer_summary.sort_values("Total_Sales", ascending=False),
    use_container_width=True,
    hide_index=True,
)

purchase_counts = (
    customer_summary["Transactions"]
    .value_counts()
    .sort_index()
    .rename_axis("Transactions")
    .reset_index(name="Customers")
)

st.plotly_chart(
    style(
        px.bar(
            purchase_counts,
            x="Transactions",
            y="Customers",
            title="Customers by Number of Recorded Transactions",
            labels={
                "Transactions": "Recorded Transactions per Customer",
                "Customers": "Number of Customers",
            },
        )
    ),
    use_container_width=True,
)

st.subheader("What Is Needed for Churn Prediction?")

st.markdown(
    """
    To build and evaluate a useful churn prediction model, collect:

    - Customer IDs linked to multiple transactions over time
    - A defined period of customer inactivity that counts as churn
    - Enough historical data to identify returning and inactive customers
    - A time-based training and testing strategy to avoid data leakage
    """
)

st.info(
    "This page reports descriptive customer statistics only. It does not "
    "display a churn probability or model accuracy because those results "
    "cannot be validated with the current data."
)

