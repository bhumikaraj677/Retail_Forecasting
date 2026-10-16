
import plotly.express as px
import streamlit as st

from utils.data import load_data, sidebar_filters
from utils.models import forecast
from utils.ui import page, style

page(
    "Executive Dashboard",
    "A summary of sales performance, customer activity and demand forecasts.",
    "📊",
)

full = load_data()
df = sidebar_filters(full)

daily = (
    df.groupby(df["InvoiceDate"].dt.normalize())["Revenue"]
    .sum()
    .sort_index()
)

st.caption(
    "Metrics are calculated from the supplied retail sales dataset. "
    "Actual stock levels, country and verified churn labels are not available."
)

c = st.columns(4)
c[0].metric("Total Sales", f"{df['Revenue'].sum():,.2f}")
c[1].metric("Transactions", f"{df['Invoice'].nunique():,}")
c[2].metric("Customers", f"{df['CustomerID'].nunique():,}")
c[3].metric("Product Categories", f"{df['ProductCategory'].nunique():,}")

st.markdown("### Sales Forecast")

if len(daily) >= 60:
    try:
        fc, mape = forecast(daily, horizon=30)
        future = fc[fc["actual"].isna()]
        a, b, c = st.columns(3)
        a.metric("Forecast Horizon", "30 days")
        b.metric("Forecast Sales", f"{future['yhat'].sum():,.2f}")
        b.help("Model-estimated sales for the next 30 days.")
        c.metric(
            "Holdout MAPE",
            f"{mape:.1f}%" if mape == mape else "Unavailable",
        )
    except (ValueError, IndexError, KeyError) as exc:
        st.warning(f"The forecast could not be calculated: {exc}")
else:
    st.info("More daily history is needed to calculate a reliable forecast.")

left, right = st.columns(2)

monthly = (
    df.groupby(df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())["Revenue"]
    .sum()
    .rename_axis("Month")
    .reset_index()
)

left.plotly_chart(
    style(px.area(monthly, x="Month", y="Revenue", title="Monthly Sales")),
    use_container_width=True,
)

category_sales = (
    df.groupby("ProductCategory", as_index=False)["Revenue"]
    .sum()
    .sort_values("Revenue", ascending=False)
)

right.plotly_chart(
    style(
        px.bar(
            category_sales,
            x="Revenue",
            y="ProductCategory",
            orientation="h",
            title="Sales by Product Category",
        )
    ),
    use_container_width=True,
)

st.markdown("### Data Availability")
st.info(
    "This dataset does not include country, individual product/SKU identifiers, "
    "or actual inventory levels. Inventory and churn metrics should not be "
    "treated as verified business measurements without additional data."
)

