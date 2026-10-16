
import plotly.express as px
import streamlit as st

from utils.data import load_data
from utils.models import psi
from utils.ui import page, style

page(
    "Sales & Model Monitoring",
    "Monitor changes in observed sales and review available data quality signals.",
    "📈",
)

df = load_data()

daily = (
    df.groupby(df["InvoiceDate"].dt.normalize())["Revenue"]
    .sum()
    .sort_index()
    .asfreq("D", fill_value=0)
)

if len(daily) < 60:
    st.warning("At least 60 calendar days are needed for the monitoring comparison.")
    st.stop()

reference = daily.iloc[-120:-30].to_numpy()
current = daily.iloc[-30:].to_numpy()

drift = psi(reference, current)

previous_week = daily.iloc[-14:-7].sum()
latest_week = daily.iloc[-7:].sum()

if previous_week:
    wow = (latest_week / previous_week - 1) * 100
else:
    wow = float("nan")

a, b, c = st.columns(3)
a.metric("Sales Drift Score (PSI)", f"{drift:.3f}")
b.metric(
    "Weekly Sales Change",
    f"{wow:+.1f}%" if wow == wow else "Unavailable",
)
c.metric("Days Monitored", f"{len(daily):,}")

st.caption(
    "PSI compares the distribution of daily sales in the reference and recent "
    "periods. It is a monitoring signal, not proof that a model has failed."
)

if drift > 0.25:
    st.error("Large change in the daily sales distribution. Investigate the data.")
elif drift > 0.1:
    st.warning("Moderate change in the daily sales distribution.")
else:
    st.success("No large sales-distribution shift detected by this PSI check.")

if wow == wow and wow < -10:
    st.warning("Sales are more than 10% lower than the previous week.")

recent = daily.tail(90).rename_axis("Date").reset_index()

st.plotly_chart(
    style(px.line(recent, x="Date", y="Revenue", title="Daily Sales — Last 90 Days")),
    use_container_width=True,
)

st.dataframe(
    {
        "Metric": ["Sales drift (PSI)", "Previous 7-day sales", "Latest 7-day sales"],
        "Value": [
            round(drift, 3),
            round(previous_week, 2),
            round(latest_week, 2),
        ],
    },
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Actual stock-on-hand and verified stockout data are not present in the CSV, "
    "so this page does not report inventory status."
)

