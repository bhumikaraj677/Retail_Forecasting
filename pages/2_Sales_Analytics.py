import plotly.express as px
import streamlit as st
from utils.data import load_data, sidebar_filters
from utils.ui import page, style

page("Sales Analytics", "Daily & monthly revenue trends, growth analysis and top sellers.", "📈")
df = sidebar_filters(load_data())

d = df.groupby(df.InvoiceDate.dt.normalize()).Revenue.sum().reset_index()
d["7-day avg"] = d.Revenue.rolling(7, min_periods=1).mean()
st.plotly_chart(style(px.line(d, x="InvoiceDate", y=["Revenue", "7-day avg"], title="Daily Revenue")), use_container_width=True)

l, r = st.columns(2)
top = df.groupby("Description").Revenue.sum().nlargest(10).sort_values().reset_index()
l.plotly_chart(style(px.bar(top, x="Revenue", y="Description", orientation="h", title="Top 10 Products")), use_container_width=True)
cat = df.groupby("ProductCategory").Revenue.sum().reset_index()
r.plotly_chart(style(px.pie(cat, names="ProductCategory", values="Revenue", hole=.5, title="Revenue by Category")), use_container_width=True)

wd = df.groupby(df.InvoiceDate.dt.day_name()).Revenue.sum().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).dropna().reset_index()
st.plotly_chart(style(px.bar(wd, x="InvoiceDate", y="Revenue", title="Revenue by Weekday")), use_container_width=True)
