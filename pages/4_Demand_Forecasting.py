import plotly.graph_objects as go
import streamlit as st
from utils.data import load_data
from utils.models import forecast
from utils.ui import page, style

page("Demand Forecasting", "Daily demand forecast with confidence band and holdout MAPE.", "🔮")
df = load_data()
cat = st.sidebar.selectbox("Category", ["All"] + sorted(df.ProductCategory.unique()))
metric = st.sidebar.radio("Forecast", ["Quantity", "Revenue"])
horizon = st.sidebar.slider("Horizon (days)", 7, 60, 30)
d = df if cat == "All" else df[df.ProductCategory == cat]
series = d.groupby(d.InvoiceDate.dt.normalize())[metric].sum()

fc, mape = forecast(series, horizon)
future = fc[fc.actual.isna()]
c = st.columns(3)
c[0].metric("MAPE", f"{mape:.2f}%")
c[1].metric(f"Avg Daily {metric}", f"{fc.actual.mean():,.0f}")
c[2].metric(f"Total Forecast ({horizon}d)", f"{future.yhat.sum():,.0f}")

fig = go.Figure()
fig.add_scatter(x=fc.ds, y=fc.yhat_upper, line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc.ds, y=fc.yhat_lower, fill="tonexty", line=dict(width=0),
                fillcolor="rgba(79,139,249,.2)", name="80% interval")
fig.add_scatter(x=fc.ds, y=fc.actual, name="Actual", line=dict(color="#E6EAF2"))
fig.add_scatter(x=fc.ds, y=fc.yhat, name="Forecast", line=dict(color="#4F8BF9", dash="dash"))
st.plotly_chart(style(fig, 440), use_container_width=True)
st.markdown("### 📋 Forecast Data")
st.dataframe(fc.round(2), use_container_width=True, hide_index=True)
