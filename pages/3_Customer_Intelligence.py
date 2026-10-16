import plotly.express as px
import streamlit as st
from utils.data import load_data, sidebar_filters
from utils.models import rfm, segment
from utils.ui import page, style

page("Customer Intelligence", "RFM behaviour and KMeans customer segmentation.", "👥")
df = sidebar_filters(load_data())
k = st.sidebar.slider("Number of segments", 3, 6, 4)

tab1, tab2 = st.tabs(["📊 RFM Analysis", "🎯 Segmentation"])
with tab1:
    r = rfm(df)
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(r, x="Recency", nbins=40, title="Recency Distribution (days)")), use_container_width=True)
    b.plotly_chart(style(px.histogram(r[r.Frequency < r.Frequency.quantile(.99)], x="Frequency", nbins=40, title="Frequency Distribution")), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(r, x="Monetary", nbins=40, title="Monetary Distribution")), use_container_width=True)
    b.plotly_chart(style(px.scatter(r, x="Frequency", y="Monetary", color="Recency", title="Frequency vs Monetary")), use_container_width=True)

with tab2:
    seg, sil = segment(df, k)
    c = st.columns(3)
    c[0].metric("Total Customers", f"{len(seg):,}")
    c[1].metric("Segments", k)
    c[2].metric("Silhouette Score", f"{sil:.2f}")
    st.plotly_chart(style(px.scatter(seg, x="PCA1", y="PCA2", color="Segment", title="Customer Segments (PCA Projection)"), 460), use_container_width=True)
    summary = seg.groupby("Segment").agg(Customers=("CustomerID", "count"), Recency=("Recency", "mean"),
                                         Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean")).round(1)
    st.dataframe(summary, use_container_width=True)
