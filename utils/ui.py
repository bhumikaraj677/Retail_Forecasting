import streamlit as st

CSS = """
<style>
div[data-testid="stMetric"]{background:#131C2E;border:1px solid #223049;
  padding:14px 16px;border-radius:12px}
div[data-testid="stMetricLabel"]{opacity:.7;text-transform:uppercase;font-size:.72rem}
.hero{background:linear-gradient(135deg,#1e3a8a22,#4F8BF922);border:1px solid #223049;
  padding:18px 22px;border-radius:12px;margin-bottom:14px}
.hero p{opacity:.75;margin:0}
</style>
"""


def page(title: str, subtitle: str, icon: str = "📊"):
    st.set_page_config(page_title=f"RetailPulse | {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h2>{icon} {title}</h2><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig
