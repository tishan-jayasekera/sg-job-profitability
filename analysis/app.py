import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analysis import (
    load_raw_data, clean_and_prep_data, 
    get_portfolio_kpis, get_waterfall_bridge, 
    get_matrix_quadrants, get_pareto_variance, 
    generate_bluf_insights, get_filters
)

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Strategic Profitability Review", layout="wide")

# Custom Styling for "Consultant" Look
st.markdown("""
<style>
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #0f172a; }
    h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 500; color: #64748b; }
    .stMetric { background-color: #f8fafc; padding: 15px; border-radius: 5px; border: 1px solid #e2e8f0; }
    .insight-box { padding: 15px; background-color: #eff6ff; border-left: 5px solid #2563eb; color: #1e3a8a; border-radius: 4px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADER
# =============================================================================
@st.cache_data
def get_data():
    raw, rates, jobs = load_raw_data()
    df = clean_and_prep_data(raw, rates)
    return df

try:
    df = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
st.sidebar.header("Scope Control")
fys, depts = get_filters(df)

sel_fy = st.sidebar.selectbox("Fiscal Year", fys)
sel_dept = st.sidebar.multiselect("Department", depts, default=depts)

# Filtering
df_scoped = df[df['FY'] == sel_fy] if sel_fy else df
if sel_dept:
    df_scoped = df_scoped[df_scoped['Department'].isin(sel_dept)]

# =============================================================================
# EXECUTIVE SUMMARY (BLUF)
# =============================================================================
st.title("Strategic Profitability Review")
st.markdown(f"### Fiscal Period: {sel_fy}")

kpis = get_portfolio_kpis(df_scoped)
pareto = get_pareto_variance(df_scoped)
insights = generate_bluf_insights(kpis, pareto)

# 1. Headline Insights
for txt in insights:
    st.markdown(f"<div class='insight-box'>{txt}</div>", unsafe_allow_html=True)

# 2. Key Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue Sold (Quoted)", f"${kpis['Revenue_Sold']:,.0f}")
c2.metric("Work Value Delivered", f"${kpis['Work_Value_Delivered']:,.0f}", 
          delta=f"${kpis['Net_Variance']:,.0f} Variance", delta_color="normal") # Green if +ve (efficient)
c3.metric("Delivery Efficiency", f"{kpis['Delivery_Efficiency']:.2f}x",
          help="Target > 1.0. (Revenue / Work Value)")
c4.metric("Realized Profit", f"${kpis['Realized_Profit']:,.0f}")

st.markdown("---")

# =============================================================================
# STRATEGIC DEEP DIVES (TABS)
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📉 Profit Bridge", "💠 Portfolio Matrix", "🚨 Critical Few"])

# --- TAB 1: WATERFALL ---
with tab1:
    st.subheader("Revenue-to-Value Bridge")
    st.caption("Visualizing the journey from Quoted Revenue to Net Delivered Value, isolating leakage.")
    
    wf_data = get_waterfall_bridge(df_scoped)
    
    fig_wf = go.Figure(go.Waterfall(
        name="Bridge", orientation="v",
        measure=wf_data['Type'].tolist(),
        x=wf_data['Category'].tolist(),
        y=wf_data['Value'].tolist(),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef4444"}}, # Red for Leakage
        increasing={"marker": {"color": "#10b981"}}, # Green for Gains
        totals={"marker": {"color": "#3b82f6"}}
    ))
    fig_wf.update_layout(template="plotly_white", height=500, yaxis_title="Financial Value ($)")
    st.plotly_chart(fig_wf, use_container_width=True)

# --- TAB 2: MATRIX ---
with tab2:
    st.subheader("Portfolio Performance Matrix")
    st.caption("Segmentation based on Strategic Importance (Revenue) vs. Execution Quality (Efficiency).")
    
    matrix_data = get_matrix_quadrants(df_scoped)
    
    fig_mat = px.scatter(
        matrix_data,
        x="Quoted_Amount",
        y="Efficiency",
        size="Base_Cost",
        color="Segment",
        hover_name="Job_Name",
        hover_data=["Client"],
        color_discrete_map={
            "Star": "#10b981",          # Green
            "Niche": "#3b82f6",         # Blue
            "Cash Cow (At Risk)": "#f59e0b", # Orange
            "Problem Child": "#ef4444", # Red
            "Unbilled / Speculative": "#94a3b8"
        },
        title="Job Segmentation"
    )
    # Reference Lines
    fig_mat.add_hline(y=1.0, line_dash="dot", annotation_text="Efficiency Parity (1.0x)")
    fig_mat.update_layout(template="plotly_white", height=600, xaxis_title="Revenue Volume ($)", yaxis_title="Delivery Efficiency (Multiplier)")
    st.plotly_chart(fig_mat, use_container_width=True)

# --- TAB 3: PARETO ---
with tab3:
    st.subheader("The Critical Few (80/20 Rule)")
    st.caption("These jobs account for the largest value leakage in the portfolio.")
    
    # Chart
    fig_par = px.bar(
        pareto,
        x="Variance",
        y="Job_Name",
        orientation='h',
        color="Variance",
        color_continuous_scale="Reds_r", # Red for negative
        text_auto='$,.0f'
    )
    fig_par.update_layout(template="plotly_white", title="Top Contributors to Leakage")
    st.plotly_chart(fig_par, use_container_width=True)
    
    st.dataframe(pareto[['Client', 'Job_Name', 'Variance']].style.format({'Variance': '${:,.0f}'}), use_container_width=True)