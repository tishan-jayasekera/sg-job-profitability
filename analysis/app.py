import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis import (
    load_raw_data, clean_and_parse, apply_filters,
    calculate_portfolio_kpis, compute_waterfall_data,
    compute_quadrant_data, identify_pareto_contributors,
    generate_strategic_insights, compute_reconciliation_totals,
    get_available_fiscal_years, get_available_departments,
    METRIC_DEFINITIONS
)

# =============================================================================
# PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Strategic Portfolio Review",
    page_icon="💼",
    layout="wide"
)

# Professional/Consulting Style CSS
st.markdown("""
<style>
    /* Headers */
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #0F172A; }
    h2 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; color: #334155; padding-top: 1rem; }
    h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 500; color: #475569; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #1E293B; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #64748B; }
    
    /* Insight Boxes */
    .insight-card {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .alert-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.5rem;
        border-radius: 4px;
        color: #991B1B;
        margin-bottom: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: white; border-radius: 4px; color: #64748B; }
    .stTabs [aria-selected="true"] { background-color: #F1F5F9; color: #0F172A; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADER
# =============================================================================
@st.cache_data
def get_data_engine():
    raw_df, df_analysis, df_rates, df_jobs, df_people = load_raw_data()
    clean_df = clean_and_parse(raw_df, df_jobs, df_rates)
    return raw_df, clean_df

try:
    raw_df, clean_df = get_data_engine()
except Exception as e:
    st.error(f"Data Connection Failed: {e}")
    st.stop()

# =============================================================================
# SIDEBAR: SCOPE DEFINITION
# =============================================================================
st.sidebar.header("Scope Definition")

# Fiscal Year Selector
fys = get_available_fiscal_years(clean_df)
selected_fy = st.sidebar.selectbox("Fiscal Period", fys, index=0 if fys else None)

# Department Selector
depts = get_available_departments(clean_df)
selected_depts = st.sidebar.multiselect("Business Unit", depts, default=depts)

# Apply Logic
df_scoped = apply_filters(clean_df, fy=selected_fy, departments=selected_depts)
recon = compute_reconciliation_totals(raw_df, df_scoped)

st.sidebar.markdown("---")
st.sidebar.caption(f"Analyzing {recon['final_records']:,} records")

# =============================================================================
# MAIN CANVAS
# =============================================================================

st.title(f"Portfolio Profitability Review: {selected_fy or 'All Time'}")
st.markdown("### Executive Performance Dashboard")

# --- KPI STRIP ---
kpis = calculate_portfolio_kpis(df_scoped)
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Revenue Sold", f"${kpis['Total_Revenue']:,.0f}", help="Total Quoted Amount")
k2.metric("Work Value Delivered", f"${kpis['Total_Work_Value']:,.0f}", help="Billable Value of Actual Hours")
k3.metric("Net Variance", f"${kpis['Net_Variance']:,.0f}", 
          delta_color="normal" if kpis['Net_Variance'] >= 0 else "inverse")
k4.metric("Delivery Efficiency", f"{kpis['Delivery_Efficiency']:.2f}x", 
          delta="Target > 1.0", delta_color="normal" if kpis['Delivery_Efficiency'] >= 1.0 else "inverse",
          help="Revenue captured per $1 of Work Value")
k5.metric("Realized Profit", f"${kpis['Realized_Profit']:,.0f}", help="Quoted Revenue - Cost Base")

st.markdown("---")

# --- INSIGHTS ENGINE ---
insights = generate_strategic_insights(df_scoped, kpis)
insight_col, _ = st.columns([2,1]) # Use mostly left side for text
with insight_col:
    for i, txt in enumerate(insights):
        # First insight is the status (Critical/Good), styled differently
        style = "alert-card" if "CRITICAL" in txt else "insight-card"
        st.markdown(f"<div class='{style}'>{txt}</div>", unsafe_allow_html=True)

# =============================================================================
# ANALYTICAL TABS
# =============================================================================
tab_bridge, tab_matrix, tab_pareto, tab_data = st.tabs([
    "📉 Value Bridge (Waterfall)", 
    "💠 Portfolio Matrix (BCG)", 
    "🚨 The Critical Few (Pareto)",
    "📋 Data Reconciliation"
])

# --- TAB 1: WATERFALL BRIDGE ---
with tab_bridge:
    st.subheader("Where is the value going?")
    st.markdown("This bridge visualizes the journey from **Sold Revenue** to **Net Delivery Value**, isolating the impact of over-servicing (leakage) and efficiency gains.")
    
    wf_data = compute_waterfall_data(df_scoped)
    
    fig_wf = go.Figure(go.Waterfall(
        name="Value Bridge",
        orientation="v",
        measure=wf_data['Measure'].tolist(),
        x=wf_data['Category'].tolist(),
        y=wf_data['Value'].tolist(),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#EF4444"}}, # Red for leakage
        increasing={"marker": {"color": "#10B981"}}, # Green for gains
        totals={"marker": {"color": "#3B82F6"}}      # Blue for totals
    ))
    
    fig_wf.update_layout(
        title="Revenue to Value Bridge",
        waterfallgap=0.3,
        height=500,
        yaxis_title="Financial Value ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# --- TAB 2: PORTFOLIO MATRIX ---
with tab_matrix:
    st.subheader("Portfolio Health Matrix")
    st.markdown("Segmentation of jobs based on **Strategic Importance (Revenue)** and **Profitability (Margin %)**.")
    
    matrix_data = compute_quadrant_data(df_scoped)
    
    # Create Scatter Plot
    fig_matrix = px.scatter(
        matrix_data,
        x="Quoted_Amount",
        y="Margin_Pct",
        size="Calculated_Base_Cost", # Size bubble by Cost (Effort)
        color="Segment",
        hover_name="Client",
        hover_data=["Job_No", "Job_Name", "Margin_Abs"],
        color_discrete_map={
            "Stars": "#10B981",          # Green
            "Cash Cows": "#3B82F6",      # Blue
            "Problem Children": "#F59E0B", # Orange
            "Standard": "#94A3B8"        # Grey
        },
        labels={"Quoted_Amount": "Revenue Volume ($)", "Margin_Pct": "Margin %", "Calculated_Base_Cost": "Cost Scale"},
        title="Job Performance Segmentation"
    )
    
    # Add Reference Lines
    fig_matrix.add_hline(y=50, line_dash="dot", annotation_text="Target Margin (50%)", annotation_position="bottom right")
    fig_matrix.add_vline(x=matrix_data['Quoted_Amount'].median(), line_dash="dot", annotation_text="Median Revenue")
    
    fig_matrix.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    with st.expander("View Matrix Data"):
        st.dataframe(matrix_data.sort_values('Quoted_Amount', ascending=False))

# --- TAB 3: PARETO ANALYSIS ---
with tab_pareto:
    st.subheader("The 'Critical Few'")
    st.markdown("Identifying the small number of jobs driving the majority of variance (80/20 Rule).")
    
    pareto_df = identify_pareto_contributors(df_scoped, top_n=10)
    
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        # Bar Chart for Variance
        fig_pareto = px.bar(
            pareto_df,
            x="Client",
            y="Margin_Variance",
            color="Margin_Variance",
            color_continuous_scale="RdYlGn",
            text_auto='.2s',
            title="Top Contributors to Variance"
        )
        fig_pareto.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    with col_p2:
        st.markdown("#### Variance Drivers")
        st.dataframe(pareto_df[['Client', 'Job_No', 'Margin_Variance', 'Cumulative_Pct']].style.format({
            'Margin_Variance': '${:,.0f}',
            'Cumulative_Pct': '{:.1%}'
        }), use_container_width=True)

# --- TAB 4: RECONCILIATION ---
with tab_data:
    st.subheader("Data Governance")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.dataframe(pd.DataFrame(list(recon['totals'].items()), columns=['Metric', 'Value']).style.format({'Value': '${:,.2f}'}), use_container_width=True)
    with r_col2:
        st.json(recon)
        
    st.subheader("Metric Dictionary")
    for k, v in METRIC_DEFINITIONS.items():
        st.markdown(f"**{v['name']}**: {v['desc']} (`{v['formula']}`)")