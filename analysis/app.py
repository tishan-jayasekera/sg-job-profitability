"""
Job Profitability Analysis Dashboard - Trend Edition
=====================================================
Interactive analysis with month-on-month trends and narrative insights.

Hierarchy: Department → Product → Job → Task
Time-Series: Month-on-Month Trend Analysis for Selected FY
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

from analysis import (
    load_raw_data, clean_and_parse, apply_filters,
    compute_reconciliation_totals, get_available_fiscal_years,
    get_available_departments, get_available_products,
    compute_department_summary, compute_product_summary,
    compute_job_summary, compute_task_summary,
    compute_monthly_summary, compute_monthly_by_department,
    compute_monthly_by_product,
    get_top_overruns, get_loss_making_jobs, get_unquoted_tasks,
    get_margin_erosion_jobs,
    calculate_overall_metrics, analyze_overrun_causes,
    generate_insights,
    METRIC_DEFINITIONS
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Job Profitability Trends",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .insight-box { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def get_data():
    raw_data, df_analysis, df_rates, df_jobs, df_people = load_raw_data()
    df_clean = clean_and_parse(raw_data, df_jobs, df_rates)
    return raw_data, df_clean

try:
    raw_df, clean_df = get_data()
except Exception as e:
    st.error(f"Failed to load data. Please ensure CSV files are in 'data/' folder. Error: {e}")
    st.stop()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

st.sidebar.title("Filters")

# FY Filter
fys = get_available_fiscal_years(clean_df)
selected_fy = st.sidebar.selectbox("Fiscal Year", fys, index=0 if fys else None)

# Department Filter
depts = get_available_departments(clean_df)
selected_depts = st.sidebar.multiselect("Department", depts, default=depts)

# Apply Filters
filtered_df = apply_filters(clean_df, fy=selected_fy, departments=selected_depts)

# Reconciliation
recon = compute_reconciliation_totals(raw_df, filtered_df)

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

st.title(f"💰 Job Profitability: {selected_fy or 'All Time'}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "📈 Trends", 
    "🏢 Hierarchy", 
    "🔍 Detailed View", 
    "⚖️ Reconciliation"
])

# =============================================================================
# TAB 1: EXECUTIVE SUMMARY
# =============================================================================
with tab1:
    metrics = calculate_overall_metrics(filtered_df)
    insights = generate_insights(filtered_df, metrics)
    
    # Top Row KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Quoted", f"${metrics['Total_Quoted']:,.0f}", help=METRIC_DEFINITIONS['Quoted_Amount']['desc'])
    c2.metric("Total Billable Value", f"${metrics['Total_Billable']:,.0f}", help=METRIC_DEFINITIONS['Billable_Value']['desc'])
    c3.metric("Net Variance", f"${metrics['Net_Variance']:,.0f}", delta_color="normal" if metrics['Net_Variance'] >= 0 else "inverse")
    c4.metric("Actual Hours", f"{metrics['Total_Hours']:,.0f}")
    
    # Insights Section
    st.subheader("💡 Key Insights")
    for insight in insights:
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bottom Half: Top Issues
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("🚨 Top 5 Overruns (Variance)")
        overruns = get_top_overruns(compute_job_summary(filtered_df), n=5)
        st.dataframe(overruns[['Job_No', 'Client', 'Margin_Variance', 'ROI']].style.format({
            'Margin_Variance': '${:,.0f}',
            'ROI': '{:.2f}x'
        }), use_container_width=True)
        
    with c_right:
        st.subheader("💸 Loss Making Jobs")
        losses = get_loss_making_jobs(compute_job_summary(filtered_df)).head(5)
        st.dataframe(losses[['Job_No', 'Client', 'Actual_Margin']].style.format({
            'Actual_Margin': '${:,.0f}'
        }), use_container_width=True)

# =============================================================================
# TAB 2: TRENDS
# =============================================================================
with tab2:
    st.subheader("Month-on-Month Performance")
    
    monthly_df = compute_monthly_summary(filtered_df)
    
    if not monthly_df.empty:
        # Altair Chart: Quoted vs Billable
        base = alt.Chart(monthly_df).encode(x='Month')
        
        line_quoted = base.mark_line(point=True, color='#2E86C1').encode(
            y=alt.Y('Quoted_Amount', title='Amount ($)'),
            tooltip=['Month', 'Quoted_Amount']
        )
        
        line_billable = base.mark_line(point=True, color='#E74C3C').encode(
            y='Calculated_Billable_Value',
            tooltip=['Month', 'Calculated_Billable_Value']
        )
        
        chart = (line_quoted + line_billable).properties(
            title="Quoted (Blue) vs Billable (Red) Trend",
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Department Split
        st.subheader("Billable Value by Department")
        dept_monthly = compute_monthly_by_department(filtered_df)
        chart_dept = alt.Chart(dept_monthly).mark_bar().encode(
            x='Month',
            y='Calculated_Billable_Value',
            color='Department',
            tooltip=['Month', 'Department', 'Calculated_Billable_Value']
        ).properties(height=400).interactive()
        st.altair_chart(chart_dept, use_container_width=True)

# =============================================================================
# TAB 3: HIERARCHY
# =============================================================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By Department")
        dept_sum = compute_department_summary(filtered_df)
        st.dataframe(dept_sum.style.format("${:,.0f}", subset=['Quoted_Amount', 'Calculated_Billable_Value', 'Actual_Margin']), use_container_width=True)
        
    with col2:
        st.subheader("By Product")
        prod_sum = compute_product_summary(filtered_df)
        st.dataframe(prod_sum.style.format("${:,.0f}", subset=['Quoted_Amount', 'Calculated_Billable_Value', 'Actual_Margin']), use_container_width=True)

# =============================================================================
# TAB 4: DETAILED VIEW
# =============================================================================
with tab4:
    st.header("Job & Task Drill-down")
    
    job_sum = compute_job_summary(filtered_df)
    search = st.text_input("Search Job Name or Client", "")
    
    if search:
        job_sum = job_sum[job_sum['Job_Name'].str.contains(search, case=False) | job_sum['Client'].str.contains(search, case=False)]
    
    selected_job_row = st.selectbox("Select Job to View Tasks", job_sum['Job_No'] + " - " + job_sum['Job_Name'])
    
    if selected_job_row:
        job_id = selected_job_row.split(" - ")[0]
        st.markdown(f"### Tasks for {job_id}")
        
        task_sum = compute_task_summary(filtered_df, job_id)
        st.dataframe(task_sum.style.format({
            'Quoted_Amount': '${:,.2f}',
            'Calculated_Billable_Value': '${:,.2f}',
            'Margin_Variance': '${:,.2f}',
            'Billable_Rate': '${:,.2f}'
        }), use_container_width=True)

# =============================================================================
# TAB 5: RECONCILIATION
# =============================================================================
with tab5:
    st.header("🔍 Reconciliation")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Records", f"{recon['raw_records']:,}")
    c2.metric("Filtered Records", f"{recon['final_records']:,}")
    c3.metric("Excluded", f"{recon['raw_records'] - recon['final_records']:,}")
    
    st.subheader("Totals Check")
    st.json(recon['totals'])
    
    st.subheader("📐 Metric Definitions")
    for k, d in METRIC_DEFINITIONS.items():
        with st.expander(d['name']):
            st.markdown(f"**Formula:** `{d['formula']}`\n\n{d['desc']}")

st.markdown("---")