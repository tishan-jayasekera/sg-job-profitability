import streamlit as st
import pandas as pd
import altair as alt
from analysis import etl, analysis

# =============================================================================
# CONFIG & STYLE
# =============================================================================
st.set_page_config(page_title="Job Profitability Engine", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def process_file(uploaded_file):
    return etl.load_and_process_data(uploaded_file)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("💰 Job Profitability Engine")
    st.markdown("### Revenue Allocation & Reconciliation Model")

    # --- SIDEBAR ---
    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload Report (.xlsx)", type=["xlsx"])
    
    if not uploaded_file:
        st.info("Please upload the `Quoted_Task_Report_FY26.xlsx` file.")
        st.stop()
    
    try:
        with st.spinner("Processing & Allocating Revenue..."):
            df_master = process_file(uploaded_file)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # --- FILTERS ---
    st.sidebar.header("Filters")
    all_fys = analysis.get_fiscal_years(df_master)
    selected_fy = st.sidebar.selectbox("Fiscal Year", all_fys, index=len(all_fys)-1 if all_fys else 0)
    
    all_depts = analysis.get_departments(df_master)
    selected_dept = st.sidebar.selectbox("Department", ["All"] + all_depts)
    
    df_filtered = df_master[df_master['Fiscal_Year'] == selected_fy].copy()
    if selected_dept != "All":
        df_filtered = df_filtered[df_filtered['Department'] == selected_dept]

    if df_filtered.empty:
        st.warning("No data found for the selected filters.")
        st.stop()

    # --- KPI HEADER ---
    metrics = analysis.compute_overall_metrics(df_filtered)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Recognized Revenue", f"${metrics['Total_Revenue']:,.0f}", help="Allocated Actuals")
    col2.metric("Total Cost", f"${metrics['Total_Cost']:,.0f}", help="Hours * Cost Rate")
    col3.metric("Realized Margin", f"${metrics['Total_Margin']:,.0f}", f"{metrics['Margin_Pct']:.1f}%")
    col4.metric("Realization Gap", f"${metrics['Realization_Gap']:,.0f}", help="Revenue - (Hours * Billable Rate)")
    col5.metric("Avg Effective Rate", f"${metrics['Avg_Effective_Rate']:.0f}/hr")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Monthly Trends", "Department Performance", "Job Explorer", "Task Drill-Down"])
    
    # --- TAB 1: MONTHLY ---
    with tab1:
        st.subheader("Monthly Performance")
        monthly_df = analysis.compute_monthly_summary(df_filtered)
        
        base = alt.Chart(monthly_df).encode(x=alt.X('Month_Label', sort=None, title='Month'))
        bars = base.mark_bar().encode(
            y=alt.Y('Allocated_Revenue', title='Revenue'),
            tooltip=['Month_Label', 'Allocated_Revenue', 'Margin', 'Realization_Gap']
        )
        line = base.mark_line(color='red').encode(
            y=alt.Y('Margin_Pct', title='Margin %', axis=alt.Axis(format='%'))
        )
        st.altair_chart((bars + line).interactive(), use_container_width=True)

    # --- TAB 2: DEPARTMENT ---
    with tab2:
        st.subheader("Department Scorecard")
        dept_df = analysis.compute_department_summary(df_filtered)
        
        scatter = alt.Chart(dept_df).mark_circle(size=100).encode(
            x=alt.X('Margin_Pct', title='Margin %', axis=alt.Axis(format='%')),
            y=alt.Y('Effective_Rate_Hr', title='Effective Rate / Hr'),
            color='Department',
            size='Allocated_Revenue',
            tooltip=['Department', 'Allocated_Revenue', 'Margin_Pct', 'Effective_Rate_Hr', 'Realization_Gap']
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)
        st.dataframe(dept_df.style.format({"Allocated_Revenue": "${:,.0f}", "Margin_Pct": "{:.1f}%"}))

    # --- TAB 3: JOB EXPLORER ---
    with tab3:
        st.subheader("Job Profitability")
        job_df = analysis.compute_job_summary(df_filtered)
        
        c1, c2 = st.columns(2)
        show_loss = c1.checkbox("Show Loss Making Only")
        show_gap = c2.checkbox("Show Under-Realized Only")
        
        if show_loss: job_df = job_df[job_df['Is_Loss']]
        if show_gap: job_df = job_df[job_df['Is_Under_Realized']]
        
        st.dataframe(job_df.sort_values('Allocated_Revenue', ascending=False).style.format({
            "Allocated_Revenue": "${:,.0f}", "Margin": "${:,.0f}", "Margin_Pct": "{:.1f}%",
            "Realization_Gap": "${:,.0f}"
        }), use_container_width=True)

    # --- TAB 4: TASK DRILL-DOWN ---
    with tab4:
        st.subheader("Task Analysis (Scope Creep)")
        # Filter for Unquoted Tasks
        task_df = analysis.compute_task_summary(df_filtered)
        unquoted = task_df[task_df['Is_Unquoted']].sort_values('Base_Cost', ascending=False)
        
        st.markdown("**Top Unquoted Tasks (Cost Impact)**")
        st.dataframe(unquoted.head(50).style.format({
            "Base_Cost": "${:,.0f}", "Hours": "{:,.1f}"
        }), use_container_width=True)

if __name__ == "__main__":
    main()