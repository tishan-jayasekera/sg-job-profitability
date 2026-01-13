"""
Job Profitability Analysis Dashboard
Streamlit application for analyzing job profitability from quote to execution.
"""

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

from analysis import (
    load_data, compute_job_summary, compute_task_summary,
    compute_category_summary, compute_client_summary,
    get_top_overruns, get_loss_making_jobs, get_unquoted_tasks,
    calculate_overall_metrics, get_available_fiscal_years
)

# Page configuration
st.set_page_config(
    page_title="Job Profitability Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-positive { color: #00c853; }
    .metric-negative { color: #ff4b4b; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(filepath):
    """Load data and compute all summaries (cached for performance)."""
    df = load_data(filepath)
    return df


def format_currency(val):
    """Format number as currency."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:,.1f}K"
    return f"${val:,.0f}"


def format_pct(val):
    """Format number as percentage."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.1f}%"


def format_hours(val):
    """Format hours."""
    if pd.isna(val):
        return "N/A"
    return f"{val:,.1f}"


def main():
    st.title("📊 Job Profitability Analysis")
    st.markdown("*Analyzing job performance from quote to execution*")
    
    # Sidebar - File Upload
    st.sidebar.header("📁 Data Source")
    
    data_path = Path("data/Quoted_Task_Report_FY26.xlsx")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel Data", type=["xlsx", "xls"],
        help="Upload the Job Task Report Excel file"
    )
    
    if uploaded_file is not None:
        data_source = uploaded_file
    elif data_path.exists():
        data_source = str(data_path)
    else:
        st.warning("⚠️ Please upload the data file or place it in `data/` folder")
        st.info("Expected file: Excel with 'Data' sheet containing job task records")
        st.stop()
    
    # Load data
    try:
        with st.spinner("Loading and processing data..."):
            df = load_and_process_data(data_source)
        st.sidebar.success(f"✅ Loaded {len(df):,} task records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar - Filters
    st.sidebar.header("🎛️ Filters")
    
    # Fiscal Year Filter
    available_fy = get_available_fiscal_years(df)
    fy_options = ["All Years"] + [f"FY{str(y)[-2:]}" for y in available_fy]
    selected_fy = st.sidebar.selectbox("Fiscal Year", fy_options, index=0)
    
    # Filter data by FY
    if selected_fy != "All Years":
        fy_num = int("20" + selected_fy[2:])
        df_filtered = df[df["Fiscal_Year"] == fy_num].copy()
    else:
        df_filtered = df.copy()
    
    if len(df_filtered) == 0:
        st.warning("No data for selected fiscal year.")
        st.stop()
    
    # Compute summaries on filtered data
    job_summary = compute_job_summary(df_filtered)
    task_summary = compute_task_summary(df_filtered)
    category_summary = compute_category_summary(job_summary)
    client_summary = compute_client_summary(job_summary)
    metrics = calculate_overall_metrics(job_summary)
    
    # Category Filter
    categories = ["All Categories"] + sorted(job_summary["Category"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Job Category", categories)
    
    # Client Filter
    clients = ["All Clients"] + sorted(job_summary["Client"].dropna().unique().tolist())
    selected_client = st.sidebar.selectbox("Client", clients)
    
    # Additional filters
    margin_threshold = st.sidebar.slider(
        "Highlight margin below %", 0, 100, 20,
        help="Jobs below this margin will be highlighted"
    )
    show_overruns = st.sidebar.checkbox("Show only overrun jobs", False)
    show_losses = st.sidebar.checkbox("Show only loss-making jobs", False)
    
    # Apply filters to job summary
    filtered_jobs = job_summary.copy()
    if selected_category != "All Categories":
        filtered_jobs = filtered_jobs[filtered_jobs["Category"] == selected_category]
    if selected_client != "All Clients":
        filtered_jobs = filtered_jobs[filtered_jobs["Client"] == selected_client]
    if show_overruns:
        filtered_jobs = filtered_jobs[filtered_jobs["Is_Overrun"]]
    if show_losses:
        filtered_jobs = filtered_jobs[filtered_jobs["Is_Loss"]]
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 **Note:** 'Social Garden Invoice Allocation' entries "
        "are excluded from this analysis."
    )
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    
    # Period indicator
    st.caption(f"📅 Period: **{selected_fy}** | {len(filtered_jobs):,} jobs displayed")
    
    # KPI Row
    st.header("📈 Overall Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.metric("Total Revenue", format_currency(metrics["total_billable_revenue"]))
    with c2:
        st.metric("Total Profit", format_currency(metrics["total_profit"]))
    with c3:
        st.metric("Overall Margin", format_pct(metrics["overall_margin_pct"]))
    with c4:
        st.metric(
            "Jobs Over Budget",
            f"{metrics['jobs_over_budget']} / {metrics['total_jobs']}",
            delta=f"{metrics['overrun_rate']:.1f}%",
            delta_color="inverse"
        )
    with c5:
        st.metric(
            "Loss-Making Jobs",
            f"{metrics['jobs_at_loss']}",
            delta=f"{metrics['loss_rate']:.1f}%",
            delta_color="inverse"
        )
    
    # Second row of metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Quoted Revenue", format_currency(metrics["total_quoted_revenue"]))
    with c2:
        variance = metrics["total_billable_revenue"] - metrics["total_quoted_revenue"]
        st.metric("Revenue vs Quote", format_currency(variance), 
                  delta="Above" if variance > 0 else "Below")
    with c3:
        st.metric("Total Hours (Actual)", format_hours(metrics["total_hours_actual"]))
    with c4:
        hrs_var = metrics["hours_variance"]
        st.metric("Hours Variance", f"{hrs_var:+,.0f} hrs",
                  delta="Over" if hrs_var > 0 else "Under",
                  delta_color="inverse" if hrs_var > 0 else "normal")
    
    st.markdown("---")
    
    # =========================================================================
    # CATEGORY ANALYSIS
    # =========================================================================
    st.header("📂 Performance by Category")
    
    tab1, tab2 = st.tabs(["📊 Charts", "📋 Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Margin % by Category")
            cat_data = category_summary.copy()
            cat_data = cat_data[cat_data["Billable_Amount"] > 0]  # Filter zero revenue
            
            margin_chart = alt.Chart(cat_data).mark_bar().encode(
                x=alt.X("Margin_Pct:Q", title="Margin %", scale=alt.Scale(domain=[-50, 100])),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.condition(
                    alt.datum.Margin_Pct < margin_threshold,
                    alt.value("#ff4b4b"),
                    alt.value("#4CAF50")
                ),
                tooltip=[
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
                    alt.Tooltip("Profit:Q", format="$,.0f"),
                    alt.Tooltip("Job_Count:Q", title="Jobs")
                ]
            ).properties(height=400)
            st.altair_chart(margin_chart, use_container_width=True)
        
        with col2:
            st.subheader("Overrun Rate by Category")
            overrun_chart = alt.Chart(cat_data).mark_bar().encode(
                x=alt.X("Overrun_Rate:Q", title="% Jobs Over Budget"),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.condition(
                    alt.datum.Overrun_Rate > 50,
                    alt.value("#ff4b4b"),
                    alt.value("#2196F3")
                ),
                tooltip=[
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Overrun_Rate:Q", format=".1f", title="Overrun Rate %"),
                    alt.Tooltip("Overrun_Count:Q", title="Overrun Jobs"),
                    alt.Tooltip("Job_Count:Q", title="Total Jobs")
                ]
            ).properties(height=400)
            st.altair_chart(overrun_chart, use_container_width=True)
    
    with tab2:
        cat_display = category_summary[[
            "Category", "Job_Count", "Quoted_Amount", "Billable_Amount",
            "Actual_Cost", "Profit", "Margin_Pct", "Overrun_Count", "Overrun_Rate"
        ]].copy()
        cat_display.columns = [
            "Category", "Jobs", "Quoted $", "Revenue $", "Cost $",
            "Profit $", "Margin %", "Overruns", "Overrun %"
        ]
        st.dataframe(
            cat_display.style.format({
                "Quoted $": "${:,.0f}", "Revenue $": "${:,.0f}",
                "Cost $": "${:,.0f}", "Profit $": "${:,.0f}",
                "Margin %": "{:.1f}%", "Overrun %": "{:.1f}%"
            }),
            use_container_width=True, height=400
        )
    
    st.markdown("---")
    
    # =========================================================================
    # JOB-LEVEL TABLE
    # =========================================================================
    st.header("📋 Job-Level Profitability")
    st.caption(f"Showing {len(filtered_jobs)} jobs")
    
    # Sort options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Margin_Pct", "Profit", "Hours_Variance", "Actual_Cost", "Billable_Amount"],
            format_func=lambda x: {
                "Margin_Pct": "Margin %", "Profit": "Profit $",
                "Hours_Variance": "Hours Variance", "Actual_Cost": "Cost $",
                "Billable_Amount": "Revenue $"
            }.get(x, x)
        )
    with col2:
        sort_asc = st.checkbox("Ascending", value=True)
    with col3:
        top_n = st.number_input("Show top N", min_value=10, max_value=500, value=50)
    
    # Prepare and display
    job_display = filtered_jobs[[
        "Job_No", "Job_Name", "Client", "Category",
        "Quoted_Hours", "Actual_Hours", "Hours_Variance",
        "Quoted_Amount", "Billable_Amount", "Actual_Cost",
        "Profit", "Margin_Pct"
    ]].copy()
    
    job_display.columns = [
        "Job #", "Job Name", "Client", "Category",
        "Quoted Hrs", "Actual Hrs", "Hrs Var",
        "Quoted $", "Revenue $", "Cost $",
        "Profit $", "Margin %"
    ]
    
    display_col = {
        "Margin_Pct": "Margin %", "Profit": "Profit $",
        "Hours_Variance": "Hrs Var", "Actual_Cost": "Cost $",
        "Billable_Amount": "Revenue $"
    }.get(sort_by, sort_by)
    
    job_display = job_display.sort_values(display_col, ascending=sort_asc).head(top_n)
    
    st.dataframe(
        job_display.style.format({
            "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
            "Quoted $": "${:,.0f}", "Revenue $": "${:,.0f}", "Cost $": "${:,.0f}",
            "Profit $": "${:,.0f}", "Margin %": "{:.1f}%"
        }),
        use_container_width=True, height=500
    )
    
    st.markdown("---")
    
    # =========================================================================
    # TASK DRILL-DOWN
    # =========================================================================
    st.header("🔍 Task-Level Drill-Down")
    
    job_options = filtered_jobs.apply(
        lambda r: f"{r['Job_No']} - {r['Job_Name'][:50]}", axis=1
    ).tolist()
    
    if job_options:
        selected_job = st.selectbox(
            "Select a job to view task breakdown",
            ["-- Select a job --"] + job_options
        )
        
        if selected_job != "-- Select a job --":
            job_no = selected_job.split(" - ")[0]
            job_tasks = task_summary[task_summary["Job_No"] == job_no].copy()
            job_info = filtered_jobs[filtered_jobs["Job_No"] == job_no].iloc[0]
            
            # Job header
            st.subheader(f"Job: {job_info['Job_Name']}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Margin", format_pct(job_info["Margin_Pct"]))
            with c2:
                st.metric("Profit", format_currency(job_info["Profit"]))
            with c3:
                st.metric("Hours Variance", f"{job_info['Hours_Variance']:+,.1f} hrs")
            with c4:
                st.metric("Client", job_info["Client"][:30] if pd.notna(job_info["Client"]) else "N/A")
            
            # Task chart
            st.subheader("Hours: Quoted vs Actual by Task")
            
            task_chart_data = job_tasks[["Task_Name", "Quoted_Hours", "Actual_Hours"]].melt(
                id_vars=["Task_Name"], var_name="Type", value_name="Hours"
            )
            task_chart_data["Type"] = task_chart_data["Type"].map({
                "Quoted_Hours": "Quoted", "Actual_Hours": "Actual"
            })
            
            chart_height = max(200, len(job_tasks) * 30)
            hours_chart = alt.Chart(task_chart_data).mark_bar().encode(
                x=alt.X("Hours:Q", title="Hours"),
                y=alt.Y("Task_Name:N", title="", sort="-x"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Quoted", "Actual"],
                    range=["#2196F3", "#4CAF50"]
                )),
                xOffset="Type:N",
                tooltip=["Task_Name", "Type", "Hours"]
            ).properties(height=chart_height)
            st.altair_chart(hours_chart, use_container_width=True)
            
            # Task table
            st.subheader("Task Details")
            task_display = job_tasks[[
                "Task_Name", "Task_Category", "Is_Billable",
                "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                "Quoted_Amount", "Actual_Cost", "Billable_Amount",
                "Profit", "Margin_Pct", "Is_Unquoted"
            ]].copy()
            
            task_display.columns = [
                "Task", "Category", "Billable?",
                "Quoted Hrs", "Actual Hrs", "Hrs Var",
                "Quoted $", "Cost $", "Revenue $",
                "Profit $", "Margin %", "Unquoted?"
            ]
            task_display["Unquoted?"] = task_display["Unquoted?"].map({True: "⚠️", False: ""})
            task_display["Billable?"] = task_display["Billable?"].map({True: "✓", False: ""})
            
            st.dataframe(
                task_display.style.format({
                    "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
                    "Quoted $": "${:,.0f}", "Cost $": "${:,.0f}", "Revenue $": "${:,.0f}",
                    "Profit $": "${:,.0f}", "Margin %": "{:.1f}%"
                }),
                use_container_width=True
            )
            
            # Warnings
            unquoted = job_tasks[job_tasks["Is_Unquoted"]]
            if len(unquoted) > 0:
                st.warning(
                    f"⚠️ **{len(unquoted)} unquoted task(s)** - scope additions totaling "
                    f"{format_currency(unquoted['Actual_Cost'].sum())} in cost"
                )
    
    st.markdown("---")
    
    # =========================================================================
    # TOP OVERRUNS & LOSSES
    # =========================================================================
    st.header("🚨 Problem Jobs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Cost Overruns")
        top_overruns = get_top_overruns(job_summary, n=10, by="Cost_Variance")
        if len(top_overruns) > 0:
            overrun_disp = top_overruns[[
                "Job_No", "Job_Name", "Client", "Cost_Variance", "Hours_Variance", "Margin_Pct"
            ]].copy()
            overrun_disp.columns = ["Job #", "Job Name", "Client", "Cost Var $", "Hrs Var", "Margin %"]
            st.dataframe(
                overrun_disp.style.format({
                    "Cost Var $": "${:+,.0f}", "Hrs Var": "{:+,.1f}", "Margin %": "{:.1f}%"
                }),
                use_container_width=True
            )
    
    with col2:
        st.subheader("Loss-Making Jobs")
        losses = get_loss_making_jobs(job_summary)
        if len(losses) > 0:
            loss_disp = losses[[
                "Job_No", "Job_Name", "Client", "Profit", "Margin_Pct", "Actual_Cost"
            ]].head(10).copy()
            loss_disp.columns = ["Job #", "Job Name", "Client", "Profit $", "Margin %", "Cost $"]
            st.dataframe(
                loss_disp.style.format({
                    "Profit $": "${:,.0f}", "Margin %": "{:.1f}%", "Cost $": "${:,.0f}"
                }),
                use_container_width=True
            )
        else:
            st.success("✅ No loss-making jobs!")
    
    # Unquoted tasks (scope creep)
    st.subheader("📋 Top Unquoted Tasks (Scope Creep)")
    unquoted_tasks = get_unquoted_tasks(task_summary).head(15)
    if len(unquoted_tasks) > 0:
        unquoted_disp = unquoted_tasks[[
            "Job_No", "Job_Name", "Task_Name", "Actual_Hours", "Actual_Cost"
        ]].copy()
        unquoted_disp.columns = ["Job #", "Job Name", "Task", "Hours", "Cost $"]
        st.dataframe(
            unquoted_disp.style.format({"Hours": "{:,.1f}", "Cost $": "${:,.0f}"}),
            use_container_width=True
        )
        total_unquoted = unquoted_tasks["Actual_Cost"].sum()
        st.warning(f"💰 Total cost of unquoted work: **{format_currency(total_unquoted)}**")
    
    # Footer
    st.markdown("---")
    st.caption(
        "Job Profitability Analysis Dashboard | "
        "Data excludes 'Social Garden Invoice Allocation' | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
