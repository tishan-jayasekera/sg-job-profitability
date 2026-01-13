"""
Job Profitability Analysis Dashboard
=====================================
Streamlit app for analyzing job profitability from quote to execution.

Structure follows hierarchical drill-down:
1. Overall KPIs
2. Level 1: Category Analysis
3. Level 2: Job Analysis (within category)
4. Level 3: Task Drill-down (within job)
5. Synthesis: Why Jobs Run Over
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

from analysis import (
    load_data, get_available_fiscal_years,
    compute_category_summary, compute_job_summary, compute_task_summary,
    get_top_overruns, get_loss_making_jobs, get_margin_erosion_jobs,
    get_unquoted_tasks, get_unbilled_tasks,
    calculate_overall_metrics, analyze_overrun_causes
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Job Profitability Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fmt_currency(val):
    if pd.isna(val) or val == 0:
        return "$0"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:,.1f}K"
    return f"${val:,.0f}"

def fmt_pct(val):
    if pd.isna(val):
        return "N/A"
    return f"{val:.1f}%"

def fmt_hours(val):
    if pd.isna(val):
        return "0"
    return f"{val:,.1f}"

def fmt_variance(val):
    if pd.isna(val):
        return "0"
    return f"{val:+,.1f}"


# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================

@st.cache_data
def load_and_process(filepath):
    """Load raw data from Excel."""
    return load_data(filepath)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------
    st.title("📊 Job Profitability Analysis")
    st.markdown("*Analyzing job performance from quote to execution — Category → Job → Task*")
    
    # -------------------------------------------------------------------------
    # SIDEBAR: DATA & FILTERS
    # -------------------------------------------------------------------------
    st.sidebar.header("📁 Data Source")
    
    data_path = Path("data/Quoted_Task_Report_FY26.xlsx")
    uploaded = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded:
        data_source = uploaded
    elif data_path.exists():
        data_source = str(data_path)
    else:
        st.warning("⚠️ Upload data file or place in `data/` folder")
        st.stop()
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            df_raw = load_and_process(data_source)
        st.sidebar.success(f"✅ {len(df_raw):,} records loaded")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # FISCAL YEAR FILTER
    st.sidebar.header("🎛️ Filters")
    fy_list = get_available_fiscal_years(df_raw)
    fy_options = ["All Years"] + [f"FY{str(y)[-2:]}" for y in fy_list]
    selected_fy = st.sidebar.selectbox("Fiscal Year ([Job] Start Date)", fy_options)
    
    # Apply FY filter
    if selected_fy != "All Years":
        fy_num = int("20" + selected_fy[2:])
        df = df_raw[df_raw["Fiscal_Year"] == fy_num].copy()
    else:
        df = df_raw.copy()
    
    if len(df) == 0:
        st.warning("No data for selected period.")
        st.stop()
    
    # Compute all summaries
    category_summary = compute_category_summary(df)
    job_summary = compute_job_summary(df)
    task_summary = compute_task_summary(df)
    metrics = calculate_overall_metrics(job_summary)
    causes = analyze_overrun_causes(task_summary)
    
    # Additional sidebar filters
    st.sidebar.markdown("---")
    margin_threshold = st.sidebar.slider("Flag margin below %", 0, 100, 20)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 **Excluded:** 'Social Garden Invoice Allocation' entries "
        "(internal allocations, not project work)"
    )
    
    # =========================================================================
    # SECTION 0: OVERALL KPIs
    # =========================================================================
    st.markdown(f"### 📅 Period: **{selected_fy}** — {metrics['total_jobs']:,} Jobs")
    
    st.subheader("📈 Overall Performance Metrics")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", fmt_currency(metrics["total_billable_revenue"]))
    c2.metric("Total Cost", fmt_currency(metrics["total_cost"]))
    c3.metric("Total Profit", fmt_currency(metrics["total_profit"]))
    c4.metric("Overall Margin", fmt_pct(metrics["overall_margin_pct"]))
    c5.metric("Profit Lost (Losses)", fmt_currency(metrics["profit_lost_to_overruns"]))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs Over Budget", f"{metrics['jobs_over_budget']} / {metrics['total_jobs']}", 
              delta=f"{metrics['overrun_rate']:.1f}%", delta_color="inverse")
    c2.metric("Jobs at Loss", str(metrics['jobs_at_loss']), 
              delta=f"{metrics['loss_rate']:.1f}%", delta_color="inverse")
    c3.metric("Avg Margin %", fmt_pct(metrics["avg_margin_pct"]),
              delta=f"{-metrics['margin_gap']:.1f}% vs quoted", delta_color="inverse" if metrics['margin_gap'] > 0 else "normal")
    c4.metric("Hours Variance", f"{metrics['hours_variance']:+,.0f} hrs",
              delta=f"{metrics['hours_variance_pct']:+.1f}%", delta_color="inverse" if metrics['hours_variance'] > 0 else "normal")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 1: CATEGORY-LEVEL ANALYSIS
    # =========================================================================
    st.header("📂 Level 1: Analysis by Job Category")
    st.markdown("*Which areas of the business are most prone to profit leaks?*")
    
    # Category selector for drill-down
    cat_list = ["All Categories"] + sorted(category_summary["Category"].dropna().unique().tolist())
    selected_category = st.selectbox("Select Category to Analyze", cat_list, key="cat_select")
    
    # Show category comparison charts when "All Categories"
    if selected_category == "All Categories":
        cat_data = category_summary[category_summary["Billable_Amount"] > 0].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Margin % by Category")
            chart1 = alt.Chart(cat_data).mark_bar().encode(
                x=alt.X("Margin_Pct:Q", title="Margin %", scale=alt.Scale(domain=[-50, 100])),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.condition(
                    alt.datum.Margin_Pct < margin_threshold,
                    alt.value("#e53935"), alt.value("#43a047")
                ),
                tooltip=["Category", alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
                         alt.Tooltip("Profit:Q", format="$,.0f"), "Job_Count"]
            ).properties(height=max(300, len(cat_data) * 25))
            st.altair_chart(chart1, use_container_width=True)
        
        with col2:
            st.subheader("Hours Variance % by Category")
            chart2 = alt.Chart(cat_data).mark_bar().encode(
                x=alt.X("Hours_Variance_Pct:Q", title="Hours Variance %"),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.condition(
                    alt.datum.Hours_Variance_Pct > 20,
                    alt.value("#e53935"), alt.value("#1e88e5")
                ),
                tooltip=["Category", alt.Tooltip("Hours_Variance_Pct:Q", format=".1f"),
                         alt.Tooltip("Hours_Variance:Q", format=",.0f", title="Hrs Over/Under")]
            ).properties(height=max(300, len(cat_data) * 25))
            st.altair_chart(chart2, use_container_width=True)
        
        # Category table
        with st.expander("📋 Category Summary Table", expanded=False):
            cat_disp = cat_data[[
                "Category", "Job_Count", "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
                "Quoted_Amount", "Billable_Amount", "Actual_Cost", "Profit", "Margin_Pct"
            ]].copy()
            cat_disp.columns = ["Category", "Jobs", "Quoted Hrs", "Actual Hrs", "Hrs Var %",
                                "Quoted $", "Revenue $", "Cost $", "Profit $", "Margin %"]
            st.dataframe(cat_disp.style.format({
                "Quoted Hrs": "{:,.0f}", "Actual Hrs": "{:,.0f}", "Hrs Var %": "{:+.1f}%",
                "Quoted $": "${:,.0f}", "Revenue $": "${:,.0f}", "Cost $": "${:,.0f}",
                "Profit $": "${:,.0f}", "Margin %": "{:.1f}%"
            }), use_container_width=True, height=400)
    
    else:
        # Show selected category metrics
        cat_row = category_summary[category_summary["Category"] == selected_category].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jobs in Category", int(cat_row["Job_Count"]))
        c2.metric("Category Margin", fmt_pct(cat_row["Margin_Pct"]))
        c3.metric("Hours Variance", f"{cat_row['Hours_Variance']:+,.0f} hrs")
        c4.metric("Revenue Variance", fmt_currency(cat_row["Revenue_Variance"]))
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 2: JOB-LEVEL ANALYSIS
    # =========================================================================
    st.header("📋 Level 2: Job-Level Profitability")
    st.markdown("*Quote vs Actual Summary per job — identify problematic projects*")
    
    # Filter jobs by category
    if selected_category != "All Categories":
        jobs_filtered = job_summary[job_summary["Category"] == selected_category].copy()
    else:
        jobs_filtered = job_summary.copy()
    
    # Quick filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_overruns = st.checkbox("Show only overrun jobs", False)
    with col2:
        show_losses = st.checkbox("Show only loss-making jobs", False)
    with col3:
        show_erosion = st.checkbox("Show margin erosion >10%", False)
    
    if show_overruns:
        jobs_filtered = jobs_filtered[jobs_filtered["Is_Overrun"]]
    if show_losses:
        jobs_filtered = jobs_filtered[jobs_filtered["Is_Loss"]]
    if show_erosion:
        jobs_filtered = jobs_filtered[jobs_filtered["Margin_Erosion"] > 10]
    
    st.caption(f"Showing {len(jobs_filtered)} jobs")
    
    # Sort controls
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox("Sort by", [
            "Margin_Pct", "Profit", "Hours_Variance", "Margin_Erosion", "Actual_Cost"
        ], format_func=lambda x: {
            "Margin_Pct": "Margin % (lowest first)",
            "Profit": "Profit $ (lowest first)",
            "Hours_Variance": "Hours Variance (highest first)",
            "Margin_Erosion": "Margin Erosion (highest first)",
            "Actual_Cost": "Cost $ (highest first)"
        }.get(x, x))
    with col2:
        top_n = st.number_input("Show top N", 10, 500, 50)
    
    # Sort logic
    ascending = sort_by in ["Margin_Pct", "Profit"]
    jobs_display = jobs_filtered.sort_values(sort_by, ascending=ascending).head(top_n)
    
    # Job table
    job_cols = ["Job_No", "Job_Name", "Client", "Category",
                "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
                "Quoted_Amount", "Billable_Amount", "Actual_Cost",
                "Profit", "Margin_Pct", "Margin_Erosion"]
    job_disp = jobs_display[job_cols].copy()
    job_disp.columns = ["Job #", "Job Name", "Client", "Category",
                        "Quoted Hrs", "Actual Hrs", "Hrs Var %",
                        "Quoted $", "Revenue $", "Cost $",
                        "Profit $", "Margin %", "Margin Erosion %"]
    
    st.dataframe(job_disp.style.format({
        "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var %": "{:+.1f}%",
        "Quoted $": "${:,.0f}", "Revenue $": "${:,.0f}", "Cost $": "${:,.0f}",
        "Profit $": "${:,.0f}", "Margin %": "{:.1f}%", "Margin Erosion %": "{:+.1f}%"
    }), use_container_width=True, height=400)
    
    # Margin distribution scatter
    with st.expander("📊 Margin Distribution (Quoted vs Actual)", expanded=False):
        scatter_data = jobs_filtered[jobs_filtered["Quoted_Amount"] > 0].copy()
        if len(scatter_data) > 0:
            scatter = alt.Chart(scatter_data).mark_circle(size=60).encode(
                x=alt.X("Quoted_Margin_Pct:Q", title="Quoted Margin %", scale=alt.Scale(domain=[-50, 100])),
                y=alt.Y("Margin_Pct:Q", title="Actual Margin %", scale=alt.Scale(domain=[-50, 100])),
                color=alt.condition(
                    alt.datum.Margin_Pct < alt.datum.Quoted_Margin_Pct,
                    alt.value("#e53935"), alt.value("#43a047")
                ),
                tooltip=["Job_No", "Job_Name", 
                         alt.Tooltip("Quoted_Margin_Pct:Q", format=".1f", title="Quoted Margin %"),
                         alt.Tooltip("Margin_Pct:Q", format=".1f", title="Actual Margin %")]
            )
            diagonal = alt.Chart(pd.DataFrame({"x": [-50, 100], "y": [-50, 100]})).mark_line(
                strokeDash=[5, 5], color="gray"
            ).encode(x="x:Q", y="y:Q")
            st.altair_chart(scatter + diagonal, use_container_width=True)
            st.caption("Points below the diagonal indicate margin erosion (actual < quoted)")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 3: TASK-LEVEL DRILL-DOWN
    # =========================================================================
    st.header("🔍 Level 3: Task-Level Drill-Down")
    st.markdown("*Which specific task or phase caused the project overrun?*")
    
    # Job selector
    job_options = jobs_filtered.apply(lambda r: f"{r['Job_No']} — {r['Job_Name'][:40]}", axis=1).tolist()
    
    if job_options:
        selected_job = st.selectbox("Select Job for Task Breakdown", 
                                    ["-- Select a job --"] + job_options, key="job_select")
        
        if selected_job != "-- Select a job --":
            job_no = selected_job.split(" — ")[0]
            job_info = jobs_filtered[jobs_filtered["Job_No"] == job_no].iloc[0]
            job_tasks = task_summary[task_summary["Job_No"] == job_no].copy()
            
            # Job header
            st.subheader(f"📁 {job_info['Job_Name']}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Margin", fmt_pct(job_info["Margin_Pct"]))
            c2.metric("Profit", fmt_currency(job_info["Profit"]))
            c3.metric("Hours Var", f"{job_info['Hours_Variance']:+,.0f} hrs")
            c4.metric("Margin Erosion", f"{job_info['Margin_Erosion']:+.1f}%")
            c5.metric("Client", str(job_info["Client"])[:20] if pd.notna(job_info["Client"]) else "N/A")
            
            # Task hours chart (quoted vs actual)
            st.subheader("Hours by Task: Quoted vs Actual")
            task_chart = job_tasks[["Task_Name", "Quoted_Hours", "Actual_Hours"]].melt(
                id_vars=["Task_Name"], var_name="Type", value_name="Hours"
            )
            task_chart["Type"] = task_chart["Type"].map({"Quoted_Hours": "Quoted", "Actual_Hours": "Actual"})
            
            bar_chart = alt.Chart(task_chart).mark_bar().encode(
                x=alt.X("Hours:Q", title="Hours"),
                y=alt.Y("Task_Name:N", title="", sort="-x"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Quoted", "Actual"], range=["#1e88e5", "#43a047"])),
                xOffset="Type:N",
                tooltip=["Task_Name", "Type", alt.Tooltip("Hours:Q", format=",.1f")]
            ).properties(height=max(200, len(job_tasks) * 28))
            st.altair_chart(bar_chart, use_container_width=True)
            
            # Task table
            st.subheader("Task Details")
            task_cols = ["Task_Name", "Task_Category", "Is_Billable",
                         "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                         "Quoted_Amount", "Actual_Cost", "Billable_Amount",
                         "Profit", "Margin_Pct", "Unbilled_Hours", "Is_Unquoted"]
            task_disp = job_tasks[task_cols].copy()
            task_disp["Is_Billable"] = task_disp["Is_Billable"].map({True: "✓", False: ""})
            task_disp["Is_Unquoted"] = task_disp["Is_Unquoted"].map({True: "⚠️ Unquoted", False: ""})
            task_disp.columns = ["Task", "Category", "Billable",
                                 "Quoted Hrs", "Actual Hrs", "Hrs Var",
                                 "Quoted $", "Cost $", "Revenue $",
                                 "Profit $", "Margin %", "Unbilled Hrs", "Flag"]
            
            st.dataframe(task_disp.style.format({
                "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
                "Quoted $": "${:,.0f}", "Cost $": "${:,.0f}", "Revenue $": "${:,.0f}",
                "Profit $": "${:,.0f}", "Margin %": "{:.1f}%", "Unbilled Hrs": "{:,.1f}"
            }), use_container_width=True)
            
            # Alerts
            unquoted = job_tasks[job_tasks["Is_Unquoted"]]
            unbilled = job_tasks[job_tasks["Unbilled_Hours"] > 1]
            
            if len(unquoted) > 0:
                st.warning(f"⚠️ **{len(unquoted)} unquoted task(s)** (scope creep) — "
                           f"Cost: {fmt_currency(unquoted['Actual_Cost'].sum())}, "
                           f"Hours: {unquoted['Actual_Hours'].sum():,.0f}")
            
            if len(unbilled) > 0:
                st.info(f"📝 **{len(unbilled)} task(s) with unbilled hours** — "
                        f"Total: {unbilled['Unbilled_Hours'].sum():,.0f} hrs")
    else:
        st.info("No jobs match current filters.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 4: SYNTHESIS — WHY DO JOBS RUN OVER?
    # =========================================================================
    st.header("🔬 Synthesis: Why Do Jobs Run Over?")
    st.markdown("*Common reasons for margin erosion identified in the data*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Scope Creep / Unquoted Work")
        st.metric("Unquoted Tasks", f"{causes['scope_creep']['task_count']:,}")
        st.metric("Cost of Unquoted Work", fmt_currency(causes['scope_creep']['total_cost']))
        st.metric("Unquoted Hours", fmt_hours(causes['scope_creep']['total_hours']))
        st.caption("Tasks with Actual Hours > 0 but Quoted Hours = 0")
        
        st.subheader("⏱️ Underestimation of Effort")
        st.metric("Overrun Tasks", f"{causes['underestimation']['task_count']:,}")
        st.metric("Excess Hours", fmt_hours(causes['underestimation']['excess_hours']))
        st.caption("Tasks where Actual Hours exceeded Quoted Hours")
    
    with col2:
        st.subheader("📝 Billing Issues")
        st.metric("Tasks with Unbilled Hours", f"{causes['billing_issues']['tasks_with_unbilled']:,}")
        st.metric("Total Unbilled Hours", fmt_hours(causes['billing_issues']['unbilled_hours']))
        st.caption("Hours logged but not invoiced")
        
        st.subheader("🚫 Non-Billable Work")
        st.metric("Non-Billable Tasks", f"{causes['non_billable_work']['task_count']:,}")
        st.metric("Cost of Non-Billable", fmt_currency(causes['non_billable_work']['total_cost']))
        st.caption("Internal work that hits profitability")
    
    # Top overruns table
    st.subheader("🚨 Top 10 Job Overruns (by Hours)")
    top_overruns = get_top_overruns(job_summary, n=10, by="Hours_Variance")
    if len(top_overruns) > 0:
        over_disp = top_overruns[["Job_No", "Job_Name", "Client", "Hours_Variance", 
                                   "Hours_Variance_Pct", "Profit", "Margin_Pct"]].copy()
        over_disp.columns = ["Job #", "Job Name", "Client", "Hrs Over", "Hrs Var %", "Profit $", "Margin %"]
        st.dataframe(over_disp.style.format({
            "Hrs Over": "{:+,.0f}", "Hrs Var %": "{:+.1f}%",
            "Profit $": "${:,.0f}", "Margin %": "{:.1f}%"
        }), use_container_width=True)
    
    # Loss-making jobs
    st.subheader("💸 Loss-Making Jobs")
    losses = get_loss_making_jobs(job_summary).head(10)
    if len(losses) > 0:
        loss_disp = losses[["Job_No", "Job_Name", "Client", "Profit", "Margin_Pct", "Actual_Cost"]].copy()
        loss_disp.columns = ["Job #", "Job Name", "Client", "Profit $", "Margin %", "Cost $"]
        st.dataframe(loss_disp.style.format({
            "Profit $": "${:,.0f}", "Margin %": "{:.1f}%", "Cost $": "${:,.0f}"
        }), use_container_width=True)
    else:
        st.success("✅ No loss-making jobs in selected period!")
    
    # Footer
    st.markdown("---")
    st.caption(
        "**Job Profitability Analysis Dashboard** | "
        "Hierarchy: Category → Job → Task | "
        "Excludes 'Social Garden Invoice Allocation' | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()