"""
Job Profitability Analysis Dashboard
=====================================
Streamlit app with full metric traceability and reconciliation.

METRIC DEFINITIONS:
-------------------
- Quoted Amount:   [Job Task] Quoted Amount
- Billable Value:  Actual Hours × Billable Rate (calculated)
- Cost (T&M):      Actual Hours × Base Rate (calculated)
- Profit:          Billable Value - Cost
- Margin %:        (Profit / Billable Value) × 100
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

from analysis import (
    load_raw_data, clean_and_parse, apply_filters,
    compute_reconciliation_totals, get_available_fiscal_years,
    compute_category_summary, compute_job_summary, compute_task_summary,
    get_top_overruns, get_loss_making_jobs, get_unquoted_tasks,
    calculate_overall_metrics, analyze_overrun_causes,
    METRIC_DEFINITIONS
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Job Profitability Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPERS
# =============================================================================

def fmt_currency(val):
    if pd.isna(val) or val == 0:
        return "$0"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:,.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:,.1f}K"
    return f"${val:,.0f}"

def fmt_pct(val):
    return f"{val:.1f}%" if pd.notna(val) else "N/A"

def fmt_hours(val):
    return f"{val:,.1f}" if pd.notna(val) else "0"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("📊 Job Profitability Analysis")
    st.markdown("*Quote to Execution — Category → Job → Task*")
    
    # =========================================================================
    # SIDEBAR: DATA SOURCE
    # =========================================================================
    st.sidebar.header("📁 Data Source")
    
    data_path = Path("data/Quoted_Task_Report_FY26.xlsx")
    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
    
    if uploaded:
        data_source = uploaded
    elif data_path.exists():
        data_source = str(data_path)
    else:
        st.warning("⚠️ Upload data file or place in `data/` folder")
        st.stop()
    
    # Load raw data
    try:
        with st.spinner("Loading raw data..."):
            df_raw = load_raw_data(data_source)
            df_parsed = clean_and_parse(df_raw)
        st.sidebar.success(f"✅ Raw: {len(df_raw):,} records")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # =========================================================================
    # SIDEBAR: FILTERS (with toggles)
    # =========================================================================
    st.sidebar.header("🎛️ Filters")
    
    # Fiscal Year
    fy_list = get_available_fiscal_years(df_parsed)
    fy_options = ["All Years"] + [f"FY{str(y)[-2:]}" for y in fy_list]
    selected_fy = st.sidebar.selectbox(
        "Fiscal Year",
        fy_options,
        help="Based on [Job] Start Date. Australian FY: Jul-Jun"
    )
    fy_num = int("20" + selected_fy[2:]) if selected_fy != "All Years" else None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔘 Data Inclusion")
    
    # Toggle: Exclude SG Allocation
    exclude_sg = st.sidebar.checkbox(
        "Exclude 'Social Garden Invoice Allocation'",
        value=True,
        help="Internal allocation entries, not actual project work"
    )
    
    # Toggle: Billable only
    billable_only = st.sidebar.checkbox(
        "Billable tasks only",
        value=True,
        help="Only tasks where Base Rate > 0 AND Billable Rate > 0"
    )
    
    # Apply filters
    df_filtered, recon = apply_filters(
        df_parsed,
        exclude_sg_allocation=exclude_sg,
        billable_only=billable_only,
        fiscal_year=fy_num
    )
    recon = compute_reconciliation_totals(df_filtered, recon)
    
    if len(df_filtered) == 0:
        st.error("No data after applying filters.")
        st.stop()
    
    # Compute summaries
    category_summary = compute_category_summary(df_filtered)
    job_summary = compute_job_summary(df_filtered)
    task_summary = compute_task_summary(df_filtered)
    metrics = calculate_overall_metrics(job_summary)
    causes = analyze_overrun_causes(task_summary)
    
    # =========================================================================
    # SIDEBAR: INFO
    # =========================================================================
    st.sidebar.markdown("---")
    st.sidebar.metric("Records After Filters", f"{recon['final_records']:,}")
    st.sidebar.metric("Unique Jobs", f"{recon['totals']['unique_jobs']:,}")
    
    # =========================================================================
    # SECTION 0: RECONCILIATION & TRACEABILITY
    # =========================================================================
    with st.expander("🔍 DATA RECONCILIATION & TRACEABILITY", expanded=False):
        st.markdown("### Filter Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Raw Records", f"{recon['raw_records']:,}")
        with col2:
            st.metric("After Filters", f"{recon['final_records']:,}")
        with col3:
            st.metric("Excluded", f"{recon['raw_records'] - recon['final_records']:,}")
        
        st.markdown("#### Exclusions Breakdown")
        excl_data = pd.DataFrame({
            "Filter": [
                "Social Garden Invoice Allocation",
                "Non-Billable Tasks (no rates)",
                "Other Fiscal Years"
            ],
            "Excluded Records": [
                recon["excluded_sg_allocation"],
                recon["excluded_non_billable"],
                recon["excluded_other_fy"]
            ],
            "Active": [
                "✓" if exclude_sg else "✗",
                "✓" if billable_only else "✗",
                "✓" if fy_num else "✗ (All Years)"
            ]
        })
        st.dataframe(excl_data, use_container_width=True, hide_index=True)
        
        st.markdown("#### Validation Totals (from filtered data)")
        st.markdown("*These should match your source data for the selected period/filters*")
        
        val_data = pd.DataFrame({
            "Metric": [
                "Sum of Quoted Hours",
                "Sum of Actual Hours",
                "Sum of Invoiced Hours",
                "Sum of Quoted Amount",
                "Sum of Billable Value (Hrs × Billable Rate)",
                "Sum of Cost T&M (Hrs × Base Rate)",
                "Unique Jobs",
                "Unique Task Names"
            ],
            "Value": [
                f"{recon['totals']['sum_quoted_hours']:,.1f}",
                f"{recon['totals']['sum_actual_hours']:,.1f}",
                f"{recon['totals']['sum_invoiced_hours']:,.1f}",
                f"${recon['totals']['sum_quoted_amount']:,.0f}",
                f"${recon['totals']['sum_calc_billable_value']:,.0f}",
                f"${recon['totals']['sum_calc_cost_tm']:,.0f}",
                f"{recon['totals']['unique_jobs']:,}",
                f"{recon['totals']['unique_tasks']:,}"
            ]
        })
        st.dataframe(val_data, use_container_width=True, hide_index=True)
        
        st.markdown("#### Comparison: Data Field vs Calculated")
        st.markdown("*Billable Amount field vs calculated (Hours × Rate)*")
        comp_data = pd.DataFrame({
            "Metric": ["Billable Amount (field)", "Billable Value (calculated)", "Difference"],
            "Value": [
                f"${recon['totals']['sum_billable_amount_field']:,.0f}",
                f"${recon['totals']['sum_calc_billable_value']:,.0f}",
                f"${recon['totals']['sum_billable_amount_field'] - recon['totals']['sum_calc_billable_value']:,.0f}"
            ]
        })
        st.dataframe(comp_data, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # SECTION: METRIC DEFINITIONS
    # =========================================================================
    with st.expander("📖 METRIC DEFINITIONS", expanded=False):
        st.markdown("### How Metrics Are Calculated")
        
        for key, defn in METRIC_DEFINITIONS.items():
            st.markdown(f"**{defn['name']}**")
            st.code(defn['formula'])
            st.caption(defn['description'])
            st.markdown("---")
    
    # =========================================================================
    # SECTION 1: OVERALL KPIs
    # =========================================================================
    st.markdown(f"### 📅 Period: **{selected_fy}** — {metrics['total_jobs']:,} Jobs | {recon['final_records']:,} Task Records")
    
    st.subheader("📈 Overall Performance")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quoted Amount", fmt_currency(metrics["total_quoted_amount"]),
              help="SUM([Job Task] Quoted Amount)")
    c2.metric("Billable Value", fmt_currency(metrics["total_billable_value"]),
              help="SUM(Actual Hours × Billable Rate)")
    c3.metric("Cost (T&M)", fmt_currency(metrics["total_cost_tm"]),
              help="SUM(Actual Hours × Base Rate)")
    c4.metric("Profit", fmt_currency(metrics["total_profit"]),
              help="Billable Value - Cost")
    c5.metric("Margin %", fmt_pct(metrics["overall_margin_pct"]),
              help="(Profit / Billable Value) × 100")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs Over Budget", f"{metrics['jobs_over_budget']} / {metrics['total_jobs']}",
              delta=f"{metrics['overrun_rate']:.1f}%", delta_color="inverse")
    c2.metric("Jobs at Loss", str(metrics['jobs_at_loss']),
              delta=f"{metrics['loss_rate']:.1f}%", delta_color="inverse")
    c3.metric("Avg Margin %", fmt_pct(metrics["avg_margin_pct"]),
              delta=f"{-metrics['margin_gap']:.1f}% vs quoted", 
              delta_color="inverse" if metrics['margin_gap'] > 0 else "normal")
    c4.metric("Hours Variance", f"{metrics['hours_variance']:+,.0f} hrs",
              delta=f"{metrics['hours_variance_pct']:+.1f}%",
              delta_color="inverse" if metrics['hours_variance'] > 0 else "normal")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 2: CATEGORY ANALYSIS
    # =========================================================================
    st.header("📂 Level 1: Category Analysis")
    st.caption("*Which areas of the business have margin issues?*")
    
    cat_list = ["All Categories"] + sorted(category_summary["Category"].dropna().unique().tolist())
    selected_category = st.selectbox("Select Category", cat_list, key="cat_sel")
    
    if selected_category == "All Categories":
        cat_data = category_summary[category_summary["Billable_Value"] > 0].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Margin % by Category")
            chart1 = alt.Chart(cat_data).mark_bar().encode(
                x=alt.X("Margin_Pct:Q", title="Margin %"),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.condition(alt.datum.Margin_Pct < 20, alt.value("#e53935"), alt.value("#43a047")),
                tooltip=["Category", alt.Tooltip("Margin_Pct:Q", format=".1f"),
                         alt.Tooltip("Profit:Q", format="$,.0f"), "Job_Count"]
            ).properties(height=max(300, len(cat_data) * 25))
            st.altair_chart(chart1, use_container_width=True)
        
        with col2:
            st.subheader("Hours Variance % by Category")
            chart2 = alt.Chart(cat_data).mark_bar().encode(
                x=alt.X("Hours_Variance_Pct:Q", title="Hours Variance %"),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.condition(alt.datum.Hours_Variance_Pct > 20, alt.value("#e53935"), alt.value("#1e88e5")),
                tooltip=["Category", alt.Tooltip("Hours_Variance_Pct:Q", format=".1f"),
                         alt.Tooltip("Hours_Variance:Q", format=",.0f")]
            ).properties(height=max(300, len(cat_data) * 25))
            st.altair_chart(chart2, use_container_width=True)
        
        with st.expander("📋 Category Table"):
            cat_disp = cat_data[[
                "Category", "Job_Count", "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
                "Quoted_Amount", "Billable_Value", "Cost_TM", "Profit", "Margin_Pct"
            ]].copy()
            cat_disp.columns = ["Category", "Jobs", "Quoted Hrs", "Actual Hrs", "Hrs Var %",
                                "Quoted $", "Billable $", "Cost $", "Profit $", "Margin %"]
            st.dataframe(cat_disp.style.format({
                "Quoted Hrs": "{:,.0f}", "Actual Hrs": "{:,.0f}", "Hrs Var %": "{:+.1f}%",
                "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
                "Profit $": "${:,.0f}", "Margin %": "{:.1f}%"
            }), use_container_width=True)
    else:
        cat_row = category_summary[category_summary["Category"] == selected_category].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jobs", int(cat_row["Job_Count"]))
        c2.metric("Margin %", fmt_pct(cat_row["Margin_Pct"]))
        c3.metric("Profit", fmt_currency(cat_row["Profit"]))
        c4.metric("Hours Var", f"{cat_row['Hours_Variance']:+,.0f} hrs")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 3: JOB-LEVEL ANALYSIS
    # =========================================================================
    st.header("📋 Level 2: Job Profitability")
    st.caption("*Quote vs Actual per job — identify problem projects*")
    
    # Filter jobs by category
    if selected_category != "All Categories":
        jobs_filtered = job_summary[job_summary["Category"] == selected_category].copy()
    else:
        jobs_filtered = job_summary.copy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        show_overruns = st.checkbox("Overruns only", False)
    with col2:
        show_losses = st.checkbox("Losses only", False)
    with col3:
        show_erosion = st.checkbox("Margin erosion >10%", False)
    
    if show_overruns:
        jobs_filtered = jobs_filtered[jobs_filtered["Is_Overrun"]]
    if show_losses:
        jobs_filtered = jobs_filtered[jobs_filtered["Is_Loss"]]
    if show_erosion:
        jobs_filtered = jobs_filtered[jobs_filtered["Margin_Erosion"] > 10]
    
    st.caption(f"Showing {len(jobs_filtered)} jobs")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox("Sort by", [
            "Margin_Pct", "Profit", "Hours_Variance", "Margin_Erosion", "Cost_TM"
        ], format_func=lambda x: {
            "Margin_Pct": "Margin % (lowest)", "Profit": "Profit (lowest)",
            "Hours_Variance": "Hours Variance (highest)", "Margin_Erosion": "Margin Erosion (highest)",
            "Cost_TM": "Cost (highest)"
        }.get(x, x))
    with col2:
        top_n = st.number_input("Show top N", 10, 500, 50)
    
    asc = sort_by in ["Margin_Pct", "Profit"]
    jobs_display = jobs_filtered.sort_values(sort_by, ascending=asc).head(top_n)
    
    job_cols = ["Job_No", "Job_Name", "Client", "Category",
                "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
                "Quoted_Amount", "Billable_Value", "Cost_TM",
                "Profit", "Margin_Pct", "Margin_Erosion"]
    job_disp = jobs_display[job_cols].copy()
    job_disp.columns = ["Job #", "Job Name", "Client", "Category",
                        "Quoted Hrs", "Actual Hrs", "Hrs Var %",
                        "Quoted $", "Billable $", "Cost $",
                        "Profit $", "Margin %", "Erosion %"]
    
    st.dataframe(job_disp.style.format({
        "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var %": "{:+.1f}%",
        "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
        "Profit $": "${:,.0f}", "Margin %": "{:.1f}%", "Erosion %": "{:+.1f}%"
    }), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 4: TASK DRILL-DOWN
    # =========================================================================
    st.header("🔍 Level 3: Task Drill-Down")
    st.caption("*Which task caused the overrun?*")
    
    job_options = jobs_filtered.apply(lambda r: f"{r['Job_No']} — {r['Job_Name'][:40]}", axis=1).tolist()
    
    if job_options:
        selected_job = st.selectbox("Select Job", ["-- Select --"] + job_options, key="job_sel")
        
        if selected_job != "-- Select --":
            job_no = selected_job.split(" — ")[0]
            job_info = jobs_filtered[jobs_filtered["Job_No"] == job_no].iloc[0]
            job_tasks = task_summary[task_summary["Job_No"] == job_no].copy()
            
            st.subheader(f"📁 {job_info['Job_Name']}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Margin %", fmt_pct(job_info["Margin_Pct"]))
            c2.metric("Profit", fmt_currency(job_info["Profit"]))
            c3.metric("Hrs Variance", f"{job_info['Hours_Variance']:+,.0f}")
            c4.metric("Erosion", f"{job_info['Margin_Erosion']:+.1f}%")
            c5.metric("Client", str(job_info["Client"])[:20] if pd.notna(job_info["Client"]) else "N/A")
            
            # Task chart
            st.subheader("Hours: Quoted vs Actual")
            task_melt = job_tasks[["Task_Name", "Quoted_Hours", "Actual_Hours"]].melt(
                id_vars=["Task_Name"], var_name="Type", value_name="Hours"
            )
            task_melt["Type"] = task_melt["Type"].map({"Quoted_Hours": "Quoted", "Actual_Hours": "Actual"})
            
            chart = alt.Chart(task_melt).mark_bar().encode(
                x=alt.X("Hours:Q"),
                y=alt.Y("Task_Name:N", sort="-x"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Quoted", "Actual"], range=["#1e88e5", "#43a047"])),
                xOffset="Type:N",
                tooltip=["Task_Name", "Type", alt.Tooltip("Hours:Q", format=",.1f")]
            ).properties(height=max(200, len(job_tasks) * 28))
            st.altair_chart(chart, use_container_width=True)
            
            # Task table
            st.subheader("Task Details")
            task_cols = ["Task_Name", "Task_Category",
                         "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                         "Quoted_Amount", "Billable_Value", "Cost_TM",
                         "Profit", "Margin_Pct", "Base_Rate", "Billable_Rate", "Is_Unquoted"]
            task_disp = job_tasks[task_cols].copy()
            task_disp["Is_Unquoted"] = task_disp["Is_Unquoted"].map({True: "⚠️", False: ""})
            task_disp.columns = ["Task", "Category",
                                 "Quoted Hrs", "Actual Hrs", "Hrs Var",
                                 "Quoted $", "Billable $", "Cost $",
                                 "Profit $", "Margin %", "Base Rate", "Bill Rate", "Flag"]
            
            st.dataframe(task_disp.style.format({
                "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
                "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
                "Profit $": "${:,.0f}", "Margin %": "{:.1f}%",
                "Base Rate": "${:,.0f}", "Bill Rate": "${:,.0f}"
            }), use_container_width=True)
            
            unquoted = job_tasks[job_tasks["Is_Unquoted"]]
            if len(unquoted) > 0:
                st.warning(f"⚠️ **{len(unquoted)} unquoted task(s)** — Cost: {fmt_currency(unquoted['Cost_TM'].sum())}")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 5: SYNTHESIS
    # =========================================================================
    st.header("🔬 Synthesis: Why Jobs Run Over")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📌 Scope Creep")
        st.metric("Unquoted Tasks", f"{causes['scope_creep']['task_count']:,}")
        st.metric("Cost", fmt_currency(causes['scope_creep']['total_cost']))
        st.metric("Hours", fmt_hours(causes['scope_creep']['total_hours']))
    with col2:
        st.subheader("⏱️ Underestimation")
        st.metric("Overrun Tasks", f"{causes['underestimation']['task_count']:,}")
        st.metric("Excess Hours", fmt_hours(causes['underestimation']['excess_hours']))
    with col3:
        st.subheader("📝 Unbilled Work")
        st.metric("Tasks", f"{causes['unbilled_work']['task_count']:,}")
        st.metric("Unbilled Hours", fmt_hours(causes['unbilled_work']['unbilled_hours']))
    
    st.subheader("🚨 Top 10 Overruns")
    top_over = get_top_overruns(job_summary, 10, "Hours_Variance")
    if len(top_over) > 0:
        st.dataframe(top_over[["Job_No", "Job_Name", "Client", "Hours_Variance", "Profit", "Margin_Pct"]].style.format({
            "Hours_Variance": "{:+,.0f}", "Profit": "${:,.0f}", "Margin_Pct": "{:.1f}%"
        }), use_container_width=True)
    
    st.subheader("💸 Loss-Making Jobs")
    losses = get_loss_making_jobs(job_summary).head(10)
    if len(losses) > 0:
        st.dataframe(losses[["Job_No", "Job_Name", "Client", "Profit", "Margin_Pct", "Cost_TM"]].style.format({
            "Profit": "${:,.0f}", "Margin_Pct": "{:.1f}%", "Cost_TM": "${:,.0f}"
        }), use_container_width=True)
    else:
        st.success("✅ No loss-making jobs!")
    
    # Footer
    st.markdown("---")
    st.caption(
        "**Job Profitability Analysis** | "
        f"Filters: {'Excl SG Alloc' if exclude_sg else 'Incl SG Alloc'}, "
        f"{'Billable Only' if billable_only else 'All Tasks'} | "
        f"Period: {selected_fy}"
    )


if __name__ == "__main__":
    main()