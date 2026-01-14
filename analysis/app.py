"""
Job Profitability Analysis Dashboard
=====================================
Hierarchy: Department → Product → Job → Task

METRIC DEFINITIONS:
-------------------
MARGINS:
- Quoted Margin %:   (Quoted Amount - Cost) / Quoted Amount × 100
- Billable Margin %: (Billable Value - Cost) / Billable Value × 100

RATES (per hour):
- Quoted Rate/Hr:    Quoted Amount / Quoted Hours
- Billable Rate/Hr:  [Task] Billable Rate
- Cost Rate/Hr:      [Task] Base Rate (T&M)

VALUES:
- Quoted Amount:     [Job Task] Quoted Amount
- Billable Value:    Actual Hours × Billable Rate/Hr
- Cost (T&M):        Actual Hours × Cost Rate/Hr
- Profit:            Billable Value - Cost
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

def fmt_rate(val):
    return f"${val:,.0f}/hr" if pd.notna(val) and val > 0 else "N/A"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("📊 Job Profitability Analysis")
    st.markdown("*Quote to Execution — Department → Product → Job → Task*")
    
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
    # SIDEBAR: FILTERS
    # =========================================================================
    st.sidebar.header("🎛️ Filters")
    
    # Department Filter
    dept_list = get_available_departments(df_parsed)
    dept_options = ["All Departments"] + dept_list
    selected_dept = st.sidebar.selectbox(
        "Department",
        dept_options,
        help="Filter by Department"
    )
    dept_filter = None if selected_dept == "All Departments" else selected_dept
    
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
        fiscal_year=fy_num,
        department=dept_filter
    )
    recon = compute_reconciliation_totals(df_filtered, recon)
    
    if len(df_filtered) == 0:
        st.error("No data after applying filters.")
        st.stop()
    
    # Compute summaries
    dept_summary = compute_department_summary(df_filtered)
    product_summary = compute_product_summary(df_filtered)
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
    st.sidebar.metric("Departments", f"{recon['totals']['unique_departments']:,}")
    st.sidebar.metric("Products", f"{recon['totals']['unique_products']:,}")
    
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
                "Other Fiscal Years",
                "Other Departments"
            ],
            "Excluded Records": [
                recon["excluded_sg_allocation"],
                recon["excluded_non_billable"],
                recon["excluded_other_fy"],
                recon["excluded_other_dept"]
            ],
            "Active": [
                "✓" if exclude_sg else "✗",
                "✓" if billable_only else "✗",
                "✓" if fy_num else "✗ (All Years)",
                "✓" if dept_filter else "✗ (All Depts)"
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
                "Sum of Billable Value (calculated)",
                "Sum of Cost T&M (calculated)",
                "Sum of Invoiced Amount",
                "Avg Quoted Rate/Hr",
                "Avg Billable Rate/Hr",
                "Avg Cost Rate/Hr",
            ],
            "Value": [
                f"{recon['totals']['sum_quoted_hours']:,.1f}",
                f"{recon['totals']['sum_actual_hours']:,.1f}",
                f"{recon['totals']['sum_invoiced_hours']:,.1f}",
                f"${recon['totals']['sum_quoted_amount']:,.0f}",
                f"${recon['totals']['sum_billable_value']:,.0f}",
                f"${recon['totals']['sum_cost_tm']:,.0f}",
                f"${recon['totals']['sum_invoiced_amount']:,.0f}",
                f"${recon['totals']['avg_quoted_rate_hr']:,.0f}/hr",
                f"${recon['totals']['avg_billable_rate_hr']:,.0f}/hr",
                f"${recon['totals']['avg_cost_rate_hr']:,.0f}/hr",
            ],
            "Source/Formula": [
                "SUM([Job Task] Quoted Time)",
                "SUM([Job Task] Actual Time (totalled))",
                "SUM([Job Task] Invoiced Time)",
                "SUM([Job Task] Quoted Amount)",
                "SUM(Actual Hrs × Billable Rate/Hr)",
                "SUM(Actual Hrs × Cost Rate/Hr)",
                "SUM([Job Task] Invoiced Amount)",
                "Quoted Amount / Quoted Hours",
                "AVG([Task] Billable Rate)",
                "AVG([Task] Base Rate)",
            ]
        })
        st.dataframe(val_data, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # SECTION 0.5: METRIC DEFINITIONS
    # =========================================================================
    with st.expander("📐 METRIC DEFINITIONS", expanded=False):
        for key, defn in METRIC_DEFINITIONS.items():
            st.markdown(f"**{defn['name']}**: `{defn['formula']}`")
            st.caption(defn['desc'])
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 1: OVERALL KPIs
    # =========================================================================
    st.header("📈 Overall Performance")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Jobs", f"{metrics['total_jobs']:,}")
    c2.metric("Quoted Amount", fmt_currency(metrics['total_quoted_amount']))
    c3.metric("Billable Value", fmt_currency(metrics['total_billable_value']))
    c4.metric("Cost (T&M)", fmt_currency(metrics['total_cost_tm']))
    c5.metric("Profit", fmt_currency(metrics['total_profit']))
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quoted Margin", fmt_pct(metrics["overall_quoted_margin_pct"]))
    c2.metric("Billable Margin", fmt_pct(metrics["overall_billable_margin_pct"]))
    c3.metric("Quoted Rate/Hr", fmt_rate(metrics["avg_quoted_rate_hr"]))
    c4.metric("Billable Rate/Hr", fmt_rate(metrics["avg_billable_rate_hr"]))
    c5.metric("Cost Rate/Hr", fmt_rate(metrics["avg_cost_rate_hr"]))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs Over Budget", f"{metrics['jobs_over_budget']} / {metrics['total_jobs']}", 
              delta=f"{metrics['overrun_rate']:.1f}%", delta_color="inverse")
    c2.metric("Jobs at Loss", str(metrics['jobs_at_loss']), 
              delta=f"{metrics['loss_rate']:.1f}%", delta_color="inverse")
    c3.metric("Quoted Hours", fmt_hours(metrics["total_hours_quoted"]))
    c4.metric("Actual Hours", fmt_hours(metrics["total_hours_actual"]), 
              delta=f"{metrics['hours_variance']:+,.0f} hrs", delta_color="inverse")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 2: DEPARTMENT ANALYSIS (Level 1)
    # =========================================================================
    st.header("🏢 Level 1: Department Analysis")
    
    if selected_dept == "All Departments":
        dept_data = dept_summary[dept_summary["Billable_Value"] > 0].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Billable Margin % by Department")
            chart = alt.Chart(dept_data).mark_bar().encode(
                x=alt.X("Billable_Margin_Pct:Q", title="Billable Margin %"),
                y=alt.Y("Department:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Billable_Margin_Pct < 20,
                    alt.value("#e53935"),
                    alt.value("#43a047")
                ),
                tooltip=["Department", 
                         alt.Tooltip("Billable_Margin_Pct:Q", format=".1f", title="Billable Margin %"),
                         alt.Tooltip("Quoted_Margin_Pct:Q", format=".1f", title="Quoted Margin %"),
                         alt.Tooltip("Job_Count:Q", title="Jobs")]
            ).properties(height=max(200, len(dept_data) * 35))
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.subheader("Hours Variance % by Department")
            chart = alt.Chart(dept_data).mark_bar().encode(
                x=alt.X("Hours_Variance_Pct:Q", title="Hours Variance %"),
                y=alt.Y("Department:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Hours_Variance_Pct > 20,
                    alt.value("#e53935"),
                    alt.value("#1e88e5")
                ),
                tooltip=["Department",
                         alt.Tooltip("Hours_Variance_Pct:Q", format="+.1f", title="Hrs Var %"),
                         alt.Tooltip("Quoted_Hours:Q", format=",.0f", title="Quoted Hrs"),
                         alt.Tooltip("Actual_Hours:Q", format=",.0f", title="Actual Hrs")]
            ).properties(height=max(200, len(dept_data) * 35))
            st.altair_chart(chart, use_container_width=True)
        
        # Department table
        st.subheader("Department Summary Table")
        dept_disp = dept_data[[
            "Department", "Job_Count", "Product_Count",
            "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
            "Quoted_Amount", "Billable_Value", "Cost_TM",
            "Profit", "Quoted_Margin_Pct", "Billable_Margin_Pct",
            "Quoted_Rate_Hr", "Billable_Rate_Hr", "Cost_Rate_Hr"
        ]].copy()
        dept_disp.columns = [
            "Department", "Jobs", "Products",
            "Quoted Hrs", "Actual Hrs", "Hrs Var %",
            "Quoted $", "Billable $", "Cost $",
            "Profit $", "Quoted Margin %", "Billable Margin %",
            "Quoted $/Hr", "Billable $/Hr", "Cost $/Hr"
        ]
        st.dataframe(dept_disp.style.format({
            "Quoted Hrs": "{:,.0f}", "Actual Hrs": "{:,.0f}", "Hrs Var %": "{:+.1f}%",
            "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
            "Profit $": "${:,.0f}", "Quoted Margin %": "{:.1f}%", "Billable Margin %": "{:.1f}%",
            "Quoted $/Hr": "${:,.0f}", "Billable $/Hr": "${:,.0f}", "Cost $/Hr": "${:,.0f}"
        }), use_container_width=True)
    else:
        dept_row = dept_summary[dept_summary["Department"] == selected_dept].iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Products", int(dept_row["Product_Count"]))
        c2.metric("Jobs", int(dept_row["Job_Count"]))
        c3.metric("Billable Margin", fmt_pct(dept_row["Billable_Margin_Pct"]))
        c4.metric("Profit", fmt_currency(dept_row["Profit"]))
        c5.metric("Hours Var", f"{dept_row['Hours_Variance']:+,.0f} hrs")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 3: PRODUCT ANALYSIS (Level 2)
    # =========================================================================
    st.header("📦 Level 2: Product Analysis")
    st.caption("*Performance by Product (within selected Department)*")
    
    # Filter products by selected department
    if selected_dept != "All Departments":
        prod_filtered = product_summary[product_summary["Department"] == selected_dept].copy()
    else:
        prod_filtered = product_summary.copy()
    
    prod_filtered = prod_filtered[prod_filtered["Billable_Value"] > 0]
    
    # Product selector
    prod_options = ["All Products"] + sorted(prod_filtered["Product"].unique().tolist())
    selected_product = st.selectbox("Select Product", prod_options, key="product_sel")
    
    if selected_product == "All Products":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Billable Margin % by Product")
            chart = alt.Chart(prod_filtered.head(20)).mark_bar().encode(
                x=alt.X("Billable_Margin_Pct:Q", title="Billable Margin %"),
                y=alt.Y("Product:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Billable_Margin_Pct < 20,
                    alt.value("#e53935"),
                    alt.value("#43a047")
                ),
                tooltip=["Product", "Department",
                         alt.Tooltip("Billable_Margin_Pct:Q", format=".1f"),
                         alt.Tooltip("Quoted_Margin_Pct:Q", format=".1f"),
                         alt.Tooltip("Job_Count:Q", title="Jobs")]
            ).properties(height=max(200, min(len(prod_filtered), 20) * 28))
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.subheader("Hours Variance % by Product")
            chart = alt.Chart(prod_filtered.head(20)).mark_bar().encode(
                x=alt.X("Hours_Variance_Pct:Q", title="Hours Variance %"),
                y=alt.Y("Product:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Hours_Variance_Pct > 20,
                    alt.value("#e53935"),
                    alt.value("#1e88e5")
                ),
                tooltip=["Product", "Department",
                         alt.Tooltip("Hours_Variance_Pct:Q", format="+.1f"),
                         alt.Tooltip("Quoted_Hours:Q", format=",.0f"),
                         alt.Tooltip("Actual_Hours:Q", format=",.0f")]
            ).properties(height=max(200, min(len(prod_filtered), 20) * 28))
            st.altair_chart(chart, use_container_width=True)
        
        # Product table
        st.subheader("Product Summary Table")
        prod_disp = prod_filtered[[
            "Department", "Product", "Job_Count",
            "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
            "Quoted_Amount", "Billable_Value", "Cost_TM",
            "Profit", "Quoted_Margin_Pct", "Billable_Margin_Pct",
            "Quoted_Rate_Hr", "Billable_Rate_Hr", "Cost_Rate_Hr"
        ]].copy()
        prod_disp.columns = [
            "Department", "Product", "Jobs",
            "Quoted Hrs", "Actual Hrs", "Hrs Var %",
            "Quoted $", "Billable $", "Cost $",
            "Profit $", "Quoted Margin %", "Billable Margin %",
            "Quoted $/Hr", "Billable $/Hr", "Cost $/Hr"
        ]
        st.dataframe(prod_disp.style.format({
            "Quoted Hrs": "{:,.0f}", "Actual Hrs": "{:,.0f}", "Hrs Var %": "{:+.1f}%",
            "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
            "Profit $": "${:,.0f}", "Quoted Margin %": "{:.1f}%", "Billable Margin %": "{:.1f}%",
            "Quoted $/Hr": "${:,.0f}", "Billable $/Hr": "${:,.0f}", "Cost $/Hr": "${:,.0f}"
        }), use_container_width=True, height=400)
    else:
        prod_row = prod_filtered[prod_filtered["Product"] == selected_product].iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jobs", int(prod_row["Job_Count"]))
        c2.metric("Billable Margin", fmt_pct(prod_row["Billable_Margin_Pct"]))
        c3.metric("Quoted Margin", fmt_pct(prod_row["Quoted_Margin_Pct"]))
        c4.metric("Profit", fmt_currency(prod_row["Profit"]))
        c5.metric("Hours Var", f"{prod_row['Hours_Variance']:+,.0f} hrs")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 4: JOB-LEVEL ANALYSIS (Level 3)
    # =========================================================================
    st.header("📋 Level 3: Job Profitability")
    st.caption("*Quote vs Actual per job — identify problem projects*")
    
    # Filter jobs by department and product
    jobs_filtered = job_summary.copy()
    if selected_dept != "All Departments":
        jobs_filtered = jobs_filtered[jobs_filtered["Department"] == selected_dept]
    if selected_product != "All Products":
        jobs_filtered = jobs_filtered[jobs_filtered["Product"] == selected_product]
    
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
            "Billable_Margin_Pct", "Profit", "Hours_Variance", "Margin_Erosion", "Cost_TM"
        ], format_func=lambda x: {
            "Billable_Margin_Pct": "Billable Margin % (lowest)", 
            "Profit": "Profit (lowest)",
            "Hours_Variance": "Hours Variance (highest)", 
            "Margin_Erosion": "Margin Erosion (highest)",
            "Cost_TM": "Cost (highest)"
        }.get(x, x))
    with col2:
        top_n = st.number_input("Show top N", 10, 500, 50)
    
    asc = sort_by in ["Billable_Margin_Pct", "Profit"]
    jobs_display = jobs_filtered.sort_values(sort_by, ascending=asc).head(top_n)
    
    job_cols = ["Job_No", "Job_Name", "Client", "Department", "Product",
                "Quoted_Hours", "Actual_Hours", "Hours_Variance_Pct",
                "Quoted_Amount", "Billable_Value", "Cost_TM",
                "Profit", "Quoted_Margin_Pct", "Billable_Margin_Pct", "Margin_Erosion"]
    job_disp = jobs_display[job_cols].copy()
    job_disp.columns = ["Job #", "Job Name", "Client", "Department", "Product",
                        "Quoted Hrs", "Actual Hrs", "Hrs Var %",
                        "Quoted $", "Billable $", "Cost $",
                        "Profit $", "Quoted Margin %", "Billable Margin %", "Erosion %"]
    
    st.dataframe(job_disp.style.format({
        "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var %": "{:+.1f}%",
        "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
        "Profit $": "${:,.0f}", "Quoted Margin %": "{:.1f}%", 
        "Billable Margin %": "{:.1f}%", "Erosion %": "{:+.1f}%"
    }), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 5: TASK DRILL-DOWN (Level 4)
    # =========================================================================
    st.header("🔍 Level 4: Task Drill-Down")
    st.caption("*Which task caused the overrun?*")
    
    job_options = jobs_filtered.apply(
        lambda r: f"{r['Job_No']} — {r['Job_Name'][:40]}", axis=1
    ).tolist()
    
    if job_options:
        selected_job = st.selectbox("Select Job", ["-- Select --"] + job_options, key="job_sel")
        
        if selected_job != "-- Select --":
            job_no = selected_job.split(" — ")[0]
            job_info = jobs_filtered[jobs_filtered["Job_No"] == job_no].iloc[0]
            job_tasks = task_summary[task_summary["Job_No"] == job_no].copy()
            
            st.subheader(f"📁 {job_info['Job_Name']}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Billable Margin", fmt_pct(job_info["Billable_Margin_Pct"]))
            c2.metric("Quoted Margin", fmt_pct(job_info["Quoted_Margin_Pct"]))
            c3.metric("Profit", fmt_currency(job_info["Profit"]))
            c4.metric("Hrs Variance", f"{job_info['Hours_Variance']:+,.0f}")
            c5.metric("Erosion", f"{job_info['Margin_Erosion']:+.1f}%")
            c6.metric("Client", str(job_info["Client"])[:20] if pd.notna(job_info["Client"]) else "N/A")
            
            # Task chart
            st.subheader("Hours: Quoted vs Actual")
            task_melt = job_tasks[["Task_Name", "Quoted_Hours", "Actual_Hours"]].melt(
                id_vars=["Task_Name"], var_name="Type", value_name="Hours"
            )
            task_melt["Type"] = task_melt["Type"].map({"Quoted_Hours": "Quoted", "Actual_Hours": "Actual"})
            
            chart = alt.Chart(task_melt).mark_bar().encode(
                x=alt.X("Hours:Q"),
                y=alt.Y("Task_Name:N", sort="-x"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Quoted", "Actual"], range=["#1e88e5", "#43a047"]
                )),
                xOffset="Type:N",
                tooltip=["Task_Name", "Type", alt.Tooltip("Hours:Q", format=",.1f")]
            ).properties(height=max(200, len(job_tasks) * 28))
            st.altair_chart(chart, use_container_width=True)
            
            # Task table
            st.subheader("Task Details")
            task_cols = ["Task_Name", "Task_Category",
                         "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                         "Quoted_Amount", "Billable_Value", "Cost_TM",
                         "Profit", "Quoted_Margin_Pct", "Billable_Margin_Pct",
                         "Cost_Rate_Hr", "Billable_Rate_Hr", "Quoted_Rate_Hr", "Is_Unquoted"]
            task_disp = job_tasks[task_cols].copy()
            task_disp["Is_Unquoted"] = task_disp["Is_Unquoted"].map({True: "⚠️", False: ""})
            task_disp.columns = ["Task", "Category",
                                 "Quoted Hrs", "Actual Hrs", "Hrs Var",
                                 "Quoted $", "Billable $", "Cost $",
                                 "Profit $", "Quoted Margin %", "Billable Margin %",
                                 "Cost $/Hr", "Billable $/Hr", "Quoted $/Hr", "Flag"]
            
            st.dataframe(task_disp.style.format({
                "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
                "Quoted $": "${:,.0f}", "Billable $": "${:,.0f}", "Cost $": "${:,.0f}",
                "Profit $": "${:,.0f}", "Quoted Margin %": "{:.1f}%", "Billable Margin %": "{:.1f}%",
                "Cost $/Hr": "${:,.0f}", "Billable $/Hr": "${:,.0f}", "Quoted $/Hr": "${:,.0f}"
            }), use_container_width=True)
            
            unquoted = job_tasks[job_tasks["Is_Unquoted"]]
            if len(unquoted) > 0:
                st.warning(f"⚠️ **{len(unquoted)} unquoted task(s)** — Cost: {fmt_currency(unquoted['Cost_TM'].sum())}")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 6: SYNTHESIS
    # =========================================================================
    st.header("🔬 Synthesis: Why Jobs Run Over")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📌 Scope Creep")
        st.metric("Unquoted Tasks", f"{causes['scope_creep']['count']:,}")
        st.metric("Cost", fmt_currency(causes['scope_creep']['cost']))
        st.metric("Hours", fmt_hours(causes['scope_creep']['hours']))
    with col2:
        st.subheader("⏱️ Underestimation")
        st.metric("Overrun Tasks", f"{causes['underestimation']['count']:,}")
        st.metric("Excess Hours", fmt_hours(causes['underestimation']['excess_hours']))
    with col3:
        st.subheader("📝 Unbilled Work")
        st.metric("Tasks", f"{causes['unbilled']['count']:,}")
        st.metric("Unbilled Hours", fmt_hours(causes['unbilled']['hours']))
    
    st.subheader("🚨 Top 10 Overruns")
    top_over = get_top_overruns(job_summary, 10, "Hours_Variance")
    if len(top_over) > 0:
        st.dataframe(top_over[[
            "Job_No", "Job_Name", "Client", "Department", "Product",
            "Hours_Variance", "Profit", "Billable_Margin_Pct"
        ]].style.format({
            "Hours_Variance": "{:+,.0f}", "Profit": "${:,.0f}", "Billable_Margin_Pct": "{:.1f}%"
        }), use_container_width=True)
    
    st.subheader("💸 Loss-Making Jobs")
    losses = get_loss_making_jobs(job_summary).head(10)
    if len(losses) > 0:
        st.dataframe(losses[[
            "Job_No", "Job_Name", "Client", "Department", "Product",
            "Profit", "Billable_Margin_Pct", "Cost_TM"
        ]].style.format({
            "Profit": "${:,.0f}", "Billable_Margin_Pct": "{:.1f}%", "Cost_TM": "${:,.0f}"
        }), use_container_width=True)
    else:
        st.success("✅ No loss-making jobs!")
    
    # Footer
    st.markdown("---")
    st.caption(
        "**Job Profitability Analysis** | "
        f"Filters: {'Excl SG Alloc' if exclude_sg else 'Incl SG Alloc'}, "
        f"{'Billable Only' if billable_only else 'All Tasks'} | "
        f"Department: {selected_dept} | "
        f"Period: {selected_fy}"
    )


if __name__ == "__main__":
    main()