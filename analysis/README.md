"""
Job Profitability Dashboard — Corrected Financial Logic
=========================================================
REVENUE = Quoted Amount (NOT Billable Value)
Billable Rate is internal only — used for rate gap analysis

Key Metrics:
- Quoted Margin = Quoted Amount - Base Cost (what we expected)
- Actual Margin = Invoiced Amount - Base Cost (what we got)
- Realization % = Invoiced / Quoted (revenue capture rate)
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
    get_margin_erosion_jobs, get_low_realization_jobs, get_write_off_jobs,
    calculate_overall_metrics, analyze_overrun_causes,
    generate_insights, diagnose_job_margin,
    METRIC_DEFINITIONS
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Job Profitability Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-help { font-size: 0.75rem; color: #666; }
    .warning-box { background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .success-box { background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .danger-box { background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

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
    return f"{val:,.0f}" if pd.notna(val) else "0"

def fmt_rate(val):
    return f"${val:,.0f}/hr" if pd.notna(val) and val > 0 else "N/A"

def status_icon(val, good_threshold, bad_threshold, higher_is_better=True):
    if higher_is_better:
        if val >= good_threshold:
            return "🟢"
        elif val >= bad_threshold:
            return "🟡"
        return "🔴"
    else:
        if val <= good_threshold:
            return "🟢"
        elif val <= bad_threshold:
            return "🟡"
        return "🔴"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("💰 Job Profitability Analysis")
    st.markdown("*Revenue = Quoted Amount | Realization = Invoiced ÷ Quoted*")
    
    # Important framing note
    with st.expander("📖 Understanding This Dashboard", expanded=False):
        st.markdown("""
        ### Key Financial Definitions
        
        | Term | Definition | Source |
        |------|------------|--------|
        | **Quoted Amount** | Client quote = **REVENUE** | `[Job Task] Quoted Amount` |
        | **Invoiced Amount** | What was actually billed | `[Job Task] Invoiced Amount` |
        | **Base Cost** | Internal labor cost | `Actual Hours × Cost Rate/Hr` |
        | **Billable Value** | ⚠️ Internal benchmark only, NOT revenue | `Hours × Billable Rate` |
        
        ### Margin Calculations
        - **Quoted Margin** = Quoted Amount − Base Cost *(what we expected to make)*
        - **Actual Margin** = Invoiced Amount − Base Cost *(what we actually made)*
        - **Realization %** = Invoiced ÷ Quoted *(how much revenue we captured)*
        
        ### Why Did Margin Erode?
        1. **Quote too low** → Quoted Rate/Hr below Billable Rate (negative rate gap)
        2. **Scope creep** → Unquoted tasks added after quote
        3. **Hour overrun** → Actual hours exceeded quoted hours
        4. **Write-off** → Invoiced less than quoted (discounts, errors)
        5. **Wrong resourcing** → Cost Rate/Hr exceeded expectations
        """)
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    st.sidebar.header("📁 Data & Filters")
    
    data_path = Path("data/Quoted_Task_Report_FY26.xlsx")
    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
    
    if uploaded:
        data_source = uploaded
    elif data_path.exists():
        data_source = str(data_path)
    else:
        st.warning("⚠️ Upload data file or place in `data/` folder")
        st.stop()
    
    try:
        with st.spinner("Loading data..."):
            df_raw = load_raw_data(data_source)
            df_parsed = clean_and_parse(df_raw)
        st.sidebar.success(f"✅ {len(df_raw):,} records")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Fiscal Year
    fy_list = get_available_fiscal_years(df_parsed)
    if not fy_list:
        st.error("No fiscal year data found")
        st.stop()
    
    selected_fy = st.sidebar.selectbox(
        "📅 Fiscal Year", fy_list,
        index=len(fy_list) - 1,
        format_func=lambda x: f"FY{str(x)[-2:]}"
    )
    
    # Department
    dept_list = get_available_departments(df_parsed)
    selected_dept = st.sidebar.selectbox("🏢 Department", ["All Departments"] + dept_list)
    dept_filter = None if selected_dept == "All Departments" else selected_dept
    
    st.sidebar.markdown("---")
    exclude_sg = st.sidebar.checkbox("Exclude SG Allocation", value=True)
    billable_only = st.sidebar.checkbox("Billable tasks only", value=True)
    
    # Apply filters
    df_filtered, recon = apply_filters(
        df_parsed, exclude_sg_allocation=exclude_sg, billable_only=billable_only,
        fiscal_year=selected_fy, department=dept_filter
    )
    recon = compute_reconciliation_totals(df_filtered, recon)
    
    if len(df_filtered) == 0:
        st.error("No data after applying filters.")
        st.stop()
    
    # Compute summaries
    with st.spinner("Computing..."):
        dept_summary = compute_department_summary(df_filtered)
        product_summary = compute_product_summary(df_filtered)
        job_summary = compute_job_summary(df_filtered)
        task_summary = compute_task_summary(df_filtered)
        monthly_summary = compute_monthly_summary(df_filtered)
        monthly_by_dept = compute_monthly_by_department(df_filtered)
        metrics = calculate_overall_metrics(job_summary)
        causes = analyze_overrun_causes(task_summary)
        insights = generate_insights(job_summary, dept_summary, monthly_summary, task_summary)
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Records", f"{recon['final_records']:,}")
    st.sidebar.metric("Jobs", f"{metrics['total_jobs']:,}")
    
    # =========================================================================
    # TABS
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary", "📈 Monthly Trends", "🏢 Drill-Down",
        "💡 Why Margins Erode", "🔍 Job Diagnosis", "📋 Reconciliation"
    ])
    
    # =========================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # =========================================================================
    with tab1:
        st.header(f"FY{str(selected_fy)[-2:]} Executive Summary")
        
        # Headlines
        if insights["headline"]:
            for h in insights["headline"]:
                st.markdown(h)
        
        st.markdown("---")
        
        # REVENUE (Quoted vs Invoiced)
        st.subheader("💵 Revenue: Quoted vs Invoiced")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Amount", fmt_currency(metrics['total_quoted_amount']),
                  help="Client quote = Expected revenue")
        c2.metric("Invoiced Amount", fmt_currency(metrics['total_invoiced_amount']),
                  delta=f"{metrics['realization_pct']:.0f}% realized",
                  delta_color="normal" if metrics['realization_pct'] >= 95 else "inverse",
                  help="What was actually billed")
        c3.metric("Write-Off", fmt_currency(metrics['write_off_total']),
                  delta="Revenue gap" if metrics['write_off_total'] > 0 else "None",
                  delta_color="inverse" if metrics['write_off_total'] > 1000 else "off",
                  help="Quoted - Invoiced = unbilled revenue")
        realization_icon = status_icon(metrics['realization_pct'], 95, 85)
        c4.metric(f"{realization_icon} Realization %", fmt_pct(metrics['realization_pct']),
                  help="Invoiced ÷ Quoted. Target: 95%+")
        
        # MARGINS
        st.subheader("📊 Margins: Quoted vs Actual")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Margin", fmt_currency(metrics['quoted_margin']),
                  delta=fmt_pct(metrics['quoted_margin_pct']),
                  help="Expected margin = Quoted - Cost")
        c2.metric("Actual Margin", fmt_currency(metrics['actual_margin']),
                  delta=fmt_pct(metrics['actual_margin_pct']),
                  help="Realized margin = Invoiced - Cost")
        margin_var = metrics['margin_variance']
        c3.metric("Margin Variance", fmt_currency(margin_var),
                  delta="Erosion" if margin_var < 0 else "Gain",
                  delta_color="inverse" if margin_var < 0 else "normal",
                  help="Actual Margin - Quoted Margin")
        c4.metric("Base Cost", fmt_currency(metrics['total_base_cost']),
                  help="Actual Hours × Cost Rate/Hr")
        
        # RATES
        st.subheader("💲 Rate Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Rate/Hr", fmt_rate(metrics['avg_quoted_rate_hr']),
                  help="Quoted Amount ÷ Quoted Hours")
        c2.metric("Actual Rate/Hr", fmt_rate(metrics['avg_actual_rate_hr']),
                  help="Invoiced Amount ÷ Actual Hours")
        c3.metric("Cost Rate/Hr", fmt_rate(metrics['avg_cost_rate_hr']),
                  help="Base Cost ÷ Actual Hours")
        spread = metrics['avg_actual_rate_hr'] - metrics['avg_cost_rate_hr']
        c4.metric("Rate Spread", fmt_rate(spread) if spread > 0 else "Negative",
                  help="Actual Rate - Cost Rate")
        
        # PERFORMANCE FLAGS
        st.subheader("⚠️ Performance Flags")
        c1, c2, c3, c4 = st.columns(4)
        loss_icon = status_icon(metrics['loss_rate'], 5, 15, higher_is_better=False)
        c1.metric(f"{loss_icon} Jobs at Loss", f"{metrics['jobs_at_loss']} / {metrics['total_jobs']}",
                  delta=f"{metrics['loss_rate']:.0f}%", delta_color="inverse")
        overrun_icon = status_icon(metrics['overrun_rate'], 30, 50, higher_is_better=False)
        c2.metric(f"{overrun_icon} Hour Overruns", f"{metrics['jobs_over_budget']}",
                  delta=f"{metrics['overrun_rate']:.0f}%", delta_color="inverse")
        c3.metric("Write-Off Jobs", str(metrics['jobs_with_write_off']),
                  help="Jobs where Invoiced < Quoted by >$100")
        c4.metric("Scope Creep Tasks", str(causes['scope_creep']['count']),
                  delta=fmt_currency(causes['scope_creep']['cost']),
                  help="Unquoted tasks = work not in original quote")
        
        st.markdown("---")
        
        # MARGIN BRIDGE
        st.subheader("🌉 Margin Bridge: Quoted → Invoiced → Cost")
        
        bridge_data = pd.DataFrame([
            {"Step": "1. Quoted Amount", "Amount": metrics['total_quoted_amount'], "Type": "Revenue"},
            {"Step": "2. Write-Off", "Amount": -metrics['write_off_total'], "Type": "Leakage"},
            {"Step": "3. Invoiced Amount", "Amount": metrics['total_invoiced_amount'], "Type": "Revenue"},
            {"Step": "4. Base Cost", "Amount": -metrics['total_base_cost'], "Type": "Cost"},
            {"Step": "5. Actual Margin", "Amount": metrics['actual_margin'], "Type": "Margin"},
        ])
        
        # Simple metrics view
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Quoted", fmt_currency(metrics['total_quoted_amount']))
        col2.metric("Write-Off", fmt_currency(-metrics['write_off_total']), delta_color="inverse")
        col3.metric("Invoiced", fmt_currency(metrics['total_invoiced_amount']))
        col4.metric("Cost", fmt_currency(-metrics['total_base_cost']))
        col5.metric("Margin", fmt_currency(metrics['actual_margin']),
                    delta=fmt_pct(metrics['actual_margin_pct']))
        
        # Waterfall chart
        bridge_chart = alt.Chart(bridge_data).mark_bar().encode(
            x=alt.X("Step:N", sort=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Amount:Q", title="Amount ($)"),
            color=alt.condition(
                alt.datum.Amount >= 0,
                alt.value("#2ecc71"),
                alt.value("#e74c3c")
            ),
            tooltip=["Step", alt.Tooltip("Amount:Q", format="$,.0f")]
        ).properties(height=300)
        st.altair_chart(bridge_chart, use_container_width=True)
    
    # =========================================================================
    # TAB 2: MONTHLY TRENDS
    # =========================================================================
    with tab2:
        st.header(f"📈 Monthly Trends — FY{str(selected_fy)[-2:]}")
        
        if len(monthly_summary) == 0:
            st.warning("No monthly data available.")
        else:
            # Metric selector
            trend_metric = st.selectbox(
                "Select Metric",
                ["Realization_Pct", "Actual_Margin_Pct", "Quoted_Margin_Pct", 
                 "Write_Off", "Invoiced_Amount", "Quoted_Amount", "Hours_Variance_Pct"],
                format_func=lambda x: {
                    "Realization_Pct": "Realization % (Invoiced ÷ Quoted)",
                    "Actual_Margin_Pct": "Actual Margin % (on Invoiced)",
                    "Quoted_Margin_Pct": "Quoted Margin % (on Quoted)",
                    "Write_Off": "Write-Off Amount",
                    "Invoiced_Amount": "Invoiced Amount",
                    "Quoted_Amount": "Quoted Amount",
                    "Hours_Variance_Pct": "Hours Variance %"
                }.get(x, x)
            )
            
            # Main trend
            st.subheader(f"📊 {trend_metric.replace('_', ' ')} by Month")
            trend_chart = alt.Chart(monthly_summary).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(f"{trend_metric}:Q"),
                tooltip=["Month", alt.Tooltip(f"{trend_metric}:Q", format=",.1f")]
            ).properties(height=350)
            
            # Reference lines
            if trend_metric == "Realization_Pct":
                rule = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y:Q")
                trend_chart = trend_chart + rule
            elif "Margin_Pct" in trend_metric:
                rule = alt.Chart(pd.DataFrame({"y": [35]})).mark_rule(color="orange", strokeDash=[5,5]).encode(y="y:Q")
                trend_chart = trend_chart + rule
            
            st.altair_chart(trend_chart, use_container_width=True)
            
            # Quoted vs Invoiced comparison
            st.subheader("📉 Quoted vs Invoiced (Revenue)")
            rev_data = monthly_summary.melt(
                id_vars=["Month"], value_vars=["Quoted_Amount", "Invoiced_Amount"],
                var_name="Type", value_name="Amount"
            )
            rev_data["Type"] = rev_data["Type"].map({
                "Quoted_Amount": "Quoted (Expected)",
                "Invoiced_Amount": "Invoiced (Actual)"
            })
            rev_chart = alt.Chart(rev_data).mark_bar().encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Amount:Q"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Quoted (Expected)", "Invoiced (Actual)"],
                    range=["#3498db", "#2ecc71"]
                )),
                xOffset="Type:N",
                tooltip=["Month", "Type", alt.Tooltip("Amount:Q", format="$,.0f")]
            ).properties(height=300)
            st.altair_chart(rev_chart, use_container_width=True)
            
            # Margin comparison
            st.subheader("📊 Quoted Margin vs Actual Margin")
            margin_data = monthly_summary.melt(
                id_vars=["Month"], value_vars=["Quoted_Margin_Pct", "Actual_Margin_Pct"],
                var_name="Type", value_name="Margin_Pct"
            )
            margin_data["Type"] = margin_data["Type"].map({
                "Quoted_Margin_Pct": "Quoted Margin %",
                "Actual_Margin_Pct": "Actual Margin %"
            })
            margin_chart = alt.Chart(margin_data).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Margin_Pct:Q", title="Margin %"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Quoted Margin %", "Actual Margin %"],
                    range=["#3498db", "#2ecc71"]
                )),
                strokeDash=alt.condition(
                    alt.datum.Type == "Quoted Margin %",
                    alt.value([5, 5]),
                    alt.value([0])
                ),
                tooltip=["Month", "Type", alt.Tooltip("Margin_Pct:Q", format=".1f")]
            ).properties(height=300)
            st.altair_chart(margin_chart, use_container_width=True)
            
            # Department trends
            if selected_dept == "All Departments" and len(monthly_by_dept) > 0:
                st.subheader("🏢 Realization by Department")
                dept_trend = alt.Chart(monthly_by_dept).mark_line(point=True).encode(
                    x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Realization_Pct:Q", title="Realization %"),
                    color="Department:N",
                    tooltip=["Month", "Department", alt.Tooltip("Realization_Pct:Q", format=".0f")]
                ).properties(height=350)
                st.altair_chart(dept_trend, use_container_width=True)
            
            with st.expander("📋 Monthly Data Table"):
                st.dataframe(monthly_summary[[
                    "Month", "Job_Count", "Quoted_Amount", "Invoiced_Amount", "Write_Off",
                    "Base_Cost", "Quoted_Margin", "Actual_Margin", "Margin_Variance",
                    "Quoted_Margin_Pct", "Actual_Margin_Pct", "Realization_Pct"
                ]], use_container_width=True)
    
    # =========================================================================
    # TAB 3: DRILL-DOWN
    # =========================================================================
    with tab3:
        st.header("🏢 Hierarchical Analysis")
        
        # Department
        st.subheader("Level 1: Department Performance")
        if len(dept_summary) > 0:
            # Realization focus
            dept_chart = alt.Chart(dept_summary).mark_bar().encode(
                y=alt.Y("Department:N", sort="-x"),
                x=alt.X("Realization_Pct:Q", title="Realization %", scale=alt.Scale(domain=[0, 120])),
                color=alt.condition(alt.datum.Realization_Pct < 90, alt.value("#e74c3c"), alt.value("#2ecc71")),
                tooltip=["Department",
                         alt.Tooltip("Realization_Pct:Q", format=".0f", title="Realization %"),
                         alt.Tooltip("Actual_Margin_Pct:Q", format=".1f", title="Actual Margin %"),
                         alt.Tooltip("Write_Off:Q", format="$,.0f", title="Write-Off")]
            ).properties(height=max(200, len(dept_summary) * 40))
            
            # Add 100% reference line
            rule = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(color="gray", strokeDash=[3,3]).encode(x="x:Q")
            st.altair_chart(dept_chart + rule, use_container_width=True)
            
            with st.expander("Department Details"):
                st.dataframe(dept_summary[[
                    "Department", "Job_Count", "Quoted_Amount", "Invoiced_Amount", "Write_Off",
                    "Base_Cost", "Actual_Margin", "Actual_Margin_Pct", "Realization_Pct"
                ]].style.format({
                    "Quoted_Amount": "${:,.0f}", "Invoiced_Amount": "${:,.0f}", 
                    "Write_Off": "${:,.0f}", "Base_Cost": "${:,.0f}",
                    "Actual_Margin": "${:,.0f}", "Actual_Margin_Pct": "{:.1f}%",
                    "Realization_Pct": "{:.0f}%"
                }), use_container_width=True)
        
        st.markdown("---")
        
        # Product
        st.subheader("Level 2: Product Performance")
        sel_dept_drill = st.selectbox("Filter by Department", ["All"] + sorted(dept_summary["Department"].unique().tolist()), key="d1")
        prod_f = product_summary if sel_dept_drill == "All" else product_summary[product_summary["Department"] == sel_dept_drill]
        
        if len(prod_f) > 0:
            prod_chart = alt.Chart(prod_f.head(15)).mark_bar().encode(
                y=alt.Y("Product:N", sort="-x"),
                x=alt.X("Realization_Pct:Q", title="Realization %"),
                color=alt.condition(alt.datum.Realization_Pct < 90, alt.value("#e74c3c"), alt.value("#2ecc71")),
                tooltip=["Product", "Department",
                         alt.Tooltip("Realization_Pct:Q", format=".0f"),
                         alt.Tooltip("Write_Off:Q", format="$,.0f")]
            ).properties(height=max(200, min(len(prod_f), 15) * 30))
            st.altair_chart(prod_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Job
        st.subheader("Level 3: Job Performance")
        sel_prod = st.selectbox("Filter by Product", ["All"] + sorted(prod_f["Product"].unique().tolist()), key="p1")
        jobs_f = job_summary.copy()
        if sel_dept_drill != "All":
            jobs_f = jobs_f[jobs_f["Department"] == sel_dept_drill]
        if sel_prod != "All":
            jobs_f = jobs_f[jobs_f["Product"] == sel_prod]
        
        # Filters
        c1, c2, c3, c4 = st.columns(4)
        show_loss = c1.checkbox("Loss only", key="jl")
        show_low_real = c2.checkbox("Low Realization (<90%)", key="jr")
        show_writeoff = c3.checkbox("Has Write-Off", key="jw")
        sort_by = c4.selectbox("Sort", ["Actual_Margin", "Realization_Pct", "Write_Off", "Margin_Variance"], key="js")
        
        if show_loss:
            jobs_f = jobs_f[jobs_f["Is_Loss"]]
        if show_low_real:
            jobs_f = jobs_f[jobs_f["Realization_Pct"] < 90]
        if show_writeoff:
            jobs_f = jobs_f[jobs_f["Has_Write_Off"]]
        
        jobs_disp = jobs_f.sort_values(sort_by, ascending=sort_by in ["Actual_Margin", "Realization_Pct"]).head(25)
        
        if len(jobs_disp) > 0:
            cols = ["Job_No", "Job_Name", "Client", "Month",
                    "Quoted_Amount", "Invoiced_Amount", "Write_Off", "Base_Cost",
                    "Quoted_Margin", "Actual_Margin", "Realization_Pct", "Actual_Margin_Pct"]
            st.dataframe(jobs_disp[cols].style.format({
                "Quoted_Amount": "${:,.0f}", "Invoiced_Amount": "${:,.0f}",
                "Write_Off": "${:,.0f}", "Base_Cost": "${:,.0f}",
                "Quoted_Margin": "${:,.0f}", "Actual_Margin": "${:,.0f}",
                "Realization_Pct": "{:.0f}%", "Actual_Margin_Pct": "{:.1f}%"
            }), use_container_width=True, height=400)
        else:
            st.info("No jobs match filters.")
        
        st.markdown("---")
        
        # Task
        st.subheader("Level 4: Task Breakdown")
        job_opts = jobs_disp.apply(lambda r: f"{r['Job_No']} — {str(r['Job_Name'])[:35]}", axis=1).tolist() if len(jobs_disp) > 0 else []
        if job_opts:
            sel_job = st.selectbox("Select Job", ["-- Select --"] + job_opts, key="tj")
            if sel_job != "-- Select --":
                job_no = sel_job.split(" — ")[0]
                job_info = jobs_disp[jobs_disp["Job_No"] == job_no].iloc[0]
                tasks = task_summary[task_summary["Job_No"] == job_no]
                
                st.markdown(f"### 📁 {job_info['Job_Name']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Quoted Margin", fmt_currency(job_info["Quoted_Margin"]),
                          delta=fmt_pct(job_info["Quoted_Margin_Pct"]))
                c2.metric("Actual Margin", fmt_currency(job_info["Actual_Margin"]),
                          delta=fmt_pct(job_info["Actual_Margin_Pct"]))
                real_icon = status_icon(job_info["Realization_Pct"], 95, 85)
                c3.metric(f"{real_icon} Realization", fmt_pct(job_info["Realization_Pct"]))
                c4.metric("Write-Off", fmt_currency(job_info["Write_Off"]))
                
                if len(tasks) > 0:
                    # Flag unquoted tasks
                    st.markdown("#### Tasks")
                    task_cols = ["Task_Name", "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                                 "Quoted_Amount", "Invoiced_Amount", "Write_Off", "Base_Cost",
                                 "Actual_Margin", "Realization_Pct", "Is_Unquoted"]
                    task_disp = tasks[task_cols].copy()
                    task_disp["Is_Unquoted"] = task_disp["Is_Unquoted"].map({True: "⚠️ SCOPE CREEP", False: ""})
                    task_disp.columns = ["Task", "Quoted Hrs", "Actual Hrs", "Hrs Var",
                                         "Quoted $", "Invoiced $", "Write-Off", "Cost",
                                         "Margin", "Realization", "Flag"]
                    st.dataframe(task_disp.style.format({
                        "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
                        "Quoted $": "${:,.0f}", "Invoiced $": "${:,.0f}",
                        "Write-Off": "${:,.0f}", "Cost": "${:,.0f}",
                        "Margin": "${:,.0f}", "Realization": "{:.0f}%"
                    }), use_container_width=True)
    
    # =========================================================================
    # TAB 4: WHY MARGINS ERODE
    # =========================================================================
    with tab4:
        st.header("💡 Why Margins Erode")
        st.markdown("*Root cause analysis for profitability issues*")
        
        # Quoting Issues
        if insights["quoting_issues"]:
            st.subheader("❓ Was the Quote Too Low?")
            for i in insights["quoting_issues"]:
                st.markdown(i)
        
        # Scope Issues
        if insights["scope_issues"]:
            st.subheader("📋 Was Scope Not Controlled?")
            for i in insights["scope_issues"]:
                st.markdown(i)
        
        # Realization Issues
        if insights["realization_issues"]:
            st.subheader("💰 Revenue Not Captured?")
            for i in insights["realization_issues"]:
                st.markdown(i)
        
        # Rate Issues
        if insights["rate_issues"]:
            st.subheader("👥 Wrong Resourcing?")
            for i in insights["rate_issues"]:
                st.markdown(i)
        
        # Margin Drivers
        if insights["margin_drivers"]:
            st.subheader("📊 Top Margin Drivers")
            for i in insights["margin_drivers"]:
                st.markdown(i)
        
        # Action Items
        if insights["action_items"]:
            st.subheader("🚨 Action Items")
            for a in insights["action_items"]:
                st.markdown(f"- {a}")
        
        st.markdown("---")
        
        # Deep dive panels
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💸 Worst Write-Offs")
            writeoffs = get_write_off_jobs(job_summary, 500).head(10)
            if len(writeoffs) > 0:
                for _, j in writeoffs.iterrows():
                    st.markdown(f"**{str(j['Job_Name'])[:35]}** — ${j['Write_Off']:,.0f} write-off")
                    st.caption(f"  Realization: {j['Realization_Pct']:.0f}%")
            else:
                st.success("No significant write-offs!")
        
        with col2:
            st.subheader("📋 Scope Creep (Unquoted Work)")
            unquoted = get_unquoted_tasks(task_summary).head(10)
            if len(unquoted) > 0:
                st.metric("Total Unquoted Cost", fmt_currency(unquoted["Base_Cost"].sum()))
                for _, t in unquoted.iterrows():
                    st.markdown(f"• **{str(t['Task_Name'])[:30]}** — {t['Actual_Hours']:.0f} hrs, ${t['Base_Cost']:,.0f}")
            else:
                st.success("No scope creep detected!")
        
        st.markdown("---")
        
        # Low Realization Jobs
        st.subheader("📉 Low Realization Jobs (<90%)")
        low_real = get_low_realization_jobs(job_summary, 90).head(10)
        if len(low_real) > 0:
            for _, j in low_real.iterrows():
                st.markdown(f"**{str(j['Job_Name'])[:40]}** ({j['Job_No']})")
                st.caption(f"  Quoted: ${j['Quoted_Amount']:,.0f} | Invoiced: ${j['Invoiced_Amount']:,.0f} | "
                           f"Realization: {j['Realization_Pct']:.0f}%")
        else:
            st.success("All jobs have good realization!")
    
    # =========================================================================
    # TAB 5: JOB DIAGNOSIS
    # =========================================================================
    with tab5:
        st.header("🔍 Job Diagnosis Tool")
        st.markdown("*Understand why a specific job performed the way it did*")
        
        # Job selector
        all_jobs = job_summary.apply(
            lambda r: f"{r['Job_No']} — {str(r['Job_Name'])[:40]} ({r['Client']})", axis=1
        ).tolist()
        
        selected_job = st.selectbox("Select a Job to Diagnose", ["-- Select --"] + all_jobs)
        
        if selected_job != "-- Select --":
            job_no = selected_job.split(" — ")[0]
            job_row = job_summary[job_summary["Job_No"] == job_no].iloc[0]
            job_tasks = task_summary[task_summary["Job_No"] == job_no]
            
            # Run diagnosis
            diagnosis = diagnose_job_margin(job_row, job_tasks)
            
            # Display job summary
            st.subheader(f"📁 {job_row['Job_Name']}")
            st.caption(f"Client: {job_row['Client']} | {job_row['Month']}")
            
            # KPIs
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Quoted", fmt_currency(job_row['Quoted_Amount']))
            c2.metric("Invoiced", fmt_currency(job_row['Invoiced_Amount']))
            c3.metric("Cost", fmt_currency(job_row['Base_Cost']))
            margin_icon = "🟢" if job_row['Actual_Margin'] > 0 else "🔴"
            c4.metric(f"{margin_icon} Margin", fmt_currency(job_row['Actual_Margin']))
            real_icon = status_icon(job_row['Realization_Pct'], 95, 85)
            c5.metric(f"{real_icon} Realization", fmt_pct(job_row['Realization_Pct']))
            
            st.markdown("---")
            
            # Diagnosis
            st.subheader("🩺 Diagnosis")
            st.markdown(f"**Summary:** {diagnosis['summary']}")
            
            if diagnosis['issues']:
                st.markdown("**Issues Identified:**")
                for issue in diagnosis['issues']:
                    st.markdown(f"- ⚠️ {issue}")
            
            if diagnosis['root_causes']:
                st.markdown("**Root Causes:**")
                for cause in diagnosis['root_causes']:
                    st.markdown(f"- 🔍 {cause}")
            
            if diagnosis['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in diagnosis['recommendations']:
                    st.markdown(f"- 💡 {rec}")
            
            # Task breakdown
            if len(job_tasks) > 0:
                st.markdown("---")
                st.subheader("📋 Task Analysis")
                
                # Highlight issues
                unquoted_tasks = job_tasks[job_tasks['Is_Unquoted']]
                overrun_tasks = job_tasks[job_tasks['Is_Overrun'] & ~job_tasks['Is_Unquoted']]
                writeoff_tasks = job_tasks[job_tasks['Has_Write_Off']]
                
                if len(unquoted_tasks) > 0:
                    st.markdown("**⚠️ Unquoted Tasks (Scope Creep):**")
                    for _, t in unquoted_tasks.iterrows():
                        st.markdown(f"- {t['Task_Name']}: {t['Actual_Hours']:.0f} hrs, ${t['Base_Cost']:,.0f} cost")
                
                if len(overrun_tasks) > 0:
                    st.markdown("**⏱️ Hour Overruns:**")
                    for _, t in overrun_tasks.iterrows():
                        st.markdown(f"- {t['Task_Name']}: {t['Hours_Variance']:+.0f} hrs over")
                
                if len(writeoff_tasks) > 0:
                    st.markdown("**💸 Tasks with Write-Offs:**")
                    for _, t in writeoff_tasks.iterrows():
                        st.markdown(f"- {t['Task_Name']}: ${t['Write_Off']:,.0f} not invoiced")
    
    # =========================================================================
    # TAB 6: RECONCILIATION
    # =========================================================================
    with tab6:
        st.header("📋 Data Reconciliation")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Records", f"{recon['raw_records']:,}")
        c2.metric("Filtered", f"{recon['final_records']:,}")
        c3.metric("Excluded", f"{recon['raw_records'] - recon['final_records']:,}")
        
        st.subheader("Exclusions")
        st.dataframe(pd.DataFrame({
            "Filter": ["SG Allocation", "Non-Billable", "Other FY", "Other Dept"],
            "Excluded": [recon["excluded_sg_allocation"], recon["excluded_non_billable"],
                         recon["excluded_other_fy"], recon["excluded_other_dept"]]
        }), use_container_width=True, hide_index=True)
        
        st.subheader("Validation Totals")
        totals_df = pd.DataFrame({
            "Metric": list(recon["totals"].keys()),
            "Value": [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in recon["totals"].values()]
        })
        st.dataframe(totals_df, use_container_width=True, hide_index=True)
        
        st.subheader("📐 Metric Definitions")
        st.markdown("""
        | Metric | Formula | Description |
        |--------|---------|-------------|
        | **Quoted Amount** | Direct from data | Client quote = REVENUE |
        | **Invoiced Amount** | Direct from data | What was actually billed |
        | **Base Cost** | Actual Hours × Cost Rate/Hr | Internal labor cost |
        | **Quoted Margin** | Quoted Amount - Base Cost | Expected margin |
        | **Actual Margin** | Invoiced Amount - Base Cost | Realized margin |
        | **Realization %** | Invoiced ÷ Quoted × 100 | Revenue capture rate |
        | **Write-Off** | Quoted - Invoiced | Revenue not captured |
        | **Billable Value** | Hours × Billable Rate | ⚠️ Internal only, NOT revenue |
        """)
        
        with st.expander("All Metric Definitions"):
            for key, defn in METRIC_DEFINITIONS.items():
                st.markdown(f"**{defn['name']}**")
                st.markdown(f"- Formula: `{defn['formula']}`")
                st.markdown(f"- {defn['desc']}")
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(
        f"💰 Job Profitability Analysis | FY{str(selected_fy)[-2:]} | {selected_dept} | "
        f"{recon['final_records']:,} records | Revenue = Quoted Amount"
    )


if __name__ == "__main__":
    main()