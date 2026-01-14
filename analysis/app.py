"""
Job Profitability Dashboard
============================
REVENUE = Quoted Amount
BENCHMARK = Expected Quote (Quoted Hours × Billable Rate)
MARGIN = Quoted Amount - Base Cost
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
    get_margin_erosion_jobs, get_underquoted_jobs, get_premium_jobs,
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
    st.markdown("*Revenue = Quoted Amount | Sanity Check = Expected Quote (Quoted Hrs × Billable Rate)*")
    
    # Key definitions
    with st.expander("📖 Understanding This Dashboard", expanded=False):
        st.markdown("""
        ### Financial Model
        
        | Term | Formula | Purpose |
        |------|---------|---------|
        | **Quoted Amount** | From data | **= REVENUE** (what we invoice the client) |
        | **Expected Quote** | Quoted Hours × Billable Rate | **Benchmark** — what we *should* have quoted |
        | **Base Cost** | Actual Hours × Cost Rate | Internal cost |
        
        ### Key Metrics
        
        | Metric | Formula | What It Tells You |
        |--------|---------|-------------------|
        | **Margin** | Quoted Amount - Base Cost | Are we profitable? |
        | **Quote Gap** | Quoted Amount - Expected Quote | Did we quote at/above/below internal rates? |
        | **Effective Rate/Hr** | Quoted Amount ÷ Actual Hours | Revenue per hour worked |
        
        ### Interpreting Quote Gap
        - **Positive** (+) = Quoted ABOVE internal benchmark (premium pricing ✅)
        - **Negative** (-) = Quoted BELOW internal benchmark (discounting ⚠️)
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
        "💡 Insights", "🔍 Job Diagnosis", "📋 Reconciliation"
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
        
        # REVENUE & MARGIN
        st.subheader("💵 Revenue & Margin")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue (Quoted)", fmt_currency(metrics['total_quoted_amount']),
                  help="Quoted Amount = What we invoice the client")
        c2.metric("Base Cost", fmt_currency(metrics['total_base_cost']),
                  help="Actual Hours × Cost Rate")
        margin_icon = status_icon(metrics['margin_pct'], 35, 20)
        c3.metric(f"{margin_icon} Margin", fmt_currency(metrics['margin']),
                  delta=fmt_pct(metrics['margin_pct']),
                  help="Quoted Amount - Base Cost")
        c4.metric("Margin %", fmt_pct(metrics['margin_pct']),
                  help="Target: 35%+")
        
        # QUOTING ACCURACY (Sanity Check)
        st.subheader("📊 Quoting Sanity Check: Quoted vs What We Should Have Quoted")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Amount", fmt_currency(metrics['total_quoted_amount']),
                  help="What we actually charged")
        c2.metric("Expected Quote", fmt_currency(metrics['total_expected_quote']),
                  help="What we SHOULD have quoted (Quoted Hours × Billable Rate)")
        gap = metrics['quote_gap']
        gap_icon = "✅" if gap >= 0 else "⚠️"
        c3.metric(f"{gap_icon} Quote Gap", fmt_currency(gap),
                  delta=f"{metrics['quote_gap_pct']:+.0f}% vs benchmark",
                  delta_color="normal" if gap >= 0 else "inverse",
                  help="Positive = quoted above internal rates. Negative = underquoting.")
        underquoted = metrics['jobs_underquoted']
        c4.metric("Underquoted Jobs", f"{underquoted} / {metrics['total_jobs']}",
                  help="Jobs quoted below internal rates")
        
        # RATES
        st.subheader("💲 Rate Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Rate/Hr", fmt_rate(metrics['avg_quoted_rate_hr']),
                  help="Quoted Amount ÷ Quoted Hours — what we charged")
        c2.metric("Billable Rate/Hr", fmt_rate(metrics['avg_billable_rate_hr']),
                  help="Internal standard rate we SHOULD charge")
        c3.metric("Effective Rate/Hr", fmt_rate(metrics['avg_effective_rate_hr']),
                  help="Quoted Amount ÷ Actual Hours (drops if hours overrun)")
        c4.metric("Cost Rate/Hr", fmt_rate(metrics['avg_cost_rate_hr']),
                  help="Internal cost per hour")
        
        # PERFORMANCE FLAGS
        st.subheader("⚠️ Performance Flags")
        c1, c2, c3, c4 = st.columns(4)
        loss_icon = status_icon(metrics['loss_rate'], 5, 15, higher_is_better=False)
        c1.metric(f"{loss_icon} Jobs at Loss", f"{metrics['jobs_at_loss']} / {metrics['total_jobs']}",
                  delta=f"{metrics['loss_rate']:.0f}%", delta_color="inverse")
        overrun_icon = status_icon(metrics['overrun_rate'], 30, 50, higher_is_better=False)
        c2.metric(f"{overrun_icon} Hour Overruns", f"{metrics['jobs_over_budget']}",
                  delta=f"{metrics['overrun_rate']:.0f}%", delta_color="inverse")
        c3.metric("Underquoted Jobs", str(metrics['jobs_underquoted']),
                  help="Jobs quoted below internal benchmark")
        c4.metric("Scope Creep Tasks", str(causes['scope_creep']['count']),
                  delta=fmt_currency(causes['scope_creep']['cost']),
                  help="Unquoted tasks = work not in original quote")
        
        st.markdown("---")
        
        # MARGIN BRIDGE
        st.subheader("🌉 Margin Bridge")
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue", fmt_currency(metrics['total_quoted_amount']))
        col2.metric("Cost", fmt_currency(-metrics['total_base_cost']))
        col3.metric("= Margin", fmt_currency(metrics['margin']), delta=fmt_pct(metrics['margin_pct']))
        
        # Waterfall
        bridge_data = pd.DataFrame([
            {"Step": "1. Revenue (Quoted)", "Amount": metrics['total_quoted_amount'], "Color": "Revenue"},
            {"Step": "2. Base Cost", "Amount": -metrics['total_base_cost'], "Color": "Cost"},
            {"Step": "3. Margin", "Amount": metrics['margin'], "Color": "Margin"},
        ])
        
        bridge_chart = alt.Chart(bridge_data).mark_bar().encode(
            x=alt.X("Step:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Amount:Q", title="Amount ($)"),
            color=alt.Color("Color:N", scale=alt.Scale(
                domain=["Revenue", "Cost", "Margin"],
                range=["#3498db", "#e74c3c", "#2ecc71"]
            )),
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
                ["Margin_Pct", "Quote_Gap_Pct", "Quoted_Amount", "Hours_Variance_Pct", "Effective_Rate_Hr"],
                format_func=lambda x: {
                    "Margin_Pct": "Margin %",
                    "Quote_Gap_Pct": "Quote Gap % (Quoting Accuracy)",
                    "Quoted_Amount": "Revenue (Quoted Amount)",
                    "Hours_Variance_Pct": "Hours Variance %",
                    "Effective_Rate_Hr": "Effective Rate/Hr"
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
            if trend_metric == "Margin_Pct":
                rule = alt.Chart(pd.DataFrame({"y": [35]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y:Q")
                trend_chart = trend_chart + rule
            elif trend_metric == "Quote_Gap_Pct":
                rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[5,5]).encode(y="y:Q")
                trend_chart = trend_chart + rule
            
            st.altair_chart(trend_chart, use_container_width=True)
            
            # Quoted vs Expected Quote
            st.subheader("📉 Quoted Amount vs Expected Quote")
            compare_data = monthly_summary.melt(
                id_vars=["Month"], value_vars=["Quoted_Amount", "Expected_Quote"],
                var_name="Type", value_name="Amount"
            )
            compare_data["Type"] = compare_data["Type"].map({
                "Quoted_Amount": "Quoted Amount (Revenue)",
                "Expected_Quote": "Expected Quote (Benchmark)"
            })
            compare_chart = alt.Chart(compare_data).mark_bar().encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Amount:Q"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Quoted Amount (Revenue)", "Expected Quote (Benchmark)"],
                    range=["#2ecc71", "#95a5a6"]
                )),
                xOffset="Type:N",
                tooltip=["Month", "Type", alt.Tooltip("Amount:Q", format="$,.0f")]
            ).properties(height=300)
            st.altair_chart(compare_chart, use_container_width=True)
            
            # Margin trend
            st.subheader("📊 Margin $ and %")
            margin_line = alt.Chart(monthly_summary).mark_line(point=True, strokeWidth=2, color="#2ecc71").encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Margin_Pct:Q", title="Margin %"),
                tooltip=["Month", alt.Tooltip("Margin_Pct:Q", format=".1f"), alt.Tooltip("Margin:Q", format="$,.0f")]
            ).properties(height=300)
            st.altair_chart(margin_line, use_container_width=True)
            
            # Department trends
            if selected_dept == "All Departments" and len(monthly_by_dept) > 0:
                st.subheader("🏢 Margin % by Department")
                dept_trend = alt.Chart(monthly_by_dept).mark_line(point=True).encode(
                    x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Margin_Pct:Q", title="Margin %"),
                    color="Department:N",
                    tooltip=["Month", "Department", alt.Tooltip("Margin_Pct:Q", format=".0f")]
                ).properties(height=350)
                st.altair_chart(dept_trend, use_container_width=True)
            
            with st.expander("📋 Monthly Data Table"):
                st.dataframe(monthly_summary[[
                    "Month", "Job_Count", "Quoted_Amount", "Expected_Quote", "Quote_Gap",
                    "Base_Cost", "Margin", "Margin_Pct", "Hours_Variance_Pct"
                ]], use_container_width=True)
    
    # =========================================================================
    # TAB 3: DRILL-DOWN
    # =========================================================================
    with tab3:
        st.header("🏢 Hierarchical Analysis")
        
        # Department
        st.subheader("Level 1: Department Performance")
        if len(dept_summary) > 0:
            dept_chart = alt.Chart(dept_summary).mark_bar().encode(
                y=alt.Y("Department:N", sort="-x"),
                x=alt.X("Margin_Pct:Q", title="Margin %"),
                color=alt.condition(alt.datum.Margin_Pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")),
                tooltip=["Department",
                         alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
                         alt.Tooltip("Margin:Q", format="$,.0f", title="Margin $"),
                         alt.Tooltip("Quote_Gap:Q", format="$,.0f", title="Quote Gap")]
            ).properties(height=max(200, len(dept_summary) * 40))
            
            rule = alt.Chart(pd.DataFrame({"x": [35]})).mark_rule(color="orange", strokeDash=[3,3]).encode(x="x:Q")
            st.altair_chart(dept_chart + rule, use_container_width=True)
            
            with st.expander("Department Details"):
                st.dataframe(dept_summary[[
                    "Department", "Job_Count", "Quoted_Amount", "Expected_Quote", "Quote_Gap",
                    "Base_Cost", "Margin", "Margin_Pct", "Hours_Variance_Pct"
                ]].style.format({
                    "Quoted_Amount": "${:,.0f}", "Expected_Quote": "${:,.0f}",
                    "Quote_Gap": "${:,.0f}", "Base_Cost": "${:,.0f}",
                    "Margin": "${:,.0f}", "Margin_Pct": "{:.1f}%",
                    "Hours_Variance_Pct": "{:+.0f}%"
                }), use_container_width=True)
        
        st.markdown("---")
        
        # Product
        st.subheader("Level 2: Product Performance")
        sel_dept_drill = st.selectbox("Filter by Department", ["All"] + sorted(dept_summary["Department"].unique().tolist()), key="d1")
        prod_f = product_summary if sel_dept_drill == "All" else product_summary[product_summary["Department"] == sel_dept_drill]
        
        if len(prod_f) > 0:
            prod_chart = alt.Chart(prod_f.head(15)).mark_bar().encode(
                y=alt.Y("Product:N", sort="-x"),
                x=alt.X("Margin_Pct:Q", title="Margin %"),
                color=alt.condition(alt.datum.Margin_Pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")),
                tooltip=["Product", "Department",
                         alt.Tooltip("Margin_Pct:Q", format=".1f"),
                         alt.Tooltip("Quote_Gap:Q", format="$,.0f")]
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
        show_underquoted = c2.checkbox("Underquoted", key="ju")
        show_overrun = c3.checkbox("Hour Overrun", key="jo")
        sort_by = c4.selectbox("Sort", ["Margin", "Quote_Gap", "Hours_Variance_Pct", "Margin_Pct"], key="js")
        
        if show_loss:
            jobs_f = jobs_f[jobs_f["Is_Loss"]]
        if show_underquoted:
            jobs_f = jobs_f[jobs_f["Is_Underquoted"]]
        if show_overrun:
            jobs_f = jobs_f[jobs_f["Is_Overrun"]]
        
        jobs_disp = jobs_f.sort_values(sort_by, ascending=sort_by in ["Margin", "Quote_Gap"]).head(25)
        
        if len(jobs_disp) > 0:
            cols = ["Job_No", "Job_Name", "Client", "Month",
                    "Quoted_Amount", "Expected_Quote", "Quote_Gap", "Base_Cost",
                    "Margin", "Margin_Pct", "Hours_Variance_Pct"]
            st.dataframe(jobs_disp[cols].style.format({
                "Quoted_Amount": "${:,.0f}", "Expected_Quote": "${:,.0f}",
                "Quote_Gap": "${:,.0f}", "Base_Cost": "${:,.0f}",
                "Margin": "${:,.0f}", "Margin_Pct": "{:.1f}%",
                "Hours_Variance_Pct": "{:+.0f}%"
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
                margin_icon = "🟢" if job_info["Margin"] > 0 else "🔴"
                c1.metric(f"{margin_icon} Margin", fmt_currency(job_info["Margin"]),
                          delta=fmt_pct(job_info["Margin_Pct"]))
                gap_icon = "✅" if job_info["Quote_Gap"] >= 0 else "⚠️"
                c2.metric(f"{gap_icon} Quote Gap", fmt_currency(job_info["Quote_Gap"]),
                          help="Quoted vs Expected Quote")
                c3.metric("Hours Var", f"{job_info['Hours_Variance_Pct']:+.0f}%")
                c4.metric("Effective Rate", fmt_rate(job_info["Effective_Rate_Hr"]))
                
                if len(tasks) > 0:
                    st.markdown("#### Tasks")
                    task_cols = ["Task_Name", "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                                 "Quoted_Amount", "Expected_Quote", "Quote_Gap", "Base_Cost",
                                 "Margin", "Is_Unquoted"]
                    task_disp = tasks[task_cols].copy()
                    task_disp["Is_Unquoted"] = task_disp["Is_Unquoted"].map({True: "⚠️ SCOPE CREEP", False: ""})
                    task_disp.columns = ["Task", "Quoted Hrs", "Actual Hrs", "Hrs Var",
                                         "Quoted $", "Expected $", "Quote Gap", "Cost",
                                         "Margin", "Flag"]
                    st.dataframe(task_disp.style.format({
                        "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
                        "Quoted $": "${:,.0f}", "Expected $": "${:,.0f}",
                        "Quote Gap": "${:,.0f}", "Cost": "${:,.0f}",
                        "Margin": "${:,.0f}"
                    }), use_container_width=True)
    
    # =========================================================================
    # TAB 4: INSIGHTS
    # =========================================================================
    with tab4:
        st.header("💡 Profitability Insights")
        
        # Quoting Issues
        if insights["quoting_issues"]:
            st.subheader("📊 Quoting Accuracy Issues")
            for i in insights["quoting_issues"]:
                st.markdown(i)
        
        # Scope Issues
        if insights["scope_issues"]:
            st.subheader("📋 Scope & Hours Issues")
            for i in insights["scope_issues"]:
                st.markdown(i)
        
        # Rate Issues
        if insights["rate_issues"]:
            st.subheader("💲 Rate Issues")
            for i in insights["rate_issues"]:
                st.markdown(i)
        
        # Margin Drivers
        if insights["margin_drivers"]:
            st.subheader("📊 Margin Drivers")
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
            st.subheader("📉 Underquoted Jobs")
            underquoted = get_underquoted_jobs(job_summary, -500).head(10)
            if len(underquoted) > 0:
                st.metric("Total Quote Gap", fmt_currency(underquoted["Quote_Gap"].sum()))
                for _, j in underquoted.iterrows():
                    st.markdown(f"**{str(j['Job_Name'])[:35]}** — ${abs(j['Quote_Gap']):,.0f} below internal rates")
            else:
                st.success("No significant underquoting!")
        
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
        
        # Loss Making Jobs
        st.subheader("💸 Loss-Making Jobs")
        losses = get_loss_making_jobs(job_summary).head(10)
        if len(losses) > 0:
            for _, j in losses.iterrows():
                st.markdown(f"**{str(j['Job_Name'])[:40]}** ({j['Job_No']}) — ${j['Margin']:,.0f}")
                reasons = []
                if j["Hours_Variance_Pct"] > 20:
                    reasons.append(f"Hours +{j['Hours_Variance_Pct']:.0f}%")
                if j["Quote_Gap"] < -500:
                    reasons.append("Underquoted")
                if reasons:
                    st.caption(f"  Drivers: {', '.join(reasons)}")
        else:
            st.success("No loss-making jobs!")
    
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
            c1.metric("Revenue", fmt_currency(job_row['Quoted_Amount']))
            c2.metric("Cost", fmt_currency(job_row['Base_Cost']))
            margin_icon = "🟢" if job_row['Margin'] > 0 else "🔴"
            c3.metric(f"{margin_icon} Margin", fmt_currency(job_row['Margin']))
            gap_icon = "✅" if job_row['Quote_Gap'] >= 0 else "⚠️"
            c4.metric(f"{gap_icon} Quote Gap", fmt_currency(job_row['Quote_Gap']))
            c5.metric("Hours Var", f"{job_row['Hours_Variance_Pct']:+.0f}%")
            
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
                
                unquoted_tasks = job_tasks[job_tasks['Is_Unquoted']]
                overrun_tasks = job_tasks[job_tasks['Is_Overrun'] & ~job_tasks['Is_Unquoted']]
                underquoted_tasks = job_tasks[job_tasks['Is_Underquoted']]
                
                if len(unquoted_tasks) > 0:
                    st.markdown("**⚠️ Unquoted Tasks (Scope Creep):**")
                    for _, t in unquoted_tasks.iterrows():
                        st.markdown(f"- {t['Task_Name']}: {t['Actual_Hours']:.0f} hrs, ${t['Base_Cost']:,.0f} cost")
                
                if len(overrun_tasks) > 0:
                    st.markdown("**⏱️ Hour Overruns:**")
                    for _, t in overrun_tasks.iterrows():
                        st.markdown(f"- {t['Task_Name']}: {t['Hours_Variance']:+.0f} hrs over")
                
                if len(underquoted_tasks) > 0:
                    st.markdown("**📉 Underquoted Tasks:**")
                    for _, t in underquoted_tasks.iterrows():
                        st.markdown(f"- {t['Task_Name']}: ${abs(t['Quote_Gap']):,.0f} below internal rates")
    
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
        | **Quoted Amount** | From data | What we charge = REVENUE |
        | **Expected Quote** | Quoted Hours × Billable Rate | What we SHOULD have quoted |
        | **Base Cost** | Actual Hours × Cost Rate | Internal cost |
        | **Margin** | Quoted Amount - Base Cost | Profit |
        | **Quote Gap** | Quoted - Expected | + = premium, - = underquoted |
        | **Effective Rate/Hr** | Quoted Amount ÷ Actual Hours | Revenue per hour worked |
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