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
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
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

def delta_color(val):
    return "🟢" if val > 0 else ("🔴" if val < 0 else "⚪")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("📈 Job Profitability Trends")
    st.markdown("*Month-on-Month Analysis with Narrative Insights*")
    
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
    
    default_fy = max(fy_list)
    selected_fy = st.sidebar.selectbox(
        "📅 Fiscal Year", fy_list,
        index=fy_list.index(default_fy) if default_fy in fy_list else 0,
        format_func=lambda x: f"FY{str(x)[-2:]}"
    )
    
    # Department
    dept_list = get_available_departments(df_parsed)
    dept_options = ["All Departments"] + dept_list
    selected_dept = st.sidebar.selectbox("🏢 Department", dept_options)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary", "📈 Monthly Trends", "🏢 Hierarchy Drill-Down",
        "💡 Insights", "🔍 Reconciliation"
    ])
    
    # =========================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # =========================================================================
    with tab1:
        st.header(f"FY{str(selected_fy)[-2:]} Executive Summary")
        
        if insights["headline"]:
            st.markdown("### 📌 Key Headlines")
            for h in insights["headline"]:
                st.markdown(f"> {h}")
        
        st.markdown("---")
        
        # Revenue & Margins
        st.subheader("💰 Revenue & Margins")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Quoted Amount", fmt_currency(metrics['total_quoted_amount']))
        c2.metric("Billable Value", fmt_currency(metrics['total_billable_value']),
                  delta=f"{metrics['revenue_realization_pct']:.0f}% realized")
        c3.metric("Base Cost", fmt_currency(metrics['total_base_cost']))
        c4.metric("Actual Margin", fmt_currency(metrics['total_profit']),
                  delta=fmt_pct(metrics['overall_billable_margin_pct']))
        margin_var = metrics.get('overall_margin_variance', 0)
        c5.metric("Margin vs Quote", fmt_currency(margin_var),
                  delta_color="normal" if margin_var >= 0 else "inverse")
        
        # Rates
        st.subheader("📊 Rate Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Rate/Hr", fmt_rate(metrics['avg_quoted_rate_hr']))
        c2.metric("Billable Rate/Hr", fmt_rate(metrics['avg_billable_rate_hr']))
        c3.metric("Cost Rate/Hr", fmt_rate(metrics['avg_cost_rate_hr']))
        spread = metrics['avg_billable_rate_hr'] - metrics['avg_cost_rate_hr']
        c4.metric("Rate Spread", fmt_rate(spread) if spread > 0 else "N/A")
        
        # Performance
        st.subheader("⚡ Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jobs Over Budget", f"{metrics['jobs_over_budget']} / {metrics['total_jobs']}",
                  delta=f"{metrics['overrun_rate']:.0f}%", delta_color="inverse")
        c2.metric("Jobs at Loss", str(metrics['jobs_at_loss']),
                  delta=f"{metrics['loss_rate']:.0f}%", delta_color="inverse")
        c3.metric("Hours Variance", f"{metrics['hours_variance']:+,.0f}",
                  delta=f"{metrics['hours_variance_pct']:+.0f}%", delta_color="inverse")
        c4.metric("Unbilled Hours", fmt_hours(causes['unbilled']['hours']))
        
        st.markdown("---")
        
        # Margin Bridge
        st.subheader("🌉 Margin Bridge")
        quoted_margin = metrics['total_quoted_amount'] - metrics['total_base_cost']
        actual_margin = metrics['total_profit']
        revenue_var = metrics['total_billable_value'] - metrics['total_quoted_amount']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Quoted Margin", fmt_currency(quoted_margin),
                    delta=fmt_pct(metrics['overall_quoted_margin_pct']))
        col2.metric("Revenue Variance", fmt_currency(revenue_var),
                    delta="Above" if revenue_var >= 0 else "Below",
                    delta_color="normal" if revenue_var >= 0 else "inverse")
        col3.metric("Actual Margin", fmt_currency(actual_margin),
                    delta=f"{delta_color(actual_margin - quoted_margin)} {fmt_currency(actual_margin - quoted_margin)} vs quoted")
        
        # Bridge chart
        bridge_data = pd.DataFrame([
            {"Step": "Quoted Margin", "Value": quoted_margin},
            {"Step": "Revenue Variance", "Value": revenue_var},
            {"Step": "Actual Margin", "Value": actual_margin},
        ])
        bridge_chart = alt.Chart(bridge_data).mark_bar(size=60).encode(
            x=alt.X("Step:N", sort=["Quoted Margin", "Revenue Variance", "Actual Margin"],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title="Amount ($)"),
            color=alt.condition(alt.datum.Value >= 0, alt.value("#2ecc71"), alt.value("#e74c3c")),
            tooltip=["Step", alt.Tooltip("Value:Q", format="$,.0f")]
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
            trend_metric = st.selectbox(
                "Select Metric",
                ["Actual_Margin_Pct", "Actual_Margin", "Billable_Value", "Quoted_Amount",
                 "Hours_Variance_Pct", "Realization_Pct", "Job_Count"],
                format_func=lambda x: x.replace("_", " ")
            )
            
            # Main trend
            st.subheader(f"📊 {trend_metric.replace('_', ' ')} by Month")
            trend_chart = alt.Chart(monthly_summary).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(f"{trend_metric}:Q"),
                tooltip=["Month", alt.Tooltip(f"{trend_metric}:Q", format=",.1f")]
            ).properties(height=350)
            
            if "Margin_Pct" in trend_metric:
                rule = alt.Chart(pd.DataFrame({"y": [30]})).mark_rule(color="orange", strokeDash=[5,5]).encode(y="y:Q")
                trend_chart = trend_chart + rule
            st.altair_chart(trend_chart, use_container_width=True)
            
            # Quoted vs Billable vs Cost
            st.subheader("📉 Quoted vs Billable vs Cost")
            compare_data = monthly_summary.melt(
                id_vars=["Month"], value_vars=["Quoted_Amount", "Billable_Value", "Base_Cost"],
                var_name="Metric", value_name="Amount"
            )
            compare_data["Metric"] = compare_data["Metric"].map({
                "Quoted_Amount": "Quoted", "Billable_Value": "Billable", "Base_Cost": "Cost"
            })
            compare_chart = alt.Chart(compare_data).mark_bar().encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Amount:Q"),
                color=alt.Color("Metric:N", scale=alt.Scale(
                    domain=["Quoted", "Billable", "Cost"], range=["#3498db", "#2ecc71", "#e74c3c"]
                )),
                xOffset="Metric:N",
                tooltip=["Month", "Metric", alt.Tooltip("Amount:Q", format="$,.0f")]
            ).properties(height=350)
            st.altair_chart(compare_chart, use_container_width=True)
            
            # Margin trend
            st.subheader("📊 Margin: Quoted vs Actual")
            margin_data = monthly_summary.melt(
                id_vars=["Month"], value_vars=["Quoted_Margin_Pct", "Actual_Margin_Pct"],
                var_name="Type", value_name="Margin_Pct"
            )
            margin_data["Type"] = margin_data["Type"].map({
                "Quoted_Margin_Pct": "Quoted", "Actual_Margin_Pct": "Actual"
            })
            margin_chart = alt.Chart(margin_data).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Margin_Pct:Q", title="Margin %"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Quoted", "Actual"], range=["#3498db", "#2ecc71"])),
                tooltip=["Month", "Type", alt.Tooltip("Margin_Pct:Q", format=".1f")]
            ).properties(height=300)
            st.altair_chart(margin_chart, use_container_width=True)
            
            # Dept trend
            if selected_dept == "All Departments" and len(monthly_by_dept) > 0:
                st.subheader("🏢 Margin by Department")
                dept_trend = alt.Chart(monthly_by_dept).mark_line(point=True).encode(
                    x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Actual_Margin_Pct:Q", title="Margin %"),
                    color="Department:N",
                    tooltip=["Month", "Department", alt.Tooltip("Actual_Margin_Pct:Q", format=".1f")]
                ).properties(height=350)
                st.altair_chart(dept_trend, use_container_width=True)
            
            with st.expander("📋 Monthly Data"):
                st.dataframe(monthly_summary, use_container_width=True)
    
    # =========================================================================
    # TAB 3: HIERARCHY DRILL-DOWN
    # =========================================================================
    with tab3:
        st.header("🏢 Hierarchical Analysis")
        
        # Department
        st.subheader("Level 1: Department")
        if len(dept_summary) > 0:
            dept_chart = alt.Chart(dept_summary).mark_bar().encode(
                y=alt.Y("Department:N", sort="-x"),
                x=alt.X("Billable_Margin_Pct:Q", title="Actual Margin %"),
                color=alt.condition(alt.datum.Billable_Margin_Pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")),
                tooltip=["Department", alt.Tooltip("Billable_Margin_Pct:Q", format=".1f"),
                         alt.Tooltip("Margin_Variance:Q", format="$,.0f")]
            ).properties(height=max(200, len(dept_summary) * 40))
            st.altair_chart(dept_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Product
        st.subheader("Level 2: Product")
        sel_dept_drill = st.selectbox("Select Department", ["All"] + sorted(dept_summary["Department"].unique().tolist()), key="d1")
        prod_f = product_summary if sel_dept_drill == "All" else product_summary[product_summary["Department"] == sel_dept_drill]
        
        if len(prod_f) > 0:
            prod_chart = alt.Chart(prod_f.head(15)).mark_bar().encode(
                y=alt.Y("Product:N", sort="-x"),
                x=alt.X("Billable_Margin_Pct:Q", title="Actual Margin %"),
                color=alt.condition(alt.datum.Billable_Margin_Pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")),
                tooltip=["Product", "Department", alt.Tooltip("Billable_Margin_Pct:Q", format=".1f")]
            ).properties(height=max(200, min(len(prod_f), 15) * 30))
            st.altair_chart(prod_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Job
        st.subheader("Level 3: Job")
        sel_prod = st.selectbox("Select Product", ["All"] + sorted(prod_f["Product"].unique().tolist()), key="p1")
        jobs_f = job_summary.copy()
        if sel_dept_drill != "All":
            jobs_f = jobs_f[jobs_f["Department"] == sel_dept_drill]
        if sel_prod != "All":
            jobs_f = jobs_f[jobs_f["Product"] == sel_prod]
        
        c1, c2, c3 = st.columns(3)
        show_loss = c1.checkbox("Loss only", key="jl")
        show_erosion = c2.checkbox("Erosion >10%", key="je")
        sort_by = c3.selectbox("Sort", ["Margin_Variance", "Billable_Margin_Pct", "Profit"], key="js")
        
        if show_loss:
            jobs_f = jobs_f[jobs_f["Is_Loss"]]
        if show_erosion:
            jobs_f = jobs_f[jobs_f["Margin_Erosion"] > 10]
        
        jobs_disp = jobs_f.sort_values(sort_by, ascending=sort_by in ["Billable_Margin_Pct", "Profit", "Margin_Variance"]).head(25)
        
        if len(jobs_disp) > 0:
            cols = ["Job_No", "Job_Name", "Client", "Month", "Quoted_Amount", "Billable_Value", "Base_Cost",
                    "Quoted_Margin", "Actual_Margin", "Margin_Variance", "Billable_Margin_Pct", "Margin_Erosion"]
            st.dataframe(jobs_disp[cols].style.format({
                "Quoted_Amount": "${:,.0f}", "Billable_Value": "${:,.0f}", "Base_Cost": "${:,.0f}",
                "Quoted_Margin": "${:,.0f}", "Actual_Margin": "${:,.0f}", "Margin_Variance": "${:+,.0f}",
                "Billable_Margin_Pct": "{:.1f}%", "Margin_Erosion": "{:+.1f}%"
            }), use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Task
        st.subheader("Level 4: Task")
        job_opts = jobs_disp.apply(lambda r: f"{r['Job_No']} — {r['Job_Name'][:35]}", axis=1).tolist()
        if job_opts:
            sel_job = st.selectbox("Select Job", ["-- Select --"] + job_opts, key="tj")
            if sel_job != "-- Select --":
                job_no = sel_job.split(" — ")[0]
                job_info = jobs_disp[jobs_disp["Job_No"] == job_no].iloc[0]
                tasks = task_summary[task_summary["Job_No"] == job_no]
                
                st.markdown(f"### 📁 {job_info['Job_Name']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Quoted Margin", fmt_currency(job_info["Quoted_Margin"]))
                c2.metric("Actual Margin", fmt_currency(job_info["Actual_Margin"]))
                c3.metric("Margin Variance", fmt_currency(job_info["Margin_Variance"]))
                c4.metric("Hours Variance", f"{job_info['Hours_Variance']:+,.0f}")
                
                if len(tasks) > 0:
                    st.dataframe(tasks[["Task_Name", "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                                        "Quoted_Amount", "Billable_Value", "Base_Cost",
                                        "Quoted_Margin", "Actual_Margin", "Margin_Variance", "Is_Unquoted"]],
                                 use_container_width=True)
    
    # =========================================================================
    # TAB 4: INSIGHTS
    # =========================================================================
    with tab4:
        st.header("💡 Insights & Narratives")
        
        if insights["margin_drivers"]:
            st.subheader("📊 Margin Drivers")
            for i in insights["margin_drivers"]:
                st.markdown(i)
        
        if insights["quoting_accuracy"]:
            st.subheader("🎯 Quoting Accuracy")
            for i in insights["quoting_accuracy"]:
                st.markdown(i)
        
        if insights["trends"]:
            st.subheader("📈 Trends")
            for i in insights["trends"]:
                st.markdown(i)
        
        if insights["action_items"]:
            st.subheader("🚨 Action Items")
            for a in insights["action_items"]:
                st.markdown(f"- {a}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💸 Margin Erosion")
            erosion = get_margin_erosion_jobs(job_summary, 10).head(10)
            if len(erosion) > 0:
                for _, j in erosion.iterrows():
                    st.markdown(f"**{j['Job_Name'][:40]}** — Erosion: {j['Margin_Erosion']:+.1f}%")
            else:
                st.success("No significant erosion!")
        
        with c2:
            st.subheader("📋 Scope Creep")
            unquoted = get_unquoted_tasks(task_summary).head(10)
            if len(unquoted) > 0:
                st.metric("Unquoted Cost", fmt_currency(unquoted["Base_Cost"].sum()))
                for _, t in unquoted.iterrows():
                    st.markdown(f"• {t['Task_Name'][:30]} ({t['Actual_Hours']:.0f} hrs)")
            else:
                st.success("No unquoted work!")
        
        st.markdown("---")
        st.subheader("📝 Executive Narrative")
        narrative = f"""
        For **FY{str(selected_fy)[-2:]}**, quoted revenue was **{fmt_currency(metrics['total_quoted_amount'])}** 
        with **{metrics['revenue_realization_pct']:.0f}%** realized ({fmt_currency(metrics['total_billable_value'])}).
        
        After costs of **{fmt_currency(metrics['total_base_cost'])}**, actual margin is **{metrics['overall_billable_margin_pct']:.1f}%** 
        ({fmt_currency(metrics['total_profit'])}).
        """
        st.markdown(narrative)
    
    # =========================================================================
    # TAB 5: RECONCILIATION
    # =========================================================================
    with tab5:
        st.header("🔍 Reconciliation")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw", f"{recon['raw_records']:,}")
        c2.metric("Filtered", f"{recon['final_records']:,}")
        c3.metric("Excluded", f"{recon['raw_records'] - recon['final_records']:,}")
        
        st.subheader("Exclusions")
        st.dataframe(pd.DataFrame({
            "Filter": ["SG Allocation", "Non-Billable", "Other FY", "Other Dept"],
            "Excluded": [recon["excluded_sg_allocation"], recon["excluded_non_billable"],
                         recon["excluded_other_fy"], recon["excluded_other_dept"]]
        }), use_container_width=True, hide_index=True)
        
        st.subheader("Totals")
        st.dataframe(pd.DataFrame({
            "Metric": list(recon["totals"].keys()),
            "Value": [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in recon["totals"].values()]
        }), use_container_width=True, hide_index=True)
        
        st.subheader("📐 Metric Definitions")
        for k, d in METRIC_DEFINITIONS.items():
            with st.expander(d['name']):
                st.markdown(f"**Formula:** `{d['formula']}`\n\n{d['desc']}")
    
    st.markdown("---")
    st.caption(f"Job Profitability Trends | FY{str(selected_fy)[-2:]} | {selected_dept} | {recon['final_records']:,} records")


if __name__ == "__main__":
    main()