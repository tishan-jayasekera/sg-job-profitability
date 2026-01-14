
"""
Job Profitability — Quote vs Ratecard vs Cost (Streamlit)
========================================================

This app is deliberately "consulting-style":
- Clear definitions up front
- Executive summary first, then trends, then drill-down, then diagnostics
- Quote is the commercial benchmark (client quote)
- Ratecard value is internal (NOT revenue)

Run:
  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st
import altair as alt

import analysis as an


# =============================================================================
# PAGE CONFIG + THEME
# =============================================================================

st.set_page_config(
    page_title="Job Profitability — Quote vs Ratecard vs Cost",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Subtle exec look */
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.kpi-card {
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(250,250,252,0.6);
}
.kpi-label { font-size: 0.85rem; opacity: 0.75; }
.kpi-value { font-size: 1.6rem; font-weight: 650; line-height: 1.1; }
.kpi-sub { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.75rem;
  border: 1px solid rgba(49,51,63,0.2);
  opacity: 0.9;
}
hr { margin: 1rem 0 1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# IO
# =============================================================================

DEFAULT_DATA_PATHS = [
    "data/Quoted_Task_Report_FY26.xlsx",
    "Quoted_Task_Report_FY26.xlsx",
]

@st.cache_data(show_spinner="Loading and parsing dataset...")
def load_and_parse(path: str) -> Dict[str, object]:
    df_raw = an.load_raw_data(path, sheet="Data")
    df_parsed, parse_report = an.clean_and_parse(df_raw)
    return {"raw": df_raw, "parsed": df_parsed, "parse_report": parse_report}

@st.cache_data(show_spinner="Computing rollups...")
def compute_rollups(df_filtered: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    task = an.compute_task_summary(df_filtered)
    job = an.compute_job_summary(df_filtered)
    product = an.compute_product_summary(df_filtered)
    dept = an.compute_department_summary(df_filtered)
    category = an.compute_category_summary(df_filtered)
    monthly = an.compute_monthly_summary(df_filtered)
    return {"task": task, "job": job, "category": category, "product": product, "dept": dept, "monthly": monthly}


# =============================================================================
# UI HELPERS
# =============================================================================

def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def fmt_money(x: float) -> str:
    return f"${x:,.0f}"

def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"

def section_header(title: str, subtitle: str = ""):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

def get_existing_default_path() -> Optional[str]:
    for p in DEFAULT_DATA_PATHS:
        if Path(p).exists():
            return p
    return None



def st_df(df, **kwargs):
    """Streamlit dataframe with forward-compatible width argument."""
    try:
        st.dataframe(df, width='stretch', **kwargs)
    except TypeError:
        # Backwards compatible

def st_chart(chart, **kwargs):
    """Altair chart helper (width compat)."""
    try:
        st.altair_chart(chart, width='stretch', **kwargs)
    except TypeError:

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Controls")

data_file = st.sidebar.file_uploader("Upload Excel dataset", type=["xlsx"])
default_path = get_existing_default_path()

path_in_use: Optional[str] = None
temp_path: Optional[str] = None

if data_file is not None:
    temp_path = str(Path("data_uploaded.xlsx"))
    with open(temp_path, "wb") as f:
        f.write(data_file.getbuffer())
    path_in_use = temp_path
elif default_path is not None:
    path_in_use = default_path
else:
    st.warning("Upload the Excel file to begin (Data sheet expected).")
    st.stop()

date_anchor = st.sidebar.radio(
    "Month-on-month anchor (approx.)",
    options=["Task Start Date", "Task Completed Date", "Job Start Date"],
    index=0,
    help="This dataset has no true timesheet entry date. MoM trends are anchored to the selected date field."
)

exclude_alloc = st.sidebar.checkbox(
    'Exclude "[Job Task] Name" = Social Garden Invoice Allocation',
    value=True
)

require_base_rate = st.sidebar.checkbox("Require Base Rate > 0", value=False)
require_billable_rate = st.sidebar.checkbox("Require Billable Rate > 0", value=False)

with st.sidebar.expander("Definitions (click to expand)", expanded=False):
    st.markdown(
        """
**Quoted Amount** = client quote value allocated to a task (commercial benchmark; expected invoiced revenue).  
**Invoiced Amount** = revenue invoiced to date.  
**Ratecard Value (Internal)** = actual hours × internal billable rate (control metric; NOT recognised revenue).  
**Base Cost** = actual hours × base rate (internal labour cost).  
        """
    )


# =============================================================================
# LOAD + PARSE
# =============================================================================

bundle = load_and_parse(path_in_use)
df_parsed: pd.DataFrame = bundle["parsed"]
parse_report: an.ParseReport = bundle["parse_report"]

# FY selection depends on date anchor
fys = an.get_available_fiscal_years(df_parsed, date_anchor=date_anchor)
if not fys:
    st.error("Could not compute fiscal years. Check date fields in dataset.")
    st.stop()

default_fy = max(fys)

fiscal_year = st.sidebar.selectbox(
    "Financial year (AU FY Jul–Jun)",
    options=fys,
    index=fys.index(default_fy),
    format_func=lambda y: an.fy_label(y),
)

# Filters based on df after FY anchoring
df_tmp = an.add_time_keys(df_parsed, anchor=date_anchor)
available_depts = sorted(df_tmp["Department"].dropna().unique().tolist())
available_products = sorted(df_tmp["Product"].dropna().unique().tolist())
available_categories = sorted(df_tmp["[Job] Category"].dropna().unique().tolist()) if "[Job] Category" in df_tmp.columns else []

departments = st.sidebar.multiselect("Department", options=available_depts, default=[])
products = st.sidebar.multiselect("Product", options=available_products, default=[])
categories = st.sidebar.multiselect("Job Category", options=available_categories, default=[])

df_filtered, filter_report = an.apply_filters(
    df_parsed,
    fiscal_year=int(fiscal_year),
    departments=departments or None,
    products=products or None,
    categories=categories or None,
    exclude_sg_allocation=exclude_alloc,
    require_base_rate=require_base_rate,
    require_billable_rate=require_billable_rate,
    date_anchor=date_anchor,
)

rollups = compute_rollups(df_filtered)
dept_df = rollups["dept"]
product_df = rollups["product"]
job_df = rollups["job"]
category_df = rollups["category"]
task_df = rollups["task"]
monthly_df = rollups["monthly"]

portfolio_metrics = an.summarize_portfolio(job_df)
headlines = an.generate_headlines(portfolio_metrics)
tops = an.top_lists(job_df, task_df, n=12)


# =============================================================================
# HEADER
# =============================================================================

st.title("📈 Job Profitability")
st.caption(
    f"{an.fy_label(int(fiscal_year))} | "
    f"Anchor: {date_anchor} | "
    f"Records used: {filter_report.final_records:,} (excluded SG allocation: {filter_report.excluded_sg_allocation:,})"
)

for h in headlines[:3]:
    st.markdown(f"- {h}")

st.markdown("---")


# =============================================================================
# TABS
# =============================================================================

tab_exec, tab_trends, tab_drill, tab_insights, tab_recon = st.tabs(
    ["Executive Summary", "Trends", "Drilldown", "Insights", "Reconciliation"]
)


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

with tab_exec:
    section_header("Executive summary", "Quote is the commercial benchmark. Ratecard value is an internal control metric.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Quoted (client)", fmt_money(portfolio_metrics["quoted_amount"]), "Allocated across tasks")
    with c2: kpi("Invoiced to date", fmt_money(portfolio_metrics["invoiced_amount"]), f"Realisation {fmt_pct(portfolio_metrics['realisation_pct'])}")
    with c3: kpi("Base cost", fmt_money(portfolio_metrics["base_cost"]), "Actual hours × base rate")
    with c4: kpi("Quote margin", fmt_pct(portfolio_metrics["quote_margin_pct"]), fmt_money(portfolio_metrics["quote_margin"]))
    with c5: kpi("Pricing adequacy", fmt_pct(portfolio_metrics["pricing_adequacy_pct"]), "Quote ÷ internal ratecard")

    st.markdown("")

    c6, c7, c8 = st.columns(3)
    with c6: kpi("Jobs at loss (quote lens)", f"{portfolio_metrics['jobs_at_loss_on_quote']:,}", "Quote < cost")
    with c7: kpi("Jobs at loss (invoiced lens)", f"{portfolio_metrics['jobs_at_loss_invoiced']:,}", "Invoiced < cost")
    with c8: kpi("Jobs with hour overruns", f"{portfolio_metrics['jobs_overrun']:,}", "Actual hours > quoted hours")

    st.markdown("")

    section_header("Where to look first", "Top offenders and structural issues.")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Worst quote-margin jobs (Quote - Cost)**")
    with colB:
        st.markdown("**Most under-priced jobs vs internal ratecard**")

    st.markdown("")

    section_header("Operating interpretation", "How to read this dashboard.")
    st.markdown(
        """
- **Quote margin** answers: *“Was the client quote sufficient for the work actually delivered?”*  
- **Pricing adequacy** answers: *“Did we price above/below our internal ratecard for the hours delivered?”*  
- **Realisation** answers: *“How much of the quote has actually been invoiced?”* (Write-offs / timing / disputes)  
- **Invoiced margin** answers: *“Given what we’ve invoiced so far, are we profitable today?”*  
        """
    )


# =============================================================================
# TRENDS
# =============================================================================

with tab_trends:
    section_header("Month-on-month trends", "Anchored to the selected date field (approximation).")

    m = monthly_df.copy()
    if m.empty:
        st.info("No rows available for the selected filters.")
    else:
        # Keep only months for selected FY, in order
        m = m.sort_values("FY_Month_Key")

        base = m[["FY_Month_Key", "FY_Month_Label", "Quoted_Amount", "Invoiced_Amount", "Base_Cost", "Ratecard_Value"]].copy()

        long = base.melt(
            id_vars=["FY_Month_Key", "FY_Month_Label"],
            value_vars=["Quoted_Amount", "Invoiced_Amount", "Base_Cost", "Ratecard_Value"],
            var_name="Metric",
            value_name="Value",
        )

        metric_labels = {
            "Quoted_Amount": "Quoted (client)",
            "Invoiced_Amount": "Invoiced",
            "Base_Cost": "Base cost",
            "Ratecard_Value": "Ratecard value (internal)",
        }
        long["Metric"] = long["Metric"].map(metric_labels)

        chart = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("FY_Month_Label:N", sort=alt.SortField("FY_Month_Key", order="ascending"), title="Fiscal month"),
            y=alt.Y("Value:Q", title="Value ($)"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["FY_Month_Label", "Metric", alt.Tooltip("Value:Q", format=",.0f")]
        ).properties(height=380)


        st.markdown("")

        # Margin trend (quote lens) month by month
        m2 = m[["FY_Month_Key", "FY_Month_Label", "Quote_Margin_Pct", "Invoiced_Margin_Pct", "Pricing_Adequacy_Pct", "Realisation_Pct"]].copy()
        long2 = m2.melt(
            id_vars=["FY_Month_Key", "FY_Month_Label"],
            value_vars=["Quote_Margin_Pct", "Invoiced_Margin_Pct", "Pricing_Adequacy_Pct", "Realisation_Pct"],
            var_name="Metric",
            value_name="Value",
        )

        metric_labels2 = {
            "Quote_Margin_Pct": "Quote margin %",
            "Invoiced_Margin_Pct": "Invoiced margin %",
            "Pricing_Adequacy_Pct": "Pricing adequacy %",
            "Realisation_Pct": "Realisation %",
        }
        long2["Metric"] = long2["Metric"].map(metric_labels2)

        chart2 = alt.Chart(long2).mark_line(point=True).encode(
            x=alt.X("FY_Month_Label:N", sort=alt.SortField("FY_Month_Key", order="ascending"), title="Fiscal month"),
            y=alt.Y("Value:Q", title="Percent (%)"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["FY_Month_Label", "Metric", alt.Tooltip("Value:Q", format=".1f")]
        ).properties(height=320)


        st.caption("Tip: If you want a more accurate MoM view, the dataset needs true timesheet-entry dates.")


# =============================================================================
# DRILLDOWN
# =============================================================================

with tab_drill:
    section_header("Drilldown", "Department → Product → Job Category → Job → Task")

    dept_focus = st.selectbox(
        "Focus department (optional)",
        options=["(All)"] + available_depts,
        index=0,
        key="dept_focus_drill",
    )

    st.markdown("### 1) Department rollup")
    st_df(dept_df.sort_values("Quote_Margin", ascending=True))

    st.markdown("")

    st.markdown("### 2) Job category rollup")
    if category_df is None or category_df.empty:
        st.info("No [Job] Category field available for this dataset / filters.")
    else:
        cat_view = category_df.copy()
        if dept_focus != "(All)":
            cat_view = cat_view.loc[cat_view["Department"] == dept_focus].copy()
        st_df(cat_view.sort_values("Quote_Margin", ascending=True))

    st.markdown("")

    st.markdown("### 3) Product rollup")
    prod_view = product_df.copy()
    if dept_focus != "(All)":
        prod_view = prod_view.loc[prod_view["Department"] == dept_focus].copy()
    st_df(prod_view.sort_values("Quote_Margin", ascending=True))

    st.markdown("")

    st.markdown("### 4) Job rollup")
    jobs_view = job_df.copy()
    if dept_focus != "(All)":
        jobs_view = jobs_view.loc[jobs_view["Department"] == dept_focus].copy()
    jobs_sorted = jobs_view.sort_values("Quote_Margin", ascending=True).copy()
    st_df(jobs_sorted)

    st.markdown("")
    job_options = jobs_sorted["[Job] Job No."].tolist()
    selected_job = st.selectbox("Select a job to inspect tasks", options=["(None)"] + job_options, index=0)

    if selected_job != "(None)":
        st.markdown("### 5) Task breakdown (selected job)")
        tasks_for_job = task_df.loc[task_df["[Job] Job No."] == selected_job].copy()
        tasks_for_job = tasks_for_job.sort_values("Quote_Margin", ascending=True)
        st_df(tasks_for_job)

        st.markdown("")
        # quick visual: quoted vs actual hours (top variance tasks)
        vis = tasks_for_job[["[Job Task] Name", "Quoted_Hours", "Actual_Hours", "Hours_Variance"]].copy()
        vis = vis.sort_values("Hours_Variance", ascending=False).head(20)
        long = vis.melt(id_vars=["[Job Task] Name"], value_vars=["Quoted_Hours", "Actual_Hours"],
                        var_name="Metric", value_name="Hours")
        long["Metric"] = long["Metric"].map({"Quoted_Hours": "Quoted hours", "Actual_Hours": "Actual hours"})

        bar = alt.Chart(long).mark_bar().encode(
            y=alt.Y("[Job Task] Name:N", sort="-x", title="Task"),
            x=alt.X("Hours:Q", title="Hours"),
            color=alt.Color("Metric:N", title=""),
            tooltip=["[Job Task] Name", "Metric", alt.Tooltip("Hours:Q", format=",.2f")],
        ).properties(height=520)
        st_chart(bar)
# =============================================================================
# INSIGHTS
# =============================================================================

with tab_insights:
    section_header("Insights", "Operational diagnostics (what to do next).")

    st.markdown("### A) Margin erosion hotspots")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Worst quote-margin jobs**")
        st_df(tops["worst_quote_margin_jobs"])
    with c2:
        st.markdown("**Worst invoiced-margin jobs**")
        st_df(tops["worst_invoiced_margin_jobs"])

    st.markdown("")

    st.markdown("### B) Root-cause patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scope creep (unquoted tasks with cost)**")
        st_df(tops["scope_creep_tasks"])
    with col2:
        st.markdown("**Task overruns (hours over quoted)**")
        st_df(tops["overrun_tasks"])

    st.markdown("")

    st.markdown("### C) Interpretation guide (actionable)")
    st.markdown(
        """
- **Low pricing adequacy (<100%)** → quote below internal ratecard value for delivered hours.  
  Action: update pricing template / task rate assumptions; enforce quote buffers for high-variance work.

- **High scope creep** (unquoted tasks with cost) → work delivered outside original allocation.  
  Action: tighten change-control; force unquoted work into explicit change orders.

- **High hour overruns** → under-scoping or delivery inefficiency.  
  Action: revise estimating factors by task type; add early warning (e.g., 80% hours consumed triggers).

- **Low realisation** → invoicing timing gaps or write-offs.  
  Action: audit invoice cadence and WIP; match “done” to billing milestones.
        """
    )
with tab_recon:
    section_header("Reconciliation", "What was included, excluded, and why.")

    st.markdown("### Data health")
    st.write("Parse notes:", parse_report.notes if parse_report.notes else "None")
    st.write("Raw records:", f"{parse_report.raw_records:,}")
    st.write("Parsed records:", f"{parse_report.parsed_records:,}")

    st.markdown("### Filter impacts")
    st.json({
        "raw_records": filter_report.raw_records,
        "excluded_sg_allocation": filter_report.excluded_sg_allocation,
        "excluded_missing_anchor_date": filter_report.excluded_missing_anchor,
        "excluded_other_fy": filter_report.excluded_other_fy,
        "excluded_other_filters": filter_report.excluded_other_filters,
        "final_records": filter_report.final_records,
    })

    st.markdown("### Metric glossary")
    defs = pd.DataFrame.from_dict(an.METRIC_DEFINITIONS, orient="index").reset_index().rename(columns={"index": "Key"})

    st.markdown("### Critical assumptions")
    st.markdown(
        f"""
1) **Quoted Amount** is treated as the commercial benchmark (client quote).  
2) **Ratecard value** is **internal** and is not recognised revenue.  
3) Month-on-month trends are anchored to **{date_anchor}** (approximation).  
4) Records with missing anchor date are excluded from trend reporting.  
5) Exclusion enforced: **{', '.join(sorted(an.EXCLUDED_TASK_NAMES))}**.
        """
    )