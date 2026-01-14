"""
Job Profitability Dashboard
============================
REVENUE = Quoted Amount
BENCHMARK = Expected Quote (Quoted Hours x Billable Rate)
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
  layout="wide",
  initial_sidebar_state="expanded"
)

# =============================================================================
# THEME / STYLES
# =============================================================================

st.markdown(
  """
  <style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  :root {
    --bg-1: #f8f4ef;
    --bg-2: #eef4f7;
    --ink-1: #1e1a18;
    --ink-2: #5a534f;
    --accent-1: #e4572e;
    --accent-2: #2e86ab;
    --accent-3: #2ecc71;
    --card: rgba(255, 255, 255, 0.85);
    --border: rgba(30, 26, 24, 0.08);
    --shadow: 0 10px 30px rgba(30, 26, 24, 0.08);
  }
  html, body, [class*="css"] {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    color: var(--ink-1);
  }
  .stApp {
    background: radial-gradient(1100px 600px at 10% -10%, #fff2e8 0%, transparent 60%),
          radial-gradient(800px 400px at 90% 10%, #e9f3ff 0%, transparent 60%),
          linear-gradient(180deg, var(--bg-1), var(--bg-2));
  }
  .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3.5rem;
  }
  h1, h2, h3, h4 {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
    letter-spacing: -0.02em;
  }
  .hero {
    padding: 1.4rem 1.8rem;
    border-radius: 18px;
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }
  .hero-sub {
    color: var(--ink-2);
    font-size: 1rem;
    margin-bottom: 0.75rem;
  }
  .pill {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(46, 134, 171, 0.12);
    color: #1f5e78;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.4rem;
  }
  .callout {
    padding: 0.85rem 1rem;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    margin: 0.6rem 0 1rem 0;
  }
  .callout-title {
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: var(--accent-2);
  }
  .callout-text {
    color: var(--ink-2);
    font-size: 0.95rem;
  }
  div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 0.9rem 0.9rem 0.7rem 0.9rem;
    border-radius: 14px;
    box-shadow: var(--shadow);
  }
  div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: var(--ink-2);
  }
  div[data-testid="stMetricValue"] {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
  }
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.01em;
  }
  .stExpander, .stDataFrame {
    background: var(--card);
    border-radius: 12px;
    border: 1px solid var(--border);
  }
  </style>
  """,
  unsafe_allow_html=True
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
      return "Good"
    elif val >= bad_threshold:
      return "Watch"
    return "Risk"
  else:
    if val <= good_threshold:
      return "Good"
    elif val <= bad_threshold:
      return "Watch"
    return "Risk"


def hero(title, subtitle, pills):
  pill_html = "".join([f"<span class='pill'>{p}</span>" for p in pills])
  st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">{title}</div>
      <div class="hero-sub">{subtitle}</div>
      <div>{pill_html}</div>
    </div>
    """,
    unsafe_allow_html=True
  )


def callout(title, body):
  st.markdown(
    f"""
    <div class="callout">
      <div class="callout-title">{title}</div>
      <div class="callout-text">{body}</div>
    </div>
    """,
    unsafe_allow_html=True
  )


def callout_list(title, items):
  items_html = "".join([f"<li>{i}</li>" for i in items])
  st.markdown(
    f"""
    <div class="callout">
      <div class="callout-title">{title}</div>
      <div class="callout-text">
        <ul style="margin: 0.2rem 0 0.2rem 1.2rem;">{items_html}</ul>
      </div>
    </div>
    """,
    unsafe_allow_html=True
  )


def metric_explainer(title, keys):
  fallback = {
    "Job_Count": {
      "name": "Job Count",
      "formula": "Count of distinct jobs",
      "desc": "Number of jobs included in the aggregation",
    }
  }
  lines = []
  for key in keys:
    defn = METRIC_DEFINITIONS.get(key, fallback.get(key))
    if not defn:
      continue
    lines.append(f"**{defn['name']}** - `{defn['formula']}` - {defn['desc']}")
  if lines:
    with st.expander(title):
      st.markdown("\n\n".join(lines))


def apply_chart_theme():
  def theme():
    return {
      "config": {
        "background": "rgba(0,0,0,0)",
        "axis": {
          "labelColor": "#4f4946",
          "titleColor": "#1e1a18",
          "gridColor": "#ece7e1",
          "domainColor": "#d7cfc6",
          "labelFont": "IBM Plex Sans",
          "titleFont": "Space Grotesk",
          "labelFontSize": 12,
          "titleFontSize": 13,
        },
        "legend": {
          "labelFont": "IBM Plex Sans",
          "titleFont": "Space Grotesk",
          "labelColor": "#4f4946",
          "titleColor": "#1e1a18",
        },
        "title": {
          "font": "Space Grotesk",
          "fontSize": 16,
          "color": "#1e1a18",
        },
        "view": {"stroke": "transparent"},
      }
    }

  alt.themes.register("profit_theme", theme)
  alt.themes.enable("profit_theme")


@st.cache_data(show_spinner=False)
def load_and_parse_data(source_key, source_payload):
  if source_key == "bytes":
    df_raw = load_raw_data(source_payload)
  else:
    df_raw = load_raw_data(source_payload)
  df_parsed = clean_and_parse(df_raw)
  return df_raw, df_parsed


@st.cache_data(show_spinner=False)
def filter_data(df_parsed, exclude_sg_allocation, billable_only, fiscal_year, department):
  df_filtered, recon = apply_filters(
    df_parsed,
    exclude_sg_allocation=exclude_sg_allocation,
    billable_only=billable_only,
    fiscal_year=fiscal_year,
    department=department
  )
  recon = compute_reconciliation_totals(df_filtered, recon)
  return df_filtered, recon


@st.cache_data(show_spinner=False)
def compute_summaries(df_filtered):
  dept_summary = compute_department_summary(df_filtered)
  product_summary = compute_product_summary(df_filtered)
  job_summary = compute_job_summary(df_filtered)
  task_summary = compute_task_summary(df_filtered)
  monthly_summary = compute_monthly_summary(df_filtered)
  monthly_by_dept = compute_monthly_by_department(df_filtered)
  metrics = calculate_overall_metrics(job_summary)
  causes = analyze_overrun_causes(task_summary)
  insights = generate_insights(job_summary, dept_summary, monthly_summary, task_summary)
  return (
    dept_summary,
    product_summary,
    job_summary,
    task_summary,
    monthly_summary,
    monthly_by_dept,
    metrics,
    causes,
    insights,
  )


@st.cache_data(show_spinner=False)
def compute_builder_task_stats(df_filtered_product):
  task_summary = compute_task_summary(df_filtered_product)
  if len(task_summary) == 0:
    return pd.DataFrame()
  total_jobs = task_summary["Job_No"].nunique()
  stats = task_summary.groupby("Task_Name").agg(
    Jobs_With_Task=("Job_No", "nunique"),
    Avg_Quoted_Hours=("Quoted_Hours", "mean"),
    Avg_Actual_Hours=("Actual_Hours", "mean"),
    Billable_Rate_Hr=("Billable_Rate_Hr", "mean"),
    Cost_Rate_Hr=("Cost_Rate_Hr", "mean"),
    Total_Actual_Hours=("Actual_Hours", "sum"),
  ).reset_index()
  stats["Frequency_Pct"] = np.where(
    total_jobs > 0, (stats["Jobs_With_Task"] / total_jobs) * 100, 0
  )
  stats = stats.sort_values(["Frequency_Pct", "Avg_Actual_Hours"], ascending=False)
  return stats


# =============================================================================
# MAIN APP
# =============================================================================

def main():
  apply_chart_theme()
  hero(
    "Job Profitability Analysis",
    "Revenue = Quoted Amount | Benchmark = Expected Quote (Quoted Hours x Billable Rate)",
    ["Pricing Discipline", "Margin Health", "Scope Control"]
  )
  
  # Key definitions
  with st.expander("Understanding This Dashboard", expanded=False):
    st.markdown("""
    ### Financial Model
    
    | Term | Formula | Purpose |
    |------|---------|---------|
    | **Quoted Amount** | From data | **= REVENUE** (what we invoice the client) |
    | **Expected Quote** | Quoted Hours x Billable Rate | **Benchmark** - what we *should* have quoted |
    | **Base Cost** | Actual Hours x Cost Rate | Internal cost |
    
    ### Key Metrics
    
    | Metric | Formula | What It Tells You |
    |--------|---------|-------------------|
    | **Margin** | Quoted Amount - Base Cost | Are we profitable? |
    | **Quote Gap** | Quoted Amount - Expected Quote | Did we quote at/above/below internal rates? |
    | **Effective Rate/Hr** | Quoted Amount / Actual Hours | Revenue per hour worked |
    
    ### Interpreting Quote Gap
    - **Positive** (+) = Quoted ABOVE internal benchmark (premium pricing )
    - **Negative** (-) = Quoted BELOW internal benchmark (discounting )
    """)
  
  callout(
    "How to read the dashboard",
    "Start with Executive Summary for topline health, then use Monthly Trends for seasonality, "
    "Drill-Down for root causes, and Job Diagnosis for single-job explanations."
  )

  # =========================================================================
  # SIDEBAR
  # =========================================================================
  st.sidebar.header("Data & Filters")
  
  data_path = Path("data/Quoted_Task_Report_FY26.xlsx")
  uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
  
  if uploaded:
    data_key = "bytes"
    data_source = uploaded.getvalue()
  elif data_path.exists():
    data_key = "path"
    data_source = str(data_path)
  else:
    st.warning(" Upload data file or place in `data/` folder")
    st.stop()
  
  try:
    with st.spinner("Loading data..."):
      df_raw, df_parsed = load_and_parse_data(data_key, data_source)
    st.sidebar.success(f" {len(df_raw):,} records")
  except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
  
  # Fiscal Year
  fy_list = get_available_fiscal_years(df_parsed)
  if not fy_list:
    st.error("No fiscal year data found")
    st.stop()
  
  selected_fy = st.sidebar.selectbox(
    " Fiscal Year", fy_list,
    index=len(fy_list) - 1,
    format_func=lambda x: f"FY{str(x)[-2:]}"
  )
  
  # Department
  dept_list = get_available_departments(df_parsed)
  selected_dept = st.sidebar.selectbox(" Department", ["All Departments"] + dept_list)
  dept_filter = None if selected_dept == "All Departments" else selected_dept
  
  st.sidebar.markdown("---")
  exclude_sg = st.sidebar.checkbox("Exclude SG Allocation", value=True)
  billable_only = st.sidebar.checkbox("Billable tasks only", value=True)
  
  # Apply filters
  df_filtered, recon = filter_data(
    df_parsed,
    exclude_sg_allocation=exclude_sg,
    billable_only=billable_only,
    fiscal_year=selected_fy,
    department=dept_filter
  )
  
  if len(df_filtered) == 0:
    st.error("No data after applying filters.")
    st.stop()
  
  # Compute summaries
  with st.spinner("Computing..."):
    (
      dept_summary,
      product_summary,
      job_summary,
      task_summary,
      monthly_summary,
      monthly_by_dept,
      metrics,
      causes,
      insights,
    ) = compute_summaries(df_filtered)
  
  st.sidebar.markdown("---")
  st.sidebar.metric("Records", f"{recon['final_records']:,}")
  st.sidebar.metric("Jobs", f"{metrics['total_jobs']:,}")
  
  # =========================================================================
  # TABS
  # =========================================================================
  tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Executive Summary", "Monthly Trends", "Drill-Down",
    "Insights", "Job Diagnosis", "Profitability Drivers", "Reconciliation", "Smart Quote Builder"
  ])
  
  # =========================================================================
  # TAB 1: EXECUTIVE SUMMARY
  # =========================================================================
  with tab1:
    st.header(f"FY{str(selected_fy)[-2:]} Executive Summary")
    callout_list(
      "Executive Summary explainer",
      [
        "All KPIs aggregate filtered jobs and tasks",
        "Margin % uses Quoted Amount as the denominator",
        "Quote Gap % uses Expected Quote as the denominator",
      ]
    )
    
    # Headlines
    if insights["headline"]:
      for h in insights["headline"]:
        st.markdown(h)
    
    st.markdown("---")
    
    # REVENUE & MARGIN
    st.subheader("Revenue & Margin")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (Quoted)", fmt_currency(metrics['total_quoted_amount']),
         help="Quoted Amount = What we invoice the client")
    c2.metric("Base Cost", fmt_currency(metrics['total_base_cost']),
         help="Actual Hours x Cost Rate")
    margin_label = status_icon(metrics['margin_pct'], 35, 20)
    c3.metric(f"Margin ({margin_label})", fmt_currency(metrics['margin']),
         delta=fmt_pct(metrics['margin_pct']),
         help="Quoted Amount - Base Cost")
    c4.metric("Margin %", fmt_pct(metrics['margin_pct']),
         help="Target: 35%+")
    callout_list(
      "Metric notes",
      [
        "Margin = Quoted Amount - Base Cost",
        "Margin % = (Margin / Quoted Amount) x 100",
        "Positive margin means revenue exceeds cost",
      ]
    )
    
    # QUOTING ACCURACY (Sanity Check)
    st.subheader("Quoting Sanity Check: Quoted vs What We Should Have Quoted")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quoted Amount", fmt_currency(metrics['total_quoted_amount']),
         help="What we actually charged")
    c2.metric("Expected Quote", fmt_currency(metrics['total_expected_quote']),
         help="What we SHOULD have quoted (Quoted Hours x Billable Rate)")
    gap = metrics['quote_gap']
    gap_label = "Above" if gap >= 0 else "Below"
    c3.metric(f"Quote Gap ({gap_label})", fmt_currency(gap),
         delta=f"{metrics['quote_gap_pct']:+.0f}% vs benchmark",
         delta_color="normal" if gap >= 0 else "inverse",
         help="Positive = quoted above internal rates. Negative = underquoting.")
    underquoted = metrics['jobs_underquoted']
    c4.metric("Underquoted Jobs", f"{underquoted} / {metrics['total_jobs']}",
         help="Jobs quoted below internal rates")
    callout_list(
      "How Quote Gap is calculated",
      [
        "Expected Quote = Quoted Hours x Billable Rate",
        "Quote Gap = Quoted Amount - Expected Quote",
        "Negative gap means discounting vs internal benchmark",
      ]
    )
    
    # RATES
    st.subheader("Rate Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quoted Rate/Hr", fmt_rate(metrics['avg_quoted_rate_hr']),
         help="Quoted Amount / Quoted Hours - what we charged")
    c2.metric("Billable Rate/Hr", fmt_rate(metrics['avg_billable_rate_hr']),
         help="Internal standard rate we SHOULD charge")
    c3.metric("Effective Rate/Hr", fmt_rate(metrics['avg_effective_rate_hr']),
         help="Quoted Amount / Actual Hours (drops if hours overrun)")
    c4.metric("Cost Rate/Hr", fmt_rate(metrics['avg_cost_rate_hr']),
         help="Internal cost per hour")
    callout_list(
      "Rate explainer",
      [
        "Quoted Rate/Hr = Quoted Amount / Quoted Hours",
        "Effective Rate/Hr = Quoted Amount / Actual Hours",
        "Overruns reduce effective rate even if quoted rate is strong",
      ]
    )
    
    # PERFORMANCE FLAGS
    st.subheader("Performance Flags")
    c1, c2, c3, c4 = st.columns(4)
    loss_label = status_icon(metrics['loss_rate'], 5, 15, higher_is_better=False)
    c1.metric(f"Jobs at Loss ({loss_label})", f"{metrics['jobs_at_loss']} / {metrics['total_jobs']}",
         delta=f"{metrics['loss_rate']:.0f}%", delta_color="inverse",
         help="Loss = Margin < 0")
    overrun_label = status_icon(metrics['overrun_rate'], 30, 50, higher_is_better=False)
    c2.metric(f"Hour Overruns ({overrun_label})", f"{metrics['jobs_over_budget']}",
         delta=f"{metrics['overrun_rate']:.0f}%", delta_color="inverse",
         help="Overrun = Actual Hours > Quoted Hours")
    c3.metric("Underquoted Jobs", str(metrics['jobs_underquoted']),
         help="Jobs quoted below internal benchmark")
    c4.metric("Scope Creep Tasks", str(causes['scope_creep']['count']),
         delta=fmt_currency(causes['scope_creep']['cost']),
         help="Unquoted tasks = work not in original quote")
    callout_list(
      "Overrun definition",
      [
        "Hours Variance = Actual Hours - Quoted Hours",
        "Hours Variance % = (Hours Variance / Quoted Hours) x 100",
        "Overrun flag = Hours Variance > 0",
      ]
    )
    
    st.markdown("---")
    
    # MARGIN BRIDGE
    st.subheader("Margin Bridge")
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
    
    bridge_chart = alt.Chart(bridge_data).mark_bar(size=45, cornerRadiusEnd=4).encode(
      x=alt.X("Step:N", sort=None, axis=alt.Axis(labelAngle=0)),
      y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="~s")),
      color=alt.Color("Color:N", scale=alt.Scale(
        domain=["Revenue", "Cost", "Margin"],
        range=["#2e86ab", "#e4572e", "#2ecc71"]
      )),
      tooltip=["Step", alt.Tooltip("Amount:Q", format="$,.0f")]
    ).properties(height=300)
    st.altair_chart(bridge_chart, width='stretch')
    callout_list(
      "Bridge chart",
      [
        "Revenue minus Base Cost yields Margin",
        "Negative bars represent costs",
      ]
    )
  
  # =========================================================================
  # TAB 2: MONTHLY TRENDS
  # =========================================================================
  with tab2:
    st.header(f"Monthly Trends - FY{str(selected_fy)[-2:]}")
    
    if len(monthly_summary) == 0:
      st.warning("No monthly data available.")
    else:
      last_month = monthly_summary["Month"].iloc[-1]
      callout_list(
        "Monthly trend explainer",
        [
          "Values are aggregated by month after filters",
          "Quote Gap % uses Expected Quote as the denominator",
          "Hours Variance % compares actual to quoted hours",
        ]
      )
      metric_explainer(
        "Monthly metrics explained",
        [
          "Quoted_Amount", "Expected_Quote", "Quote_Gap", "Quote_Gap_Pct",
          "Margin", "Margin_Pct", "Hours_Variance_Pct", "Effective_Rate_Hr",
          "Billable_Rate_Hr", "Cost_Rate_Hr", "Job_Count",
        ],
      )
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
      st.subheader(f"{trend_metric.replace('_', ' ')} by Month")
      format_map = {
        "Margin_Pct": ".1f",
        "Quote_Gap_Pct": ".1f",
        "Hours_Variance_Pct": ".1f",
        "Quoted_Amount": "$,.0f",
        "Effective_Rate_Hr": "$,.0f",
      }
      metric_format = format_map.get(trend_metric, ",.1f")
      trend_chart = alt.Chart(monthly_summary).mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=3).encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{trend_metric}:Q", axis=alt.Axis(format=metric_format)),
        color=alt.value("#2e86ab"),
        tooltip=["Month", alt.Tooltip(f"{trend_metric}:Q", format=metric_format)]
      ).properties(height=350)
      trend_label = alt.Chart(monthly_summary).transform_filter(
        alt.datum.Month == last_month
      ).mark_text(align="left", dx=8, dy=-6, fontSize=12, fontWeight="bold").encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"])),
        y=alt.Y(f"{trend_metric}:Q"),
        text=alt.Text(f"{trend_metric}:Q", format=metric_format)
      )
      
      # Reference lines
      if trend_metric == "Margin_Pct":
        rule = alt.Chart(pd.DataFrame({"y": [35]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y:Q")
        trend_chart = trend_chart + rule
      elif trend_metric == "Quote_Gap_Pct":
        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[5,5]).encode(y="y:Q")
        trend_chart = trend_chart + rule
      
      st.altair_chart(trend_chart + trend_label, width='stretch')
      callout_list(
        "Chart guide",
        [
          "Selector switches KPIs",
          "Margin % target line is 35%",
          "Quote Gap % zero line indicates at-benchmark quoting",
        ]
      )
      
      st.subheader("Monthly Health Panel")
      monthly_health = monthly_summary.copy()
      monthly_health["Overrun_Rate"] = np.where(
        monthly_health["Hours_Variance_Pct"] > 0, monthly_health["Hours_Variance_Pct"], 0
      )
      health_fields = [
        "Margin_Pct", "Quote_Gap_Pct", "Hours_Variance_Pct", "Job_Count"
      ]
      health_melt = monthly_health.melt(
        id_vars=["Month"], value_vars=health_fields, var_name="Metric", value_name="Value"
      )
      metric_labels = {
        "Margin_Pct": "Margin %",
        "Quote_Gap_Pct": "Quote Gap %",
        "Hours_Variance_Pct": "Hours Variance %",
        "Job_Count": "# Jobs",
      }
      health_melt["Metric"] = health_melt["Metric"].map(metric_labels)
      health_chart = alt.Chart(health_melt).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2.5).encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Value:Q", axis=alt.Axis(title=None)),
        color=alt.Color("Metric:N", scale=alt.Scale(
          range=["#2ecc71", "#e4572e", "#2e86ab", "#6b705c"]
        )),
        tooltip=["Month", "Metric", alt.Tooltip("Value:Q", format=",.1f")]
      ).properties(height=320)
      health_label = alt.Chart(health_melt).transform_filter(
        alt.datum.Month == last_month
      ).mark_text(align="left", dx=6, dy=-5, fontSize=11).encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"])),
        y=alt.Y("Value:Q"),
        color=alt.Color("Metric:N", scale=alt.Scale(
          range=["#2ecc71", "#e4572e", "#2e86ab", "#6b705c"]
        )),
        text=alt.Text("Value:Q", format=",.1f")
      )
      st.altair_chart(health_chart + health_label, width='stretch')
      callout_list(
        "Health panel",
        [
          "Compare pricing (Quote Gap) vs delivery (Hours Variance)",
          "Job count spikes indicate volume-driven months",
        ]
      )

      # Quoted vs Expected Quote
      st.subheader("Quoted Amount vs Expected Quote")
      compare_data = monthly_summary.melt(
        id_vars=["Month"], value_vars=["Quoted_Amount", "Expected_Quote"],
        var_name="Type", value_name="Amount"
      )
      compare_data["Type"] = compare_data["Type"].map({
        "Quoted_Amount": "Quoted Amount (Revenue)",
        "Expected_Quote": "Expected Quote (Benchmark)"
      })
      compare_chart = alt.Chart(compare_data).mark_bar(size=18, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Amount:Q", axis=alt.Axis(format="~s")),
        color=alt.Color("Type:N", scale=alt.Scale(
          domain=["Quoted Amount (Revenue)", "Expected Quote (Benchmark)"],
          range=["#2ecc71", "#a8a29e"]
        )),
        xOffset="Type:N",
        tooltip=["Month", "Type", alt.Tooltip("Amount:Q", format="$,.0f")]
      ).properties(height=300)
      st.altair_chart(compare_chart, width='stretch')
      callout_list(
        "Quoted vs Expected",
        [
          "The gap between bars is the Quote Gap",
          "Larger Expected Quote suggests underpricing",
        ]
      )
      
      st.subheader("Rate & Cost Trends")
      rate_data = monthly_summary.melt(
        id_vars=["Month"],
        value_vars=["Quoted_Rate_Hr", "Effective_Rate_Hr", "Billable_Rate_Hr", "Cost_Rate_Hr"],
        var_name="Type",
        value_name="Rate"
      )
      rate_labels = {
        "Quoted_Rate_Hr": "Quoted Rate/Hr",
        "Effective_Rate_Hr": "Effective Rate/Hr",
        "Billable_Rate_Hr": "Billable Rate/Hr",
        "Cost_Rate_Hr": "Cost Rate/Hr",
      }
      rate_data["Type"] = rate_data["Type"].map(rate_labels)
      rate_chart = alt.Chart(rate_data).mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5).encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Rate:Q", title="Rate per hour ($)"),
        color=alt.Color("Type:N", scale=alt.Scale(
          range=["#2e86ab", "#2ecc71", "#6c757d", "#e4572e"]
        )),
        tooltip=["Month", "Type", alt.Tooltip("Rate:Q", format="$,.0f")]
      ).properties(height=320)
      rate_label = alt.Chart(rate_data).transform_filter(
        alt.datum.Month == last_month
      ).mark_text(align="left", dx=6, dy=-5, fontSize=11).encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"])),
        y=alt.Y("Rate:Q"),
        color=alt.Color("Type:N", scale=alt.Scale(
          range=["#2e86ab", "#2ecc71", "#6c757d", "#e4572e"]
        )),
        text=alt.Text("Rate:Q", format="$,.0f")
      )
      st.altair_chart(rate_chart + rate_label, width='stretch')
      callout_list(
        "Rate story",
        [
          "Effective Rate/Hr below Cost Rate/Hr signals loss pressure",
          "Divergence between Quoted vs Billable suggests discounting",
        ]
      )

      # Margin trend
      st.subheader("Margin $ and %")
      margin_line = alt.Chart(monthly_summary).mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.5, color="#2ecc71").encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Margin_Pct:Q", title="Margin %"),
        tooltip=["Month", alt.Tooltip("Margin_Pct:Q", format=".1f"), alt.Tooltip("Margin:Q", format="$,.0f")]
      ).properties(height=300)
      margin_label = alt.Chart(monthly_summary).transform_filter(
        alt.datum.Month == last_month
      ).mark_text(align="left", dx=8, dy=-6, fontSize=12, fontWeight="bold", color="#2ecc71").encode(
        x=alt.X("Month:N", sort=list(monthly_summary["Month"])),
        y=alt.Y("Margin_Pct:Q"),
        text=alt.Text("Margin_Pct:Q", format=".1f")
      )
      st.altair_chart(margin_line + margin_label, width='stretch')
      callout_list(
        "Margin line",
        [
          "Line shows Margin % by month",
          "Hover reveals Margin $ and %",
          "Use dips to spot margin pressure",
        ]
      )
      
      # Department trends
      if selected_dept == "All Departments" and len(monthly_by_dept) > 0:
        st.subheader("Margin % by Department")
        dept_trend = alt.Chart(monthly_by_dept).mark_line(point=alt.OverlayMarkDef(size=40)).encode(
          x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
          y=alt.Y("Margin_Pct:Q", title="Margin %"),
          color="Department:N",
          tooltip=["Month", "Department", alt.Tooltip("Margin_Pct:Q", format=".0f")]
        ).properties(height=350)
        st.altair_chart(dept_trend, width='stretch')
        callout_list(
          "Department trend",
          [
            "Comparative trajectories show volatility drivers",
            "Use this to isolate high-variance departments",
          ]
        )
        
        st.subheader("Quote Gap % by Department")
        dept_quote = alt.Chart(monthly_by_dept).mark_line(point=alt.OverlayMarkDef(size=40)).encode(
          x=alt.X("Month:N", sort=list(monthly_summary["Month"]), axis=alt.Axis(labelAngle=-45)),
          y=alt.Y("Quote_Gap_Pct:Q", title="Quote Gap %"),
          color="Department:N",
          tooltip=["Month", "Department", alt.Tooltip("Quote_Gap_Pct:Q", format=".0f")]
        ).properties(height=350)
        st.altair_chart(dept_quote, width='stretch')
        callout_list(
          "Department pricing",
          [
            "Negative Quote Gap % indicates discounting by department",
            "Persistent gaps are candidates for rate adjustments",
          ]
        )
      
      with st.expander(" Monthly Data Table"):
        st.dataframe(monthly_summary[[
          "Month", "Job_Count", "Quoted_Amount", "Expected_Quote", "Quote_Gap",
          "Base_Cost", "Margin", "Margin_Pct", "Hours_Variance_Pct"
        ]], width='stretch')
  
  # =========================================================================
  # TAB 3: DRILL-DOWN
  # =========================================================================
  with tab3:
    st.header("Hierarchical Analysis")
    callout_list(
      "Drill-down explainer",
      [
        "Each level inherits the filters above",
        "Use Margin % to spot weak performers",
        "Use Quote Gap to spot pricing issues",
      ]
    )
    metric_explainer(
      "Drill-down metrics explained",
      [
        "Margin", "Margin_Pct", "Quote_Gap", "Quote_Gap_Pct",
        "Effective_Rate_Hr", "Hours_Variance_Pct", "Job_Count",
      ],
    )
    dept_select = alt.selection_point(fields=["Department"], on="click", clear="dblclick", empty="all")
    
    if len(dept_summary) > 0:
      st.subheader("Department Scoreboard")
      dept_metrics = dept_summary.copy()
      dept_metrics["Margin_Band"] = np.where(
        dept_metrics["Margin_Pct"] >= 35, "Healthy",
        np.where(dept_metrics["Margin_Pct"] < 20, "At Risk", "Watch")
      )
      dept_scatter = alt.Chart(dept_metrics).mark_circle(size=240).encode(
        x=alt.X("Quote_Gap_Pct:Q", title="Quote Gap %"),
        y=alt.Y("Margin_Pct:Q", title="Margin %"),
        color=alt.Color("Margin_Band:N", scale=alt.Scale(
          domain=["Healthy", "Watch", "At Risk"],
          range=["#2ecc71", "#f4d35e", "#e4572e"]
        )),
        size=alt.Size("Job_Count:Q", title="# Jobs", scale=alt.Scale(range=[200, 1200])),
        tooltip=[
          "Department",
          alt.Tooltip("Job_Count:Q", format=",.0f", title="# Jobs"),
          alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
          alt.Tooltip("Quote_Gap_Pct:Q", format=".1f", title="Quote Gap %"),
          alt.Tooltip("Hours_Variance_Pct:Q", format=".0f", title="Hours Var %"),
        ],
      ).properties(height=360)
      st.altair_chart(dept_scatter, width='stretch')
      callout_list(
        "How to use",
        [
          "Right side = premium pricing, left side = discounting",
          "Lower margin + negative quote gap indicates urgent pricing fixes",
        ]
      )

    # Department
    st.subheader("Level 1: Department Performance")
    if len(dept_summary) > 0:
      dept_chart = alt.Chart(dept_summary).mark_bar(size=20, cornerRadiusEnd=3).encode(
        y=alt.Y("Department:N", sort="-x"),
        x=alt.X("Margin_Pct:Q", title="Margin %", axis=alt.Axis(format="~s")),
        color=alt.condition(
          alt.datum.Margin_Pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")
        ),
        opacity=alt.condition(dept_select, alt.value(1), alt.value(0.35)),
        tooltip=["Department",
             alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
             alt.Tooltip("Margin:Q", format="$,.0f", title="Margin $"),
             alt.Tooltip("Quote_Gap:Q", format="$,.0f", title="Quote Gap")]
      ).add_params(dept_select).properties(height=max(200, len(dept_summary) * 40))
      
      rule = alt.Chart(pd.DataFrame({"x": [35]})).mark_rule(color="orange", strokeDash=[3,3]).encode(x="x:Q")
      st.altair_chart(dept_chart + rule, width='stretch')
      callout_list(
        "Department bar chart",
        [
          "Bars show Margin %",
          "Orange line is the 35% target benchmark",
          "Click a department bar to filter charts below (double-click to reset)",
        ]
      )
      
      with st.expander("Department Details"):
        st.dataframe(dept_summary[[
          "Department", "Job_Count", "Quoted_Amount", "Expected_Quote", "Quote_Gap",
          "Base_Cost", "Margin", "Margin_Pct", "Hours_Variance_Pct"
        ]].style.format({
          "Quoted_Amount": "${:,.0f}", "Expected_Quote": "${:,.0f}",
          "Quote_Gap": "${:,.0f}", "Base_Cost": "${:,.0f}",
          "Margin": "${:,.0f}", "Margin_Pct": "{:.1f}%",
          "Hours_Variance_Pct": "{:+.0f}%"
        }), width='stretch')
    
    st.markdown("---")
    
    # Product
    st.subheader("Level 2: Product Performance")
    sel_dept_drill = st.selectbox("Filter by Department", ["All"] + sorted(dept_summary["Department"].unique().tolist()), key="d1")
    prod_f = product_summary if sel_dept_drill == "All" else product_summary[product_summary["Department"] == sel_dept_drill]
    
    if len(product_summary) > 0:
      prod_base = alt.Chart(product_summary).transform_filter(dept_select)
      prod_chart = prod_base.mark_bar(size=16, cornerRadiusEnd=3).encode(
        y=alt.Y("Product:N", sort="-x"),
        x=alt.X("Margin_Pct:Q", title="Margin %", axis=alt.Axis(format="~s")),
        color=alt.condition(alt.datum.Margin_Pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")),
        tooltip=["Product", "Department",
             alt.Tooltip("Margin_Pct:Q", format=".1f"),
             alt.Tooltip("Quote_Gap:Q", format="$,.0f")]
      ).properties(height=320)
      st.altair_chart(prod_chart, width='stretch')
      callout_list(
        "Product bar chart",
        [
          "Filtered by selected department",
          "Use Quote Gap in tooltips to validate pricing",
        ]
      )
    
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
    
    if len(job_summary) > 0:
      st.subheader("Job Performance (Interactive)")
      job_chart = alt.Chart(job_summary).transform_filter(dept_select).mark_circle(size=110).encode(
        x=alt.X("Quote_Gap_Pct:Q", title="Quote Gap %"),
        y=alt.Y("Margin_Pct:Q", title="Margin %"),
        color=alt.condition(alt.datum.Margin_Pct < 20, alt.value("#e4572e"), alt.value("#2ecc71")),
        tooltip=[
          "Job_Name",
          "Department",
          "Product",
          alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
          alt.Tooltip("Quote_Gap_Pct:Q", format=".1f", title="Quote Gap %"),
          alt.Tooltip("Hours_Variance_Pct:Q", format=".0f", title="Hours Var %"),
          alt.Tooltip("Effective_Rate_Hr:Q", format="$,.0f", title="Effective Rate/Hr"),
        ],
      ).properties(height=360)
      st.altair_chart(job_chart, width='stretch')
      callout_list(
        "Job bubble chart",
        [
          "Upper-right = high margin and premium pricing",
          "Lower-left = discounted and low margin (urgent review)",
        ]
      )
    
    if len(jobs_disp) > 0:
      cols = ["Job_No", "Job_Name", "Client", "Month",
          "Quoted_Amount", "Expected_Quote", "Quote_Gap", "Base_Cost",
          "Margin", "Margin_Pct", "Hours_Variance_Pct"]
      st.dataframe(jobs_disp[cols].style.format({
        "Quoted_Amount": "${:,.0f}", "Expected_Quote": "${:,.0f}",
        "Quote_Gap": "${:,.0f}", "Base_Cost": "${:,.0f}",
        "Margin": "${:,.0f}", "Margin_Pct": "{:.1f}%",
        "Hours_Variance_Pct": "{:+.0f}%"
      }), width='stretch', height=400)
      callout_list(
        "Job table",
        [
          "Hours Variance % above 0 indicates overruns",
          "Quote Gap shows discounting vs benchmark",
        ]
      )
    else:
      st.info("No jobs match filters.")
    
    st.markdown("---")
    
    # Task
    st.subheader("Level 4: Task Breakdown")
    job_opts = jobs_disp.apply(lambda r: f"{r['Job_No']} - {str(r['Job_Name'])[:35]}", axis=1).tolist() if len(jobs_disp) > 0 else []
    if job_opts:
      sel_job = st.selectbox("Select Job", ["-- Select --"] + job_opts, key="tj")
      if sel_job != "-- Select --":
        job_no = sel_job.split(" - ")[0]
        job_info = jobs_disp[jobs_disp["Job_No"] == job_no].iloc[0]
        tasks = task_summary[task_summary["Job_No"] == job_no]
        
        st.markdown(f"### {job_info['Job_Name']}")
        c1, c2, c3, c4 = st.columns(4)
        margin_label = "Positive" if job_info["Margin"] > 0 else "Negative"
        c1.metric(f"Margin ({margin_label})", fmt_currency(job_info["Margin"]),
             delta=fmt_pct(job_info["Margin_Pct"]))
        gap_label = "Above" if job_info["Quote_Gap"] >= 0 else "Below"
        c2.metric(f"Quote Gap ({gap_label})", fmt_currency(job_info["Quote_Gap"]),
             help="Quoted vs Expected Quote")
        c3.metric("Hours Var", f"{job_info['Hours_Variance_Pct']:+.0f}%")
        c4.metric("Effective Rate", fmt_rate(job_info["Effective_Rate_Hr"]))
        
        if len(tasks) > 0:
          st.markdown("#### Tasks")
          callout_list(
            "Task breakdown",
            [
              "Quoted Hrs vs Actual Hrs shows overruns",
              "Quote Gap is task-level pricing vs benchmark",
              "SCOPE CREEP indicates unquoted tasks",
            ]
          )
          task_cols = ["Task_Name", "Quoted_Hours", "Actual_Hours", "Hours_Variance",
                 "Quoted_Amount", "Expected_Quote", "Quote_Gap", "Base_Cost",
                 "Margin", "Is_Unquoted"]
          task_disp = tasks[task_cols].copy()
          task_disp["Is_Unquoted"] = task_disp["Is_Unquoted"].map({True: "SCOPE CREEP", False: ""})
          task_disp.columns = ["Task", "Quoted Hrs", "Actual Hrs", "Hrs Var",
                     "Quoted $", "Expected $", "Quote Gap", "Cost",
                     "Margin", "Flag"]
          st.dataframe(task_disp.style.format({
            "Quoted Hrs": "{:,.1f}", "Actual Hrs": "{:,.1f}", "Hrs Var": "{:+,.1f}",
            "Quoted $": "${:,.0f}", "Expected $": "${:,.0f}",
            "Quote Gap": "${:,.0f}", "Cost": "${:,.0f}",
            "Margin": "${:,.0f}"
          }), width='stretch')
  
  # =========================================================================
  # TAB 4: INSIGHTS
  # =========================================================================
  with tab4:
    st.header("Profitability Insights")
    callout_list(
      "Insights explainer",
      [
        "Narratives are generated from filtered data",
        "Focus on pricing, scope, and margin drivers",
      ]
    )
    
    # Quoting Issues
    if insights["quoting_issues"]:
      st.subheader("Quoting Accuracy Issues")
      for i in insights["quoting_issues"]:
        st.markdown(i)
    
    # Scope Issues
    if insights["scope_issues"]:
      st.subheader("Scope & Hours Issues")
      for i in insights["scope_issues"]:
        st.markdown(i)
    
    # Rate Issues
    if insights["rate_issues"]:
      st.subheader("Rate Issues")
      for i in insights["rate_issues"]:
        st.markdown(i)
    
    # Margin Drivers
    if insights["margin_drivers"]:
      st.subheader("Margin Drivers")
      for i in insights["margin_drivers"]:
        st.markdown(i)
    
    # Action Items
    if insights["action_items"]:
      st.subheader("Action Items")
      for a in insights["action_items"]:
        st.markdown(f"- {a}")
    
    st.markdown("---")
    
    # Deep dive panels
    col1, col2 = st.columns(2)
    
    with col1:
      st.subheader("Underquoted Jobs")
      underquoted = get_underquoted_jobs(job_summary, -500).head(10)
      if len(underquoted) > 0:
        st.metric("Total Quote Gap", fmt_currency(underquoted["Quote_Gap"].sum()))
        for _, j in underquoted.iterrows():
          st.markdown(f"**{str(j['Job_Name'])[:35]}** - ${abs(j['Quote_Gap']):,.0f} below internal rates")
      else:
        st.success("No significant underquoting!")
      callout_list(
        "Underquoted list",
        [
          "Jobs here have Quote Gap below -$500",
          "Threshold is configurable",
        ]
      )
    
    with col2:
      st.subheader("Scope Creep (Unquoted Work)")
      unquoted = get_unquoted_tasks(task_summary).head(10)
      if len(unquoted) > 0:
        st.metric("Total Unquoted Cost", fmt_currency(unquoted["Base_Cost"].sum()))
        for _, t in unquoted.iterrows():
          st.markdown(f"- **{str(t['Task_Name'])[:30]}** - {t['Actual_Hours']:.0f} hrs, ${t['Base_Cost']:,.0f}")
      else:
        st.success("No scope creep detected!")
      callout_list(
        "Scope creep list",
        [
          "Unquoted tasks have no quoted hours/amount",
          "Actual hours recorded indicate scope creep",
        ]
      )
    
    st.markdown("---")
    
    # Loss Making Jobs
    st.subheader("Loss-Making Jobs")
    losses = get_loss_making_jobs(job_summary).head(10)
    if len(losses) > 0:
      for _, j in losses.iterrows():
        st.markdown(f"**{str(j['Job_Name'])[:40]}** ({j['Job_No']}) - ${j['Margin']:,.0f}")
        reasons = []
        if j["Hours_Variance_Pct"] > 20:
          reasons.append(f"Hours +{j['Hours_Variance_Pct']:.0f}%")
        if j["Quote_Gap"] < -500:
          reasons.append("Underquoted")
        if reasons:
          st.caption(f" Drivers: {', '.join(reasons)}")
    else:
      st.success("No loss-making jobs!")
    callout_list(
      "Loss-making jobs",
      [
        "Loss = Margin < 0",
        "Drivers use Hours Variance %, Quote Gap, Effective Rate vs Cost Rate",
      ]
    )
  
  # =========================================================================
  # TAB 5: JOB DIAGNOSIS
  # =========================================================================
  with tab5:
    st.header("Job Diagnosis Tool")
    st.markdown("*Understand why a specific job performed the way it did*")
    callout_list(
      "Diagnosis explainer",
      [
        "Combines job-level and task-level indicators",
        "Overruns, underquoting, scope creep inform root causes",
      ]
    )
    
    # Job selector
    all_jobs = job_summary.apply(
      lambda r: f"{r['Job_No']} - {str(r['Job_Name'])[:40]} ({r['Client']})", axis=1
    ).tolist()
    
    selected_job = st.selectbox("Select a Job to Diagnose", ["-- Select --"] + all_jobs)
    
    if selected_job != "-- Select --":
      job_no = selected_job.split(" - ")[0]
      job_row = job_summary[job_summary["Job_No"] == job_no].iloc[0]
      job_tasks = task_summary[task_summary["Job_No"] == job_no]
      
      # Run diagnosis
      diagnosis = diagnose_job_margin(job_row, job_tasks)
      
      # Display job summary
      st.subheader(f"{job_row['Job_Name']}")
      st.caption(f"Client: {job_row['Client']} | {job_row['Month']}")
      
      # KPIs
      c1, c2, c3, c4, c5 = st.columns(5)
      c1.metric("Revenue", fmt_currency(job_row['Quoted_Amount']))
      c2.metric("Cost", fmt_currency(job_row['Base_Cost']))
      margin_label = "Positive" if job_row['Margin'] > 0 else "Negative"
      c3.metric(f"Margin ({margin_label})", fmt_currency(job_row['Margin']))
      gap_label = "Above" if job_row['Quote_Gap'] >= 0 else "Below"
      c4.metric(f"Quote Gap ({gap_label})", fmt_currency(job_row['Quote_Gap']))
      c5.metric("Hours Var", f"{job_row['Hours_Variance_Pct']:+.0f}%")
      
      st.markdown("---")
      
      # Diagnosis
      st.subheader("Diagnosis")
      st.markdown(f"**Summary:** {diagnosis['summary']}")
      
      if diagnosis['issues']:
        st.markdown("**Issues Identified:**")
        for issue in diagnosis['issues']:
          st.markdown(f"- {issue}")
      
      if diagnosis['root_causes']:
        st.markdown("**Root Causes:**")
        for cause in diagnosis['root_causes']:
          st.markdown(f"- {cause}")
      
      if diagnosis['recommendations']:
        st.markdown("**Recommendations:**")
        for rec in diagnosis['recommendations']:
          st.markdown(f"- {rec}")
      
      # Task breakdown
      if len(job_tasks) > 0:
        st.markdown("---")
        st.subheader("Task Analysis")
        callout_list(
          "Task flags",
          [
            "Unquoted tasks are scope creep",
            "Overrun tasks have Actual Hours > Quoted Hours",
            "Underquoted tasks have Quote Gap below benchmark",
          ]
        )
        
        unquoted_tasks = job_tasks[job_tasks['Is_Unquoted']]
        overrun_tasks = job_tasks[job_tasks['Is_Overrun'] & ~job_tasks['Is_Unquoted']]
        underquoted_tasks = job_tasks[job_tasks['Is_Underquoted']]
        
        if len(unquoted_tasks) > 0:
          st.markdown("** Unquoted Tasks (Scope Creep):**")
          for _, t in unquoted_tasks.iterrows():
            st.markdown(f"- {t['Task_Name']}: {t['Actual_Hours']:.0f} hrs, ${t['Base_Cost']:,.0f} cost")
        
        if len(overrun_tasks) > 0:
          st.markdown("** Hour Overruns:**")
          for _, t in overrun_tasks.iterrows():
            st.markdown(f"- {t['Task_Name']}: {t['Hours_Variance']:+.0f} hrs over")
        
        if len(underquoted_tasks) > 0:
          st.markdown("** Underquoted Tasks:**")
          for _, t in underquoted_tasks.iterrows():
            st.markdown(f"- {t['Task_Name']}: ${abs(t['Quote_Gap']):,.0f} below internal rates")
  
  # =========================================================================
  # TAB 6: PROFITABILITY DRIVERS
  # =========================================================================
  with tab6:
    st.header("Profitability Drivers")
    callout_list(
      "How to use this section",
      [
        "Each lens isolates a different margin driver",
        "Use the erosion summary to prioritize operational fixes",
        "Focus on consistent patterns, not single outliers",
      ]
    )
    st.markdown("**Definitions and formulas**")
    st.markdown(
      """
      - Delivery Overrun Cost = max(Actual Hours - Quoted Hours, 0) x Cost Rate/Hr
      - Underquote Cost = max(Expected Quote - Quoted Amount, 0)
      - Rate Erosion = max(Billable Rate/Hr - Quoted Rate/Hr, 0) x Quoted Hours
      - Total Erosion = Delivery Overrun Cost + Underquote Cost + Rate Erosion
      - Rate Delta = Effective Rate/Hr - Cost Rate/Hr
      """
    )
    metric_explainer(
      "Core metrics explained",
      [
        "Margin", "Margin_Pct", "Quote_Gap", "Quote_Gap_Pct",
        "Hours_Variance", "Hours_Variance_Pct",
        "Quoted_Rate_Hr", "Billable_Rate_Hr", "Effective_Rate_Hr", "Cost_Rate_Hr",
      ],
    )
    
    driver_df = job_summary.copy()
    driver_df["Overrun_Cost"] = np.where(
      driver_df["Hours_Variance"] > 0,
      driver_df["Hours_Variance"] * driver_df["Cost_Rate_Hr"],
      0
    )
    driver_df["Underquote_Cost"] = np.where(driver_df["Quote_Gap"] < 0, -driver_df["Quote_Gap"], 0)
    driver_df["Rate_Erosion"] = np.where(
      driver_df["Quoted_Rate_Hr"] < driver_df["Billable_Rate_Hr"],
      (driver_df["Billable_Rate_Hr"] - driver_df["Quoted_Rate_Hr"]) * driver_df["Quoted_Hours"],
      0
    )
    driver_df["Total_Erosion"] = driver_df["Overrun_Cost"] + driver_df["Underquote_Cost"] + driver_df["Rate_Erosion"]
    
    erosion_totals = pd.DataFrame({
      "Driver": ["Delivery Overruns", "Underquoting", "Rate Erosion"],
      "Amount": [
        driver_df["Overrun_Cost"].sum(),
        driver_df["Underquote_Cost"].sum(),
        driver_df["Rate_Erosion"].sum(),
      ],
    })
    st.subheader("Margin Erosion Summary")
    erosion_chart = alt.Chart(erosion_totals).mark_bar(size=38, cornerRadiusEnd=4).encode(
      y=alt.Y("Driver:N", sort="-x"),
      x=alt.X("Amount:Q", title="Estimated margin erosion ($)", axis=alt.Axis(format="$~s")),
      color=alt.Color("Driver:N", scale=alt.Scale(range=["#e4572e", "#f4d35e", "#6c757d"])),
      tooltip=["Driver", alt.Tooltip("Amount:Q", format="$,.0f")]
    ).properties(height=240)
    st.altair_chart(erosion_chart, width='stretch')
    callout_list(
      "What it means",
      [
        "Delivery overruns reflect extra hours beyond the quote",
        "Underquoting shows pricing below internal benchmarks",
        "Rate erosion captures discounting vs billable rates",
      ]
    )
    
    st.subheader("Pricing vs Margin Lens")
    pricing_chart = alt.Chart(driver_df).mark_circle(size=120).encode(
      x=alt.X("Quote_Gap_Pct:Q", title="Quote Gap %"),
      y=alt.Y("Margin_Pct:Q", title="Margin %"),
      color=alt.Color("Department:N"),
      size=alt.Size("Quoted_Amount:Q", title="Revenue", scale=alt.Scale(range=[80, 800])),
      tooltip=[
        "Job_Name", "Department", "Product",
        alt.Tooltip("Quote_Gap_Pct:Q", format=".1f", title="Quote Gap %"),
        alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
        alt.Tooltip("Quoted_Amount:Q", format="$,.0f", title="Revenue"),
      ],
    ).properties(height=340)
    st.altair_chart(pricing_chart, width='stretch')
    callout_list(
      "Operational insight",
      [
        "Bottom-left quadrant indicates discounting and poor margin",
        "Right shift without margin lift suggests delivery issues",
      ]
    )
    
    st.subheader("Delivery Efficiency Lens")
    delivery_chart = alt.Chart(driver_df).mark_circle(size=120).encode(
      x=alt.X("Hours_Variance_Pct:Q", title="Hours Variance %"),
      y=alt.Y("Margin_Pct:Q", title="Margin %"),
      color=alt.Color("Department:N"),
      size=alt.Size("Actual_Hours:Q", title="Actual Hours", scale=alt.Scale(range=[80, 800])),
      tooltip=[
        "Job_Name", "Department", "Product",
        alt.Tooltip("Hours_Variance_Pct:Q", format=".0f", title="Hours Var %"),
        alt.Tooltip("Margin_Pct:Q", format=".1f", title="Margin %"),
        alt.Tooltip("Actual_Hours:Q", format=",.0f", title="Actual Hours"),
      ],
    ).properties(height=340)
    st.altair_chart(delivery_chart, width='stretch')
    callout_list(
      "Operational insight",
      [
        "High variance with low margin points to estimation or scope control issues",
        "High variance with healthy margin may indicate premium pricing",
      ]
    )
    
    st.subheader("Rate Realization Lens")
    rate_gap = driver_df.copy()
    rate_gap["Rate_Delta"] = rate_gap["Effective_Rate_Hr"] - rate_gap["Cost_Rate_Hr"]
    rate_chart = alt.Chart(rate_gap).mark_circle(size=120).encode(
      x=alt.X("Rate_Delta:Q", title="Effective Rate - Cost Rate ($/hr)"),
      y=alt.Y("Margin_Pct:Q", title="Margin %"),
      color=alt.Color("Department:N"),
      size=alt.Size("Actual_Hours:Q", title="Actual Hours", scale=alt.Scale(range=[80, 800])),
      tooltip=[
        "Job_Name", "Department", "Product",
        alt.Tooltip("Rate_Delta:Q", format="$,.0f", title="Rate Delta"),
        alt.Tooltip("Effective_Rate_Hr:Q", format="$,.0f", title="Effective Rate/Hr"),
        alt.Tooltip("Cost_Rate_Hr:Q", format="$,.0f", title="Cost Rate/Hr"),
      ],
    ).properties(height=340)
    st.altair_chart(rate_chart, width='stretch')
    callout_list(
      "Operational insight",
      [
        "Negative rate delta indicates work priced below cost",
        "Improve rate discipline or reduce cost to restore margin",
      ]
    )
    
    st.subheader("Scope Creep Lens")
    unquoted = get_unquoted_tasks(task_summary).head(15)
    if len(unquoted) > 0:
      scope_chart = alt.Chart(unquoted).mark_bar(size=20, cornerRadiusEnd=3).encode(
        y=alt.Y("Task_Name:N", sort="-x", title="Task"),
        x=alt.X("Base_Cost:Q", title="Unquoted Cost ($)", axis=alt.Axis(format="$~s")),
        color=alt.value("#e4572e"),
        tooltip=["Task_Name", alt.Tooltip("Actual_Hours:Q", format=",.0f", title="Hours"), alt.Tooltip("Base_Cost:Q", format="$,.0f")]
      ).properties(height=320)
      st.altair_chart(scope_chart, width='stretch')
    else:
      st.info("No unquoted tasks found for the current filters.")
    callout_list(
      "Operational insight",
      [
        "Repeated unquoted tasks signal scope control gaps",
        "Use this list to update templates or pricing rules",
      ]
    )
    
    st.subheader("Top Margin Erosion Jobs")
    erosion_jobs = driver_df.sort_values("Total_Erosion", ascending=False).head(15)
    st.dataframe(
      erosion_jobs[[
        "Job_No", "Job_Name", "Department", "Product",
        "Overrun_Cost", "Underquote_Cost", "Rate_Erosion", "Total_Erosion", "Margin_Pct"
      ]].style.format({
        "Overrun_Cost": "${:,.0f}",
        "Underquote_Cost": "${:,.0f}",
        "Rate_Erosion": "${:,.0f}",
        "Total_Erosion": "${:,.0f}",
        "Margin_Pct": "{:.1f}%",
      }),
      width='stretch',
      height=320
    )
  
  # =========================================================================
  # TAB 7: RECONCILIATION
  # =========================================================================
  with tab7:
    st.header("Data Reconciliation")
    callout_list(
      "Reconciliation explainer",
      [
        "Shows how many rows were excluded by filters",
        "Validates totals across key fields",
      ]
    )
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Records", f"{recon['raw_records']:,}")
    c2.metric("Filtered", f"{recon['final_records']:,}")
    c3.metric("Excluded", f"{recon['raw_records'] - recon['final_records']:,}")
    
    st.subheader("Exclusions")
    st.dataframe(pd.DataFrame({
      "Filter": ["SG Allocation", "Non-Billable", "Other FY", "Other Dept"],
      "Excluded": [recon["excluded_sg_allocation"], recon["excluded_non_billable"],
             recon["excluded_other_fy"], recon["excluded_other_dept"]]
    }), width='stretch', hide_index=True)
    
    st.subheader("Validation Totals")
    totals_df = pd.DataFrame({
      "Metric": list(recon["totals"].keys()),
      "Value": [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in recon["totals"].values()]
    })
    st.dataframe(totals_df, width='stretch', hide_index=True)
    
    st.subheader("Metric Definitions")
    st.markdown("""
    | Metric | Formula | Description |
    |--------|---------|-------------|
    | **Quoted Amount** | From data | What we charge = REVENUE |
    | **Expected Quote** | Quoted Hours x Billable Rate | What we SHOULD have quoted |
    | **Base Cost** | Actual Hours x Cost Rate | Internal cost |
    | **Margin** | Quoted Amount - Base Cost | Profit |
    | **Quote Gap** | Quoted - Expected | + = premium, - = underquoted |
    | **Effective Rate/Hr** | Quoted Amount / Actual Hours | Revenue per hour worked |
    """)
    
    with st.expander("All Metric Definitions"):
      for key, defn in METRIC_DEFINITIONS.items():
        st.markdown(f"**{defn['name']}**")
        st.markdown(f"- Formula: `{defn['formula']}`")
        st.markdown(f"- {defn['desc']}")
        st.markdown("---")
  
  # =========================================================================
  # TAB 8: SMART QUOTE BUILDER
  # =========================================================================
  with tab8:
    st.header("Smart Quote Builder")
    callout_list(
      "How this works",
      [
        "Select a department, fiscal year, and product to anchor historical tasks",
        "Pick tasks, adjust proposed hours, and add custom lines",
        "Quote uses standard billable rates for pricing",
      ]
    )
    
    c1, c2, c3 = st.columns(3)
    builder_dept = c1.selectbox("Department", ["All Departments"] + dept_list, key="b_dept")
    builder_dept_filter = None if builder_dept == "All Departments" else builder_dept
    builder_fy = c2.selectbox(
      "Reference Fiscal Year", fy_list, index=len(fy_list) - 1, key="b_fy",
      format_func=lambda x: f"FY{str(x)[-2:]}"
    )
    
    base_filtered, _ = filter_data(
      df_parsed,
      exclude_sg_allocation=exclude_sg,
      billable_only=billable_only,
      fiscal_year=builder_fy,
      department=builder_dept_filter
    )
    products = sorted(base_filtered["Product"].dropna().unique().tolist())
    builder_product = c3.selectbox("Product", products if products else ["None"], key="b_prod")
    
    if builder_product == "None" or len(base_filtered) == 0:
      st.info("No data available for the selected context.")
    else:
      product_filtered = base_filtered[base_filtered["Product"] == builder_product]
      task_stats = compute_builder_task_stats(product_filtered)
      
      if len(task_stats) == 0:
        st.warning("No tasks found for this product and fiscal year.")
      else:
        st.subheader("Historical Task Library")
        callout_list(
          "Historical metrics",
          [
            "Frequency = % of jobs where the task appears",
            "Avg Quoted/Actual Hrs are per-job averages",
            "Rates are historical averages for the task",
          ]
        )
        
        st.dataframe(
          task_stats[[
            "Task_Name", "Frequency_Pct", "Avg_Quoted_Hours", "Avg_Actual_Hours",
            "Billable_Rate_Hr", "Cost_Rate_Hr"
          ]].style.format({
            "Frequency_Pct": "{:.0f}%",
            "Avg_Quoted_Hours": "{:,.1f}",
            "Avg_Actual_Hours": "{:,.1f}",
            "Billable_Rate_Hr": "${:,.0f}",
            "Cost_Rate_Hr": "${:,.0f}",
          }),
          width='stretch',
          height=320
        )
        
        default_tasks = task_stats[task_stats["Frequency_Pct"] >= 75]["Task_Name"].tolist()
        selected_tasks = st.multiselect(
          "Select tasks to include",
          task_stats["Task_Name"].tolist(),
          default=default_tasks
        )
        
        if selected_tasks:
          builder_df = task_stats[task_stats["Task_Name"].isin(selected_tasks)].copy()
          builder_df["Proposed_Hours"] = builder_df["Avg_Actual_Hours"].round(1)
          builder_df = builder_df[[
            "Task_Name", "Frequency_Pct", "Avg_Quoted_Hours", "Avg_Actual_Hours",
            "Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr", "Total_Actual_Hours"
          ]]
          
          st.subheader("Quote Builder")
          callout_list(
            "Input guidance",
            [
              "Proposed Hours default to Avg Actual Hours for realism",
              "Adjust hours to reflect scope differences for this quote",
            ]
          )
          
          edited = st.data_editor(
            builder_df,
            width='stretch',
            hide_index=True,
            column_config={
              "Task_Name": st.column_config.TextColumn("Task"),
              "Frequency_Pct": st.column_config.NumberColumn("Frequency %", format="%.0f"),
              "Avg_Quoted_Hours": st.column_config.NumberColumn("Avg Quoted Hrs", format="%.1f"),
              "Avg_Actual_Hours": st.column_config.NumberColumn("Avg Actual Hrs", format="%.1f"),
              "Proposed_Hours": st.column_config.NumberColumn("Proposed Hrs", min_value=0.0, step=0.5),
              "Billable_Rate_Hr": st.column_config.NumberColumn("Billable Rate/Hr", format="$%.0f"),
              "Cost_Rate_Hr": st.column_config.NumberColumn("Cost Rate/Hr", format="$%.0f"),
            },
            disabled=[
              "Task_Name", "Frequency_Pct", "Avg_Quoted_Hours",
              "Avg_Actual_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr", "Total_Actual_Hours"
            ],
          )
          
          st.subheader("Custom Line Items")
          callout_list(
            "Custom entries",
            [
              "Add unique tasks not in the history",
              "Provide hours and rates for pricing",
            ]
          )
          
          if "custom_items" not in st.session_state:
            st.session_state["custom_items"] = pd.DataFrame(
              columns=["Task_Name", "Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr"]
            )
          
          custom_items = st.data_editor(
            st.session_state["custom_items"],
            width='stretch',
            num_rows="dynamic",
            hide_index=True,
            column_config={
              "Task_Name": st.column_config.TextColumn("Task"),
              "Proposed_Hours": st.column_config.NumberColumn("Proposed Hrs", min_value=0.0, step=0.5),
              "Billable_Rate_Hr": st.column_config.NumberColumn("Billable Rate/Hr", min_value=0.0, format="$%.0f"),
              "Cost_Rate_Hr": st.column_config.NumberColumn("Cost Rate/Hr", min_value=0.0, format="$%.0f"),
            },
          )
          st.session_state["custom_items"] = custom_items
          
          custom_clean = custom_items.copy()
          for col in ["Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr"]:
            custom_clean[col] = pd.to_numeric(custom_clean[col], errors="coerce")
          custom_clean["Task_Name"] = custom_clean["Task_Name"].fillna("").astype(str).str.strip()
          valid_custom = custom_clean.dropna(
            subset=["Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr"]
          )
          valid_custom = valid_custom[
            (valid_custom["Proposed_Hours"] > 0) & (valid_custom["Task_Name"] != "")
          ]
          
          edited["Revenue"] = edited["Proposed_Hours"] * edited["Billable_Rate_Hr"]
          edited["Cost"] = edited["Proposed_Hours"] * edited["Cost_Rate_Hr"]
          
          custom_revenue = (valid_custom["Proposed_Hours"] * valid_custom["Billable_Rate_Hr"]).sum()
          custom_cost = (valid_custom["Proposed_Hours"] * valid_custom["Cost_Rate_Hr"]).sum()
          
          total_revenue = edited["Revenue"].sum() + custom_revenue
          total_cost = edited["Cost"].sum() + custom_cost
          margin = total_revenue - total_cost
          margin_pct = (margin / total_revenue * 100) if total_revenue > 0 else 0
          
          total_proposed_hours = edited["Proposed_Hours"].sum() + valid_custom["Proposed_Hours"].sum()
          total_hist_actual = edited["Total_Actual_Hours"].sum()
          
          st.subheader("Final Recommendation")
          col_a, col_b, col_c = st.columns(3)
          col_a.metric("Total Recommended Quote", fmt_currency(total_revenue))
          
          if margin_pct >= 35:
            margin_label = f"Good {margin_pct:.1f}%"
          elif margin_pct < 20:
            margin_label = f"Risk {margin_pct:.1f}%"
          else:
            margin_label = f"Watch {margin_pct:.1f}%"
          col_b.metric("Projected Margin %", margin_label)
          col_c.metric("Total Proposed Hours", f"{total_proposed_hours:,.1f}")
          
          if total_hist_actual > 0 and total_proposed_hours < total_hist_actual * 0.8:
            st.warning(
              "Proposed hours are >20% below historical actual hours for these tasks. "
              "Consider increasing scope or validating assumptions."
            )
  
  # Footer
  st.markdown("---")
  st.caption(
    f"Job Profitability Analysis | FY{str(selected_fy)[-2:]} | {selected_dept} | "
    f"{recon['final_records']:,} records | Revenue = Quoted Amount"
  )


if __name__ == "__main__":
  main()
