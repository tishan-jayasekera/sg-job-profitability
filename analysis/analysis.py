
"""
Job Profitability — Quote vs Ratecard vs Cost
============================================

This module powers the Streamlit dashboard.

Core business definitions (IMPORTANT)
------------------------------------
1) QUOTED (client quote) = the commercial value agreed with the client.
   - This is the "real" contracted number we expect to invoice and recognise as revenue.
   - In data: [Job Task] Quoted Amount (task allocation of the client quote)

2) INVOICED = revenue that has actually been invoiced to date.
   - In data: [Job Task] Invoiced Amount

3) RATECARD VALUE (internal) = Actual Hours × Billable Rate.
   - This is NOT recognised revenue.
   - It is an internal "control" metric to ensure work is priced to sustain target margins.
   - In data: derived from [Task] Billable Rate and actual hours.

4) BASE COST = Actual Hours × Base Rate (time & materials cost).
   - In data: derived from [Task] Base Rate and actual hours,
             or the provided Time+Material (Base) field when present.

Key profit lenses
-----------------
A) Quote Margin ($)        = Quoted Amount - Base Cost
B) Quote Margin (%)        = (Quoted Amount - Base Cost) / Quoted Amount
C) Invoiced Margin ($)     = Invoiced Amount - Base Cost
D) Invoiced Margin (%)     = (Invoiced Amount - Base Cost) / Invoiced Amount
E) Realisation (%)         = Invoiced Amount / Quoted Amount
F) Pricing Adequacy (%)    = Quoted Amount / Ratecard Value
   - < 100% implies the quote is below internal ratecard value for delivered hours

Time series note
----------------
This dataset does not include a true timesheet-entry date.
Month-on-month is therefore anchored to a configurable date (Task Start Date by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

EXCLUDED_TASK_NAMES = {"Social Garden Invoice Allocation"}

DEFAULT_DATE_ANCHOR = "Task Start Date"  # other options: Task Completed Date, Job Start Date


# =============================================================================
# HELPERS
# =============================================================================

def _as_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def _as_float(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    # remove currency commas
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0

def _as_bool_like(x) -> Optional[bool]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return None

def safe_div(n, d):
    """Vector-safe divide. Supports scalars, numpy arrays, and pandas Series."""
    n_arr = np.asarray(n, dtype="float64")
    d_arr = np.asarray(d, dtype="float64")
    out = np.zeros_like(n_arr, dtype="float64")
    np.divide(n_arr, d_arr, out=out, where=d_arr != 0)
    # Preserve scalar return type when inputs are scalar
    return float(out) if out.shape == () else out

def pct(n, d):
    """Percent = (n/d)*100 with vector-safe divide."""
    return safe_div(n, d) * 100.0


# =============================================================================
# FISCAL YEAR (AU) + MONTH KEYS
# =============================================================================

def get_fiscal_year(dt: pd.Timestamp) -> Optional[int]:
    """Australian FY (Jul-Jun). FY26 = Jul 2025 - Jun 2026 => represented as 2026."""
    if pd.isna(dt):
        return None
    return int(dt.year + 1) if int(dt.month) >= 7 else int(dt.year)

def get_fiscal_month(dt: pd.Timestamp) -> Optional[int]:
    """Return FY month index 1..12 where Jul=1, Aug=2, ..., Jun=12."""
    if pd.isna(dt):
        return None
    m = int(dt.month)
    return m - 6 if m >= 7 else m + 6

def fy_label(fy: Optional[int]) -> str:
    if fy is None or pd.isna(fy):
        return "Unknown"
    return f"FY{str(int(fy))[-2:]}"


# =============================================================================
# IO
# =============================================================================

def load_raw_data(path: str, sheet: str = "Data") -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)

@dataclass
class ParseReport:
    raw_records: int
    parsed_records: int
    notes: List[str]

def clean_and_parse(df: pd.DataFrame) -> Tuple[pd.DataFrame, ParseReport]:
    """
    Light cleaning and typing. Keeps all original columns, adds canonical fields.
    """
    notes: List[str] = []
    out = df.copy()

    # Required columns (soft checks)
    required = [
        "[Job] Job No.", "[Job Task] Name",
        "[Job Task] Quoted Time", "[Job Task] Quoted Amount",
        "[Job Task] Actual Time (totalled)", "[Job Task] Actual Time",
        "[Task] Base Rate", "[Task] Billable Rate",
        "Time+Material (Base)",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        notes.append(f"Missing expected columns: {missing}")

    # Standardise key strings
    for col in ["[Job] Job No.", "[Job Task] Name", "[Job] Category", "[Job] Name",
                "Department", "Product", "[Job] Client"]:
        if col in out.columns:
            out[col] = out[col].map(_as_str)

    # Date columns
    date_cols = [
        "[Job Task] Start Date", "[Job Task] Date Completed",
        "[Job] Start Date", "[Job] Completed Date",
    ]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Numerics (safe)
    num_cols = [
        "[Job Task] Quoted Time", "[Job Task] Quoted Amount",
        "[Job Task] Actual Time (totalled)", "[Job Task] Actual Time",
        "[Job Task] Invoiced Amount", "[Job Task] Invoiced Time",
        "[Task] Base Rate", "[Task] Billable Rate",
        "Time+Material (Base)", "[Job Task] Cost",
    ]
    for col in num_cols:
        if col in out.columns:
            out[col] = out[col].map(_as_float)

    # Flags
    for col in ["[Job Task] Billable", "[Job Task] Completed", "[Job Task] Allocated"]:
        if col in out.columns:
            out[col + "__bool"] = out[col].map(_as_bool_like)

    # Canonical hours fields
    out["Quoted_Hours"] = out.get("[Job Task] Quoted Time", 0.0)
    out["Quoted_Amount"] = out.get("[Job Task] Quoted Amount", 0.0)

    # Choose actual hours
    a_tot = out.get("[Job Task] Actual Time (totalled)", pd.Series(0.0, index=out.index))
    a_raw = out.get("[Job Task] Actual Time", pd.Series(0.0, index=out.index))
    out["Actual_Hours"] = np.where(a_tot > 0, a_tot, a_raw).astype(float)

    out["Base_Rate"] = out.get("[Task] Base Rate", 0.0)
    out["Billable_Rate"] = out.get("[Task] Billable Rate", 0.0)

    # Cost: prefer provided Time+Material (Base) if present and non-zero, else compute
    tm = out.get("Time+Material (Base)", pd.Series(0.0, index=out.index))
    out["Base_Cost"] = np.where(tm > 0, tm, out["Actual_Hours"] * out["Base_Rate"]).astype(float)

    # Internal ratecard value (NOT revenue)
    out["Ratecard_Value"] = (out["Actual_Hours"] * out["Billable_Rate"]).astype(float)

    # Revenue (to date)
    out["Invoiced_Amount"] = out.get("[Job Task] Invoiced Amount", 0.0)

    # Derived rate and ratios (task-grain)
    out["Quoted_Rate_Hr"] = np.where(out["Quoted_Hours"] > 0, out["Quoted_Amount"] / out["Quoted_Hours"], 0.0)
    out["Pricing_Adequacy_Pct"] = np.where(out["Ratecard_Value"] > 0, pct(out["Quoted_Amount"], out["Ratecard_Value"]), 0.0)
    out["Realisation_Pct"] = np.where(out["Quoted_Amount"] > 0, pct(out["Invoiced_Amount"], out["Quoted_Amount"]), 0.0)

    # Profit lenses
    out["Quote_Margin"] = out["Quoted_Amount"] - out["Base_Cost"]
    out["Quote_Margin_Pct"] = np.where(out["Quoted_Amount"] > 0, pct(out["Quote_Margin"], out["Quoted_Amount"]), 0.0)

    out["Invoiced_Margin"] = out["Invoiced_Amount"] - out["Base_Cost"]
    out["Invoiced_Margin_Pct"] = np.where(out["Invoiced_Amount"] > 0, pct(out["Invoiced_Margin"], out["Invoiced_Amount"]), 0.0)

    # Execution variance
    out["Hours_Variance"] = out["Actual_Hours"] - out["Quoted_Hours"]
    out["Hours_Variance_Pct"] = np.where(out["Quoted_Hours"] > 0, pct(out["Hours_Variance"], out["Quoted_Hours"]), 0.0)

    # Scope flags
    out["Is_Unquoted"] = (out["Quoted_Amount"] <= 0) & (out["Actual_Hours"] > 0)
    out["Is_Overrun"] = (out["Quoted_Hours"] > 0) & (out["Actual_Hours"] > out["Quoted_Hours"] + 1e-9)

    # Row-level hierarchy fallbacks
    if "Department" not in out.columns:
        out["Department"] = "Unknown"
    if "Product" not in out.columns:
        out["Product"] = "Unknown"

    rep = ParseReport(raw_records=len(df), parsed_records=len(out), notes=notes)
    return out, rep


# =============================================================================
# TIME ANCHORING + FILTERING
# =============================================================================

def add_time_keys(df: pd.DataFrame, anchor: str = DEFAULT_DATE_ANCHOR) -> pd.DataFrame:
    out = df.copy()

    if anchor == "Task Completed Date":
        anchor_col = "[Job Task] Date Completed"
    elif anchor == "Job Start Date":
        anchor_col = "[Job] Start Date"
    else:
        anchor_col = "[Job Task] Start Date"

    if anchor_col not in out.columns:
        out["Anchor_Date"] = pd.NaT
    else:
        out["Anchor_Date"] = pd.to_datetime(out[anchor_col], errors="coerce")

    # If anchor is missing, fall back to Job Start Date if present
    if "[Job] Start Date" in out.columns:
        out["Anchor_Date"] = out["Anchor_Date"].fillna(out["[Job] Start Date"])

    out["Fiscal_Year"] = out["Anchor_Date"].apply(get_fiscal_year)
    out["Fiscal_Month"] = out["Anchor_Date"].apply(get_fiscal_month)

    # For chart ordering
    out["FY_Month_Key"] = out.apply(
        lambda r: (int(r["Fiscal_Year"]) * 100 + int(r["Fiscal_Month"]))
        if (not pd.isna(r["Fiscal_Year"])) and (not pd.isna(r["Fiscal_Month"]))
        else np.nan,
        axis=1
    )
    out["FY_Label"] = out["Fiscal_Year"].apply(fy_label)
    out["FY_Month_Label"] = out.apply(
        lambda r: f"{r['FY_Label']} M{int(r['Fiscal_Month']):02d}" if not pd.isna(r["Fiscal_Month"]) else "Unknown",
        axis=1
    )

    return out


@dataclass
class FilterReport:
    raw_records: int
    excluded_sg_allocation: int
    excluded_missing_anchor: int
    excluded_other_fy: int
    excluded_other_filters: int
    final_records: int

def apply_filters(
    df: pd.DataFrame,
    *,
    fiscal_year: Optional[int] = None,
    departments: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    exclude_sg_allocation: bool = True,
    require_base_rate: bool = False,
    require_billable_rate: bool = False,
    date_anchor: str = DEFAULT_DATE_ANCHOR,
) -> Tuple[pd.DataFrame, FilterReport]:
    """
    Filter after parsing. Returns filtered df + reconciliation counts.
    """
    base = add_time_keys(df, anchor=date_anchor)

    raw_n = len(base)

    # Exclude SG allocation
    if exclude_sg_allocation:
        mask_excl = base["[Job Task] Name"].isin(EXCLUDED_TASK_NAMES)
        excl_sg = int(mask_excl.sum())
        base = base.loc[~mask_excl].copy()
    else:
        excl_sg = 0

    # Require anchor date (for trend views)
    missing_anchor = int(base["Anchor_Date"].isna().sum())
    base = base.loc[~base["Anchor_Date"].isna()].copy()

    # FY filter
    if fiscal_year is not None:
        other_fy = int((base["Fiscal_Year"] != fiscal_year).sum())
        base = base.loc[base["Fiscal_Year"] == fiscal_year].copy()
    else:
        other_fy = 0

    # Other filters
    before_other = len(base)

    if departments:
        base = base.loc[base["Department"].isin(departments)].copy()
    if products:
        base = base.loc[base["Product"].isin(products)].copy()
    if categories and "[Job] Category" in base.columns:
        base = base.loc[base["[Job] Category"].isin(categories)].copy()

    if require_base_rate:
        base = base.loc[base["Base_Rate"] > 0].copy()
    if require_billable_rate:
        base = base.loc[base["Billable_Rate"] > 0].copy()

    after_other = len(base)
    excl_other = int(before_other - after_other)

    rep = FilterReport(
        raw_records=raw_n,
        excluded_sg_allocation=excl_sg,
        excluded_missing_anchor=missing_anchor,
        excluded_other_fy=other_fy,
        excluded_other_filters=excl_other,
        final_records=len(base),
    )
    return base, rep


def get_available_fiscal_years(df: pd.DataFrame, date_anchor: str = DEFAULT_DATE_ANCHOR) -> List[int]:
    tmp = add_time_keys(df, anchor=date_anchor)
    yrs = sorted([int(y) for y in tmp["Fiscal_Year"].dropna().unique()])
    return yrs


# =============================================================================
# AGGREGATIONS (Task → Job → Product → Department)
# =============================================================================

AGG_SUM_COLS = [
    "Quoted_Hours", "Quoted_Amount",
    "Actual_Hours", "Base_Cost",
    "Ratecard_Value", "Invoiced_Amount",
    "Hours_Variance",
    "Quote_Margin", "Invoiced_Margin",
]

def _weighted_rate(sum_value: pd.Series, sum_hours: pd.Series) -> pd.Series:
    return np.where(sum_hours > 0, sum_value / sum_hours, 0.0)

def _post_agg_metrics(df_agg: pd.DataFrame) -> pd.DataFrame:
    out = df_agg.copy()
    out["Quote_Margin_Pct"] = np.where(out["Quoted_Amount"] > 0, pct(out["Quote_Margin"], out["Quoted_Amount"]), 0.0)
    out["Invoiced_Margin_Pct"] = np.where(out["Invoiced_Amount"] > 0, pct(out["Invoiced_Margin"], out["Invoiced_Amount"]), 0.0)
    out["Hours_Variance_Pct"] = np.where(out["Quoted_Hours"] > 0, pct(out["Hours_Variance"], out["Quoted_Hours"]), 0.0)
    out["Realisation_Pct"] = np.where(out["Quoted_Amount"] > 0, pct(out["Invoiced_Amount"], out["Quoted_Amount"]), 0.0)
    out["Pricing_Adequacy_Pct"] = np.where(out["Ratecard_Value"] > 0, pct(out["Quoted_Amount"], out["Ratecard_Value"]), 0.0)
    # Implied average rates
    out["Avg_Base_Rate"] = _weighted_rate(out["Base_Cost"], out["Actual_Hours"])
    out["Avg_Billable_Rate"] = _weighted_rate(out["Ratecard_Value"], out["Actual_Hours"])
    out["Avg_Quoted_Rate"] = _weighted_rate(out["Quoted_Amount"], out["Quoted_Hours"])
    return out

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Department", "Product", "[Job] Category",
        "[Job] Job No.", "[Job] Name", "[Job] Client",
        "[Job Task] Name",
    ]
    keep_first = {
        "[Job] Client Manager": "first",
        "[Job] Status": "first",
    }
    agg = df.groupby(keys, dropna=False).agg(
        {**{c: "sum" for c in AGG_SUM_COLS}, **keep_first}
    ).reset_index()

    agg["Task_Count"] = 1  # already task-grain per row group
    agg = _post_agg_metrics(agg)

    # Flags at task rollup
    agg["Is_Unquoted"] = (agg["Quoted_Amount"] <= 0) & (agg["Actual_Hours"] > 0)
    agg["Is_Overrun"] = (agg["Quoted_Hours"] > 0) & (agg["Actual_Hours"] > agg["Quoted_Hours"] + 1e-9)
    return agg

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Department", "Product", "[Job] Category",
        "[Job] Job No.", "[Job] Name", "[Job] Client",
    ]
    keep_first = {
        "[Job] Client Manager": "first",
        "[Job] Status": "first",
    }
    agg = df.groupby(keys, dropna=False).agg(
        {**{c: "sum" for c in AGG_SUM_COLS}, **keep_first}
    ).reset_index()

    agg["Task_Count"] = df.groupby(keys, dropna=False)["[Job Task] Name"].nunique().values
    agg = _post_agg_metrics(agg)

    agg["Is_At_Loss_On_Quote"] = agg["Quote_Margin"] < 0
    agg["Is_At_Loss_Invoiced"] = agg["Invoiced_Margin"] < 0
    agg["Is_Overrun"] = (agg["Quoted_Hours"] > 0) & (agg["Actual_Hours"] > agg["Quoted_Hours"] + 1e-9)
    agg["Is_Unquoted_Work"] = (agg["Quoted_Amount"] <= 0) & (agg["Actual_Hours"] > 0)

    # "Margin erosion" as a planning vs delivery delta:
    # If you wanted: quote margin % vs invoiced margin % (only where both available)
    agg["Margin_Erosion_PctPts"] = np.where(
        (agg["Quoted_Amount"] > 0) & (agg["Invoiced_Amount"] > 0),
        agg["Invoiced_Margin_Pct"] - agg["Quote_Margin_Pct"],
        0.0
    )
    return agg

def compute_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["Department", "Product"]
    agg = df.groupby(keys, dropna=False).agg({c: "sum" for c in AGG_SUM_COLS}).reset_index()
    agg["Job_Count"] = df.groupby(keys, dropna=False)["[Job] Job No."].nunique().values
    agg["Task_Count"] = df.groupby(keys, dropna=False)["[Job Task] Name"].nunique().values
    agg = _post_agg_metrics(agg)
    return agg

def compute_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["Department"]
    agg = df.groupby(keys, dropna=False).agg({c: "sum" for c in AGG_SUM_COLS}).reset_index()
    agg["Job_Count"] = df.groupby(keys, dropna=False)["[Job] Job No."].nunique().values
    agg["Task_Count"] = df.groupby(keys, dropna=False)["[Job Task] Name"].nunique().values
    agg = _post_agg_metrics(agg)
    return agg


def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Roll up by job category (and department/product for context)."""
    if "[Job] Category" not in df.columns:
        return pd.DataFrame()
    keys = ["Department", "Product", "[Job] Category"]
    agg = df.groupby(keys, dropna=False).agg({c: "sum" for c in AGG_SUM_COLS}).reset_index()
    agg["Job_Count"] = df.groupby(keys, dropna=False)["[Job] Job No."].nunique().values
    agg["Task_Count"] = df.groupby(keys, dropna=False)["[Job Task] Name"].nunique().values
    agg = _post_agg_metrics(agg)
    return agg
def compute_monthly_summary(
    df: pd.DataFrame,
    *,
    level: str = "All",
    department: Optional[str] = None,
    product: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns month-on-month aggregates using Fiscal_Year + Fiscal_Month.
    level: "All" | "Department" | "Product"
    """
    base = df.copy()
    if department:
        base = base.loc[base["Department"] == department].copy()
    if product:
        base = base.loc[base["Product"] == product].copy()

    keys = ["Fiscal_Year", "Fiscal_Month", "FY_Month_Key", "FY_Month_Label"]

    if level == "Department":
        keys = ["Department"] + keys
    elif level == "Product":
        keys = ["Department", "Product"] + keys

    agg = base.groupby(keys, dropna=False).agg({c: "sum" for c in AGG_SUM_COLS}).reset_index()
    agg = _post_agg_metrics(agg)
    agg = agg.sort_values(keys)
    return agg


# =============================================================================
# INSIGHTS (LIGHTWEIGHT, OPERATIONAL)
# =============================================================================

def summarize_portfolio(job_summary: pd.DataFrame) -> Dict[str, float]:
    js = job_summary.copy()
    total_jobs = len(js)
    quoted = float(js["Quoted_Amount"].sum())
    invoiced = float(js["Invoiced_Amount"].sum())
    cost = float(js["Base_Cost"].sum())
    ratecard = float(js["Ratecard_Value"].sum())
    quote_margin = float(js["Quote_Margin"].sum())
    invoiced_margin = float(js["Invoiced_Margin"].sum())

    return {
        "total_jobs": total_jobs,
        "quoted_amount": quoted,
        "invoiced_amount": invoiced,
        "ratecard_value": ratecard,
        "base_cost": cost,
        "quote_margin": quote_margin,
        "invoiced_margin": invoiced_margin,
        "quote_margin_pct": pct(quote_margin, quoted) if quoted else 0.0,
        "invoiced_margin_pct": pct(invoiced_margin, invoiced) if invoiced else 0.0,
        "realisation_pct": pct(invoiced, quoted) if quoted else 0.0,
        "pricing_adequacy_pct": pct(quoted, ratecard) if ratecard else 0.0,
        "jobs_at_loss_on_quote": int((js["Quote_Margin"] < 0).sum()),
        "jobs_at_loss_invoiced": int((js["Invoiced_Margin"] < 0).sum()),
        "jobs_overrun": int(js["Is_Overrun"].sum()) if "Is_Overrun" in js.columns else int((js["Hours_Variance"] > 0).sum()),
    }

def top_lists(job_summary: pd.DataFrame, task_summary: pd.DataFrame, n: int = 10) -> Dict[str, pd.DataFrame]:
    js = job_summary.copy()
    ts = task_summary.copy()

    out: Dict[str, pd.DataFrame] = {}

    # Biggest quote margin losses
    out["worst_quote_margin_jobs"] = js.nsmallest(n, "Quote_Margin")[
        ["Department", "Product", "[Job] Job No.", "[Job] Name", "[Job] Client",
         "Quoted_Amount", "Base_Cost", "Quote_Margin", "Quote_Margin_Pct", "Pricing_Adequacy_Pct", "Hours_Variance_Pct"]
    ]

    # Biggest invoiced margin losses
    out["worst_invoiced_margin_jobs"] = js.nsmallest(n, "Invoiced_Margin")[
        ["Department", "Product", "[Job] Job No.", "[Job] Name", "[Job] Client",
         "Invoiced_Amount", "Base_Cost", "Invoiced_Margin", "Invoiced_Margin_Pct", "Realisation_Pct"]
    ]

    # Under-priced vs ratecard (pricing adequacy low) among meaningful hours
    meaningful = js[js["Ratecard_Value"] > 0].copy()
    out["most_underpriced_jobs"] = meaningful.nsmallest(n, "Pricing_Adequacy_Pct")[
        ["Department", "Product", "[Job] Job No.", "[Job] Name", "[Job] Client",
         "Quoted_Amount", "Ratecard_Value", "Pricing_Adequacy_Pct", "Quote_Margin_Pct"]
    ]

    # Scope creep: unquoted tasks with cost
    scope = ts[(ts["Is_Unquoted"]) & (ts["Base_Cost"] > 0)].copy()
    out["scope_creep_tasks"] = scope.nlargest(n, "Base_Cost")[
        ["Department", "Product", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
         "Actual_Hours", "Base_Cost", "Ratecard_Value"]
    ]

    # Overrun tasks: hours variance
    overrun = ts[(ts["Is_Overrun"]) & (ts["Quoted_Hours"] > 0)].copy()
    out["overrun_tasks"] = overrun.nlargest(n, "Hours_Variance")[
        ["Department", "Product", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
         "Quoted_Hours", "Actual_Hours", "Hours_Variance", "Hours_Variance_Pct", "Base_Cost"]
    ]

    return out

def generate_headlines(metrics: Dict[str, float]) -> List[str]:
    """
    Simple narrative bullets.
    """
    headlines: List[str] = []

    # Core
    headlines.append(
        f"Portfolio: {metrics['total_jobs']:,} jobs | "
        f"Quoted ${metrics['quoted_amount']:,.0f} | "
        f"Invoiced ${metrics['invoiced_amount']:,.0f} "
        f"({metrics['realisation_pct']:.1f}% realised)."
    )

    headlines.append(
        f"Cost base: ${metrics['base_cost']:,.0f} | "
        f"Quote margin: {metrics['quote_margin_pct']:.1f}% | "
        f"Invoiced margin: {metrics['invoiced_margin_pct']:.1f}%."
    )

    # Pricing adequacy
    pa = metrics["pricing_adequacy_pct"]
    if pa > 0:
        if pa < 90:
            headlines.append(
                f"Pricing adequacy is weak: quoted value is only {pa:.1f}% of internal ratecard value "
                f"(quote below ratecard for delivered hours)."
            )
        elif pa < 110:
            headlines.append(
                f"Pricing adequacy is tight: quote is {pa:.1f}% of internal ratecard value "
                f"(limited buffer vs standard ratecard for delivered hours)."
            )
        else:
            headlines.append(
                f"Pricing adequacy is strong: quote is {pa:.1f}% of internal ratecard value."
            )

    # Losses
    if metrics["jobs_at_loss_on_quote"] > 0:
        headlines.append(
            f"{metrics['jobs_at_loss_on_quote']:,} jobs are loss-making on the quote margin lens (Quote < Cost)."
        )
    if metrics["jobs_at_loss_invoiced"] > 0:
        headlines.append(
            f"{metrics['jobs_at_loss_invoiced']:,} jobs are loss-making based on invoiced-to-date (Invoiced < Cost)."
        )

    # Overruns
    headlines.append(
        f"{metrics['jobs_overrun']:,} jobs show hours overruns vs quoted hours (where quoted hours exist)."
    )

    return headlines


METRIC_DEFINITIONS = {
    "Quoted_Amount": {"name": "Quoted Amount", "formula": "[Job Task] Quoted Amount", "description": "Client quote value allocated to task (commercial benchmark)."},
    "Invoiced_Amount": {"name": "Invoiced Amount", "formula": "[Job Task] Invoiced Amount", "description": "Revenue invoiced to date (realised)."},
    "Ratecard_Value": {"name": "Ratecard Value (Internal)", "formula": "Actual Hours × [Task] Billable Rate", "description": "Internal value at standard billable rate; NOT recognised revenue."},
    "Base_Cost": {"name": "Base Cost", "formula": "Actual Hours × [Task] Base Rate", "description": "Internal labour cost (T&M base)."},
    "Quote_Margin": {"name": "Quote Margin ($)", "formula": "Quoted Amount - Base Cost", "description": "Margin implied if quote is the commercial revenue benchmark."},
    "Invoiced_Margin": {"name": "Invoiced Margin ($)", "formula": "Invoiced Amount - Base Cost", "description": "Margin based on revenue invoiced to date."},
    "Realisation_Pct": {"name": "Realisation (%)", "formula": "(Invoiced Amount / Quoted Amount) × 100", "description": "How much of quoted value has been invoiced to date."},
    "Pricing_Adequacy_Pct": {"name": "Pricing Adequacy (%)", "formula": "(Quoted Amount / Ratecard Value) × 100", "description": "Is the quote above or below internal ratecard value for delivered hours? <100% suggests underpricing vs ratecard."},
}