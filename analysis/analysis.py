"""
Job Profitability Analysis Module
==================================
Hierarchy: Department → Product → Job → Task
Time-Series: Month-on-Month Trend Analysis

FINANCIAL MODEL:
----------------
1. Quoted Amount    = What we charged the client = REVENUE
2. Expected Quote   = Quoted Hours × Billable Rate = What we SHOULD have quoted
3. Base Cost        = Actual Hours × Cost Rate = What it cost us

SANITY CHECK (Quoting Accuracy):
--------------------------------
- Quote Gap = Quoted Amount - Expected Quote
  - Positive (+) = Quoted ABOVE internal rates (premium pricing)
  - Negative (-) = Quoted BELOW internal rates (discounting / underquoting)

MARGIN:
-------
- Margin = Quoted Amount - Base Cost
- Margin % = Margin / Quoted Amount × 100
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional, List


# =============================================================================
# DATA PARSING
# =============================================================================

def parse_numeric(val) -> float:
    """Parse numeric, handling commas and #N/A."""
    if pd.isna(val) or str(val).strip() in ("#N/A", "", "N/A", "-"):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(",", "").strip())
    except:
        return 0.0


def parse_date(val):
    """Parse date in various formats."""
    if pd.isna(val) or str(val).strip() in ("", "#N/A", "N/A"):
        return pd.NaT
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.to_datetime(val)
    try:
        for fmt in ["%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]:
            try:
                return pd.to_datetime(val, format=fmt)
            except:
                continue
        return pd.to_datetime(val, dayfirst=True)
    except:
        return pd.NaT


def get_fiscal_year(date) -> Optional[int]:
    """Australian FY (Jul-Jun). FY26 = Jul 2025 - Jun 2026."""
    if pd.isna(date):
        return None
    return date.year + 1 if date.month >= 7 else date.year


def get_fy_label(fy) -> str:
    if pd.isna(fy):
        return "Unknown"
    return f"FY{str(int(fy))[-2:]}"


def get_fy_month(date) -> Optional[int]:
    """Get fiscal year month (1=Jul, 2=Aug, ..., 12=Jun)."""
    if pd.isna(date):
        return None
    month = date.month
    return month - 6 if month >= 7 else month + 6


def get_fy_month_label(fy_month) -> str:
    """Convert FY month number to label."""
    if pd.isna(fy_month):
        return "Unknown"
    months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    return months[int(fy_month) - 1] if 1 <= int(fy_month) <= 12 else "Unknown"


def get_calendar_month_label(date) -> str:
    """Get calendar month label (e.g., 'Jul 2025')."""
    if pd.isna(date):
        return "Unknown"
    return date.strftime("%b %Y")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data(filepath, sheet_name: str = "Data") -> pd.DataFrame:
    """Load raw Excel data."""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse columns and compute derived metrics.
    
    REVENUE = Quoted Amount
    BENCHMARK = Expected Quote (Quoted Hours × Billable Rate)
    """
    df = df.copy()
    
    # Parse numeric columns
    numeric_cols = [
        "[Job Task] Quoted Time", "[Job Task] Remaining Time",
        "[Job Task] Quoted Amount", "[Job Task] Actual Time",
        "[Job Task] Actual Time (totalled)", "[Job Task] % Complete",
        "[Job Task] Billable Amount", "[Job Task] Invoiced Time",
        "[Job Task] Invoiced Amount", "[Job Task] Cost",
        "[Task] Base Rate", "[Task] Billable Rate",
        "Time+Material (Base)", "[Job] Budget"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric)
    
    # Parse date columns
    date_cols = [
        "[Job] Start Date", "[Job] Due Date", "[Job] Completed Date",
        "[Job Task] Start Date", "[Job Task] Due Date", "[Job Task] Date Completed"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)
    
    # Fiscal year and month
    df["Fiscal_Year"] = df["[Job] Start Date"].apply(get_fiscal_year)
    df["FY_Label"] = df["Fiscal_Year"].apply(get_fy_label)
    df["FY_Month"] = df["[Job] Start Date"].apply(get_fy_month)
    df["FY_Month_Label"] = df["FY_Month"].apply(get_fy_month_label)
    df["Calendar_Month"] = df["[Job] Start Date"].apply(get_calendar_month_label)
    df["Year_Month"] = df["[Job] Start Date"].dt.to_period('M')
    
    # Clean Product and Department
    if "Product" in df.columns:
        df["Product"] = df["Product"].fillna("Unknown").astype(str).str.strip()
    else:
        df["Product"] = "Unknown"
    
    if "Department" in df.columns:
        df["Department"] = df["Department"].fillna("Unknown").astype(str).str.strip()
    else:
        df["Department"] = "Unknown"
    
    # =========================================================================
    # RATES (per hour)
    # =========================================================================
    df["Billable_Rate_Hr"] = df["[Task] Billable Rate"]  # Internal standard rate
    df["Cost_Rate_Hr"] = df["[Task] Base Rate"]  # Internal cost rate
    
    # Quoted Rate = What we actually charged per hour
    df["Quoted_Rate_Hr"] = np.where(
        df["[Job Task] Quoted Time"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Quoted Time"],
        0
    )
    
    # Effective Rate = Revenue per actual hour worked
    df["Effective_Rate_Hr"] = np.where(
        df["[Job Task] Actual Time (totalled)"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Actual Time (totalled)"],
        0
    )
    
    # =========================================================================
    # EXPECTED QUOTE = What we SHOULD have quoted (Quoted Hours × Billable Rate)
    # =========================================================================
    df["Expected_Quote"] = df["[Job Task] Quoted Time"] * df["Billable_Rate_Hr"]
    
    # =========================================================================
    # QUOTE GAP = Sanity check: Quoted Amount vs Expected Quote
    # =========================================================================
    df["Quote_Gap"] = df["[Job Task] Quoted Amount"] - df["Expected_Quote"]
    df["Quote_Gap_Pct"] = np.where(
        df["Expected_Quote"] > 0,
        (df["Quote_Gap"] / df["Expected_Quote"]) * 100,
        0
    )
    
    # Rate Gap = Quoted Rate vs Billable Rate
    df["Rate_Gap"] = df["Quoted_Rate_Hr"] - df["Billable_Rate_Hr"]
    df["Rate_Gap_Pct"] = np.where(
        df["Billable_Rate_Hr"] > 0,
        (df["Rate_Gap"] / df["Billable_Rate_Hr"]) * 100,
        0
    )
    
    # =========================================================================
    # COSTS
    # =========================================================================
    df["Base_Cost"] = df["[Job Task] Actual Time (totalled)"] * df["Cost_Rate_Hr"]
    df["Quoted_Cost"] = df["[Job Task] Quoted Time"] * df["Cost_Rate_Hr"]
    
    return df


# =============================================================================
# FILTERING
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    exclude_sg_allocation: bool = True,
    billable_only: bool = True,
    fiscal_year: int = None,
    department: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """Apply filters and return reconciliation."""
    recon = {
        "raw_records": len(df),
        "excluded_sg_allocation": 0,
        "excluded_non_billable": 0,
        "excluded_other_fy": 0,
        "excluded_other_dept": 0,
        "final_records": 0,
    }
    
    df_f = df.copy()
    
    if exclude_sg_allocation:
        if "[Job Task] Name" in df_f.columns:
            task_name = df_f["[Job Task] Name"].fillna("").astype(str).str.strip()
            mask = (
                task_name.str.contains("social garden invoice allocation", case=False) |
                task_name.str.contains("social garden allocation", case=False) |
                task_name.str.contains("sg allocation", case=False)
            )
            recon["excluded_sg_allocation"] = mask.sum()
            df_f = df_f[~mask]
    
    if billable_only:
        mask = (df_f["Cost_Rate_Hr"] > 0) & (df_f["Billable_Rate_Hr"] > 0)
        recon["excluded_non_billable"] = (~mask).sum()
        df_f = df_f[mask]
    
    if fiscal_year is not None:
        mask = df_f["Fiscal_Year"] == fiscal_year
        recon["excluded_other_fy"] = (~mask).sum()
        df_f = df_f[mask]
    
    if department is not None:
        mask = df_f["Department"] == department
        recon["excluded_other_dept"] = (~mask).sum()
        df_f = df_f[mask]
    
    recon["final_records"] = len(df_f)
    return df_f, recon


def get_available_fiscal_years(df: pd.DataFrame) -> list:
    return sorted([int(y) for y in df["Fiscal_Year"].dropna().unique() if pd.notna(y)])


def get_available_departments(df: pd.DataFrame) -> list:
    return sorted(df["Department"].dropna().unique().tolist())


def get_available_products(df: pd.DataFrame, department: str = None) -> list:
    if department:
        prods = df[df["Department"] == department]["Product"].dropna().unique()
    else:
        prods = df["Product"].dropna().unique()
    return sorted(prods.tolist())


# =============================================================================
# RECONCILIATION
# =============================================================================

def compute_reconciliation_totals(df: pd.DataFrame, recon: Dict) -> Dict:
    """Compute validation totals."""
    recon["totals"] = {
        # Hours
        "sum_quoted_hours": df["[Job Task] Quoted Time"].sum(),
        "sum_actual_hours": df["[Job Task] Actual Time (totalled)"].sum(),
        # Revenue
        "sum_quoted_amount": df["[Job Task] Quoted Amount"].sum(),
        # Benchmark
        "sum_expected_quote": df["Expected_Quote"].sum(),
        # Quote Gap
        "sum_quote_gap": df["Quote_Gap"].sum(),
        # Cost
        "sum_base_cost": df["Base_Cost"].sum(),
        # Rates
        "avg_quoted_rate_hr": df[df["Quoted_Rate_Hr"] > 0]["Quoted_Rate_Hr"].mean() if len(df[df["Quoted_Rate_Hr"] > 0]) > 0 else 0,
        "avg_billable_rate_hr": df["Billable_Rate_Hr"].mean(),
        "avg_cost_rate_hr": df["Cost_Rate_Hr"].mean(),
        # Counts
        "unique_jobs": df["[Job] Job No."].nunique(),
        "unique_products": df["Product"].nunique(),
        "unique_departments": df["Department"].nunique(),
    }
    return recon


# =============================================================================
# MONTHLY TREND SUMMARIES
# =============================================================================

def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Fiscal_Year", "FY_Month"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Fiscal_Year", "FY_Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Expected_Quote", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values("Month_Sort").reset_index(drop=True)
    
    # Margin
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    
    # Quote Gap (sanity check)
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    # Rates
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Effective_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Quoted_Amount"] / g["Actual_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Expected_Quote"] / g["Quoted_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    
    # Hours variance
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


def compute_monthly_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trends by department."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Department"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Department",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Expected_Quote", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values(["Month_Sort", "Department"]).reset_index(drop=True)
    
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    return g


def compute_monthly_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trends by product."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Department", "Product"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Department", "Product",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Expected_Quote", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values(["Month_Sort", "Department", "Product"]).reset_index(drop=True)
    
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    
    return g


# =============================================================================
# DEPARTMENT SUMMARY
# =============================================================================

def compute_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Department summary."""
    g = df.groupby("Department").agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
        "Product": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Expected_Quote", "Base_Cost", "Job_Count", "Product_Count"]
    
    # Margin
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    
    # Quote Gap (sanity check)
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    # Rates
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Effective_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Quoted_Amount"] / g["Actual_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Expected_Quote"] / g["Quoted_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    
    g["Rate_Gap"] = g["Quoted_Rate_Hr"] - g["Billable_Rate_Hr"]
    
    # Hours
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


# =============================================================================
# PRODUCT SUMMARY
# =============================================================================

def compute_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Product summary."""
    g = df.groupby(["Department", "Product"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Product", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Expected_Quote", "Base_Cost", "Job_Count"]
    
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Effective_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Quoted_Amount"] / g["Actual_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Expected_Quote"] / g["Quoted_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


# =============================================================================
# JOB SUMMARY
# =============================================================================

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Job summary."""
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name",
        "[Job] Client", "[Job] Client Manager", "[Job] Status",
        "[Job] Start Date", "Fiscal_Year", "FY_Label", "Calendar_Month"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "[Job] Budget": "first",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Client", "Client_Manager", "Status",
        "Start_Date", "Fiscal_Year", "FY_Label", "Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Expected_Quote", "Base_Cost", "Budget"
    ]
    
    # Margin
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    
    # Quote Gap (sanity check)
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    # Rates
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Effective_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Quoted_Amount"] / g["Actual_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Expected_Quote"] / g["Quoted_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    
    g["Rate_Gap"] = g["Quoted_Rate_Hr"] - g["Billable_Rate_Hr"]
    
    # Hours
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    
    # FLAGS
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Is_Loss"] = g["Margin"] < 0
    g["Is_Underquoted"] = g["Quote_Gap"] < -100  # Quoted below internal benchmark
    g["Is_Premium"] = g["Quote_Gap"] > 100  # Quoted above internal benchmark
    g["Significant_Overrun"] = g["Hours_Variance_Pct"] > 20
    
    return g


# =============================================================================
# TASK SUMMARY
# =============================================================================

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Task summary."""
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
        "Task Category", "Fiscal_Year", "FY_Label", "Calendar_Month"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Expected_Quote": "sum",
        "Base_Cost": "sum",
        "Cost_Rate_Hr": "mean",
        "Billable_Rate_Hr": "mean",
        "Quoted_Rate_Hr": "mean",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Fiscal_Year", "FY_Label", "Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Expected_Quote", "Base_Cost", "Cost_Rate_Hr", "Billable_Rate_Hr", "Quoted_Rate_Hr"
    ]
    
    # Margin
    g["Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Margin"] / g["Quoted_Amount"]) * 100, 0)
    
    # Quote Gap
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    
    # Effective Rate
    g["Effective_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Quoted_Amount"] / g["Actual_Hours"], 0)
    
    # Rate gap
    g["Rate_Gap"] = g["Quoted_Rate_Hr"] - g["Billable_Rate_Hr"]
    
    # Hours
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    
    # Flags
    g["Is_Unquoted"] = (g["Quoted_Hours"] == 0) & (g["Actual_Hours"] > 0)
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Is_Underquoted"] = g["Quote_Gap"] < -50
    
    return g


# =============================================================================
# NARRATIVE INSIGHTS
# =============================================================================

def generate_insights(
    job_summary: pd.DataFrame,
    dept_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    task_summary: pd.DataFrame
) -> Dict:
    """Generate narrative insights explaining profitability drivers."""
    insights = {
        "headline": [],
        "margin_drivers": [],
        "quoting_issues": [],
        "scope_issues": [],
        "rate_issues": [],
        "action_items": []
    }
    
    if len(job_summary) == 0:
        return insights
    
    # Overall metrics
    total_quoted = job_summary["Quoted_Amount"].sum()
    total_cost = job_summary["Base_Cost"].sum()
    total_expected = job_summary["Expected_Quote"].sum()
    
    margin = total_quoted - total_cost
    margin_pct = (margin / total_quoted * 100) if total_quoted > 0 else 0
    quote_gap = total_quoted - total_expected
    quote_gap_pct = (quote_gap / total_expected * 100) if total_expected > 0 else 0
    
    # HEADLINE INSIGHTS
    if margin_pct < 20:
        insights["headline"].append(f"🔴 **Low Margin Alert**: Overall margin at {margin_pct:.1f}% — below 20% threshold")
    elif margin_pct < 35:
        insights["headline"].append(f"🟡 **Margin Below Target**: {margin_pct:.1f}% (target: 35%+)")
    else:
        insights["headline"].append(f"🟢 **Healthy Margin**: {margin_pct:.1f}%")
    
    # Quoting accuracy
    if quote_gap < -10000:
        insights["headline"].append(
            f"⚠️ **Underquoting**: ${abs(quote_gap):,.0f} below internal rates ({quote_gap_pct:+.0f}%)"
        )
    elif quote_gap > 10000:
        insights["headline"].append(
            f"✅ **Premium Pricing**: ${quote_gap:,.0f} above internal rates ({quote_gap_pct:+.0f}%)"
        )
    
    # Loss-making jobs
    loss_jobs = job_summary[job_summary["Is_Loss"]]
    if len(loss_jobs) > 0:
        total_losses = loss_jobs["Margin"].sum()
        insights["headline"].append(f"💸 **{len(loss_jobs)} loss-making jobs** totaling ${abs(total_losses):,.0f} in losses")
    
    # MARGIN DRIVER ANALYSIS
    if len(dept_summary) > 0:
        worst_dept = dept_summary.loc[dept_summary["Margin_Pct"].idxmin()]
        if worst_dept["Margin_Pct"] < 20:
            insights["margin_drivers"].append(
                f"🔻 **{worst_dept['Department']}** dragging margins at {worst_dept['Margin_Pct']:.1f}%"
            )
        
        best_dept = dept_summary.loc[dept_summary["Margin_Pct"].idxmax()]
        if best_dept["Margin_Pct"] > 40 and best_dept["Quoted_Amount"] > 10000:
            insights["margin_drivers"].append(
                f"🔺 **{best_dept['Department']}** leading with {best_dept['Margin_Pct']:.1f}% margin"
            )
        
        # Departments quoting below internal rates
        underquoting_depts = dept_summary[dept_summary["Quote_Gap_Pct"] < -10]
        for _, d in underquoting_depts.iterrows():
            insights["quoting_issues"].append(
                f"📉 **{d['Department']}**: Quoting {abs(d['Quote_Gap_Pct']):.0f}% below internal rates"
            )
    
    # QUOTING ISSUES
    underquoted_jobs = job_summary[job_summary["Is_Underquoted"]]
    if len(underquoted_jobs) > 0:
        total_underquote = abs(underquoted_jobs["Quote_Gap"].sum())
        insights["quoting_issues"].append(
            f"⚠️ **{len(underquoted_jobs)} jobs quoted below internal rates** — ${total_underquote:,.0f} left on table"
        )
    
    low_rate_jobs = job_summary[job_summary["Rate_Gap"] < -20]
    if len(low_rate_jobs) > 0:
        insights["quoting_issues"].append(
            f"📊 **{len(low_rate_jobs)} jobs quoted >$20/hr below billable rate**"
        )
    
    # SCOPE ISSUES
    unquoted_tasks = task_summary[task_summary["Is_Unquoted"]]
    if len(unquoted_tasks) > 0:
        unquoted_cost = unquoted_tasks["Base_Cost"].sum()
        unquoted_hours = unquoted_tasks["Actual_Hours"].sum()
        insights["scope_issues"].append(
            f"📋 **Scope Creep**: {len(unquoted_tasks)} unquoted tasks — {unquoted_hours:,.0f} hrs, ${unquoted_cost:,.0f}"
        )
    
    overrun_jobs = job_summary[job_summary["Significant_Overrun"]]
    if len(overrun_jobs) > 0:
        excess_hours = overrun_jobs["Hours_Variance"].sum()
        insights["scope_issues"].append(
            f"⏱️ **{len(overrun_jobs)} jobs with >20% hour overrun** — {excess_hours:,.0f} excess hours"
        )
    
    # RATE ISSUES
    low_effective_rate = job_summary[
        (job_summary["Effective_Rate_Hr"] > 0) & 
        (job_summary["Effective_Rate_Hr"] < job_summary["Cost_Rate_Hr"])
    ]
    if len(low_effective_rate) > 0:
        insights["rate_issues"].append(
            f"👥 **{len(low_effective_rate)} jobs earning below cost rate**"
        )
    
    # ACTION ITEMS
    if len(loss_jobs) > 0:
        top_losses = loss_jobs.nsmallest(3, "Margin")
        for _, job in top_losses.iterrows():
            reasons = []
            if job["Hours_Variance_Pct"] > 20:
                reasons.append(f"hour overrun ({job['Hours_Variance_Pct']:+.0f}%)")
            if job["Quote_Gap"] < -500:
                reasons.append("underquoted")
            if job["Effective_Rate_Hr"] < job["Cost_Rate_Hr"] and job["Effective_Rate_Hr"] > 0:
                reasons.append("effective rate below cost")
            
            reason_str = ", ".join(reasons) if reasons else "review needed"
            insights["action_items"].append(
                f"Review **{str(job['Job_Name'])[:35]}** ({job['Job_No']}) — ${job['Margin']:,.0f} loss: {reason_str}"
            )
    
    return insights


def diagnose_job_margin(job_row: pd.Series, tasks: pd.DataFrame) -> Dict:
    """Diagnose WHY a specific job has margin issues."""
    diagnosis = {
        "summary": "",
        "issues": [],
        "root_causes": [],
        "recommendations": []
    }
    
    margin = job_row.get("Margin", 0)
    margin_pct = job_row.get("Margin_Pct", 0)
    quote_gap = job_row.get("Quote_Gap", 0)
    quote_gap_pct = job_row.get("Quote_Gap_Pct", 0)
    hours_var_pct = job_row.get("Hours_Variance_Pct", 0)
    rate_gap = job_row.get("Rate_Gap", 0)
    
    # Summary
    if job_row.get("Is_Loss", False):
        diagnosis["summary"] = f"Loss-making job with ${margin:,.0f} negative margin"
    elif margin_pct < 20:
        diagnosis["summary"] = f"Low margin job at {margin_pct:.1f}%"
    else:
        diagnosis["summary"] = f"Acceptable margin at {margin_pct:.1f}%"
    
    # Issue 1: Quoted below internal benchmark
    if quote_gap < -500:
        diagnosis["issues"].append(f"Quoted ${abs(quote_gap):,.0f} below internal rates ({quote_gap_pct:+.0f}%)")
        diagnosis["root_causes"].append("Pricing discounted vs standard rates")
        diagnosis["recommendations"].append("Review pricing approval process")
    
    # Issue 2: Quoted rate below billable rate
    if rate_gap < -15:
        diagnosis["issues"].append(f"Quoted rate ${abs(rate_gap):.0f}/hr below billable rate")
        diagnosis["root_causes"].append("Rate discount applied")
        diagnosis["recommendations"].append("Document discount justification")
    
    # Issue 3: Scope creep
    if len(tasks) > 0:
        unquoted = tasks[tasks.get("Is_Unquoted", False)]
        if len(unquoted) > 0:
            unquoted_cost = unquoted["Base_Cost"].sum()
            diagnosis["issues"].append(f"{len(unquoted)} unquoted tasks added (${unquoted_cost:,.0f} cost)")
            diagnosis["root_causes"].append("Scope expanded beyond original quote")
            diagnosis["recommendations"].append("Implement change order process")
    
    # Issue 4: Hour overrun
    if hours_var_pct > 20:
        diagnosis["issues"].append(f"Hours {hours_var_pct:+.0f}% over quoted")
        if len(tasks) > 0:
            overrun_tasks = tasks[(tasks.get("Is_Overrun", False)) & (~tasks.get("Is_Unquoted", True))]
            if len(overrun_tasks) > 0:
                top_overrun = overrun_tasks.nlargest(1, "Hours_Variance").iloc[0]
                diagnosis["root_causes"].append(f"'{str(top_overrun['Task_Name'])[:30]}' had {top_overrun['Hours_Variance']:+.0f}hr overrun")
        diagnosis["recommendations"].append("Review estimation accuracy")
    
    # Issue 5: Effective rate below cost
    effective_rate = job_row.get("Effective_Rate_Hr", 0)
    cost_rate = job_row.get("Cost_Rate_Hr", 0)
    if effective_rate > 0 and effective_rate < cost_rate:
        diagnosis["issues"].append(f"Effective rate ${effective_rate:.0f}/hr below cost rate ${cost_rate:.0f}/hr")
        diagnosis["root_causes"].append("Not earning enough per hour worked")
        diagnosis["recommendations"].append("Review scope or renegotiate")
    
    return diagnosis


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def get_top_overruns(js: pd.DataFrame, n: int = 10, by: str = "Hours_Variance") -> pd.DataFrame:
    return js.nlargest(n, by)


def get_loss_making_jobs(js: pd.DataFrame) -> pd.DataFrame:
    return js[js["Is_Loss"]].sort_values("Margin")


def get_unquoted_tasks(ts: pd.DataFrame) -> pd.DataFrame:
    return ts[ts["Is_Unquoted"]].sort_values("Base_Cost", ascending=False)


def get_underquoted_jobs(js: pd.DataFrame, threshold: float = -500) -> pd.DataFrame:
    """Jobs quoted below internal benchmark."""
    return js[js["Quote_Gap"] < threshold].sort_values("Quote_Gap")


def get_margin_erosion_jobs(js: pd.DataFrame, threshold: float = 10) -> pd.DataFrame:
    """Jobs with significant hour overruns."""
    return js[js["Hours_Variance_Pct"] > threshold].sort_values("Hours_Variance_Pct", ascending=False)


def get_premium_jobs(js: pd.DataFrame, threshold: float = 500) -> pd.DataFrame:
    """Jobs quoted above internal benchmark."""
    return js[js["Quote_Gap"] > threshold].sort_values("Quote_Gap", ascending=False)


def calculate_overall_metrics(js: pd.DataFrame) -> dict:
    """Calculate overall metrics."""
    n = len(js)
    if n == 0:
        return {k: 0 for k in [
            "total_jobs", "total_quoted_amount", "total_base_cost", "total_expected_quote",
            "margin", "margin_pct", "quote_gap", "quote_gap_pct",
            "avg_quoted_rate_hr", "avg_effective_rate_hr", "avg_billable_rate_hr", "avg_cost_rate_hr",
            "jobs_over_budget", "jobs_at_loss", "jobs_underquoted", "jobs_premium",
            "overrun_rate", "loss_rate",
            "total_hours_quoted", "total_hours_actual", "hours_variance", "hours_variance_pct"
        ]}
    
    # Revenue & Cost
    q = js["Quoted_Amount"].sum()
    c = js["Base_Cost"].sum()
    eq = js["Expected_Quote"].sum()
    
    # Margin
    margin = q - c
    
    # Quote Gap
    qg = q - eq
    
    # Hours
    hq = js["Quoted_Hours"].sum()
    ha = js["Actual_Hours"].sum()
    
    return {
        "total_jobs": n,
        # Revenue
        "total_quoted_amount": q,
        "total_base_cost": c,
        "total_expected_quote": eq,
        # Margin
        "margin": margin,
        "margin_pct": (margin / q * 100) if q > 0 else 0,
        # Quote Gap
        "quote_gap": qg,
        "quote_gap_pct": (qg / eq * 100) if eq > 0 else 0,
        # Rates
        "avg_quoted_rate_hr": (q / hq) if hq > 0 else 0,
        "avg_effective_rate_hr": (q / ha) if ha > 0 else 0,
        "avg_billable_rate_hr": (eq / hq) if hq > 0 else 0,
        "avg_cost_rate_hr": (c / ha) if ha > 0 else 0,
        # Counts
        "jobs_over_budget": int(js["Is_Overrun"].sum()),
        "jobs_at_loss": int(js["Is_Loss"].sum()),
        "jobs_underquoted": int(js["Is_Underquoted"].sum()),
        "jobs_premium": int(js["Is_Premium"].sum()),
        "overrun_rate": (js["Is_Overrun"].sum() / n * 100) if n > 0 else 0,
        "loss_rate": (js["Is_Loss"].sum() / n * 100) if n > 0 else 0,
        # Hours
        "total_hours_quoted": hq,
        "total_hours_actual": ha,
        "hours_variance": ha - hq,
        "hours_variance_pct": ((ha - hq) / hq * 100) if hq > 0 else 0,
    }


def analyze_overrun_causes(ts: pd.DataFrame) -> dict:
    """Analyze root causes of overruns."""
    unq = ts[ts["Is_Unquoted"]]
    ovr = ts[(ts["Is_Overrun"]) & (~ts["Is_Unquoted"])]
    
    return {
        "scope_creep": {
            "count": len(unq), 
            "cost": unq["Base_Cost"].sum(), 
            "hours": unq["Actual_Hours"].sum()
        },
        "underestimation": {
            "count": len(ovr), 
            "excess_hours": ovr["Hours_Variance"].sum()
        },
    }


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

METRIC_DEFINITIONS = {
    "Quoted_Amount": {
        "name": "Quoted Amount (Revenue)", 
        "formula": "[Job Task] Quoted Amount", 
        "desc": "What we charge the client = REVENUE"
    },
    "Expected_Quote": {
        "name": "Expected Quote (Benchmark)", 
        "formula": "Quoted Hours × Billable Rate", 
        "desc": "What we SHOULD have quoted based on internal rates"
    },
    "Base_Cost": {
        "name": "Base Cost", 
        "formula": "Actual Hours × Cost Rate/Hr", 
        "desc": "What the work cost us internally"
    },
    "Margin": {
        "name": "Margin", 
        "formula": "Quoted Amount - Base Cost", 
        "desc": "Profit on the job"
    },
    "Margin_Pct": {
        "name": "Margin %", 
        "formula": "(Margin / Quoted Amount) × 100", 
        "desc": "Profit margin percentage. Target: 35%+"
    },
    "Quote_Gap": {
        "name": "Quote Gap", 
        "formula": "Quoted Amount - Expected Quote", 
        "desc": "How much above (+) or below (-) internal rates we quoted"
    },
    "Quote_Gap_Pct": {
        "name": "Quote Gap %", 
        "formula": "(Quote Gap / Expected Quote) × 100", 
        "desc": "Percentage above/below internal benchmark"
    },
    "Quoted_Rate_Hr": {
        "name": "Quoted Rate/Hr", 
        "formula": "Quoted Amount / Quoted Hours", 
        "desc": "What we charged per quoted hour"
    },
    "Effective_Rate_Hr": {
        "name": "Effective Rate/Hr", 
        "formula": "Quoted Amount / Actual Hours", 
        "desc": "Revenue per hour actually worked"
    },
    "Billable_Rate_Hr": {
        "name": "Billable Rate/Hr", 
        "formula": "[Task] Billable Rate", 
        "desc": "Internal standard rate we should be charging"
    },
    "Cost_Rate_Hr": {
        "name": "Cost Rate/Hr", 
        "formula": "[Task] Base Rate", 
        "desc": "What each hour costs us internally"
    },
    "Rate_Gap": {
        "name": "Rate Gap", 
        "formula": "Quoted Rate/Hr - Billable Rate/Hr", 
        "desc": "Difference between quoted and internal rate. Negative = discounting"
    },
    "Hours_Variance": {
        "name": "Hours Variance", 
        "formula": "Actual Hours - Quoted Hours", 
        "desc": "Positive values indicate hours overrun"
    },
    "Hours_Variance_Pct": {
        "name": "Hours Variance %", 
        "formula": "(Hours Variance / Quoted Hours) × 100", 
        "desc": "Overrun percentage vs quoted hours"
    },
    "Is_Overrun": {
        "name": "Hour Overrun Flag", 
        "formula": "Hours Variance > 0", 
        "desc": "True when actual hours exceed quoted hours"
    },
    "Is_Loss": {
        "name": "Loss Flag", 
        "formula": "Margin < 0", 
        "desc": "True when the job loses money"
    },
    "Overrun_Rate": {
        "name": "Overrun Rate", 
        "formula": "Jobs with overruns / Total jobs", 
        "desc": "Percentage of jobs exceeding quoted hours"
    },
    "Loss_Rate": {
        "name": "Loss Rate", 
        "formula": "Jobs at loss / Total jobs", 
        "desc": "Percentage of jobs with negative margin"
    },
}
