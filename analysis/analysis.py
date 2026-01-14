"""
Job Profitability Analysis Module — Corrected Financial Logic
==============================================================
Hierarchy: Department → Product → Job → Task
Time-Series: Month-on-Month Trend Analysis

CRITICAL FINANCIAL DEFINITIONS:
-------------------------------
- Quoted Amount:    The client quote — this IS revenue (what will be invoiced)
- Invoiced Amount:  What was actually billed to client
- Billable Rate:    Internal control rate (NOT revenue — used for margin management)
- Base Rate:        Internal cost rate (labor cost per hour)

MARGIN DEFINITIONS:
-------------------
- Quoted Margin:    Quoted Amount - Base Cost (expected margin at quote)
- Actual Margin:    Invoiced Amount - Base Cost (realized margin)
- Margin Variance:  Actual Margin - Quoted Margin (erosion or gain)

REALIZATION:
------------
- Realization %:    (Invoiced Amount / Quoted Amount) × 100
  - 100% = billed exactly what was quoted
  - <100% = discounting, write-offs, or scope reduction
  - >100% = change orders or additional billing

RATE ANALYSIS (Internal Control):
---------------------------------
- Quoted Rate/Hr:   Quoted Amount / Quoted Hours (implied rate from quote)
- Billable Rate/Hr: [Task] Billable Rate (internal benchmark)
- Cost Rate/Hr:     [Task] Base Rate (internal cost)
- Rate Gap:         Quoted Rate/Hr - Billable Rate/Hr (quoting vs standard)
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
    
    REVENUE = Quoted Amount (NOT Billable Rate × Hours)
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
    # RATE CALCULATIONS (per hour) - FOR INTERNAL ANALYSIS ONLY
    # =========================================================================
    # Billable Rate/Hr = [Task] Billable Rate (internal control rate)
    df["Billable_Rate_Hr"] = df["[Task] Billable Rate"]
    
    # Cost Rate/Hr = [Task] Base Rate (internal cost)
    df["Cost_Rate_Hr"] = df["[Task] Base Rate"]
    
    # Quoted Rate/Hr = Quoted Amount / Quoted Hours (implied rate from quote)
    df["Quoted_Rate_Hr"] = np.where(
        df["[Job Task] Quoted Time"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Quoted Time"],
        0
    )
    
    # Actual Rate/Hr = Invoiced Amount / Actual Hours (realized rate)
    df["Actual_Rate_Hr"] = np.where(
        df["[Job Task] Actual Time (totalled)"] > 0,
        df["[Job Task] Invoiced Amount"] / df["[Job Task] Actual Time (totalled)"],
        0
    )
    
    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================
    # Base Cost = Actual Hours × Cost Rate/Hr (what it actually cost us)
    df["Base_Cost"] = df["[Job Task] Actual Time (totalled)"] * df["Cost_Rate_Hr"]
    
    # Quoted Base Cost = Quoted Hours × Cost Rate (expected cost at quote)
    df["Quoted_Base_Cost"] = df["[Job Task] Quoted Time"] * df["Cost_Rate_Hr"]
    
    # =========================================================================
    # INTERNAL BENCHMARK (NOT REVENUE - for rate gap analysis only)
    # =========================================================================
    df["Billable_Value"] = df["[Job Task] Actual Time (totalled)"] * df["Billable_Rate_Hr"]
    
    # =========================================================================
    # RATE GAP ANALYSIS
    # =========================================================================
    # Gap between quoted rate and internal billable rate
    df["Rate_Gap"] = df["Quoted_Rate_Hr"] - df["Billable_Rate_Hr"]
    df["Rate_Gap_Pct"] = np.where(
        df["Billable_Rate_Hr"] > 0,
        (df["Rate_Gap"] / df["Billable_Rate_Hr"]) * 100,
        0
    )
    
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
        mask = df_f["[Job Task] Name"] == "Social Garden Invoice Allocation"
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
    """Compute validation totals - emphasizing Quoted vs Invoiced (real revenue)."""
    recon["totals"] = {
        # Hours
        "sum_quoted_hours": df["[Job Task] Quoted Time"].sum(),
        "sum_actual_hours": df["[Job Task] Actual Time (totalled)"].sum(),
        "sum_invoiced_hours": df["[Job Task] Invoiced Time"].sum(),
        # Revenue (Quoted = Expected, Invoiced = Actual)
        "sum_quoted_amount": df["[Job Task] Quoted Amount"].sum(),
        "sum_invoiced_amount": df["[Job Task] Invoiced Amount"].sum(),
        # Cost
        "sum_base_cost": df["Base_Cost"].sum(),
        # Internal benchmark (NOT revenue)
        "sum_billable_value": df["Billable_Value"].sum(),
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
    """Monthly aggregation with correct financial logic."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Fiscal_Year", "FY_Month"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "Billable_Value": "sum",  # Internal benchmark only
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Fiscal_Year", "FY_Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Hours", "Invoiced_Amount",
        "Base_Cost", "Billable_Value", "Job_Count"
    ]
    
    g = g.sort_values("Month_Sort").reset_index(drop=True)
    
    # =========================================================================
    # CORRECT MARGIN CALCULATIONS
    # =========================================================================
    # Quoted Margin = Expected margin at time of quote
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    
    # Actual Margin = Realized margin based on what was invoiced
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    
    # Margin Variance = How much margin eroded vs quote
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    
    # Margin percentages
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    
    # =========================================================================
    # REALIZATION = Invoiced / Quoted (the real measure of revenue capture)
    # =========================================================================
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    
    # Write-off = Quoted - Invoiced (revenue not captured)
    g["Write_Off"] = g["Quoted_Amount"] - g["Invoiced_Amount"]
    g["Write_Off_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Write_Off"] / g["Quoted_Amount"]) * 100, 0)
    
    # Rates
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Actual_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Invoiced_Amount"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    
    # Hours variance
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


def compute_monthly_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trends by department with correct financial logic."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Department"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Department",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Amount", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values(["Month_Sort", "Department"]).reset_index(drop=True)
    
    # Margins
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    
    return g


def compute_monthly_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trends by product with correct financial logic."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Department", "Product"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Department", "Product",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Amount", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values(["Month_Sort", "Department", "Product"]).reset_index(drop=True)
    
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    
    return g


# =============================================================================
# DEPARTMENT SUMMARY
# =============================================================================

def compute_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Department summary with correct financial logic."""
    g = df.groupby("Department").agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "Billable_Value": "sum",
        "[Job] Job No.": pd.Series.nunique,
        "Product": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Invoiced_Hours", "Invoiced_Amount", "Base_Cost", "Billable_Value",
                 "Job_Count", "Product_Count"]
    
    # Correct margins
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    
    # Realization
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    g["Write_Off"] = g["Quoted_Amount"] - g["Invoiced_Amount"]
    
    # Rate analysis (internal)
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Actual_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Invoiced_Amount"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    
    # Rate gap (quoted vs internal standard)
    g["Rate_Gap"] = g["Quoted_Rate_Hr"] - g["Billable_Rate_Hr"]
    
    # Hours variance
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


# =============================================================================
# PRODUCT SUMMARY
# =============================================================================

def compute_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Product summary with correct financial logic."""
    g = df.groupby(["Department", "Product"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "Billable_Value": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Product", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Invoiced_Hours", "Invoiced_Amount", "Base_Cost", "Billable_Value", "Job_Count"]
    
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    g["Write_Off"] = g["Quoted_Amount"] - g["Invoiced_Amount"]
    
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Actual_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Invoiced_Amount"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


# =============================================================================
# JOB SUMMARY
# =============================================================================

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Job summary with correct financial logic."""
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name",
        "[Job] Client", "[Job] Client Manager", "[Job] Status",
        "[Job] Start Date", "Fiscal_Year", "FY_Label", "Calendar_Month"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "Billable_Value": "sum",
        "[Job] Budget": "first",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Client", "Client_Manager", "Status",
        "Start_Date", "Fiscal_Year", "FY_Label", "Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours", "Invoiced_Hours", "Invoiced_Amount",
        "Base_Cost", "Billable_Value", "Budget"
    ]
    
    # =========================================================================
    # CORRECT MARGIN CALCULATIONS
    # =========================================================================
    # Quoted Margin = What we expected to make
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    
    # Actual Margin = What we actually made (based on invoiced, not billable)
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    
    # Margin Variance = Erosion or gain vs quote
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    
    # Margin percentages
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    
    # Margin Erosion (positive = margin got worse)
    g["Margin_Erosion"] = g["Quoted_Margin_Pct"] - g["Actual_Margin_Pct"]
    
    # =========================================================================
    # REALIZATION = Invoiced / Quoted
    # =========================================================================
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    g["Write_Off"] = g["Quoted_Amount"] - g["Invoiced_Amount"]
    g["Write_Off_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Write_Off"] / g["Quoted_Amount"]) * 100, 0)
    
    # Rates
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Actual_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Invoiced_Amount"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    
    # Rate gap analysis
    g["Rate_Gap"] = g["Quoted_Rate_Hr"] - g["Billable_Rate_Hr"]
    
    # Hours
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    g["Unbilled_Hours"] = g["Actual_Hours"] - g["Invoiced_Hours"]
    
    # Flags
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Is_Loss"] = g["Actual_Margin"] < 0
    g["Has_Write_Off"] = g["Write_Off"] > 100
    g["Has_Margin_Erosion"] = g["Margin_Erosion"] > 10
    g["Low_Realization"] = g["Realization_Pct"] < 90
    
    return g


# =============================================================================
# TASK SUMMARY
# =============================================================================

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Task summary with correct financial logic."""
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
        "Task Category", "Fiscal_Year", "FY_Label", "Calendar_Month"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Base_Cost": "sum",
        "Billable_Value": "sum",
        "Cost_Rate_Hr": "mean",
        "Billable_Rate_Hr": "mean",
        "Quoted_Rate_Hr": "mean",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Fiscal_Year", "FY_Label", "Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours", "Invoiced_Hours", "Invoiced_Amount",
        "Base_Cost", "Billable_Value", "Cost_Rate_Hr", "Billable_Rate_Hr", "Quoted_Rate_Hr"
    ]
    
    # Margins
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Invoiced_Amount"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Invoiced_Amount"] > 0, (g["Actual_Margin"] / g["Invoiced_Amount"]) * 100, 0)
    
    # Realization
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Invoiced_Amount"] / g["Quoted_Amount"]) * 100, 0)
    g["Write_Off"] = g["Quoted_Amount"] - g["Invoiced_Amount"]
    
    # Actual rate
    g["Actual_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Invoiced_Amount"] / g["Actual_Hours"], 0)
    
    # Rate gap
    g["Rate_Gap"] = g["Quoted_Rate_Hr"] - g["Billable_Rate_Hr"]
    
    # Hours
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    g["Unbilled_Hours"] = g["Actual_Hours"] - g["Invoiced_Hours"]
    
    # Flags
    g["Is_Unquoted"] = (g["Quoted_Hours"] == 0) & (g["Actual_Hours"] > 0)
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Has_Unbilled"] = g["Unbilled_Hours"] > 0.5
    g["Has_Write_Off"] = g["Write_Off"] > 50
    
    return g


# =============================================================================
# NARRATIVE INSIGHTS - WHY DID MARGIN ERODE?
# =============================================================================

def generate_insights(
    job_summary: pd.DataFrame,
    dept_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    task_summary: pd.DataFrame
) -> Dict:
    """Generate narrative insights explaining WHY margins are good or bad."""
    insights = {
        "headline": [],
        "margin_drivers": [],
        "quoting_issues": [],
        "scope_issues": [],
        "realization_issues": [],
        "rate_issues": [],
        "action_items": []
    }
    
    if len(job_summary) == 0:
        return insights
    
    # Overall metrics
    total_quoted = job_summary["Quoted_Amount"].sum()
    total_invoiced = job_summary["Invoiced_Amount"].sum()
    total_cost = job_summary["Base_Cost"].sum()
    
    quoted_margin = total_quoted - total_cost
    actual_margin = total_invoiced - total_cost
    margin_variance = actual_margin - quoted_margin
    
    realization = (total_invoiced / total_quoted * 100) if total_quoted > 0 else 0
    quoted_margin_pct = (quoted_margin / total_quoted * 100) if total_quoted > 0 else 0
    actual_margin_pct = (actual_margin / total_invoiced * 100) if total_invoiced > 0 else 0
    
    # =========================================================================
    # HEADLINE INSIGHTS
    # =========================================================================
    if realization < 90:
        gap = total_quoted - total_invoiced
        insights["headline"].append(
            f"⚠️ **Revenue Leakage**: Only {realization:.0f}% of quoted revenue realized "
            f"(${gap:,.0f} write-off/discount)"
        )
    elif realization > 105:
        over = total_invoiced - total_quoted
        insights["headline"].append(
            f"✅ **Revenue Exceeded**: {realization:.0f}% realization — ${over:,.0f} above quotes "
            f"(change orders or additional work)"
        )
    
    if actual_margin_pct < 20:
        insights["headline"].append(f"🔴 **Low Margin**: Actual margin at {actual_margin_pct:.1f}% — needs attention")
    elif actual_margin_pct < 35:
        insights["headline"].append(f"🟡 **Margin Below Target**: {actual_margin_pct:.1f}% (target: 35%+)")
    else:
        insights["headline"].append(f"🟢 **Healthy Margin**: {actual_margin_pct:.1f}%")
    
    if margin_variance < -10000:
        insights["headline"].append(
            f"💸 **Margin Erosion**: ${abs(margin_variance):,.0f} less profit than quoted "
            f"({quoted_margin_pct:.0f}% quoted → {actual_margin_pct:.0f}% actual)"
        )
    
    # Loss-making jobs
    loss_jobs = job_summary[job_summary["Is_Loss"]]
    if len(loss_jobs) > 0:
        total_losses = loss_jobs["Actual_Margin"].sum()
        insights["headline"].append(f"💸 **{len(loss_jobs)} loss-making jobs** totaling ${abs(total_losses):,.0f}")
    
    # =========================================================================
    # MARGIN DRIVER ANALYSIS
    # =========================================================================
    if len(dept_summary) > 0:
        # Worst department
        worst_dept = dept_summary.loc[dept_summary["Actual_Margin_Pct"].idxmin()]
        if worst_dept["Actual_Margin_Pct"] < 20:
            insights["margin_drivers"].append(
                f"🔻 **{worst_dept['Department']}** is dragging margins at {worst_dept['Actual_Margin_Pct']:.1f}% "
                f"(${worst_dept['Actual_Margin']:,.0f} on ${worst_dept['Invoiced_Amount']:,.0f})"
            )
        
        # Best department
        best_dept = dept_summary.loc[dept_summary["Actual_Margin_Pct"].idxmax()]
        if best_dept["Actual_Margin_Pct"] > 40 and best_dept["Invoiced_Amount"] > 10000:
            insights["margin_drivers"].append(
                f"🔺 **{best_dept['Department']}** leading with {best_dept['Actual_Margin_Pct']:.1f}% margin"
            )
        
        # Departments with low realization
        low_real_depts = dept_summary[dept_summary["Realization_Pct"] < 85]
        for _, d in low_real_depts.iterrows():
            insights["realization_issues"].append(
                f"📉 **{d['Department']}**: {d['Realization_Pct']:.0f}% realization — "
                f"${d['Write_Off']:,.0f} not invoiced"
            )
    
    # =========================================================================
    # QUOTING ISSUES - Was the quote too low?
    # =========================================================================
    # Jobs with negative quoted margin (quoted below cost)
    underquoted = job_summary[job_summary["Quoted_Margin"] < 0]
    if len(underquoted) > 0:
        insights["quoting_issues"].append(
            f"⚠️ **{len(underquoted)} jobs quoted below cost** — "
            f"${abs(underquoted['Quoted_Margin'].sum()):,.0f} in negative quoted margin"
        )
    
    # Jobs with very low quoted rate vs billable rate
    rate_gap_jobs = job_summary[(job_summary["Rate_Gap"] < -20) & (job_summary["Quoted_Amount"] > 1000)]
    if len(rate_gap_jobs) > 0:
        insights["quoting_issues"].append(
            f"📊 **{len(rate_gap_jobs)} jobs quoted below standard rate** — "
            f"average ${rate_gap_jobs['Rate_Gap'].mean():,.0f}/hr below billable rate"
        )
    
    # =========================================================================
    # SCOPE ISSUES - Was scope not controlled?
    # =========================================================================
    # Unquoted tasks (scope creep)
    unquoted_tasks = task_summary[task_summary["Is_Unquoted"]]
    if len(unquoted_tasks) > 0:
        unquoted_cost = unquoted_tasks["Base_Cost"].sum()
        unquoted_hours = unquoted_tasks["Actual_Hours"].sum()
        insights["scope_issues"].append(
            f"📋 **Scope Creep**: {len(unquoted_tasks)} unquoted tasks — "
            f"{unquoted_hours:,.0f} hours at ${unquoted_cost:,.0f} cost"
        )
    
    # Jobs with high hour overruns
    overrun_jobs = job_summary[job_summary["Hours_Variance_Pct"] > 50]
    if len(overrun_jobs) > 0:
        excess_hours = overrun_jobs["Hours_Variance"].sum()
        insights["scope_issues"].append(
            f"⏱️ **{len(overrun_jobs)} jobs with >50% hour overrun** — "
            f"{excess_hours:,.0f} excess hours"
        )
    
    # =========================================================================
    # REALIZATION ISSUES - Write-offs and discounts
    # =========================================================================
    low_real_jobs = job_summary[job_summary["Low_Realization"]]
    if len(low_real_jobs) > 0:
        total_writeoff = low_real_jobs["Write_Off"].sum()
        insights["realization_issues"].append(
            f"💰 **{len(low_real_jobs)} jobs with <90% realization** — "
            f"${total_writeoff:,.0f} not invoiced"
        )
    
    # =========================================================================
    # RATE ISSUES - Were base rates too high (wrong resourcing)?
    # =========================================================================
    high_cost_jobs = job_summary[
        (job_summary["Cost_Rate_Hr"] > job_summary["Billable_Rate_Hr"]) &
        (job_summary["Base_Cost"] > 5000)
    ]
    if len(high_cost_jobs) > 0:
        insights["rate_issues"].append(
            f"👥 **{len(high_cost_jobs)} jobs with cost rate > billable rate** — "
            f"indicates over-resourcing with senior staff"
        )
    
    # =========================================================================
    # ACTION ITEMS
    # =========================================================================
    if len(loss_jobs) > 0:
        top_losses = loss_jobs.nsmallest(3, "Actual_Margin")
        for _, job in top_losses.iterrows():
            reason = []
            if job["Realization_Pct"] < 90:
                reason.append(f"low realization ({job['Realization_Pct']:.0f}%)")
            if job["Hours_Variance_Pct"] > 30:
                reason.append(f"hour overrun ({job['Hours_Variance_Pct']:+.0f}%)")
            if job["Rate_Gap"] < -20:
                reason.append(f"underquoted rate")
            
            reason_str = ", ".join(reason) if reason else "review needed"
            insights["action_items"].append(
                f"Review **{str(job['Job_Name'])[:35]}** ({job['Job_No']}) — "
                f"${job['Actual_Margin']:,.0f} loss due to {reason_str}"
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
    
    # Basic metrics
    quoted_margin_pct = job_row.get("Quoted_Margin_Pct", 0)
    actual_margin_pct = job_row.get("Actual_Margin_Pct", 0)
    margin_erosion = job_row.get("Margin_Erosion", 0)
    realization = job_row.get("Realization_Pct", 100)
    hours_var_pct = job_row.get("Hours_Variance_Pct", 0)
    rate_gap = job_row.get("Rate_Gap", 0)
    
    # Summary
    if job_row.get("Is_Loss", False):
        diagnosis["summary"] = f"Loss-making job with ${job_row['Actual_Margin']:,.0f} negative margin"
    elif margin_erosion > 15:
        diagnosis["summary"] = f"Significant margin erosion: {margin_erosion:.1f}pp"
    else:
        diagnosis["summary"] = f"Margin performance acceptable: {actual_margin_pct:.1f}%"
    
    # Issue 1: Quote was too low
    if quoted_margin_pct < 25:
        diagnosis["issues"].append(f"Quote margin was only {quoted_margin_pct:.0f}% (target: 35%+)")
        if rate_gap < -15:
            diagnosis["root_causes"].append(f"Quoted ${abs(rate_gap):.0f}/hr below standard rate")
        diagnosis["recommendations"].append("Review quoting process for this job type")
    
    # Issue 2: Scope creep
    unquoted = tasks[tasks.get("Is_Unquoted", False)] if len(tasks) > 0 else pd.DataFrame()
    if len(unquoted) > 0:
        unquoted_cost = unquoted["Base_Cost"].sum()
        diagnosis["issues"].append(f"{len(unquoted)} unquoted tasks added (${unquoted_cost:,.0f})")
        diagnosis["root_causes"].append("Scope expanded beyond original quote")
        diagnosis["recommendations"].append("Implement change order process")
    
    # Issue 3: Hour overrun
    if hours_var_pct > 30:
        diagnosis["issues"].append(f"Hours {hours_var_pct:+.0f}% over quoted")
        overrun_tasks = tasks[tasks.get("Is_Overrun", False) & ~tasks.get("Is_Unquoted", True)] if len(tasks) > 0 else pd.DataFrame()
        if len(overrun_tasks) > 0:
            top_overrun = overrun_tasks.nlargest(1, "Hours_Variance").iloc[0] if len(overrun_tasks) > 0 else None
            if top_overrun is not None:
                diagnosis["root_causes"].append(f"'{str(top_overrun['Task_Name'])[:30]}' had {top_overrun['Hours_Variance']:+.0f}hr variance")
        diagnosis["recommendations"].append("Review estimation accuracy for this job type")
    
    # Issue 4: Revenue not captured
    if realization < 90:
        diagnosis["issues"].append(f"Only {realization:.0f}% of quoted amount invoiced")
        diagnosis["root_causes"].append(f"${job_row['Write_Off']:,.0f} not billed (discount/write-off)")
        diagnosis["recommendations"].append("Investigate billing gaps and approval process")
    
    # Issue 5: Wrong resourcing
    cost_rate = job_row.get("Cost_Rate_Hr", 0)
    billable_rate = job_row.get("Billable_Rate_Hr", 0)
    if cost_rate > billable_rate and cost_rate > 0:
        diagnosis["issues"].append(f"Cost rate (${cost_rate:.0f}/hr) exceeds billable rate (${billable_rate:.0f}/hr)")
        diagnosis["root_causes"].append("Over-resourcing with senior staff")
        diagnosis["recommendations"].append("Review resource allocation for cost efficiency")
    
    return diagnosis


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def get_top_overruns(js: pd.DataFrame, n: int = 10, by: str = "Hours_Variance") -> pd.DataFrame:
    return js.nlargest(n, by)


def get_loss_making_jobs(js: pd.DataFrame) -> pd.DataFrame:
    return js[js["Is_Loss"]].sort_values("Actual_Margin")


def get_unquoted_tasks(ts: pd.DataFrame) -> pd.DataFrame:
    return ts[ts["Is_Unquoted"]].sort_values("Base_Cost", ascending=False)


def get_margin_erosion_jobs(js: pd.DataFrame, threshold: float = 10) -> pd.DataFrame:
    return js[js["Margin_Erosion"] > threshold].sort_values("Margin_Erosion", ascending=False)


def get_low_realization_jobs(js: pd.DataFrame, threshold: float = 90) -> pd.DataFrame:
    return js[js["Realization_Pct"] < threshold].sort_values("Realization_Pct")


def get_write_off_jobs(js: pd.DataFrame, min_amount: float = 1000) -> pd.DataFrame:
    return js[js["Write_Off"] > min_amount].sort_values("Write_Off", ascending=False)


def calculate_overall_metrics(js: pd.DataFrame) -> dict:
    """Calculate overall metrics with correct financial logic."""
    n = len(js)
    if n == 0:
        return {k: 0 for k in [
            "total_jobs", "total_quoted_amount", "total_invoiced_amount", "total_base_cost",
            "quoted_margin", "actual_margin", "margin_variance",
            "quoted_margin_pct", "actual_margin_pct",
            "realization_pct", "write_off_total",
            "avg_quoted_rate_hr", "avg_actual_rate_hr", "avg_cost_rate_hr",
            "jobs_over_budget", "jobs_at_loss", "jobs_with_write_off",
            "overrun_rate", "loss_rate",
            "total_hours_quoted", "total_hours_actual", "hours_variance", "hours_variance_pct"
        ]}
    
    # Revenue (Quoted = Expected, Invoiced = Actual)
    q = js["Quoted_Amount"].sum()
    inv = js["Invoiced_Amount"].sum()
    c = js["Base_Cost"].sum()
    
    # Margins
    quoted_margin = q - c
    actual_margin = inv - c
    margin_var = actual_margin - quoted_margin
    
    # Hours
    hq = js["Quoted_Hours"].sum()
    ha = js["Actual_Hours"].sum()
    
    return {
        "total_jobs": n,
        # Revenue
        "total_quoted_amount": q,
        "total_invoiced_amount": inv,
        "total_base_cost": c,
        # Margins
        "quoted_margin": quoted_margin,
        "actual_margin": actual_margin,
        "margin_variance": margin_var,
        "quoted_margin_pct": (quoted_margin / q * 100) if q > 0 else 0,
        "actual_margin_pct": (actual_margin / inv * 100) if inv > 0 else 0,
        # Realization
        "realization_pct": (inv / q * 100) if q > 0 else 0,
        "write_off_total": q - inv,
        # Rates
        "avg_quoted_rate_hr": (q / hq) if hq > 0 else 0,
        "avg_actual_rate_hr": (inv / ha) if ha > 0 else 0,
        "avg_cost_rate_hr": (c / ha) if ha > 0 else 0,
        # Counts
        "jobs_over_budget": int(js["Is_Overrun"].sum()),
        "jobs_at_loss": int(js["Is_Loss"].sum()),
        "jobs_with_write_off": int(js["Has_Write_Off"].sum()),
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
    unb = ts[ts["Has_Unbilled"]]
    wo = ts[ts["Has_Write_Off"]]
    
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
        "unbilled": {
            "count": len(unb), 
            "hours": unb["Unbilled_Hours"].sum()
        },
        "write_offs": {
            "count": len(wo),
            "amount": wo["Write_Off"].sum()
        }
    }


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

METRIC_DEFINITIONS = {
    # Revenue
    "Quoted_Amount": {
        "name": "Quoted Amount (Revenue)", 
        "formula": "[Job Task] Quoted Amount", 
        "desc": "The client quote — this IS revenue. What will be invoiced and recognized."
    },
    "Invoiced_Amount": {
        "name": "Invoiced Amount", 
        "formula": "[Job Task] Invoiced Amount", 
        "desc": "What was actually billed to the client."
    },
    # Cost
    "Base_Cost": {
        "name": "Base Cost", 
        "formula": "Actual Hours × Cost Rate/Hr", 
        "desc": "Internal labor cost based on actual hours worked."
    },
    # Margins
    "Quoted_Margin": {
        "name": "Quoted Margin", 
        "formula": "Quoted Amount - Base Cost", 
        "desc": "Expected margin at time of quote. What we should have made."
    },
    "Actual_Margin": {
        "name": "Actual Margin", 
        "formula": "Invoiced Amount - Base Cost", 
        "desc": "Realized margin. What we actually made."
    },
    "Margin_Variance": {
        "name": "Margin Variance", 
        "formula": "Actual Margin - Quoted Margin", 
        "desc": "How much margin eroded (negative) or improved (positive) vs quote."
    },
    "Margin_Erosion": {
        "name": "Margin Erosion", 
        "formula": "Quoted Margin % - Actual Margin %", 
        "desc": "Percentage point drop in margin vs quote. Positive = margin got worse."
    },
    # Realization
    "Realization_Pct": {
        "name": "Realization %", 
        "formula": "(Invoiced Amount / Quoted Amount) × 100", 
        "desc": "Percentage of quoted revenue actually billed. <100% = write-off/discount. >100% = change orders."
    },
    "Write_Off": {
        "name": "Write-Off", 
        "formula": "Quoted Amount - Invoiced Amount", 
        "desc": "Revenue not captured. Could be discounts, scope reductions, or unbilled work."
    },
    # Rates
    "Quoted_Rate_Hr": {
        "name": "Quoted Rate/Hr", 
        "formula": "Quoted Amount / Quoted Hours", 
        "desc": "Implied hourly rate from the quote."
    },
    "Actual_Rate_Hr": {
        "name": "Actual Rate/Hr", 
        "formula": "Invoiced Amount / Actual Hours", 
        "desc": "Realized hourly rate based on what was actually invoiced."
    },
    "Billable_Rate_Hr": {
        "name": "Billable Rate/Hr (Internal)", 
        "formula": "[Task] Billable Rate", 
        "desc": "Internal control rate used for margin management. NOT external revenue."
    },
    "Cost_Rate_Hr": {
        "name": "Cost Rate/Hr", 
        "formula": "[Task] Base Rate", 
        "desc": "Internal cost per hour."
    },
    "Rate_Gap": {
        "name": "Rate Gap", 
        "formula": "Quoted Rate/Hr - Billable Rate/Hr", 
        "desc": "Difference between quoted rate and internal standard. Negative = quoting below standard."
    },
    # Internal (NOT Revenue)
    "Billable_Value": {
        "name": "Billable Value (Internal)", 
        "formula": "Actual Hours × Billable Rate/Hr", 
        "desc": "⚠️ INTERNAL ONLY. Not actual revenue. Used for rate gap analysis."
    },
}