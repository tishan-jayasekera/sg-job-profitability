"""
Job Profitability Analysis Module - Enhanced Edition
=====================================================
Hierarchy: Department → Product → Job → Task
Time-Series: Month-on-Month Trend Analysis

MARGIN DEFINITIONS:
- Quoted Margin:     Quoted Amount - Base Cost (Base Rate × Hours)
- Actual Margin:     Billable Value - Base Cost
- Margin Variance:   Actual Margin - Quoted Margin

RATE DEFINITIONS (per hour):
- Quoted Rate/Hr:    Quoted Amount / Quoted Hours
- Billable Rate/Hr:  [Task] Billable Rate
- Cost Rate/Hr:      [Task] Base Rate (T&M cost per hour)
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
    # RATE CALCULATIONS (per hour)
    # =========================================================================
    df["Billable_Rate_Hr"] = df["[Task] Billable Rate"]
    df["Cost_Rate_Hr"] = df["[Task] Base Rate"]
    df["Quoted_Rate_Hr"] = np.where(
        df["[Job Task] Quoted Time"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Quoted Time"],
        0
    )
    
    # =========================================================================
    # VALUE CALCULATIONS
    # =========================================================================
    # Base Cost = Actual Hours × Cost Rate/Hr
    df["Calc_Base_Cost"] = df["[Job Task] Actual Time (totalled)"] * df["Cost_Rate_Hr"]
    
    # Billable Value = Actual Hours × Billable Rate/Hr
    df["Calc_Billable_Value"] = df["[Job Task] Actual Time (totalled)"] * df["Billable_Rate_Hr"]
    
    # Quoted Base Cost = Quoted Hours × Cost Rate/Hr (for comparison)
    df["Calc_Quoted_Base_Cost"] = df["[Job Task] Quoted Time"] * df["Cost_Rate_Hr"]
    
    # Legacy alias
    df["Calc_Cost_TM"] = df["Calc_Base_Cost"]
    
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
    recon["totals"] = {
        "sum_quoted_hours": df["[Job Task] Quoted Time"].sum(),
        "sum_actual_hours": df["[Job Task] Actual Time (totalled)"].sum(),
        "sum_invoiced_hours": df["[Job Task] Invoiced Time"].sum(),
        "sum_quoted_amount": df["[Job Task] Quoted Amount"].sum(),
        "sum_billable_value": df["Calc_Billable_Value"].sum(),
        "sum_base_cost": df["Calc_Base_Cost"].sum(),
        "sum_cost_tm": df["Calc_Cost_TM"].sum(),
        "sum_invoiced_amount": df["[Job Task] Invoiced Amount"].sum(),
        "avg_quoted_rate_hr": df[df["Quoted_Rate_Hr"] > 0]["Quoted_Rate_Hr"].mean() if len(df[df["Quoted_Rate_Hr"] > 0]) > 0 else 0,
        "avg_billable_rate_hr": df["Billable_Rate_Hr"].mean(),
        "avg_cost_rate_hr": df["Cost_Rate_Hr"].mean(),
        "unique_jobs": df["[Job] Job No."].nunique(),
        "unique_products": df["Product"].nunique(),
        "unique_departments": df["Department"].nunique(),
    }
    return recon


# =============================================================================
# MONTHLY TREND SUMMARIES
# =============================================================================

def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by calendar month for trend analysis."""
    # Create a sortable month key
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Fiscal_Year", "FY_Month"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "Calc_Quoted_Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Fiscal_Year", "FY_Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Base_Cost", "Quoted_Base_Cost", "Job_Count"
    ]
    
    # Sort by month
    g = g.sort_values("Month_Sort").reset_index(drop=True)
    
    # Calculate margins
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    
    # Margin percentages
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Actual_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    
    # Alias columns for convenience
    g["Margin"] = g["Actual_Margin"]
    g["Margin_Pct"] = g["Actual_Margin_Pct"]
    
    # Rates
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    g["Effective_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Quoted_Amount"] / g["Actual_Hours"], 0)
    
    # Variances
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    g["Revenue_Variance"] = g["Billable_Value"] - g["Quoted_Amount"]
    g["Realization_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Billable_Value"] / g["Quoted_Amount"]) * 100, 0)
    g["Expected_Quote"] = g["Quoted_Hours"] * g["Billable_Rate_Hr"]
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    return g


def compute_monthly_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trends broken down by department."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Department"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Department",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Billable_Value", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values(["Month_Sort", "Department"]).reset_index(drop=True)
    
    # Margins
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Actual_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    
    return g


def compute_monthly_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trends broken down by product."""
    df = df.copy()
    df["Month_Sort"] = df["[Job] Start Date"].dt.to_period('M')
    
    g = df.groupby(["Month_Sort", "Calendar_Month", "Department", "Product"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = [
        "Month_Sort", "Month", "Department", "Product",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Billable_Value", "Base_Cost", "Job_Count"
    ]
    
    g = g.sort_values(["Month_Sort", "Department", "Product"]).reset_index(drop=True)
    
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Actual_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    
    return g


# =============================================================================
# DEPARTMENT SUMMARY
# =============================================================================

def compute_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Department").agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "Calc_Cost_TM": "sum",
        "[Job] Job No.": pd.Series.nunique,
        "Product": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Invoiced_Hours", "Billable_Value", "Base_Cost", "Cost_TM", "Job_Count", "Product_Count"]
    
    g["Profit"] = g["Billable_Value"] - g["Base_Cost"]
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    g["Expected_Quote"] = g["Quoted_Hours"] * g["Billable_Rate_Hr"]
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    # Alias columns
    g["Margin"] = g["Actual_Margin"]
    g["Margin_Pct"] = g["Billable_Margin_Pct"]
    
    return g


# =============================================================================
# PRODUCT SUMMARY
# =============================================================================

def compute_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["Department", "Product"]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "Calc_Cost_TM": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Product", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Invoiced_Hours", "Billable_Value", "Base_Cost", "Cost_TM", "Job_Count"]
    
    g["Profit"] = g["Billable_Value"] - g["Base_Cost"]
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    g["Expected_Quote"] = g["Quoted_Hours"] * g["Billable_Rate_Hr"]
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    # Alias columns
    g["Margin"] = g["Actual_Margin"]
    g["Margin_Pct"] = g["Billable_Margin_Pct"]
    
    return g


# =============================================================================
# JOB SUMMARY
# =============================================================================

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
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
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "Calc_Cost_TM": "sum",
        "[Job] Budget": "first",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Client", "Client_Manager", "Status",
        "Start_Date", "Fiscal_Year", "FY_Label", "Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours", "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Base_Cost", "Cost_TM", "Budget"
    ]
    
    g["Profit"] = g["Billable_Value"] - g["Base_Cost"]
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    g["Margin_Erosion"] = g["Quoted_Margin_Pct"] - g["Billable_Margin_Pct"]
    
    # Alias columns for convenience
    g["Margin"] = g["Actual_Margin"]
    g["Margin_Pct"] = g["Billable_Margin_Pct"]
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Base_Cost"] / g["Actual_Hours"], 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    g["Unbilled_Hours"] = g["Actual_Hours"] - g["Invoiced_Hours"]
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Is_Loss"] = g["Profit"] < 0
    g["Has_Margin_Erosion"] = g["Margin_Erosion"] > 10
    g["Expected_Quote"] = g["Quoted_Hours"] * g["Billable_Rate_Hr"]
    g["Quote_Gap"] = g["Quoted_Amount"] - g["Expected_Quote"]
    g["Quote_Gap_Pct"] = np.where(g["Expected_Quote"] > 0, (g["Quote_Gap"] / g["Expected_Quote"]) * 100, 0)
    
    return g


# =============================================================================
# TASK SUMMARY
# =============================================================================

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
        "Task Category", "Fiscal_Year", "FY_Label", "Calendar_Month"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Base_Cost": "sum",
        "Calc_Cost_TM": "sum",
        "Cost_Rate_Hr": "mean",
        "Billable_Rate_Hr": "mean",
        "Quoted_Rate_Hr": "mean",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Fiscal_Year", "FY_Label", "Month",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours", "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Base_Cost", "Cost_TM", "Cost_Rate_Hr", "Billable_Rate_Hr", "Quoted_Rate_Hr"
    ]
    
    g["Profit"] = g["Billable_Value"] - g["Base_Cost"]
    g["Quoted_Margin"] = g["Quoted_Amount"] - g["Base_Cost"]
    g["Actual_Margin"] = g["Billable_Value"] - g["Base_Cost"]
    g["Margin_Variance"] = g["Actual_Margin"] - g["Quoted_Margin"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, (g["Quoted_Margin"] / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Actual_Margin"] / g["Billable_Value"]) * 100, 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    g["Unbilled_Hours"] = g["Actual_Hours"] - g["Invoiced_Hours"]
    g["Is_Unquoted"] = (g["Quoted_Hours"] == 0) & (g["Actual_Hours"] > 0)
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Has_Unbilled"] = g["Unbilled_Hours"] > 0.5
    
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
    """Generate narrative insights from the data."""
    insights = {
        "headline": [],
        "margin_drivers": [],
        "quoting_accuracy": [],
        "department_performance": [],
        "trends": [],
        "action_items": []
    }
    
    # Overall metrics
    total_quoted = job_summary["Quoted_Amount"].sum()
    total_billable = job_summary["Billable_Value"].sum()
    total_cost = job_summary["Base_Cost"].sum()
    total_profit = total_billable - total_cost
    overall_margin = (total_profit / total_billable * 100) if total_billable > 0 else 0
    realization = (total_billable / total_quoted * 100) if total_quoted > 0 else 0
    
    # Headline insights
    if realization < 90:
        insights["headline"].append(f"⚠️ Revenue realization at {realization:.0f}% — billing {100-realization:.0f}% less than quoted")
    elif realization > 110:
        insights["headline"].append(f"✅ Strong revenue realization at {realization:.0f}% — exceeding quotes by {realization-100:.0f}%")
    
    if overall_margin < 20:
        insights["headline"].append(f"🔴 Overall margin critically low at {overall_margin:.1f}%")
    elif overall_margin < 35:
        insights["headline"].append(f"🟡 Overall margin below target at {overall_margin:.1f}%")
    else:
        insights["headline"].append(f"🟢 Healthy overall margin at {overall_margin:.1f}%")
    
    # Loss-making jobs
    loss_jobs = job_summary[job_summary["Is_Loss"]]
    if len(loss_jobs) > 0:
        total_losses = loss_jobs["Profit"].sum()
        insights["headline"].append(f"💸 {len(loss_jobs)} jobs running at a loss, totaling ${abs(total_losses):,.0f}")
    
    # Margin drivers (which departments/products are hurting or helping)
    if len(dept_summary) > 0:
        worst_dept = dept_summary.loc[dept_summary["Billable_Margin_Pct"].idxmin()]
        best_dept = dept_summary.loc[dept_summary["Billable_Margin_Pct"].idxmax()]
        
        if worst_dept["Billable_Margin_Pct"] < 15:
            insights["margin_drivers"].append(
                f"🔻 **{worst_dept['Department']}** dragging margins at {worst_dept['Billable_Margin_Pct']:.1f}% "
                f"(${worst_dept['Actual_Margin']:,.0f} on ${worst_dept['Billable_Value']:,.0f} revenue)"
            )
        
        if best_dept["Billable_Margin_Pct"] > 40:
            insights["margin_drivers"].append(
                f"🔺 **{best_dept['Department']}** leading with {best_dept['Billable_Margin_Pct']:.1f}% margin "
                f"(${best_dept['Actual_Margin']:,.0f} profit)"
            )
    
    # Quoting accuracy
    overquoted = job_summary[job_summary["Hours_Variance"] < -5]
    underquoted = job_summary[job_summary["Hours_Variance_Pct"] > 25]
    
    if len(underquoted) > 0:
        excess_hours = underquoted["Hours_Variance"].sum()
        excess_cost = (underquoted["Hours_Variance"] * underquoted["Cost_Rate_Hr"]).sum()
        insights["quoting_accuracy"].append(
            f"📉 {len(underquoted)} jobs significantly underquoted (+{excess_hours:,.0f} excess hours, "
            f"~${excess_cost:,.0f} unrecovered cost)"
        )
    
    # Unquoted work (scope creep)
    unquoted_tasks = task_summary[task_summary["Is_Unquoted"]]
    if len(unquoted_tasks) > 0:
        unquoted_cost = unquoted_tasks["Base_Cost"].sum()
        unquoted_hours = unquoted_tasks["Actual_Hours"].sum()
        insights["quoting_accuracy"].append(
            f"📋 {len(unquoted_tasks)} unquoted tasks detected (scope creep) — "
            f"{unquoted_hours:,.0f} hours at ${unquoted_cost:,.0f} cost"
        )
    
    # Monthly trends
    if len(monthly_summary) >= 3:
        recent = monthly_summary.tail(3)
        margin_trend = recent["Actual_Margin_Pct"].values
        if len(margin_trend) >= 3:
            if margin_trend[-1] > margin_trend[-3] + 5:
                insights["trends"].append(f"📈 Margins improving — up {margin_trend[-1] - margin_trend[-3]:.1f}pp over last 3 months")
            elif margin_trend[-1] < margin_trend[-3] - 5:
                insights["trends"].append(f"📉 Margins declining — down {margin_trend[-3] - margin_trend[-1]:.1f}pp over last 3 months")
    
    # Action items
    if len(loss_jobs) > 0:
        top_loss = loss_jobs.nsmallest(3, "Profit")
        for _, job in top_loss.iterrows():
            insights["action_items"].append(
                f"Review **{job['Job_Name'][:40]}** ({job['Job_No']}) — "
                f"${job['Profit']:,.0f} loss, {job['Hours_Variance_Pct']:+.0f}% hours variance"
            )
    
    return insights


def compute_waterfall_data(
    quoted_amount: float,
    billable_value: float,
    base_cost: float,
    hours_variance_cost: float = 0,
    rate_variance: float = 0
) -> pd.DataFrame:
    """Create waterfall chart data for margin bridge."""
    data = []
    
    # Start with quoted margin
    quoted_margin = quoted_amount - base_cost
    data.append({"Category": "Quoted Margin", "Amount": quoted_margin, "Type": "start"})
    
    # Revenue variance
    revenue_var = billable_value - quoted_amount
    data.append({"Category": "Revenue Variance", "Amount": revenue_var, "Type": "delta"})
    
    # End with actual margin
    actual_margin = billable_value - base_cost
    data.append({"Category": "Actual Margin", "Amount": actual_margin, "Type": "end"})
    
    return pd.DataFrame(data)


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def get_top_overruns(js: pd.DataFrame, n: int = 10, by: str = "Hours_Variance") -> pd.DataFrame:
    return js.nlargest(n, by)


def get_loss_making_jobs(js: pd.DataFrame) -> pd.DataFrame:
    return js[js["Is_Loss"]].sort_values("Profit")


def get_unquoted_tasks(ts: pd.DataFrame) -> pd.DataFrame:
    return ts[ts["Is_Unquoted"]].sort_values("Base_Cost", ascending=False)


def get_margin_erosion_jobs(js: pd.DataFrame, threshold: float = 10) -> pd.DataFrame:
    return js[js["Margin_Erosion"] > threshold].sort_values("Margin_Erosion", ascending=False)


def get_underquoted_jobs(js: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return js[js["Quote_Gap"] < threshold].sort_values("Quote_Gap")


def get_premium_jobs(js: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return js[js["Quote_Gap"] > threshold].sort_values("Quote_Gap", ascending=False)


def diagnose_job_margin(job_row: pd.Series, job_tasks: pd.DataFrame) -> Dict:
    """Diagnose margin issues for a specific job."""
    diagnosis = {
        "summary": "",
        "issues": [],
        "root_causes": [],
        "recommendations": []
    }
    
    # Basic checks
    margin = job_row["Actual_Margin"]
    margin_pct = job_row["Billable_Margin_Pct"]
    quote_gap = job_row["Quote_Gap"]
    hours_var = job_row["Hours_Variance"]
    hours_var_pct = job_row["Hours_Variance_Pct"]
    
    # Determine overall health
    if margin < 0:
        diagnosis["summary"] = "Job is running at a loss"
        diagnosis["issues"].append("Negative margin")
    elif margin_pct < 20:
        diagnosis["summary"] = "Job has low profitability"
        diagnosis["issues"].append("Low margin percentage")
    else:
        diagnosis["summary"] = "Job is profitable"
    
    # Quote gap analysis
    if quote_gap < -1000:
        diagnosis["issues"].append("Significantly underquoted")
        diagnosis["root_causes"].append("Pricing too low compared to internal rates")
        diagnosis["recommendations"].append("Review pricing strategy for similar jobs")
    elif quote_gap > 1000:
        diagnosis["issues"].append("Premium pricing applied")
    
    # Hours variance analysis
    if hours_var_pct > 50:
        diagnosis["issues"].append("Major scope overrun")
        diagnosis["root_causes"].append("Underestimated effort requirements")
        diagnosis["recommendations"].append("Improve effort estimation process")
    elif hours_var_pct < -20:
        diagnosis["issues"].append("Significant underrun")
    
    # Task-level analysis
    if len(job_tasks) > 0:
        unquoted_tasks = job_tasks[job_tasks["Is_Unquoted"]]
        if len(unquoted_tasks) > 0:
            diagnosis["issues"].append(f"{len(unquoted_tasks)} unquoted tasks")
            diagnosis["root_causes"].append("Scope changes not properly quoted")
            diagnosis["recommendations"].append("Implement change order process")
    
    return diagnosis


def calculate_overall_metrics(js: pd.DataFrame) -> dict:
    n = len(js)
    if n == 0:
        return {k: 0 for k in [
            "total_jobs", "total_quoted_amount", "total_expected_quote", "total_billable_value", "total_base_cost", "total_profit",
            "margin", "margin_pct", "quote_gap", "quote_gap_pct", "jobs_underquoted",
            "overall_quoted_margin", "overall_actual_margin", "overall_margin_variance",
            "overall_quoted_margin_pct", "overall_billable_margin_pct", "revenue_realization_pct",
            "avg_quoted_rate_hr", "avg_billable_rate_hr", "avg_effective_rate_hr", "avg_cost_rate_hr",
            "jobs_over_budget", "jobs_at_loss", "overrun_rate", "loss_rate",
            "total_hours_quoted", "total_hours_actual", "hours_variance", "hours_variance_pct",
            "total_margin_variance"
        ]}
    
    q, b, c = js["Quoted_Amount"].sum(), js["Billable_Value"].sum(), js["Base_Cost"].sum()
    p = b - c
    hq, ha = js["Quoted_Hours"].sum(), js["Actual_Hours"].sum()
    eq = js["Expected_Quote"].sum()
    
    quoted_margin = q - c
    actual_margin = b - c
    
    quote_gap = q - eq
    quote_gap_pct = (quote_gap / eq * 100) if eq > 0 else 0
    jobs_underquoted = int((js["Quote_Gap"] < 0).sum())
    avg_effective_rate_hr = (q / ha) if ha > 0 else 0
    
    return {
        "total_jobs": n,
        "total_quoted_amount": q,
        "total_expected_quote": eq,
        "total_billable_value": b,
        "total_base_cost": c,
        "total_cost_tm": c,
        "total_profit": p,
        "margin": quoted_margin,
        "margin_pct": (quoted_margin / q * 100) if q > 0 else 0,
        "quote_gap": quote_gap,
        "quote_gap_pct": quote_gap_pct,
        "jobs_underquoted": jobs_underquoted,
        "overall_quoted_margin": quoted_margin,
        "overall_actual_margin": actual_margin,
        "overall_margin_variance": actual_margin - quoted_margin,
        "overall_quoted_margin_pct": (quoted_margin / q * 100) if q > 0 else 0,
        "overall_billable_margin_pct": (p / b * 100) if b > 0 else 0,
        "revenue_realization_pct": (b / q * 100) if q > 0 else 0,
        "avg_quoted_rate_hr": (q / hq) if hq > 0 else 0,
        "avg_billable_rate_hr": (b / ha) if ha > 0 else 0,
        "avg_effective_rate_hr": avg_effective_rate_hr,
        "avg_cost_rate_hr": (c / ha) if ha > 0 else 0,
        "jobs_over_budget": int(js["Is_Overrun"].sum()),
        "jobs_at_loss": int(js["Is_Loss"].sum()),
        "overrun_rate": (js["Is_Overrun"].sum() / n * 100) if n > 0 else 0,
        "loss_rate": (js["Is_Loss"].sum() / n * 100) if n > 0 else 0,
        "total_hours_quoted": hq,
        "total_hours_actual": ha,
        "hours_variance": ha - hq,
        "hours_variance_pct": ((ha - hq) / hq * 100) if hq > 0 else 0,
    }


def analyze_overrun_causes(ts: pd.DataFrame) -> dict:
    unq = ts[ts["Is_Unquoted"]]
    ovr = ts[(ts["Is_Overrun"]) & (~ts["Is_Unquoted"])]
    unb = ts[ts["Has_Unbilled"]]
    return {
        "scope_creep": {"count": len(unq), "cost": unq["Base_Cost"].sum(), "hours": unq["Actual_Hours"].sum()},
        "underestimation": {"count": len(ovr), "excess_hours": ovr["Hours_Variance"].sum()},
        "unbilled": {"count": len(unb), "hours": unb["Unbilled_Hours"].sum()},
    }


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

METRIC_DEFINITIONS = {
    "Quoted_Amount": {"name": "Quoted Amount", "formula": "[Job Task] Quoted Amount", "desc": "Revenue from original quote"},
    "Billable_Value": {"name": "Billable Value", "formula": "Actual Hours × Billable Rate/Hr", "desc": "Value at standard billing rate"},
    "Base_Cost": {"name": "Base Cost", "formula": "Actual Hours × Cost Rate/Hr", "desc": "Internal labor cost"},
    "Quoted_Margin": {"name": "Quoted Margin", "formula": "Quoted Amount - Base Cost", "desc": "Margin if we achieved quoted revenue"},
    "Actual_Margin": {"name": "Actual Margin", "formula": "Billable Value - Base Cost", "desc": "True margin at billing rates"},
    "Margin_Variance": {"name": "Margin Variance", "formula": "Actual Margin - Quoted Margin", "desc": "Difference from quoted expectations"},
    "Quoted_Margin_Pct": {"name": "Quoted Margin %", "formula": "(Quoted Margin / Quoted Amount) × 100", "desc": "Margin percentage if quoted"},
    "Billable_Margin_Pct": {"name": "Actual Margin %", "formula": "(Actual Margin / Billable Value) × 100", "desc": "Actual margin percentage"},
    "Quoted_Rate_Hr": {"name": "Quoted Rate/Hr", "formula": "Quoted Amount / Quoted Hours", "desc": "Implied rate from quote"},
    "Billable_Rate_Hr": {"name": "Billable Rate/Hr", "formula": "[Task] Billable Rate", "desc": "Standard client rate"},
    "Cost_Rate_Hr": {"name": "Cost Rate/Hr", "formula": "[Task] Base Rate", "desc": "Internal cost per hour"},
    "Realization_Pct": {"name": "Revenue Realization", "formula": "(Billable Value / Quoted Amount) × 100", "desc": "How much of quote was realized"},
}
