"""
Job Profitability Analysis Module
=================================
Hierarchy: Department → Product → Job → Task

MARGIN DEFINITIONS:
- Quoted Margin %:   (Quoted Amount - Cost) / Quoted Amount × 100
- Billable Margin %: (Billable Value - Cost) / Billable Value × 100

RATE DEFINITIONS (per hour):
- Quoted Rate/Hr:    Quoted Amount / Quoted Hours
- Billable Rate/Hr:  [Task] Billable Rate
- Cost Rate/Hr:      [Task] Base Rate (T&M cost per hour)

VALUE DEFINITIONS:
- Quoted Amount:     [Job Task] Quoted Amount
- Billable Value:    Actual Hours × Billable Rate/Hr
- Cost (T&M):        Actual Hours × Cost Rate/Hr
- Profit:            Billable Value - Cost
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional


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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data(filepath, sheet_name: str = "Data") -> pd.DataFrame:
    """Load raw Excel data."""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse columns and compute derived metrics.
    
    Uses existing columns:
    - "Product" (direct from data)
    - "Department" (direct from data)
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
    
    # Fiscal year
    df["Fiscal_Year"] = df["[Job] Start Date"].apply(get_fiscal_year)
    df["FY_Label"] = df["Fiscal_Year"].apply(get_fy_label)
    
    # Clean Product and Department (direct from data)
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
    # Billable Rate/Hr = [Task] Billable Rate
    df["Billable_Rate_Hr"] = df["[Task] Billable Rate"]
    
    # Cost Rate/Hr = [Task] Base Rate
    df["Cost_Rate_Hr"] = df["[Task] Base Rate"]
    
    # Quoted Rate/Hr = Quoted Amount / Quoted Hours
    df["Quoted_Rate_Hr"] = np.where(
        df["[Job Task] Quoted Time"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Quoted Time"],
        0
    )
    
    # =========================================================================
    # VALUE CALCULATIONS
    # =========================================================================
    # Billable Value = Actual Hours × Billable Rate/Hr
    df["Calc_Billable_Value"] = df["[Job Task] Actual Time (totalled)"] * df["Billable_Rate_Hr"]
    
    # Cost (T&M) = Actual Hours × Cost Rate/Hr
    df["Calc_Cost_TM"] = df["[Job Task] Actual Time (totalled)"] * df["Cost_Rate_Hr"]
    
    return df


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
# DEPARTMENT SUMMARY
# =============================================================================

def compute_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Department").agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Cost_TM": "sum",
        "[Job] Job No.": pd.Series.nunique,
        "Product": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Invoiced_Hours", "Billable_Value", "Cost_TM", "Job_Count", "Product_Count"]
    
    g["Profit"] = g["Billable_Value"] - g["Cost_TM"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, ((g["Quoted_Amount"] - g["Cost_TM"]) / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Profit"] / g["Billable_Value"]) * 100, 0)
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Cost_TM"] / g["Actual_Hours"], 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
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
        "Calc_Cost_TM": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    g.columns = ["Department", "Product", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
                 "Invoiced_Hours", "Billable_Value", "Cost_TM", "Job_Count"]
    
    g["Profit"] = g["Billable_Value"] - g["Cost_TM"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, ((g["Quoted_Amount"] - g["Cost_TM"]) / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Profit"] / g["Billable_Value"]) * 100, 0)
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Cost_TM"] / g["Actual_Hours"], 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, 0)
    
    return g


# =============================================================================
# JOB SUMMARY
# =============================================================================

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name",
        "[Job] Client", "[Job] Client Manager", "[Job] Status",
        "[Job] Start Date", "Fiscal_Year", "FY_Label"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Cost_TM": "sum",
        "[Job] Budget": "first",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Client", "Client_Manager", "Status",
        "Start_Date", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours", "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Cost_TM", "Budget"
    ]
    
    g["Profit"] = g["Billable_Value"] - g["Cost_TM"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, ((g["Quoted_Amount"] - g["Cost_TM"]) / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Profit"] / g["Billable_Value"]) * 100, 0)
    g["Margin_Erosion"] = g["Quoted_Margin_Pct"] - g["Billable_Margin_Pct"]
    g["Quoted_Rate_Hr"] = np.where(g["Quoted_Hours"] > 0, g["Quoted_Amount"] / g["Quoted_Hours"], 0)
    g["Billable_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Billable_Value"] / g["Actual_Hours"], 0)
    g["Cost_Rate_Hr"] = np.where(g["Actual_Hours"] > 0, g["Cost_TM"] / g["Actual_Hours"], 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    g["Unbilled_Hours"] = g["Actual_Hours"] - g["Invoiced_Hours"]
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Is_Loss"] = g["Profit"] < 0
    g["Has_Margin_Erosion"] = g["Margin_Erosion"] > 10
    
    return g


# =============================================================================
# TASK SUMMARY
# =============================================================================

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([
        "Department", "Product", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
        "Task Category", "Fiscal_Year", "FY_Label"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Cost_TM": "sum",
        "Cost_Rate_Hr": "mean",
        "Billable_Rate_Hr": "mean",
        "Quoted_Rate_Hr": "mean",
    }).reset_index()
    
    g.columns = [
        "Department", "Product", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours", "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Cost_TM", "Cost_Rate_Hr", "Billable_Rate_Hr", "Quoted_Rate_Hr"
    ]
    
    g["Profit"] = g["Billable_Value"] - g["Cost_TM"]
    g["Quoted_Margin_Pct"] = np.where(g["Quoted_Amount"] > 0, ((g["Quoted_Amount"] - g["Cost_TM"]) / g["Quoted_Amount"]) * 100, 0)
    g["Billable_Margin_Pct"] = np.where(g["Billable_Value"] > 0, (g["Profit"] / g["Billable_Value"]) * 100, 0)
    g["Hours_Variance"] = g["Actual_Hours"] - g["Quoted_Hours"]
    g["Hours_Variance_Pct"] = np.where(g["Quoted_Hours"] > 0, (g["Hours_Variance"] / g["Quoted_Hours"]) * 100, np.where(g["Actual_Hours"] > 0, 100, 0))
    g["Unbilled_Hours"] = g["Actual_Hours"] - g["Invoiced_Hours"]
    g["Is_Unquoted"] = (g["Quoted_Hours"] == 0) & (g["Actual_Hours"] > 0)
    g["Is_Overrun"] = g["Hours_Variance"] > 0
    g["Has_Unbilled"] = g["Unbilled_Hours"] > 0.5
    
    return g


# =============================================================================
# ANALYSIS
# =============================================================================

def get_top_overruns(js: pd.DataFrame, n: int = 10, by: str = "Hours_Variance") -> pd.DataFrame:
    return js.nlargest(n, by)

def get_loss_making_jobs(js: pd.DataFrame) -> pd.DataFrame:
    return js[js["Is_Loss"]].sort_values("Profit")

def get_unquoted_tasks(ts: pd.DataFrame) -> pd.DataFrame:
    return ts[ts["Is_Unquoted"]].sort_values("Cost_TM", ascending=False)

def calculate_overall_metrics(js: pd.DataFrame) -> dict:
    n = len(js)
    if n == 0:
        return {k: 0 for k in ["total_jobs", "total_quoted_amount", "total_billable_value", "total_cost_tm", "total_profit",
                               "overall_quoted_margin_pct", "overall_billable_margin_pct",
                               "avg_quoted_rate_hr", "avg_billable_rate_hr", "avg_cost_rate_hr",
                               "jobs_over_budget", "jobs_at_loss", "overrun_rate", "loss_rate",
                               "total_hours_quoted", "total_hours_actual", "hours_variance", "hours_variance_pct"]}
    
    q, b, c, p = js["Quoted_Amount"].sum(), js["Billable_Value"].sum(), js["Cost_TM"].sum(), js["Profit"].sum()
    hq, ha = js["Quoted_Hours"].sum(), js["Actual_Hours"].sum()
    
    return {
        "total_jobs": n,
        "total_quoted_amount": q,
        "total_billable_value": b,
        "total_cost_tm": c,
        "total_profit": p,
        "overall_quoted_margin_pct": ((q - c) / q * 100) if q > 0 else 0,
        "overall_billable_margin_pct": (p / b * 100) if b > 0 else 0,
        "avg_quoted_rate_hr": (q / hq) if hq > 0 else 0,
        "avg_billable_rate_hr": (b / ha) if ha > 0 else 0,
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
        "scope_creep": {"count": len(unq), "cost": unq["Cost_TM"].sum(), "hours": unq["Actual_Hours"].sum()},
        "underestimation": {"count": len(ovr), "excess_hours": ovr["Hours_Variance"].sum()},
        "unbilled": {"count": len(unb), "hours": unb["Unbilled_Hours"].sum()},
    }


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

METRIC_DEFINITIONS = {
    "Quoted_Amount": {"name": "Quoted Amount", "formula": "[Job Task] Quoted Amount", "desc": "Revenue from original quote"},
    "Billable_Value": {"name": "Billable Value", "formula": "Actual Hours × Billable Rate/Hr", "desc": "Value at standard billing rate"},
    "Cost_TM": {"name": "Cost (T&M)", "formula": "Actual Hours × Cost Rate/Hr", "desc": "Internal labor cost"},
    "Profit": {"name": "Profit", "formula": "Billable Value - Cost", "desc": "Gross profit"},
    "Quoted_Margin_Pct": {"name": "Quoted Margin %", "formula": "(Quoted Amount - Cost) / Quoted Amount × 100", "desc": "Margin if we billed quoted amount"},
    "Billable_Margin_Pct": {"name": "Billable Margin %", "formula": "(Billable Value - Cost) / Billable Value × 100", "desc": "Margin at standard billing rates"},
    "Quoted_Rate_Hr": {"name": "Quoted Rate/Hr", "formula": "Quoted Amount / Quoted Hours", "desc": "Implied rate from quote"},
    "Billable_Rate_Hr": {"name": "Billable Rate/Hr", "formula": "[Task] Billable Rate", "desc": "Standard client rate"},
    "Cost_Rate_Hr": {"name": "Cost Rate/Hr (T&M)", "formula": "[Task] Base Rate", "desc": "Internal cost per hour"},
}