"""
Job Profitability Analysis Module
=================================
Data loading, cleaning, and metric calculations for job profitability analysis.

METRIC DEFINITIONS (IMPORTANT):
-------------------------------
- Quoted Amount:   [Job Task] Quoted Amount (from quote/estimate)
- Quoted Hours:    [Job Task] Quoted Time (estimated hours)
- Actual Hours:    [Job Task] Actual Time (totalled) (logged hours)
- Billable Value:  Actual Hours × [Task] Billable Rate (what SHOULD be billed)
- Cost (T&M):      Actual Hours × [Task] Base Rate (internal labor cost)
- Profit:          Billable Value - Cost
- Margin %:        (Profit / Billable Value) × 100

FILTERING:
----------
- Billable tasks only: Tasks where BOTH [Task] Base Rate > 0 AND [Task] Billable Rate > 0
- Optional: Exclude "Social Garden Invoice Allocation" entries (toggle)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict


# =============================================================================
# DATA PARSING UTILITIES
# =============================================================================

def parse_numeric(val) -> float:
    """
    Parse numeric values, handling commas and #N/A.
    Returns 0.0 for invalid/missing values.
    """
    if pd.isna(val) or str(val).strip() in ("#N/A", "", "N/A", "-"):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def parse_date(val):
    """Parse date values in various formats (dd-Mon-yy, etc.)."""
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


def get_fiscal_year(date) -> int:
    """
    Get Australian fiscal year from date.
    FY runs July 1 - June 30. 
    Example: FY26 = Jul 2025 - Jun 2026 → returns 2026
    """
    if pd.isna(date):
        return None
    return date.year + 1 if date.month >= 7 else date.year


def get_fy_label(fy) -> str:
    """Convert fiscal year int to label (e.g., 2026 -> 'FY26')."""
    if pd.isna(fy):
        return "Unknown"
    return f"FY{str(int(fy))[-2:]}"


# =============================================================================
# DATA LOADING AND FILTERING
# =============================================================================

def load_raw_data(filepath, sheet_name: str = "Data") -> pd.DataFrame:
    """
    Load raw Excel data without any filtering.
    Returns unmodified dataframe for reconciliation purposes.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


def clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse numeric and date columns. Does NOT filter any rows.
    
    Numeric columns parsed:
    - [Job Task] Quoted Time, Quoted Amount, Actual Time, etc.
    - [Task] Base Rate, Billable Rate
    - Time+Material (Base)
    
    Date columns parsed:
    - [Job] Start Date, Due Date, Completed Date
    - [Job Task] Start Date, Due Date, Date Completed
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
    
    # Add fiscal year columns
    df["Fiscal_Year"] = df["[Job] Start Date"].apply(get_fiscal_year)
    df["FY_Label"] = df["Fiscal_Year"].apply(get_fy_label)
    
    # ==========================================================================
    # COMPUTED METRICS (based on rates × hours)
    # ==========================================================================
    
    # Billable Value = Actual Hours × Billable Rate
    # This is what SHOULD be billed based on time worked at standard rate
    df["Calc_Billable_Value"] = df["[Job Task] Actual Time (totalled)"] * df["[Task] Billable Rate"]
    
    # Cost (Time & Materials) = Actual Hours × Base Rate
    # This is the internal labor cost
    df["Calc_Cost_TM"] = df["[Job Task] Actual Time (totalled)"] * df["[Task] Base Rate"]
    
    # Quoted Rate (implied) = Quoted Amount / Quoted Time
    df["Calc_Quoted_Rate"] = np.where(
        df["[Job Task] Quoted Time"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Quoted Time"],
        0
    )
    
    return df


def apply_filters(
    df: pd.DataFrame,
    exclude_sg_allocation: bool = True,
    billable_only: bool = True,
    fiscal_year: int = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply filters to the dataset and return reconciliation info.
    
    Filters:
    1. exclude_sg_allocation: Remove "Social Garden Invoice Allocation" tasks
    2. billable_only: Keep only tasks where Base Rate > 0 AND Billable Rate > 0
    3. fiscal_year: Filter to specific FY (None = all years)
    
    Returns:
    - Filtered DataFrame
    - Reconciliation dict with counts
    """
    recon = {
        "raw_records": len(df),
        "excluded_sg_allocation": 0,
        "excluded_non_billable": 0,
        "excluded_other_fy": 0,
        "final_records": 0,
    }
    
    df_filtered = df.copy()
    
    # Filter 1: Social Garden Invoice Allocation
    if exclude_sg_allocation:
        mask_sg = df_filtered["[Job Task] Name"] == "Social Garden Invoice Allocation"
        recon["excluded_sg_allocation"] = mask_sg.sum()
        df_filtered = df_filtered[~mask_sg]
    
    # Filter 2: Billable tasks only (both rates must be > 0)
    if billable_only:
        mask_billable = (df_filtered["[Task] Base Rate"] > 0) & (df_filtered["[Task] Billable Rate"] > 0)
        recon["excluded_non_billable"] = (~mask_billable).sum()
        df_filtered = df_filtered[mask_billable]
    
    # Filter 3: Fiscal year
    if fiscal_year is not None:
        mask_fy = df_filtered["Fiscal_Year"] == fiscal_year
        recon["excluded_other_fy"] = (~mask_fy).sum()
        df_filtered = df_filtered[mask_fy]
    
    recon["final_records"] = len(df_filtered)
    
    return df_filtered, recon


def get_available_fiscal_years(df: pd.DataFrame) -> list:
    """Get sorted list of unique fiscal years in the data."""
    years = df["Fiscal_Year"].dropna().unique()
    return sorted([int(y) for y in years if pd.notna(y)])


# =============================================================================
# RECONCILIATION & VALIDATION
# =============================================================================

def compute_reconciliation_totals(df: pd.DataFrame, recon: Dict) -> Dict:
    """
    Compute totals for reconciliation/validation.
    These totals should match the underlying data exactly.
    """
    recon["totals"] = {
        # Hours
        "sum_quoted_hours": df["[Job Task] Quoted Time"].sum(),
        "sum_actual_hours": df["[Job Task] Actual Time (totalled)"].sum(),
        "sum_invoiced_hours": df["[Job Task] Invoiced Time"].sum(),
        
        # Amounts from data
        "sum_quoted_amount": df["[Job Task] Quoted Amount"].sum(),
        "sum_billable_amount_field": df["[Job Task] Billable Amount"].sum(),  # From data field
        "sum_invoiced_amount": df["[Job Task] Invoiced Amount"].sum(),
        "sum_tm_base_field": df["Time+Material (Base)"].sum(),  # From data field
        
        # Calculated amounts (rate × hours)
        "sum_calc_billable_value": df["Calc_Billable_Value"].sum(),  # Hours × Billable Rate
        "sum_calc_cost_tm": df["Calc_Cost_TM"].sum(),  # Hours × Base Rate
        
        # Counts
        "unique_jobs": df["[Job] Job No."].nunique(),
        "unique_tasks": df["[Job Task] Name"].nunique(),
        "unique_categories": df["[Job] Category"].nunique(),
    }
    return recon


# =============================================================================
# LEVEL 1: CATEGORY-LEVEL AGGREGATION
# =============================================================================

def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-level summary by Job Category.
    
    METRICS COMPUTED:
    -----------------
    - Quoted_Hours:     SUM([Job Task] Quoted Time)
    - Quoted_Amount:    SUM([Job Task] Quoted Amount)
    - Actual_Hours:     SUM([Job Task] Actual Time (totalled))
    - Billable_Value:   SUM(Actual Hours × Billable Rate) — calculated
    - Cost_TM:          SUM(Actual Hours × Base Rate) — calculated
    - Profit:           Billable_Value - Cost_TM
    - Margin_Pct:       (Profit / Billable_Value) × 100
    """
    cat_group = df.groupby("[Job] Category").agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Cost_TM": "sum",
        "[Job] Job No.": pd.Series.nunique,
    }).reset_index()
    
    cat_group.columns = [
        "Category", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Cost_TM", "Job_Count"
    ]
    
    # Profit = Billable Value - Cost
    cat_group["Profit"] = cat_group["Billable_Value"] - cat_group["Cost_TM"]
    
    # Margin % = Profit / Billable Value × 100
    cat_group["Margin_Pct"] = np.where(
        cat_group["Billable_Value"] > 0,
        (cat_group["Profit"] / cat_group["Billable_Value"]) * 100,
        0
    )
    
    # Variances
    cat_group["Hours_Variance"] = cat_group["Actual_Hours"] - cat_group["Quoted_Hours"]
    cat_group["Hours_Variance_Pct"] = np.where(
        cat_group["Quoted_Hours"] > 0,
        (cat_group["Hours_Variance"] / cat_group["Quoted_Hours"]) * 100,
        0
    )
    
    cat_group["Amount_Variance"] = cat_group["Billable_Value"] - cat_group["Quoted_Amount"]
    cat_group["Amount_Variance_Pct"] = np.where(
        cat_group["Quoted_Amount"] > 0,
        (cat_group["Amount_Variance"] / cat_group["Quoted_Amount"]) * 100,
        0
    )
    
    return cat_group


# =============================================================================
# LEVEL 2: JOB-LEVEL AGGREGATION
# =============================================================================

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Job-level profitability summary.
    
    METRICS COMPUTED:
    -----------------
    - Quoted_Hours:     SUM([Job Task] Quoted Time)
    - Quoted_Amount:    SUM([Job Task] Quoted Amount)
    - Actual_Hours:     SUM([Job Task] Actual Time (totalled))
    - Billable_Value:   SUM(Actual Hours × Billable Rate)
    - Cost_TM:          SUM(Actual Hours × Base Rate)
    - Profit:           Billable_Value - Cost_TM
    - Margin_Pct:       (Profit / Billable_Value) × 100
    - Quoted_Margin:    (Quoted_Amount - Cost_TM) / Quoted_Amount × 100
    - Margin_Erosion:   Quoted_Margin - Actual_Margin
    """
    job_group = df.groupby([
        "[Job] Category", "[Job] Job No.", "[Job] Name",
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
    
    job_group.columns = [
        "Category", "Job_No", "Job_Name", "Client", "Client_Manager", "Status",
        "Start_Date", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Cost_TM", "Budget"
    ]
    
    # Profit = Billable Value - Cost
    job_group["Profit"] = job_group["Billable_Value"] - job_group["Cost_TM"]
    
    # Margin % = Profit / Billable Value × 100
    job_group["Margin_Pct"] = np.where(
        job_group["Billable_Value"] > 0,
        (job_group["Profit"] / job_group["Billable_Value"]) * 100,
        0
    )
    
    # Quoted Margin = (Quoted Amount - Cost) / Quoted Amount × 100
    job_group["Quoted_Margin_Pct"] = np.where(
        job_group["Quoted_Amount"] > 0,
        ((job_group["Quoted_Amount"] - job_group["Cost_TM"]) / job_group["Quoted_Amount"]) * 100,
        0
    )
    
    # Margin Erosion = Quoted Margin - Actual Margin
    job_group["Margin_Erosion"] = job_group["Quoted_Margin_Pct"] - job_group["Margin_Pct"]
    
    # Hour variances
    job_group["Hours_Variance"] = job_group["Actual_Hours"] - job_group["Quoted_Hours"]
    job_group["Hours_Variance_Pct"] = np.where(
        job_group["Quoted_Hours"] > 0,
        (job_group["Hours_Variance"] / job_group["Quoted_Hours"]) * 100,
        np.where(job_group["Actual_Hours"] > 0, 100, 0)
    )
    
    # Amount variance
    job_group["Amount_Variance"] = job_group["Billable_Value"] - job_group["Quoted_Amount"]
    
    # Billing efficiency
    job_group["Unbilled_Hours"] = job_group["Actual_Hours"] - job_group["Invoiced_Hours"]
    
    # Flags
    job_group["Is_Overrun"] = job_group["Hours_Variance"] > 0
    job_group["Is_Loss"] = job_group["Profit"] < 0
    job_group["Has_Margin_Erosion"] = job_group["Margin_Erosion"] > 10
    
    return job_group


# =============================================================================
# LEVEL 3: TASK-LEVEL AGGREGATION
# =============================================================================

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task-level breakdown within each job.
    
    METRICS COMPUTED (per task):
    ----------------------------
    - Quoted_Hours:     [Job Task] Quoted Time
    - Quoted_Amount:    [Job Task] Quoted Amount
    - Actual_Hours:     [Job Task] Actual Time (totalled)
    - Billable_Value:   Actual Hours × [Task] Billable Rate
    - Cost_TM:          Actual Hours × [Task] Base Rate
    - Profit:           Billable_Value - Cost_TM
    - Margin_Pct:       (Profit / Billable_Value) × 100
    """
    task_group = df.groupby([
        "[Job] Category", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
        "Task Category", "Fiscal_Year", "FY_Label"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Calc_Billable_Value": "sum",
        "Calc_Cost_TM": "sum",
        "[Task] Base Rate": "mean",
        "[Task] Billable Rate": "mean",
        "Calc_Quoted_Rate": "mean",
    }).reset_index()
    
    task_group.columns = [
        "Category", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Invoiced_Hours", "Invoiced_Amount",
        "Billable_Value", "Cost_TM",
        "Base_Rate", "Billable_Rate", "Quoted_Rate"
    ]
    
    # Profit and Margin
    task_group["Profit"] = task_group["Billable_Value"] - task_group["Cost_TM"]
    task_group["Margin_Pct"] = np.where(
        task_group["Billable_Value"] > 0,
        (task_group["Profit"] / task_group["Billable_Value"]) * 100,
        0
    )
    
    # Variances
    task_group["Hours_Variance"] = task_group["Actual_Hours"] - task_group["Quoted_Hours"]
    task_group["Hours_Variance_Pct"] = np.where(
        task_group["Quoted_Hours"] > 0,
        (task_group["Hours_Variance"] / task_group["Quoted_Hours"]) * 100,
        np.where(task_group["Actual_Hours"] > 0, 100, 0)
    )
    
    task_group["Amount_Variance"] = task_group["Billable_Value"] - task_group["Quoted_Amount"]
    
    # Unbilled
    task_group["Unbilled_Hours"] = task_group["Actual_Hours"] - task_group["Invoiced_Hours"]
    
    # Flags
    task_group["Is_Unquoted"] = (task_group["Quoted_Hours"] == 0) & (task_group["Actual_Hours"] > 0)
    task_group["Is_Overrun"] = task_group["Hours_Variance"] > 0
    task_group["Has_Unbilled"] = task_group["Unbilled_Hours"] > 0.5  # > 30 min
    
    return task_group


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def get_top_overruns(job_summary: pd.DataFrame, n: int = 10, by: str = "Hours_Variance") -> pd.DataFrame:
    """Get top N jobs by overrun metric."""
    return job_summary.nlargest(n, by)


def get_loss_making_jobs(job_summary: pd.DataFrame) -> pd.DataFrame:
    """Get all jobs with negative profit (losses)."""
    return job_summary[job_summary["Is_Loss"]].sort_values("Profit")


def get_unquoted_tasks(task_summary: pd.DataFrame) -> pd.DataFrame:
    """Get tasks not in original quote (scope creep indicators)."""
    return task_summary[task_summary["Is_Unquoted"]].sort_values("Cost_TM", ascending=False)


def calculate_overall_metrics(job_summary: pd.DataFrame) -> dict:
    """
    Calculate high-level KPIs across all jobs.
    
    METRICS:
    --------
    - Total Quoted Amount
    - Total Billable Value (Actual Hours × Billable Rate)
    - Total Cost (Actual Hours × Base Rate)
    - Total Profit (Billable Value - Cost)
    - Overall Margin % (Profit / Billable Value × 100)
    - Jobs over budget count
    - Jobs at loss count
    - Hours variance
    """
    n = len(job_summary)
    if n == 0:
        return {k: 0 for k in [
            "total_jobs", "total_quoted_amount", "total_billable_value",
            "total_cost_tm", "total_profit", "overall_margin_pct",
            "avg_margin_pct", "avg_quoted_margin_pct", "margin_gap",
            "jobs_over_budget", "jobs_at_loss", "overrun_rate", "loss_rate",
            "total_hours_quoted", "total_hours_actual", "hours_variance",
            "profit_lost_to_losses"
        ]}
    
    total_quoted = job_summary["Quoted_Amount"].sum()
    total_billable = job_summary["Billable_Value"].sum()
    total_cost = job_summary["Cost_TM"].sum()
    total_profit = job_summary["Profit"].sum()
    
    jobs_over = int(job_summary["Is_Overrun"].sum())
    jobs_loss = int(job_summary["Is_Loss"].sum())
    
    hrs_quoted = job_summary["Quoted_Hours"].sum()
    hrs_actual = job_summary["Actual_Hours"].sum()
    
    losses_sum = job_summary[job_summary["Profit"] < 0]["Profit"].sum()
    
    return {
        "total_jobs": n,
        "total_quoted_amount": total_quoted,
        "total_billable_value": total_billable,
        "total_cost_tm": total_cost,
        "total_profit": total_profit,
        "overall_margin_pct": (total_profit / total_billable * 100) if total_billable > 0 else 0,
        "avg_margin_pct": job_summary["Margin_Pct"].mean(),
        "avg_quoted_margin_pct": job_summary["Quoted_Margin_Pct"].mean(),
        "margin_gap": job_summary["Quoted_Margin_Pct"].mean() - job_summary["Margin_Pct"].mean(),
        "jobs_over_budget": jobs_over,
        "jobs_at_loss": jobs_loss,
        "overrun_rate": (jobs_over / n * 100) if n > 0 else 0,
        "loss_rate": (jobs_loss / n * 100) if n > 0 else 0,
        "total_hours_quoted": hrs_quoted,
        "total_hours_actual": hrs_actual,
        "hours_variance": hrs_actual - hrs_quoted,
        "hours_variance_pct": ((hrs_actual - hrs_quoted) / hrs_quoted * 100) if hrs_quoted > 0 else 0,
        "profit_lost_to_losses": abs(losses_sum),
    }


def analyze_overrun_causes(task_summary: pd.DataFrame) -> dict:
    """
    Analyze common reasons for margin erosion.
    
    Categories:
    - Scope creep (unquoted tasks)
    - Underestimation (overrun tasks)
    - Unbilled work
    """
    unquoted = task_summary[task_summary["Is_Unquoted"]]
    overrun = task_summary[(task_summary["Is_Overrun"]) & (~task_summary["Is_Unquoted"])]
    unbilled = task_summary[task_summary["Has_Unbilled"]]
    
    return {
        "scope_creep": {
            "task_count": len(unquoted),
            "total_cost": unquoted["Cost_TM"].sum(),
            "total_hours": unquoted["Actual_Hours"].sum(),
        },
        "underestimation": {
            "task_count": len(overrun),
            "excess_hours": overrun["Hours_Variance"].sum(),
        },
        "unbilled_work": {
            "task_count": len(unbilled),
            "unbilled_hours": unbilled["Unbilled_Hours"].sum(),
        },
    }


# =============================================================================
# METRIC DEFINITIONS (for display in UI)
# =============================================================================

METRIC_DEFINITIONS = {
    "Quoted_Hours": {
        "name": "Quoted Hours",
        "formula": "SUM([Job Task] Quoted Time)",
        "description": "Total estimated hours from the original quote"
    },
    "Quoted_Amount": {
        "name": "Quoted Amount",
        "formula": "SUM([Job Task] Quoted Amount)",
        "description": "Total estimated revenue from the original quote"
    },
    "Actual_Hours": {
        "name": "Actual Hours",
        "formula": "SUM([Job Task] Actual Time (totalled))",
        "description": "Total hours actually logged/worked"
    },
    "Billable_Value": {
        "name": "Billable Value",
        "formula": "Actual Hours × [Task] Billable Rate",
        "description": "Revenue that SHOULD be billed based on hours worked at standard billable rate"
    },
    "Cost_TM": {
        "name": "Cost (Time & Materials)",
        "formula": "Actual Hours × [Task] Base Rate",
        "description": "Internal labor cost based on hours worked at base cost rate"
    },
    "Profit": {
        "name": "Profit",
        "formula": "Billable Value - Cost (T&M)",
        "description": "Gross profit = what can be billed minus internal labor cost"
    },
    "Margin_Pct": {
        "name": "Margin %",
        "formula": "(Profit / Billable Value) × 100",
        "description": "Profit as a percentage of billable value"
    },
    "Quoted_Margin_Pct": {
        "name": "Quoted Margin %",
        "formula": "(Quoted Amount - Cost) / Quoted Amount × 100",
        "description": "Expected margin based on quoted amount vs actual cost incurred"
    },
    "Margin_Erosion": {
        "name": "Margin Erosion",
        "formula": "Quoted Margin % - Actual Margin %",
        "description": "How much margin was lost compared to what was expected from quote"
    },
    "Hours_Variance": {
        "name": "Hours Variance",
        "formula": "Actual Hours - Quoted Hours",
        "description": "Positive = overrun (worked more than quoted)"
    },
    "Unbilled_Hours": {
        "name": "Unbilled Hours",
        "formula": "Actual Hours - Invoiced Hours",
        "description": "Hours worked but not yet invoiced to client"
    },
}