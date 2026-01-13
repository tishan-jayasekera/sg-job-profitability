"""
Job Profitability Analysis Module
=================================
Data loading, cleaning, and metric calculations for job profitability analysis.
Follows structured approach: Category → Job → Task hierarchy.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# =============================================================================
# DATA PARSING UTILITIES
# =============================================================================

def parse_numeric(val):
    """Parse numeric values, handling commas and #N/A."""
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


def get_fiscal_year(date):
    """
    Get Australian fiscal year from date.
    FY runs July 1 - June 30. E.g., FY26 = Jul 2025 - Jun 2026.
    """
    if pd.isna(date):
        return None
    return date.year + 1 if date.month >= 7 else date.year


def get_fy_label(fy):
    """Convert fiscal year int to label (e.g., 2026 -> 'FY26')."""
    if pd.isna(fy):
        return "Unknown"
    return f"FY{str(int(fy))[-2:]}"


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_data(filepath, sheet_name: str = "Data") -> pd.DataFrame:
    """
    Load the Excel dataset and perform initial cleaning.
    
    Key steps per the analysis plan:
    1. Load Data sheet with granular task records
    2. Filter out "Social Garden Invoice Allocation" (internal allocations)
    3. Parse numeric and date columns
    4. Add fiscal year columns for period filtering
    5. Clean Yes/No flags to boolean
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # EXCLUSION: Remove internal allocation entries
    allocation_mask = df["[Job Task] Name"] == "Social Garden Invoice Allocation"
    df = df[~allocation_mask].copy()
    
    # Parse numeric columns (handle commas, #N/A)
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
    
    # Add fiscal year columns based on [Job] Start Date
    df["Fiscal_Year"] = df["[Job] Start Date"].apply(get_fiscal_year)
    df["FY_Label"] = df["Fiscal_Year"].apply(get_fy_label)
    
    # Clean Yes/No columns to boolean
    bool_cols = ["[Job Task] Billable", "[Job Task] Completed", "[Job Task] Allocated"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
    
    # Derive additional columns for analysis
    # Quoted Rate = Quoted Amount / Quoted Time (if quoted time > 0)
    df["Quoted_Rate"] = np.where(
        df["[Job Task] Quoted Time"] > 0,
        df["[Job Task] Quoted Amount"] / df["[Job Task] Quoted Time"],
        0
    )
    
    return df


def get_available_fiscal_years(df: pd.DataFrame) -> list:
    """Get sorted list of unique fiscal years in the data."""
    years = df["Fiscal_Year"].dropna().unique()
    return sorted([int(y) for y in years if pd.notna(y)])


# =============================================================================
# LEVEL 1: CATEGORY-LEVEL AGGREGATION
# =============================================================================

def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-level summary by Job Category.
    
    Per analysis plan: "For each category, compile aggregate metrics:
    total quoted revenue, total actual billable revenue, total quoted 
    hours vs actual hours, and overall margin %."
    """
    cat_group = df.groupby("[Job] Category").agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Billable Amount": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "Time+Material (Base)": "sum",
        "[Job Task] Cost": "sum",
        "[Job] Job No.": pd.Series.nunique,  # Count unique jobs
    }).reset_index()
    
    cat_group.columns = [
        "Category", "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Billable_Amount", "Invoiced_Amount", "TM_Cost", "Task_Cost", "Job_Count"
    ]
    
    # Use Time+Material (Base) as primary cost; fallback to Task_Cost
    cat_group["Actual_Cost"] = cat_group["TM_Cost"].where(
        cat_group["TM_Cost"] > 0, cat_group["Task_Cost"]
    )
    
    # Profit and Margin
    cat_group["Profit"] = cat_group["Billable_Amount"] - cat_group["Actual_Cost"]
    cat_group["Margin_Pct"] = np.where(
        cat_group["Billable_Amount"] > 0,
        (cat_group["Profit"] / cat_group["Billable_Amount"]) * 100,
        0
    )
    
    # Quoted Margin (expected based on quote)
    cat_group["Quoted_Margin_Pct"] = np.where(
        cat_group["Quoted_Amount"] > 0,
        ((cat_group["Quoted_Amount"] - cat_group["Actual_Cost"]) / cat_group["Quoted_Amount"]) * 100,
        0
    )
    
    # Variances
    cat_group["Hours_Variance"] = cat_group["Actual_Hours"] - cat_group["Quoted_Hours"]
    cat_group["Hours_Variance_Pct"] = np.where(
        cat_group["Quoted_Hours"] > 0,
        (cat_group["Hours_Variance"] / cat_group["Quoted_Hours"]) * 100,
        0
    )
    
    cat_group["Revenue_Variance"] = cat_group["Billable_Amount"] - cat_group["Quoted_Amount"]
    cat_group["Revenue_Variance_Pct"] = np.where(
        cat_group["Quoted_Amount"] > 0,
        (cat_group["Revenue_Variance"] / cat_group["Quoted_Amount"]) * 100,
        0
    )
    
    return cat_group.drop(columns=["TM_Cost", "Task_Cost"])


# =============================================================================
# LEVEL 2: JOB-LEVEL AGGREGATION
# =============================================================================

def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Job-level profitability summary.
    
    Per analysis plan: "For each project: Quoted $, Actual $, Quoted Hours,
    Actual Hours, Cost, Profit, Margin %."
    """
    job_group = df.groupby([
        "[Job] Category", "[Job] Job No.", "[Job] Name",
        "[Job] Client", "[Job] Client Manager", "[Job] Status",
        "[Job] Start Date", "Fiscal_Year", "FY_Label"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Billable Amount": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Cost": "sum",
        "Time+Material (Base)": "sum",
        "[Job] Budget": "first",
    }).reset_index()
    
    job_group.columns = [
        "Category", "Job_No", "Job_Name", "Client", "Client_Manager", "Status",
        "Start_Date", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Billable_Amount", "Invoiced_Amount", "Invoiced_Hours",
        "Task_Cost", "TM_Cost", "Budget"
    ]
    
    # Primary cost: Time+Material (Base), fallback to Task_Cost
    job_group["Actual_Cost"] = job_group["TM_Cost"].where(
        job_group["TM_Cost"] > 0, job_group["Task_Cost"]
    )
    
    # Profit and Margin calculations
    job_group["Profit"] = job_group["Billable_Amount"] - job_group["Actual_Cost"]
    job_group["Margin_Pct"] = np.where(
        job_group["Billable_Amount"] > 0,
        (job_group["Profit"] / job_group["Billable_Amount"]) * 100,
        0
    )
    
    # Expected margin from quote
    job_group["Quoted_Margin_Pct"] = np.where(
        job_group["Quoted_Amount"] > 0,
        ((job_group["Quoted_Amount"] - job_group["Actual_Cost"]) / job_group["Quoted_Amount"]) * 100,
        0
    )
    
    # Margin erosion (difference between expected and actual margin)
    job_group["Margin_Erosion"] = job_group["Quoted_Margin_Pct"] - job_group["Margin_Pct"]
    
    # Hour variances
    job_group["Hours_Variance"] = job_group["Actual_Hours"] - job_group["Quoted_Hours"]
    job_group["Hours_Variance_Pct"] = np.where(
        job_group["Quoted_Hours"] > 0,
        (job_group["Hours_Variance"] / job_group["Quoted_Hours"]) * 100,
        np.where(job_group["Actual_Hours"] > 0, 100, 0)
    )
    
    # Revenue variance
    job_group["Revenue_Variance"] = job_group["Billable_Amount"] - job_group["Quoted_Amount"]
    
    # Cost variance (actual cost vs quoted amount)
    job_group["Cost_Variance"] = job_group["Actual_Cost"] - job_group["Quoted_Amount"]
    
    # Flags for filtering
    job_group["Is_Overrun"] = job_group["Hours_Variance"] > 0
    job_group["Is_Loss"] = job_group["Profit"] < 0
    job_group["Is_Unquoted"] = (job_group["Quoted_Hours"] == 0) & (job_group["Actual_Hours"] > 0)
    job_group["Has_Margin_Erosion"] = job_group["Margin_Erosion"] > 10  # >10% erosion
    
    return job_group.drop(columns=["TM_Cost", "Task_Cost"])


# =============================================================================
# LEVEL 3: TASK-LEVEL AGGREGATION
# =============================================================================

def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task-level breakdown within each job.
    
    Per analysis plan: "List each task under the job with its quoted vs 
    actual hours and costs... identify which specific task or phase 
    caused the project overrun."
    """
    task_group = df.groupby([
        "[Job] Category", "[Job] Job No.", "[Job] Name", "[Job Task] Name",
        "Task Category", "[Job Task] Billable", "Fiscal_Year", "FY_Label"
    ]).agg({
        "[Job Task] Quoted Time": "sum",
        "[Job Task] Quoted Amount": "sum",
        "[Job Task] Actual Time (totalled)": "sum",
        "[Job Task] Billable Amount": "sum",
        "[Job Task] Invoiced Amount": "sum",
        "[Job Task] Invoiced Time": "sum",
        "[Job Task] Cost": "sum",
        "Time+Material (Base)": "sum",
        "[Task] Base Rate": "mean",
        "[Task] Billable Rate": "mean",
        "Quoted_Rate": "mean",
    }).reset_index()
    
    task_group.columns = [
        "Category", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Is_Billable", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Billable_Amount", "Invoiced_Amount", "Invoiced_Hours",
        "Task_Cost", "TM_Cost", "Base_Rate", "Billable_Rate", "Quoted_Rate"
    ]
    
    # Cost calculation
    task_group["Actual_Cost"] = task_group["TM_Cost"].where(
        task_group["TM_Cost"] > 0, task_group["Task_Cost"]
    )
    
    # Task profit and margin
    task_group["Profit"] = task_group["Billable_Amount"] - task_group["Actual_Cost"]
    task_group["Margin_Pct"] = np.where(
        task_group["Billable_Amount"] > 0,
        (task_group["Profit"] / task_group["Billable_Amount"]) * 100,
        0
    )
    
    # Hour variance
    task_group["Hours_Variance"] = task_group["Actual_Hours"] - task_group["Quoted_Hours"]
    task_group["Hours_Variance_Pct"] = np.where(
        task_group["Quoted_Hours"] > 0,
        (task_group["Hours_Variance"] / task_group["Quoted_Hours"]) * 100,
        np.where(task_group["Actual_Hours"] > 0, 100, 0)
    )
    
    # Cost variance
    task_group["Cost_Variance"] = task_group["Actual_Cost"] - task_group["Quoted_Amount"]
    
    # Billing efficiency: invoiced vs billable
    task_group["Billing_Efficiency"] = np.where(
        task_group["Billable_Amount"] > 0,
        (task_group["Invoiced_Amount"] / task_group["Billable_Amount"]) * 100,
        0
    )
    
    # Unbilled hours
    task_group["Unbilled_Hours"] = task_group["Actual_Hours"] - task_group["Invoiced_Hours"]
    
    # Flags
    task_group["Is_Unquoted"] = (task_group["Quoted_Hours"] == 0) & (task_group["Actual_Hours"] > 0)
    task_group["Is_Overrun"] = task_group["Hours_Variance"] > 0
    task_group["Has_Unbilled"] = task_group["Unbilled_Hours"] > 0
    
    return task_group.drop(columns=["TM_Cost", "Task_Cost"])


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def get_top_overruns(job_summary: pd.DataFrame, n: int = 10, by: str = "Hours_Variance") -> pd.DataFrame:
    """Get top N jobs by overrun metric."""
    return job_summary.nlargest(n, by)


def get_loss_making_jobs(job_summary: pd.DataFrame) -> pd.DataFrame:
    """Get all jobs with negative profit (losses)."""
    return job_summary[job_summary["Is_Loss"]].sort_values("Profit")


def get_margin_erosion_jobs(job_summary: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Get jobs where margin eroded more than threshold %."""
    return job_summary[job_summary["Margin_Erosion"] > threshold].sort_values("Margin_Erosion", ascending=False)


def get_unquoted_tasks(task_summary: pd.DataFrame) -> pd.DataFrame:
    """Get all tasks not in original quote (scope creep indicators)."""
    return task_summary[task_summary["Is_Unquoted"]].sort_values("Actual_Cost", ascending=False)


def get_unbilled_tasks(task_summary: pd.DataFrame) -> pd.DataFrame:
    """Get tasks with significant unbilled hours."""
    return task_summary[task_summary["Unbilled_Hours"] > 1].sort_values("Unbilled_Hours", ascending=False)


def calculate_overall_metrics(job_summary: pd.DataFrame) -> dict:
    """
    Calculate high-level KPIs across all jobs.
    
    Per analysis plan: "Total profit lost due to overruns, number of 
    projects over budget vs on budget, average margin % across all jobs 
    vs average quoted margin %."
    """
    total_jobs = len(job_summary)
    if total_jobs == 0:
        return {k: 0 for k in [
            "total_jobs", "total_quoted_revenue", "total_billable_revenue",
            "total_cost", "total_profit", "overall_margin_pct",
            "avg_margin_pct", "avg_quoted_margin_pct", "margin_gap",
            "jobs_over_budget", "jobs_on_budget", "jobs_at_loss",
            "overrun_rate", "loss_rate", "total_hours_quoted",
            "total_hours_actual", "hours_variance", "hours_variance_pct",
            "profit_lost_to_overruns"
        ]}
    
    total_quoted = job_summary["Quoted_Amount"].sum()
    total_billable = job_summary["Billable_Amount"].sum()
    total_cost = job_summary["Actual_Cost"].sum()
    total_profit = job_summary["Profit"].sum()
    
    jobs_over = int(job_summary["Is_Overrun"].sum())
    jobs_loss = int(job_summary["Is_Loss"].sum())
    
    hrs_quoted = job_summary["Quoted_Hours"].sum()
    hrs_actual = job_summary["Actual_Hours"].sum()
    hrs_variance = hrs_actual - hrs_quoted
    
    # Profit lost to overruns (sum of negative profits)
    losses = job_summary[job_summary["Profit"] < 0]["Profit"].sum()
    
    return {
        "total_jobs": total_jobs,
        "total_quoted_revenue": total_quoted,
        "total_billable_revenue": total_billable,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "overall_margin_pct": (total_profit / total_billable * 100) if total_billable > 0 else 0,
        "avg_margin_pct": job_summary["Margin_Pct"].mean(),
        "avg_quoted_margin_pct": job_summary["Quoted_Margin_Pct"].mean(),
        "margin_gap": job_summary["Quoted_Margin_Pct"].mean() - job_summary["Margin_Pct"].mean(),
        "jobs_over_budget": jobs_over,
        "jobs_on_budget": total_jobs - jobs_over,
        "jobs_at_loss": jobs_loss,
        "overrun_rate": (jobs_over / total_jobs * 100) if total_jobs > 0 else 0,
        "loss_rate": (jobs_loss / total_jobs * 100) if total_jobs > 0 else 0,
        "total_hours_quoted": hrs_quoted,
        "total_hours_actual": hrs_actual,
        "hours_variance": hrs_variance,
        "hours_variance_pct": (hrs_variance / hrs_quoted * 100) if hrs_quoted > 0 else 0,
        "profit_lost_to_overruns": abs(losses),
    }


def analyze_overrun_causes(task_summary: pd.DataFrame) -> dict:
    """
    Synthesis: Analyze common reasons for margin erosion.
    
    Per analysis plan: "Underestimation of effort, Scope Creep / Unquoted Work,
    Billing Issues, Rate Misalignment."
    """
    # Unquoted tasks (scope creep)
    unquoted = task_summary[task_summary["Is_Unquoted"]]
    unquoted_cost = unquoted["Actual_Cost"].sum()
    unquoted_hours = unquoted["Actual_Hours"].sum()
    
    # Overrun tasks (underestimation)
    overrun = task_summary[(task_summary["Is_Overrun"]) & (~task_summary["Is_Unquoted"])]
    overrun_hours = overrun["Hours_Variance"].sum()
    
    # Unbilled work (billing issues)
    unbilled = task_summary[task_summary["Has_Unbilled"]]
    unbilled_hours = unbilled["Unbilled_Hours"].sum()
    
    # Non-billable tasks with cost
    non_billable = task_summary[(~task_summary["Is_Billable"]) & (task_summary["Actual_Cost"] > 0)]
    non_billable_cost = non_billable["Actual_Cost"].sum()
    
    return {
        "scope_creep": {
            "task_count": len(unquoted),
            "total_cost": unquoted_cost,
            "total_hours": unquoted_hours,
        },
        "underestimation": {
            "task_count": len(overrun),
            "excess_hours": overrun_hours,
        },
        "billing_issues": {
            "tasks_with_unbilled": len(unbilled),
            "unbilled_hours": unbilled_hours,
        },
        "non_billable_work": {
            "task_count": len(non_billable),
            "total_cost": non_billable_cost,
        }
    }