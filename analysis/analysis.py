"""
Job Profitability Analysis Module
Data loading, cleaning, and metric calculations for job profitability analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime


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
    """Parse date values in various formats."""
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
    FY runs July 1 - June 30. FY26 = Jul 2025 - Jun 2026.
    Returns integer like 2026 for FY26.
    """
    if pd.isna(date):
        return None
    if date.month >= 7:
        return date.year + 1
    return date.year


def load_data(filepath, sheet_name: str = "Data") -> pd.DataFrame:
    """
    Load the Excel dataset and perform initial cleaning.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Filter out internal allocation entries
    df = df[df["[Job Task] Name"] != "Social Garden Invoice Allocation"].copy()
    
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
    
    # Add fiscal year column based on Job Start Date
    df["Fiscal_Year"] = df["[Job] Start Date"].apply(get_fiscal_year)
    
    # Create FY label (e.g., "FY26")
    df["FY_Label"] = df["Fiscal_Year"].apply(
        lambda x: f"FY{str(int(x))[-2:]}" if pd.notna(x) else "Unknown"
    )
    
    # Clean Yes/No columns to boolean
    bool_cols = ["[Job Task] Billable", "[Job Task] Completed", "[Job Task] Allocated"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
    
    return df


def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data at the job level with profitability metrics.
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
        "Task_Cost", "Actual_Cost", "Budget"
    ]
    
    # Use Time+Material (Base) as primary cost; fallback to Task_Cost
    job_group["Actual_Cost"] = job_group["Actual_Cost"].where(
        job_group["Actual_Cost"] > 0, job_group["Task_Cost"]
    )
    
    # Calculate profit and margins
    job_group["Profit"] = job_group["Billable_Amount"] - job_group["Actual_Cost"]
    
    job_group["Margin_Pct"] = np.where(
        job_group["Billable_Amount"] > 0,
        (job_group["Profit"] / job_group["Billable_Amount"]) * 100,
        0
    )
    
    # Expected margin based on quote
    job_group["Expected_Profit"] = job_group["Quoted_Amount"] - job_group["Actual_Cost"]
    job_group["Quoted_Margin_Pct"] = np.where(
        job_group["Quoted_Amount"] > 0,
        ((job_group["Quoted_Amount"] - job_group["Actual_Cost"]) / job_group["Quoted_Amount"]) * 100,
        0
    )
    
    # Variances
    job_group["Hours_Variance"] = job_group["Actual_Hours"] - job_group["Quoted_Hours"]
    job_group["Hours_Variance_Pct"] = np.where(
        job_group["Quoted_Hours"] > 0,
        ((job_group["Actual_Hours"] - job_group["Quoted_Hours"]) / job_group["Quoted_Hours"]) * 100,
        np.where(job_group["Actual_Hours"] > 0, 100, 0)
    )
    
    job_group["Revenue_Variance"] = job_group["Billable_Amount"] - job_group["Quoted_Amount"]
    job_group["Cost_Variance"] = job_group["Actual_Cost"] - job_group["Quoted_Amount"]
    
    # Flags
    job_group["Is_Overrun"] = job_group["Hours_Variance"] > 0
    job_group["Is_Loss"] = job_group["Profit"] < 0
    job_group["Is_Under_Quoted"] = (job_group["Quoted_Hours"] == 0) & (job_group["Actual_Hours"] > 0)
    
    return job_group


def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data at the task level within each job.
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
    }).reset_index()
    
    task_group.columns = [
        "Category", "Job_No", "Job_Name", "Task_Name",
        "Task_Category", "Is_Billable", "Fiscal_Year", "FY_Label",
        "Quoted_Hours", "Quoted_Amount", "Actual_Hours",
        "Billable_Amount", "Invoiced_Amount", "Invoiced_Hours",
        "Task_Cost", "Actual_Cost", "Base_Rate", "Billable_Rate"
    ]
    
    # Use Time+Material (Base) as primary cost
    task_group["Actual_Cost"] = task_group["Actual_Cost"].where(
        task_group["Actual_Cost"] > 0, task_group["Task_Cost"]
    )
    
    # Task-level metrics
    task_group["Profit"] = task_group["Billable_Amount"] - task_group["Actual_Cost"]
    task_group["Margin_Pct"] = np.where(
        task_group["Billable_Amount"] > 0,
        (task_group["Profit"] / task_group["Billable_Amount"]) * 100,
        0
    )
    
    task_group["Hours_Variance"] = task_group["Actual_Hours"] - task_group["Quoted_Hours"]
    task_group["Hours_Variance_Pct"] = np.where(
        task_group["Quoted_Hours"] > 0,
        ((task_group["Actual_Hours"] - task_group["Quoted_Hours"]) / task_group["Quoted_Hours"]) * 100,
        np.where(task_group["Actual_Hours"] > 0, 100, 0)
    )
    
    task_group["Cost_Variance"] = task_group["Actual_Cost"] - task_group["Quoted_Amount"]
    
    # Flags
    task_group["Is_Unquoted"] = (task_group["Quoted_Hours"] == 0) & (task_group["Actual_Hours"] > 0)
    task_group["Is_Overrun"] = task_group["Hours_Variance"] > 0
    
    return task_group


def compute_category_summary(job_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate job summary to category level.
    """
    cat_group = job_summary.groupby("Category").agg({
        "Quoted_Hours": "sum",
        "Quoted_Amount": "sum",
        "Actual_Hours": "sum",
        "Actual_Cost": "sum",
        "Billable_Amount": "sum",
        "Profit": "sum",
        "Job_No": "count",
        "Is_Overrun": "sum",
        "Is_Loss": "sum"
    }).reset_index()
    
    cat_group.columns = [
        "Category", "Quoted_Hours", "Quoted_Amount",
        "Actual_Hours", "Actual_Cost", "Billable_Amount",
        "Profit", "Job_Count", "Overrun_Count", "Loss_Count"
    ]
    
    cat_group["Margin_Pct"] = np.where(
        cat_group["Billable_Amount"] > 0,
        (cat_group["Profit"] / cat_group["Billable_Amount"]) * 100,
        0
    )
    
    cat_group["Hours_Variance_Pct"] = np.where(
        cat_group["Quoted_Hours"] > 0,
        ((cat_group["Actual_Hours"] - cat_group["Quoted_Hours"]) / cat_group["Quoted_Hours"]) * 100,
        0
    )
    
    cat_group["Overrun_Rate"] = np.where(
        cat_group["Job_Count"] > 0,
        (cat_group["Overrun_Count"] / cat_group["Job_Count"]) * 100,
        0
    )
    
    return cat_group


def compute_client_summary(job_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate job summary to client level.
    """
    client_group = job_summary.groupby("Client").agg({
        "Quoted_Amount": "sum",
        "Actual_Cost": "sum",
        "Billable_Amount": "sum",
        "Profit": "sum",
        "Job_No": "count",
        "Is_Overrun": "sum",
        "Is_Loss": "sum"
    }).reset_index()
    
    client_group.columns = [
        "Client", "Quoted_Amount", "Actual_Cost", "Billable_Amount",
        "Profit", "Job_Count", "Overrun_Count", "Loss_Count"
    ]
    
    client_group["Margin_Pct"] = np.where(
        client_group["Billable_Amount"] > 0,
        (client_group["Profit"] / client_group["Billable_Amount"]) * 100,
        0
    )
    
    client_group["Overrun_Rate"] = np.where(
        client_group["Job_Count"] > 0,
        (client_group["Overrun_Count"] / client_group["Job_Count"]) * 100,
        0
    )
    
    return client_group.sort_values("Profit", ascending=True)


def get_top_overruns(job_summary: pd.DataFrame, n: int = 10, by: str = "Cost_Variance") -> pd.DataFrame:
    """Get the top N jobs by overrun amount."""
    return job_summary.nlargest(n, by)


def get_loss_making_jobs(job_summary: pd.DataFrame) -> pd.DataFrame:
    """Get all jobs with negative profit."""
    return job_summary[job_summary["Profit"] < 0].sort_values("Profit")


def get_unquoted_tasks(task_summary: pd.DataFrame) -> pd.DataFrame:
    """Get all tasks that were not in the original quote (scope creep)."""
    return task_summary[task_summary["Is_Unquoted"]].sort_values("Actual_Cost", ascending=False)


def calculate_overall_metrics(job_summary: pd.DataFrame) -> dict:
    """Calculate high-level KPIs across all jobs."""
    total_quoted = job_summary["Quoted_Amount"].sum()
    total_billable = job_summary["Billable_Amount"].sum()
    total_cost = job_summary["Actual_Cost"].sum()
    total_profit = job_summary["Profit"].sum()
    total_jobs = len(job_summary)
    
    return {
        "total_jobs": total_jobs,
        "total_quoted_revenue": total_quoted,
        "total_billable_revenue": total_billable,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "overall_margin_pct": (total_profit / total_billable * 100) if total_billable > 0 else 0,
        "jobs_over_budget": int(job_summary["Is_Overrun"].sum()),
        "jobs_at_loss": int(job_summary["Is_Loss"].sum()),
        "total_hours_quoted": job_summary["Quoted_Hours"].sum(),
        "total_hours_actual": job_summary["Actual_Hours"].sum(),
        "hours_variance": job_summary["Hours_Variance"].sum(),
        "revenue_variance": job_summary["Revenue_Variance"].sum(),
        "overrun_rate": (job_summary["Is_Overrun"].sum() / total_jobs * 100) if total_jobs > 0 else 0,
        "loss_rate": (job_summary["Is_Loss"].sum() / total_jobs * 100) if total_jobs > 0 else 0,
    }


def get_available_fiscal_years(df: pd.DataFrame) -> list:
    """Get list of unique fiscal years in the data."""
    years = df["Fiscal_Year"].dropna().unique()
    return sorted([int(y) for y in years if pd.notna(y)])
