"""
Job Profitability Analysis Logic
================================
Refactored to support the Revenue Allocation Model while maintaining
the granular drill-down capabilities of the original dashboard.

CORE CONCEPTS:
1. Allocated_Revenue (Realized): Monthly Revenue prorated to tasks based on timesheet effort.
   - REPLACES: 'Quoted Amount' as the revenue source.
2. Quoted_Amount (Budget): The static budget from the Quotation sheet.
   - USED FOR: Budget utilization tracking (capped revenue potential).
3. Billable_Value (Benchmark): Actual Hours * Billable Rate.
   - USED FOR: 'Revenue Realization Gap' (Did we earn what we should have?).
"""

import pandas as pd
import numpy as np

# Updated Metric Definitions Dictionary
METRIC_DEFINITIONS = {
    "Allocated_Revenue": {
        "name": "Recognized Revenue",
        "formula": "Monthly Revenue × (Task Hours / Monthly Job Hours)",
        "desc": "Actual revenue recognized, allocated to tasks based on effort."
    },
    "Quoted_Amount": {
        "name": "Quoted Budget",
        "formula": "Static value from Quote Sheet",
        "desc": "The original budget sold to the client."
    },
    "Margin": {
        "name": "Realized Margin",
        "formula": "Allocated_Revenue - Base_Cost",
        "desc": "True Profitability based on recognized revenue."
    },
    "Margin_Pct": {
        "name": "Realized Margin %",
        "formula": "Margin / Allocated_Revenue",
        "desc": "Profit margin percentage."
    },
    "Billable_Value": {
        "name": "Billable Value",
        "formula": "Actual Hours × Billable Rate",
        "desc": "The theoretical value of the work done (Benchmark)."
    },
    "Realization_Gap": {
        "name": "Realization Gap",
        "formula": "Allocated_Revenue - Billable_Value",
        "desc": "Difference between recognized revenue and standard billable value."
    },
    "Cost_Rate_Hr": {
        "name": "Avg Cost Rate/Hr",
        "formula": "Total Cost / Total Hours",
        "desc": "Weighted average internal cost per hour."
    },
    "Effective_Rate_Hr": {
        "name": "Effective Rate/Hr",
        "formula": "Allocated_Revenue / Total Hours",
        "desc": "Actual revenue earned per hour worked."
    }
}

# =============================================================================
# HELPER: DEDUPLICATE QUOTES
# =============================================================================
def get_unique_quotes(df, group_cols):
    """
    Extracts unique quote values to avoid double counting static data.
    """
    # We group by Job+Task to get unique line items, then aggregate up to the requested level
    unique_tasks = df.groupby(['Job_Number', 'Task_Name'] + group_cols, as_index=False).agg({
        'Quoted_Amount': 'max',
        'Quoted_Hours': 'max'
    })
    return unique_tasks.groupby(group_cols).agg({
        'Quoted_Amount': 'sum',
        'Quoted_Hours': 'sum'
    }).reset_index()

# =============================================================================
# AGGREGATION FUNCTIONS (Aligned with Original Structure)
# =============================================================================

def compute_overall_metrics(df):
    """Calculates high-level KPIs."""
    if df.empty:
        return {}
    
    total_revenue = df['Allocated_Revenue'].sum()
    total_cost = df['Base_Cost'].sum()
    total_margin = total_revenue - total_cost
    total_billable = df['Billable_Value'].sum()
    total_hours = df['Hours'].sum()
    
    # Quote Total (Deduped)
    # Aggregating whole DF to get global unique tasks
    unique_tasks = df.groupby(['Job_Number', 'Task_Name'], as_index=False).agg({
        'Quoted_Amount': 'max'
    })
    total_budget = unique_tasks['Quoted_Amount'].sum()
    
    return {
        "Total_Revenue": total_revenue,
        "Total_Cost": total_cost,
        "Total_Margin": total_margin,
        "Margin_Pct": (total_margin / total_revenue * 100) if total_revenue != 0 else 0,
        "Total_Billable_Value": total_billable,
        "Total_Quoted_Budget": total_budget,
        "Realization_Gap": total_revenue - total_billable,
        "Avg_Effective_Rate": (total_revenue / total_hours) if total_hours > 0 else 0
    }

def compute_department_summary(df):
    """Aggregates by Department."""
    group_col = 'Department'
    
    # Transactional Data
    trans = df.groupby(group_col).agg({
        'Allocated_Revenue': 'sum',
        'Base_Cost': 'sum',
        'Billable_Value': 'sum',
        'Hours': 'sum',
        'Job_Number': 'nunique'
    }).reset_index()
    
    # Static Data (Quotes)
    quotes = get_unique_quotes(df, [group_col])
    
    # Merge
    final = pd.merge(trans, quotes, on=group_col, how='left')
    
    # Metrics
    final['Margin'] = final['Allocated_Revenue'] - final['Base_Cost']
    final['Margin_Pct'] = np.where(final['Allocated_Revenue'] != 0, (final['Margin'] / final['Allocated_Revenue'] * 100), 0)
    final['Realization_Gap'] = final['Allocated_Revenue'] - final['Billable_Value']
    final['Effective_Rate_Hr'] = np.where(final['Hours'] > 0, final['Allocated_Revenue'] / final['Hours'], 0)
    
    return final

def compute_product_summary(df):
    """Aggregates by Department & Product."""
    group_cols = ['Department', 'Product']
    
    trans = df.groupby(group_cols).agg({
        'Allocated_Revenue': 'sum',
        'Base_Cost': 'sum',
        'Billable_Value': 'sum',
        'Hours': 'sum',
        'Job_Number': 'nunique'
    }).reset_index()
    
    quotes = get_unique_quotes(df, group_cols)
    final = pd.merge(trans, quotes, on=group_cols, how='left')
    
    final['Margin'] = final['Allocated_Revenue'] - final['Base_Cost']
    final['Margin_Pct'] = np.where(final['Allocated_Revenue'] != 0, (final['Margin'] / final['Allocated_Revenue'] * 100), 0)
    
    return final

def compute_job_summary(df):
    """
    Aggregates by Job. 
    Includes flags for Loss, Overrun, etc.
    """
    group_cols = ['Job_Number', 'Job_Name', 'Client', 'Department', 'Product']
    
    # Transactional
    trans = df.groupby(group_cols).agg({
        'Allocated_Revenue': 'sum',
        'Base_Cost': 'sum',
        'Billable_Value': 'sum',
        'Hours': 'sum',
        'Date': 'max'
    }).reset_index()
    
    # Static
    quotes = get_unique_quotes(df, group_cols)
    final = pd.merge(trans, quotes, on=group_cols, how='left')
    
    # Metrics
    final['Margin'] = final['Allocated_Revenue'] - final['Base_Cost']
    final['Margin_Pct'] = np.where(final['Allocated_Revenue'] != 0, (final['Margin'] / final['Allocated_Revenue'] * 100), 0)
    final['Realization_Gap'] = final['Allocated_Revenue'] - final['Billable_Value']
    final['Hours_Variance'] = final['Hours'] - final['Quoted_Hours']
    
    # Flags (Aligned with original repo concepts)
    final['Is_Loss'] = final['Margin'] < 0
    final['Is_Overrun'] = final['Hours'] > final['Quoted_Hours']
    final['Is_Under_Realized'] = final['Realization_Gap'] < -500 # Significant gap
    
    return final

def compute_task_summary(df):
    """
    Aggregates by Task.
    """
    group_cols = ['Job_Number', 'Task_Name', 'Department']
    
    trans = df.groupby(group_cols).agg({
        'Allocated_Revenue': 'sum',
        'Base_Cost': 'sum',
        'Billable_Value': 'sum',
        'Hours': 'sum'
    }).reset_index()
    
    # For tasks, we can just grab max of static cols since we group by Job+Task
    quotes = df.groupby(group_cols).agg({
        'Quoted_Amount': 'max',
        'Quoted_Hours': 'max'
    }).reset_index()
    
    final = pd.merge(trans, quotes, on=group_cols, how='left')
    
    final['Margin'] = final['Allocated_Revenue'] - final['Base_Cost']
    final['Realization_Gap'] = final['Allocated_Revenue'] - final['Billable_Value']
    final['Is_Unquoted'] = (final['Quoted_Amount'] == 0) & (final['Hours'] > 0)
    
    return final

def compute_monthly_summary(df):
    """
    Aggregates by Month.
    """
    g = df.groupby('Month_Key').agg({
        'Allocated_Revenue': 'sum',
        'Base_Cost': 'sum',
        'Billable_Value': 'sum',
        'Hours': 'sum',
        'Job_Number': 'nunique'
    }).reset_index()
    
    g['Margin'] = g['Allocated_Revenue'] - g['Base_Cost']
    g['Margin_Pct'] = np.where(g['Allocated_Revenue'] != 0, (g['Margin'] / g['Allocated_Revenue'] * 100), 0)
    g['Realization_Gap'] = g['Allocated_Revenue'] - g['Billable_Value']
    g['Month_Label'] = g['Month_Key'].dt.strftime('%b %Y')
    
    return g.sort_values('Month_Key')

def get_fiscal_years(df):
    if 'Fiscal_Year' in df.columns:
        return sorted(df['Fiscal_Year'].unique().tolist())
    return []

def get_departments(df):
    if 'Department' in df.columns:
        return sorted(df['Department'].dropna().unique().tolist())
    return []