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
# CONSTANTS & CONFIG
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
    "Cost_Rate_Hr": {"name": "Cost Rate/Hr", "formula": "[Task] Base Rate", "desc": "Internal cost rate"}
}

# =============================================================================
# DATA PARSING Helpers
# =============================================================================

def parse_numeric(val) -> float:
    """Parse numeric, handling commas and #N/A."""
    if pd.isna(val) or str(val).strip() in ("#N/A", "", "N/A", "-"):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        clean_str = str(val).replace(",", "").replace("$", "").strip()
        return float(clean_str)
    except:
        return 0.0

def parse_date(val):
    """Parse date in various formats."""
    if pd.isna(val) or str(val).strip() in ("", "#N/A", "N/A"):
        return pd.NaT
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.to_datetime(val)
    try:
        return pd.to_datetime(val, errors='coerce')
    except:
        return pd.NaT

# =============================================================================
# DATA LOADING & CLEANING
# =============================================================================

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw datasets.
    Assumes standard file naming conventions in 'data/' folder.
    """
    base_path = "data/"
    
    # Using the filenames provided in context
    try:
        df_data = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Data.csv")
        df_analysis = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Analysis.csv", header=4)
        df_rates = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Bill_Base_Rates.csv")
        df_jobs = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Job Classification.csv")
        df_people = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - People.csv")
        return df_data, df_analysis, df_rates, df_jobs, df_people
    except FileNotFoundError:
        # Fallback for testing environment if paths differ
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def clean_and_parse(df_data, df_jobs, df_rates) -> pd.DataFrame:
    """
    Main cleaning pipeline.
    1. Standardize columns
    2. Parse numerics/dates
    3. Merge Classification and Rates
    4. Calculate calculated columns (Billable Value, Costs)
    """
    if df_data.empty: return df_data

    # 1. Clean Column Names
    df = df_data.copy()
    # Remove brackets for easier access: [Job] Job No. -> Job_No
    df.columns = [c.replace('[Job]', '').replace('[Job Task]', '').replace('[Task]', '').replace('.', '').strip().replace(' ', '_') for c in df.columns]
    
    # 2. Parse Types
    numeric_cols = ['Quoted_Amount', 'Billable_Amount', 'Actual_Time', 'Quoted_Time', 'Base_Rate', 'Billable_Rate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric)

    date_cols = ['Date_Completed', 'Start_Date', 'Due_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)
            
    # Derive Month (use Date Completed, fallback to Start Date)
    df['Month'] = df['Date_Completed'].fillna(df['Start_Date']).dt.to_period('M').astype(str)
    df['FY'] = df['Date_Completed'].fillna(df['Start_Date']).apply(
        lambda x: f"FY{x.year + 1}" if x.month >= 7 else f"FY{x.year}" if pd.notnull(x) else "Unknown"
    )

    # 3. Merge Classification (Department, Product) if not present or to enrich
    # Checking if 'Job_Category' exists to map to Dept/Product
    if 'Job_Category' in df.columns:
        # Basic mapping logic if external file isn't perfect
        # Ideally we merge df_jobs here. Assuming df_jobs has 'Category' -> 'Department'
        pass 

    # 4. Calculate Financials
    # Ensure rates exist
    if 'Billable_Rate' not in df.columns: df['Billable_Rate'] = 0
    if 'Base_Rate' not in df.columns: df['Base_Rate'] = 0

    # Calculated Fields
    df['Calculated_Billable_Value'] = df['Actual_Time'] * df['Billable_Rate']
    df['Calculated_Base_Cost'] = df['Actual_Time'] * df['Base_Rate']
    
    # Handle Quoted Amount (fill NaNs)
    if 'Quoted_Amount' in df.columns:
        df['Quoted_Amount'] = df['Quoted_Amount'].fillna(0)
    
    # Margins
    df['Quoted_Margin'] = df['Quoted_Amount'] - df['Calculated_Base_Cost']
    df['Actual_Margin'] = df['Calculated_Billable_Value'] - df['Calculated_Base_Cost']
    df['Margin_Variance'] = df['Actual_Margin'] - df['Quoted_Margin']
    
    return df

# =============================================================================
# FILTERING & SELECTION
# =============================================================================

def get_available_fiscal_years(df):
    return sorted(df['FY'].unique().tolist(), reverse=True)

def get_available_departments(df):
    if 'Department' in df.columns:
        return sorted(df['Department'].dropna().unique().tolist())
    return []

def get_available_products(df):
    if 'Product' in df.columns:
        return sorted(df['Product'].dropna().unique().tolist())
    return []

def apply_filters(df, fy=None, departments=None, products=None):
    mask = pd.Series(True, index=df.index)
    if fy:
        mask &= (df['FY'] == fy)
    if departments:
        mask &= (df['Department'].isin(departments))
    if products:
        mask &= (df['Product'].isin(products))
    return df[mask]

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def calculate_overall_metrics(df):
    """Returns dictionary of high-level KPIs."""
    return {
        "Total_Quoted": df['Quoted_Amount'].sum(),
        "Total_Billable": df['Calculated_Billable_Value'].sum(),
        "Total_Cost": df['Calculated_Base_Cost'].sum(),
        "Total_Hours": df['Actual_Time'].sum(),
        "Quoted_Margin": df['Quoted_Margin'].sum(),
        "Actual_Margin": df['Actual_Margin'].sum(),
        "Net_Variance": df['Margin_Variance'].sum()
    }

def compute_monthly_summary(df):
    """Aggregates metrics by Month."""
    g = df.groupby('Month').agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum',
        'Calculated_Base_Cost': 'sum',
        'Actual_Time': 'sum'
    }).reset_index()
    g = g.sort_values('Month')
    return g

def compute_department_summary(df):
    if 'Department' not in df.columns: return pd.DataFrame()
    return df.groupby('Department').agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum',
        'Actual_Margin': 'sum',
        'Margin_Variance': 'sum'
    }).reset_index().sort_values('Calculated_Billable_Value', ascending=False)

def compute_product_summary(df):
    if 'Product' not in df.columns: return pd.DataFrame()
    return df.groupby('Product').agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum',
        'Actual_Margin': 'sum'
    }).reset_index().sort_values('Calculated_Billable_Value', ascending=False)

def compute_job_summary(df):
    """Job level roll-up."""
    cols = ['Job_No', 'Client', 'Job_Name']
    # Filter valid cols
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols: return pd.DataFrame()
    
    g = df.groupby(valid_cols).agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum',
        'Calculated_Base_Cost': 'sum',
        'Actual_Time': 'sum',
        'Margin_Variance': 'sum'
    }).reset_index()
    
    # Add KPIs
    g['ROI'] = np.where(g['Calculated_Billable_Value'] > 0, g['Quoted_Amount'] / g['Calculated_Billable_Value'], 0)
    g = g.sort_values('Margin_Variance', ascending=True) # Worst performers first
    return g

def compute_task_summary(df, job_no):
    """Task level detail for a specific job."""
    d = df[df['Job_No'] == job_no].copy()
    return d.groupby('Name').agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum',
        'Actual_Time': 'sum',
        'Billable_Rate': 'mean',
        'Margin_Variance': 'sum'
    }).reset_index().sort_values('Margin_Variance')

def compute_monthly_by_department(df):
    if 'Department' not in df.columns: return pd.DataFrame()
    return df.groupby(['Month', 'Department'])['Calculated_Billable_Value'].sum().reset_index()

def compute_monthly_by_product(df):
    if 'Product' not in df.columns: return pd.DataFrame()
    return df.groupby(['Month', 'Product'])['Calculated_Billable_Value'].sum().reset_index()

# =============================================================================
# INSIGHTS & ANALYTICS
# =============================================================================

def get_top_overruns(job_summary_df, n=5, metric='Margin_Variance'):
    """Returns bottom n jobs by variance (negative variance = overrun)."""
    return job_summary_df.sort_values(metric, ascending=True).head(n)

def get_loss_making_jobs(job_summary_df):
    """Jobs where Actual Margin is Negative."""
    g = job_summary_df.copy()
    g['Actual_Margin'] = g['Calculated_Billable_Value'] - g['Calculated_Base_Cost']
    return g[g['Actual_Margin'] < 0].sort_values('Actual_Margin')

def get_unquoted_tasks(df):
    """Tasks with Actual Time > 0 but Quoted Amount = 0."""
    return df[(df['Actual_Time'] > 0) & (df['Quoted_Amount'] == 0)]

def get_margin_erosion_jobs(job_summary_df):
    """Jobs where Quoted Margin was positive but Actual Margin is significantly lower."""
    # Simplified proxy
    return job_summary_df[job_summary_df['Margin_Variance'] < -1000]

def analyze_overrun_causes(df):
    """Rough bucket analysis of where hours are going."""
    causes = {
        'unbilled_work': {'unbilled_hours': 0},
        'rate_variance': 0
    }
    # Logic placeholders
    unquoted = df[(df['Actual_Time'] > 0) & (df['Quoted_Amount'] == 0)]
    causes['unbilled_work']['unbilled_hours'] = unquoted['Actual_Time'].sum()
    return causes

def generate_insights(df, metrics):
    """Generates text narratives."""
    insights = []
    
    # 1. Variance Insight
    var = metrics['Net_Variance']
    if var < 0:
        insights.append(f"⚠️ **Margin Erosion:** Net variance is negative (${var:,.0f}). Actual work value exceeds quoted revenue, indicating scope creep or under-quoting.")
    else:
        insights.append(f"✅ **Positive Variance:** Net variance is positive (${var:,.0f}). Delivery is efficient relative to quotes.")

    # 2. Utilization / Yield
    if metrics['Total_Billable'] > 0:
        yield_pct = (metrics['Total_Quoted'] / metrics['Total_Billable']) * 100
        if yield_pct < 90:
            insights.append(f"📉 **Low Yield ({yield_pct:.1f}%):** We are quoting significantly less than the internal value of work delivered.")
        elif yield_pct > 110:
             insights.append(f"💰 **High Yield ({yield_pct:.1f}%):** We are quoting higher than the internal cost of delivery.")

    return insights

# =============================================================================
# RECONCILIATION
# =============================================================================

def compute_reconciliation_totals(raw_df, final_df):
    """
    Compares raw input rows to final filtered rows to track data loss/filtering.
    """
    return {
        "raw_records": len(raw_df),
        "final_records": len(final_df),
        "excluded_sg_allocation": 0, # Placeholder
        "excluded_non_billable": 0,  # Placeholder
        "excluded_other_fy": 0,      # Placeholder
        "excluded_other_dept": 0,    # Placeholder
        "totals": {
            "Raw Quoted Sum": raw_df['Quoted_Amount'].sum() if 'Quoted_Amount' in raw_df else 0,
            "Final Quoted Sum": final_df['Quoted_Amount'].sum()
        }
    }