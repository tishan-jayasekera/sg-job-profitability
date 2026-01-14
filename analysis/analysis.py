"""
Strategic Job Profitability Module
==================================
Focus: Portfolio Diagnostics & Strategic Insights
Frameworks: Pareto (80/20), Value Bridge, Portfolio Matrix

CORE METRIC DEFINITIONS:
1.  Delivery Efficiency (DE) = Quoted Revenue / Billable Value
    - DE < 1.0: Value Leakage (Over-servicing)
    - DE > 1.0: Value Capture (Efficient Delivery)
    - DE = 1.0: Parity (T&M Equivalent)

2.  Revenue Leakage = Abs(Variance) where Billable > Quoted
3.  Realized Profit = Quoted Revenue - Base Cost
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List

# =============================================================================
# 1. DATA INGESTION & CLEANING
# =============================================================================

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw CSVs from the data directory."""
    base = "data/"
    try:
        # Load core files
        df_data = pd.read_csv(base + "Quoted_Task_Report_FY26.xlsx - Data.csv")
        df_rates = pd.read_csv(base + "Quoted_Task_Report_FY26.xlsx - Bill_Base_Rates.csv")
        
        # Load Job Classification (optional but recommended)
        try:
            df_jobs = pd.read_csv(base + "Quoted_Task_Report_FY26.xlsx - Job Classification.csv")
        except FileNotFoundError:
            df_jobs = pd.DataFrame()
            
        return df_data, df_rates, df_jobs
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def clean_and_prep_data(df_data, df_rates) -> pd.DataFrame:
    """
    Prepares the 'Master Table' with all calculated financial metrics.
    """
    if df_data.empty: return df_data
    
    df = df_data.copy()
    
    # Standardize Columns
    df.columns = [
        c.replace('[Job]', '').replace('[Job Task]', '').replace('[Task]', '')
        .replace('.', '').strip().replace(' ', '_') 
        for c in df.columns
    ]
    
    # Parse Numerics
    numeric_cols = ['Quoted_Amount', 'Actual_Time', 'Billable_Rate', 'Base_Rate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            
    # Parse Dates for FY/Month Logic
    if 'Date_Completed' in df.columns:
        df['Date_Obj'] = pd.to_datetime(df['Date_Completed'], errors='coerce')
        # Fallback to Start Date if Date Completed is missing
        if 'Start_Date' in df.columns:
            df['Date_Obj'] = df['Date_Obj'].fillna(pd.to_datetime(df['Start_Date'], errors='coerce'))
            
        # FY Calculation (Australia: July-June)
        df['FY'] = df['Date_Obj'].apply(
            lambda x: f"FY{x.year + 1}" if pd.notnull(x) and x.month >= 7 
            else (f"FY{x.year}" if pd.notnull(x) else "Unknown")
        )

    # --- FINANCIAL CALCULATIONS ---
    
    # 1. Ensure Rates Exist
    if 'Billable_Rate' not in df.columns: df['Billable_Rate'] = 0
    if 'Base_Rate' not in df.columns: df['Base_Rate'] = 0

    # 2. Compute Values
    df['Billable_Value'] = df['Actual_Time'] * df['Billable_Rate']  # The "Work Value"
    df['Base_Cost'] = df['Actual_Time'] * df['Base_Rate']           # The "Cost to Serve"
    
    # 3. Handle Quoted Amount
    if 'Quoted_Amount' in df.columns:
        df['Quoted_Amount'] = df['Quoted_Amount'].fillna(0)
    else:
        df['Quoted_Amount'] = 0

    # 4. Variance (Quoted - Billable Value)
    # Positive = Efficiency Gain (Under Budget)
    # Negative = Leakage (Over Budget)
    df['Variance'] = df['Quoted_Amount'] - df['Billable_Value']
    
    return df

# =============================================================================
# 2. STRATEGIC AGGREGATIONS
# =============================================================================

def get_portfolio_kpis(df):
    """High-level executive metrics."""
    quoted = df['Quoted_Amount'].sum()
    billable = df['Billable_Value'].sum()
    cost = df['Base_Cost'].sum()
    
    # Delivery Efficiency: How much revenue do we capture per $1 of work value?
    efficiency = quoted / billable if billable > 0 else 0
    
    return {
        "Revenue_Sold": quoted,
        "Work_Value_Delivered": billable,
        "Cost_Base": cost,
        "Realized_Profit": quoted - cost,
        "Delivery_Efficiency": efficiency,
        "Net_Variance": quoted - billable
    }

def get_waterfall_bridge(df):
    """
    Constructs the data for a P&L Waterfall Chart.
    Walks from 'Sold Revenue' to 'Net Value Delivered' by splitting Variance.
    """
    # Group by Job to isolate over/under performance at the job level
    job_view = df.groupby('Job_No').agg({
        'Quoted_Amount': 'sum',
        'Billable_Value': 'sum'
    }).reset_index()
    
    job_view['Job_Variance'] = job_view['Quoted_Amount'] - job_view['Billable_Value']
    
    # Split Variance
    leakage = job_view[job_view['Job_Variance'] < 0]['Job_Variance'].sum() # Negative Number
    gains = job_view[job_view['Job_Variance'] >= 0]['Job_Variance'].sum()  # Positive Number
    
    total_revenue = df['Quoted_Amount'].sum()
    
    return pd.DataFrame([
        {"Category": "Revenue Sold (Quoted)", "Value": total_revenue, "Type": "absolute"},
        {"Category": "Value Leakage (Over-Servicing)", "Value": leakage, "Type": "relative"},
        {"Category": "Efficiency Gains", "Value": gains, "Type": "relative"},
        {"Category": "Net Value Delivered", "Value": total_revenue + leakage + gains, "Type": "total"} 
        # Note: Net Value Delivered theoretically equals Total Billable Value
    ])

def get_matrix_quadrants(df):
    """
    Prepares data for Growth/Share Matrix.
    X-Axis: Revenue Volume (Strategic Importance)
    Y-Axis: Delivery Efficiency (Profitability Proxy)
    Size: Cost (Resource Drain)
    """
    jobs = df.groupby(['Job_No', 'Client', 'Job_Name']).agg({
        'Quoted_Amount': 'sum',
        'Billable_Value': 'sum',
        'Base_Cost': 'sum'
    }).reset_index()
    
    # Avoid div by zero
    jobs['Efficiency'] = jobs.apply(lambda x: x['Quoted_Amount'] / x['Billable_Value'] if x['Billable_Value'] > 0 else 0, axis=1)
    
    # Segmentation Logic
    median_rev = jobs[jobs['Quoted_Amount'] > 0]['Quoted_Amount'].median()
    
    def classify(row):
        if row['Quoted_Amount'] == 0: return "Unbilled / Speculative"
        if row['Quoted_Amount'] >= median_rev:
            return "Star" if row['Efficiency'] >= 1.0 else "Cash Cow (At Risk)"
        else:
            return "Niche" if row['Efficiency'] >= 1.0 else "Problem Child"
            
    jobs['Segment'] = jobs.apply(classify, axis=1)
    return jobs

def get_pareto_variance(df, top_n=5):
    """
    Identifies the 'Critical Few' jobs driving negative variance.
    """
    jobs = df.groupby(['Job_No', 'Client', 'Job_Name'])['Variance'].sum().reset_index()
    
    # Filter for negative variance only (The "Problem" jobs)
    negative_jobs = jobs[jobs['Variance'] < 0].sort_values('Variance', ascending=True)
    
    return negative_jobs.head(top_n)

# =============================================================================
# 3. INSIGHT GENERATION (NLG)
# =============================================================================

def generate_bluf_insights(kpis, pareto_df):
    """
    Generates 'Bottom Line Up Front' text.
    """
    eff = kpis['Delivery_Efficiency']
    insights = []
    
    # 1. The Headline
    if eff < 0.9:
        insights.append(f"**CRITICAL:** Portfolio is leaking value. For every $1.00 sold, we are delivering **${1/eff:.2f}** of work.")
    elif eff > 1.1:
        insights.append(f"**STRONG:** High efficiency detected. We are delivering work at **{(1/eff)*100:.0f}%** of its billable value.")
    else:
        insights.append(f"**STABLE:** Portfolio is operating near parity (1.0x efficiency).")
        
    # 2. The Driver
    if not pareto_df.empty:
        top_loss = abs(pareto_df.iloc[0]['Variance'])
        client = pareto_df.iloc[0]['Client']
        insights.append(f"**Primary Drag:** The single largest contributor to variance is **{client}**, accounting for **${top_loss:,.0f}** in leakage.")
        
    return insights

# =============================================================================
# 4. UTILS
# =============================================================================

def get_filters(df):
    fys = sorted(df['FY'].unique().tolist(), reverse=True) if 'FY' in df.columns else []
    depts = sorted(df['Department'].dropna().unique().tolist()) if 'Department' in df.columns else []
    return fys, depts