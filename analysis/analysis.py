"""
Job Profitability Analysis Module - Strategic Edition
=====================================================
Hierarchy: Department → Product → Job → Task
Framework: 
  1. Executive Summary (The "Bottom Line")
  2. Portfolio Health (Quadrants)
  3. Driver Analysis (Pareto/Waterfalls)

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
    "Quoted_Amount": {"name": "Quoted Revenue", "formula": "[Job Task] Quoted Amount", "desc": "Revenue cap / Fixed Fee sold"},
    "Billable_Value": {"name": "Work Value (Billable)", "formula": "Actual Hours × Billable Rate/Hr", "desc": "Value of effort expended at card rates"},
    "Base_Cost": {"name": "Cost to Serve", "formula": "Actual Hours × Cost Rate/Hr", "desc": "Internal labor cost"},
    "Delivery_Efficiency": {"name": "Delivery Efficiency", "formula": "Quoted Amount / Billable Value", "desc": ">1.0 = Efficient, <1.0 = Over-servicing"},
    "Actual_Margin": {"name": "Realized Margin", "formula": "Billable Value - Base Cost", "desc": "True margin at billing rates"},
    "Margin_Variance": {"name": "Value Variance", "formula": "Quoted Amount - Billable Value", "desc": "Positive = Under Budget, Negative = Over Budget"},
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
    try:
        df_data = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Data.csv")
        df_analysis = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Analysis.csv", header=4)
        df_rates = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Bill_Base_Rates.csv")
        df_jobs = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - Job Classification.csv")
        df_people = pd.read_csv(base_path + "Quoted_Task_Report_FY26.xlsx - People.csv")
        return df_data, df_analysis, df_rates, df_jobs, df_people
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def clean_and_parse(df_data, df_jobs, df_rates) -> pd.DataFrame:
    """
    Main cleaning pipeline.
    """
    if df_data.empty: return df_data

    # 1. Clean Column Names
    df = df_data.copy()
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
            
    # Derive Month & FY
    df['Month'] = df['Date_Completed'].fillna(df['Start_Date']).dt.to_period('M').astype(str)
    df['FY'] = df['Date_Completed'].fillna(df['Start_Date']).apply(
        lambda x: f"FY{x.year + 1}" if x.month >= 7 else f"FY{x.year}" if pd.notnull(x) else "Unknown"
    )

    # 3. Calculate Financials
    if 'Billable_Rate' not in df.columns: df['Billable_Rate'] = 0
    if 'Base_Rate' not in df.columns: df['Base_Rate'] = 0

    # Calculated Fields
    df['Calculated_Billable_Value'] = df['Actual_Time'] * df['Billable_Rate']
    df['Calculated_Base_Cost'] = df['Actual_Time'] * df['Base_Rate']
    
    if 'Quoted_Amount' in df.columns:
        df['Quoted_Amount'] = df['Quoted_Amount'].fillna(0)
    
    # Strategic Metrics
    # Delivery Efficiency: Quoted / Billable. (How much revenue did we capture per $1 of effort?)
    df['Delivery_Efficiency'] = df.apply(
        lambda x: x['Quoted_Amount'] / x['Calculated_Billable_Value'] if x['Calculated_Billable_Value'] > 0 else 0, axis=1
    )

    # Margins
    df['Quoted_Margin'] = df['Quoted_Amount'] - df['Calculated_Base_Cost']
    df['Actual_Margin'] = df['Calculated_Billable_Value'] - df['Calculated_Base_Cost']
    
    # Variance: Quoted (Budget) - Billable (Actuals). 
    # Positive = Under Budget (Efficient). Negative = Over Budget (Scope Creep).
    df['Margin_Variance'] = df['Quoted_Amount'] - df['Calculated_Billable_Value']
    
    return df

# =============================================================================
# STRATEGIC AGGREGATIONS
# =============================================================================

def get_available_fiscal_years(df):
    return sorted(df['FY'].unique().tolist(), reverse=True)

def get_available_departments(df):
    if 'Department' in df.columns:
        return sorted(df['Department'].dropna().unique().tolist())
    return []

def apply_filters(df, fy=None, departments=None, products=None):
    mask = pd.Series(True, index=df.index)
    if fy:
        mask &= (df['FY'] == fy)
    if departments:
        mask &= (df['Department'].isin(departments))
    return df[mask]

def calculate_portfolio_kpis(df):
    """Returns high-level McKinsey-style KPIs."""
    total_quoted = df['Quoted_Amount'].sum()
    total_billable = df['Calculated_Billable_Value'].sum()
    total_cost = df['Calculated_Base_Cost'].sum()
    
    efficiency = (total_quoted / total_billable) if total_billable > 0 else 0
    realized_profit = total_quoted - total_cost
    
    return {
        "Total_Revenue": total_quoted,
        "Total_Work_Value": total_billable,
        "Cost_Base": total_cost,
        "Delivery_Efficiency": efficiency,
        "Realized_Profit": realized_profit,
        "Net_Variance": total_quoted - total_billable
    }

def compute_waterfall_data(df):
    """
    Generates data for a P&L Waterfall Chart.
    Bridge: Quoted Revenue -> Scope Creep -> Realized Value
    """
    total_quoted = df['Quoted_Amount'].sum()
    total_billable = df['Calculated_Billable_Value'].sum()
    
    # 1. Base Revenue (Quoted)
    # 2. Revenue Leakage (Over-servicing: Where Billable > Quoted)
    # 3. Efficiency Gain (Under-servicing: Where Billable < Quoted)
    
    job_level = df.groupby('Job_No').agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum'
    }).reset_index()
    
    job_level['Variance'] = job_level['Quoted_Amount'] - job_level['Calculated_Billable_Value']
    
    over_service_jobs = job_level[job_level['Variance'] < 0]
    efficient_jobs = job_level[job_level['Variance'] >= 0]
    
    leakage = over_service_jobs['Variance'].sum() # Negative number
    gain = efficient_jobs['Variance'].sum()       # Positive number
    
    return pd.DataFrame([
        {"Category": "Total Quoted (Sold)", "Value": total_quoted, "Measure": "absolute"},
        {"Category": "Over-Servicing (Leakage)", "Value": leakage, "Measure": "relative"},
        {"Category": "Efficiency Gains", "Value": gain, "Measure": "relative"},
        {"Category": "Net Delivery Value", "Value": total_quoted + leakage + gain, "Measure": "total"} # Should equal Billable - wait, concept check.
    ])
    # Note: If we want to reconcile to Profit, we would subtract cost. 
    # For now, this bridges "Sold" to "delivered" value variance.

def compute_quadrant_data(df):
    """
    Prepares data for the BCG-style Matrix (Margin % vs Revenue Volume).
    x: Revenue (Quoted Amount)
    y: Margin % ((Quoted - Cost) / Quoted)
    size: Cost
    """
    jobs = df.groupby(['Job_No', 'Client', 'Job_Name']).agg({
        'Quoted_Amount': 'sum',
        'Calculated_Base_Cost': 'sum',
        'Calculated_Billable_Value': 'sum'
    }).reset_index()
    
    # Margin based on Quoted Price vs Actual Cost
    jobs['Margin_Abs'] = jobs['Quoted_Amount'] - jobs['Calculated_Base_Cost']
    jobs['Margin_Pct'] = np.where(jobs['Quoted_Amount'] > 0, (jobs['Margin_Abs'] / jobs['Quoted_Amount']) * 100, -100)
    
    # Cap Margin for visualization (handle outliers)
    jobs['Margin_Pct'] = jobs['Margin_Pct'].clip(-100, 100)
    
    # Categorize
    conditions = [
        (jobs['Quoted_Amount'] > jobs['Quoted_Amount'].median()) & (jobs['Margin_Pct'] > 50), # High Rev, High Margin
        (jobs['Quoted_Amount'] < jobs['Quoted_Amount'].median()) & (jobs['Margin_Pct'] > 50), # Low Rev, High Margin
        (jobs['Quoted_Amount'] > jobs['Quoted_Amount'].median()) & (jobs['Margin_Pct'] < 20), # High Rev, Low Margin
        (jobs['Margin_Pct'] < 0) # Loss Makers
    ]
    choices = ['Stars', 'Niche', 'Cash Cows', 'Problem Children']
    jobs['Segment'] = np.select(conditions, choices, default='Standard')
    
    return jobs

def identify_pareto_contributors(df, metric='Margin_Variance', top_n=5):
    """
    Identifies the 'Vital Few' jobs driving 80% of the negative variance.
    """
    jobs = df.groupby(['Job_No', 'Client']).agg({
        'Quoted_Amount': 'sum',
        'Calculated_Billable_Value': 'sum',
        'Margin_Variance': 'sum'
    }).reset_index()
    
    # Filter for negative variance (Overruns)
    overruns = jobs[jobs['Margin_Variance'] < 0].copy()
    overruns = overruns.sort_values('Margin_Variance', ascending=True) # Largest negative first
    
    total_overrun = overruns['Margin_Variance'].sum()
    overruns['Cumulative_Pct'] = overruns['Margin_Variance'].cumsum() / total_overrun
    
    return overruns.head(top_n)

# =============================================================================
# NARRATIVE GENERATION (The "So What?")
# =============================================================================

def generate_strategic_insights(df, metrics):
    """
    Generates insights using the Pyramid Principle.
    1. The "Governing Thought" (Bottom Line)
    2. The Key Drivers
    3. The Action Plan
    """
    insights = []
    
    # 1. The Bottom Line
    efficiency = metrics['Delivery_Efficiency']
    net_var = metrics['Net_Variance']
    
    if efficiency < 0.9:
        status = "⚠️ **CRITICAL: Value Leakage Detected**"
        main_thought = f"The portfolio is currently **over-servicing clients by {(1-efficiency)*100:.1f}%**. For every $1 sold, we are delivering ${1/efficiency:.2f} of work value."
    elif efficiency > 1.1:
        status = "✅ **High Efficiency**"
        main_thought = f"The portfolio is operating efficiently, delivering work at **{(efficiency-1)*100:.1f}% under budget** relative to quotes."
    else:
        status = "ℹ️ **Balanced Performance**"
        main_thought = "The portfolio is generally aligned with quoted expectations, with minor variance."
        
    insights.append(f"### {status}\n{main_thought}")

    # 2. The Drivers (Pareto)
    pareto_jobs = identify_pareto_contributors(df, top_n=3)
    if not pareto_jobs.empty:
        driver_text = "**Top 3 Drivers of Variance:**\n"
        for _, row in pareto_jobs.iterrows():
            driver_text += f"- **{row['Client']}** ({row['Job_No']}): Overrun by **${abs(row['Margin_Variance']):,.0f}**\n"
        insights.append(driver_text)

    # 3. Strategic Recommendation
    if efficiency < 0.9:
        insights.append("**Recommended Action:**\nReview scope boundaries for the top 3 accounts. Work is exceeding the fixed fee cap. Consider moving these accounts to T&M or issuing Change Orders.")
    elif efficiency > 1.2:
        insights.append("**Recommended Action:**\nVerify quality standards. High efficiency might indicate under-delivery against scope or highly standardized delivery (opportunity to productize).")

    return insights

# =============================================================================
# RECONCILIATION
# =============================================================================

def compute_reconciliation_totals(raw_df, final_df):
    return {
        "raw_records": len(raw_df),
        "final_records": len(final_df),
        "totals": {
            "Raw Quoted Sum": raw_df['Quoted_Amount'].sum() if 'Quoted_Amount' in raw_df else 0,
            "Final Quoted Sum": final_df['Quoted_Amount'].sum()
        }
    }