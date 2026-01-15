import pandas as pd
import numpy as np

def load_and_process_data(file_source):
    """
    Reads a single Excel file with 3 specific tabs, standardizes them,
    and performs the Revenue Allocation Join.
    
    Returns:
        df_master (pd.DataFrame): Daily transaction table with allocated revenue.
    """
    # 1. Load the Excel File
    try:
        xls = pd.ExcelFile(file_source)
    except Exception as e:
        raise ValueError(f"Invalid Excel file: {e}")

    # Validate Sheets
    required_sheets = ['Monthly Revenue', 'Timesheet Data', 'Quotation Data']
    missing_sheets = [s for s in required_sheets if s not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {', '.join(missing_sheets)}")

    # Read Dataframes
    df_rev = pd.read_excel(xls, 'Monthly Revenue')
    df_time = pd.read_excel(xls, 'Timesheet Data')
    df_quote = pd.read_excel(xls, 'Quotation Data')

    # --- 2. STANDARDIZATION ---
    
    # Timesheet: Clean & Rename
    # Target Structure: Job_Number, Date, Hours, Task_Name, Cost_Rate, Billable_Rate
    df_time = df_time.rename(columns={
        '[Job] Job No.': 'Job_Number',
        '[Time] Date': 'Date',
        '[Time] Time': 'Hours',
        '[Job Task] Name': 'Task_Name',
        '[Task] Base Rate': 'Cost_Rate',
        '[Task] Billable Rate': 'Billable_Rate',
        '[Job] Name': 'Job_Name',
        '[Job] Client': 'Client',
        '[Job] Client Manager': 'Client_Manager',
        'Department': 'Department', 
        'Product': 'Product'
    })
    
    # Ensure Date format
    df_time['Date'] = pd.to_datetime(df_time['Date'])
    
    # Create MonthKey for joining (1st of the month)
    df_time['Month_Key'] = df_time['Date'].values.astype('datetime64[M]')
    
    # Ensure Numeric Rates
    for col in ['Cost_Rate', 'Billable_Rate', 'Hours']:
        df_time[col] = pd.to_numeric(df_time[col], errors='coerce').fillna(0)

    # Revenue: Clean & Rename
    df_rev = df_rev.rename(columns={
        'Job Number': 'Job_Number',
        'Month': 'Revenue_Month',
        'Amount': 'Revenue_Amount'
    })
    df_rev['Revenue_Month'] = pd.to_datetime(df_rev['Revenue_Month'])
    df_rev['Month_Key'] = df_rev['Revenue_Month'].values.astype('datetime64[M]')
    df_rev['Revenue_Amount'] = pd.to_numeric(df_rev['Revenue_Amount'], errors='coerce').fillna(0)

    # Quotation: Clean & Rename
    df_quote = df_quote.rename(columns={
        '[Job] Job No.': 'Job_Number',
        '[Job Task] Name': 'Task_Name',
        '[Job Task] Quoted Amount': 'Quoted_Amount',
        '[Job Task] Quoted Time': 'Quoted_Hours'
    })
    df_quote['Quoted_Amount'] = pd.to_numeric(df_quote['Quoted_Amount'], errors='coerce').fillna(0)
    df_quote['Quoted_Hours'] = pd.to_numeric(df_quote['Quoted_Hours'], errors='coerce').fillna(0)

    # --- 3. REVENUE ALLOCATION LOGIC ---

    # A. Calculate "Driver Weights" (Task Hours / Total Job-Month Hours)
    # Group by Job & Month to get total effort
    monthly_job_hours = df_time.groupby(['Job_Number', 'Month_Key'])['Hours'].sum().reset_index()
    monthly_job_hours.rename(columns={'Hours': 'Total_Job_Month_Hours'}, inplace=True)

    # Join totals back to the daily timesheet lines
    df_master = pd.merge(df_time, monthly_job_hours, on=['Job_Number', 'Month_Key'], how='left')

    # Calculate Weight (Protect against division by zero)
    # If Total Hours is 0 (unlikely if rows exist), weight is 0.
    df_master['Revenue_Weight'] = np.where(
        df_master['Total_Job_Month_Hours'] > 0,
        df_master['Hours'] / df_master['Total_Job_Month_Hours'],
        0
    )

    # B. Map Revenue (Join Monthly Revenue to Daily Tasks)
    # Left join ensures we keep timesheet data even if no revenue was recognized that month
    df_master = pd.merge(df_master, 
                         df_rev[['Job_Number', 'Month_Key', 'Revenue_Amount']], 
                         on=['Job_Number', 'Month_Key'], 
                         how='left')

    # Fill NaN revenue with 0 (work done, but no revenue recognized)
    df_master['Revenue_Amount'] = df_master['Revenue_Amount'].fillna(0)

    # C. Calculate Allocated Revenue
    # This is the "True Revenue" for that specific timesheet entry
    df_master['Allocated_Revenue'] = df_master['Revenue_Amount'] * df_master['Revenue_Weight']

    # --- 4. RECONCILIATION MERGE (QUOTATION) ---

    # Join Quotation Data (Budget)
    # Left join on Job + Task Name
    # Note: This creates repeated data for Quoted Amount since one task has multiple time entries.
    # Analysis logic must handle this deduplication.
    df_master = pd.merge(df_master, 
                         df_quote[['Job_Number', 'Task_Name', 'Quoted_Amount', 'Quoted_Hours']], 
                         on=['Job_Number', 'Task_Name'], 
                         how='left')
    
    # Fill NaN quotes with 0 (Unquoted / Scope Creep)
    df_master['Quoted_Amount'] = df_master['Quoted_Amount'].fillna(0)
    df_master['Quoted_Hours'] = df_master['Quoted_Hours'].fillna(0)

    # --- 5. FINAL METRIC CALCS ---
    
    # Base Cost = Actual Hours * Cost Rate
    df_master['Base_Cost'] = df_master['Hours'] * df_master['Cost_Rate']
    
    # Margin = Allocated Revenue - Base Cost
    df_master['Margin'] = df_master['Allocated_Revenue'] - df_master['Base_Cost']

    # Billable Value (Benchmark) = Actual Hours * Billable Rate
    df_master['Billable_Value'] = df_master['Hours'] * df_master['Billable_Rate']
    
    # Add Fiscal Year Helper
    # AU FY: Jul-Jun. If month >= 7, FY = Year + 1
    df_master['Fiscal_Year'] = np.where(
        df_master['Date'].dt.month >= 7, 
        df_master['Date'].dt.year + 1, 
        df_master['Date'].dt.year
    )

    return df_master