# Job Profitability Analysis Dashboard

An interactive Streamlit dashboard for analyzing job profitability from quote to execution. Identify margin erosion, scope creep, and problematic projects by comparing quoted estimates against actual performance.

## Features

- **Fiscal Year Filtering**: Analyze any period using `[Job] Start Date` (Australian FY: Jul-Jun)
- **Hierarchical Analysis**: Drill down from Category → Job → Task
- **Multi-Dimensional Filtering**: Filter by FY, Category, Client, and job status
- **Key Metrics**: Quoted vs Actual hours, revenue, cost, and margin calculations
- **Scope Creep Detection**: Flag unquoted tasks that weren't in the original estimate
- **Client Analysis**: View profitability by client to identify problematic accounts
- **Visual Insights**: Interactive charts comparing margins and overrun rates

## Repository Structure

```
job-profitability-analysis/
├── data/
│   └── Quoted_Task_Report_FY26.xlsx   # Your dataset
├── notebooks/
│   └── exploration.ipynb              # Optional: exploratory analysis
├── app.py                             # Streamlit application
├── analysis.py                        # Data processing module
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Data Requirements

The application expects an Excel file with a **"Data"** sheet containing these columns:

### Job-Level Fields
| Column | Type | Description |
|--------|------|-------------|
| `[Job] Job No.` | string | Unique job identifier |
| `[Job] Name` | string | Job/project name |
| `[Job] Category` | string | Business category |
| `[Job] Client` | string | Client name |
| `[Job] Client Manager` | string | Account manager |
| `[Job] Start Date` | date | Job start date (used for FY filtering) |
| `[Job] Status` | string | Job status |
| `[Job] Budget` | numeric | Internal budget |

### Task-Level Fields
| Column | Type | Description |
|--------|------|-------------|
| `[Job Task] Name` | string | Task name |
| `[Job Task] Quoted Time` | numeric | Estimated hours |
| `[Job Task] Quoted Amount` | numeric | Estimated revenue |
| `[Job Task] Actual Time (totalled)` | numeric | Actual hours logged |
| `[Job Task] Billable Amount` | numeric | Billable revenue |
| `[Job Task] Invoiced Amount` | numeric | Amount invoiced |
| `[Job Task] Cost` | numeric | Task labor cost |
| `[Job Task] Billable` | Yes/No | Whether task is billable |
| `Task Category` | string | Task classification |

### Rate & Cost Fields
| Column | Type | Description |
|--------|------|-------------|
| `[Task] Base Rate` | numeric | Internal hourly rate |
| `[Task] Billable Rate` | numeric | Client billing rate |
| `Time+Material (Base)` | numeric | Actual cost (Base Rate × Hours) |

**Note**: Entries where `[Job Task] Name` = "Social Garden Invoice Allocation" are automatically excluded.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/job-profitability-analysis.git
   cd job-profitability-analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your data**
   ```bash
   mkdir -p data
   # Place your Excel file in data/ folder
   ```

## Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. You can also upload data directly via the sidebar.

## Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Margin %** | (Billable - Cost) / Billable × 100 | Profit as % of revenue |
| **Hours Variance** | Actual Hours - Quoted Hours | Positive = overrun |
| **Cost Variance** | Actual Cost - Quoted Amount | Positive = over budget |
| **Overrun Rate** | Overrun Jobs / Total Jobs × 100 | Category health indicator |

## Fiscal Year Logic

The dashboard uses Australian fiscal year convention:
- **FY26** = July 1, 2025 – June 30, 2026
- Jobs are assigned to FY based on `[Job] Start Date`

## Analysis Workflow

1. **Select Period**: Choose fiscal year to focus on
2. **Category Overview**: Review margin and overrun rates by category
3. **Identify Problem Jobs**: Sort by margin or profit to find worst performers
4. **Task Drill-Down**: Select a job to see which tasks drove variances
5. **Scope Creep**: Review unquoted tasks for hidden costs

## Common Issues Identified

| Issue | Indicator | Action |
|-------|-----------|--------|
| **Scope Creep** | Unquoted tasks with actual hours | Review change management |
| **Hour Overruns** | Actual Hours >> Quoted Hours | Improve estimation |
| **Margin Erosion** | Low/negative margin % | Review pricing |
| **Unbilled Work** | Billable Amount < Cost | Check billing process |

## Deployment to Streamlit Cloud

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, set main file to `app.py`
4. Deploy

For sensitive data, keep repo private and use Streamlit secrets management.

## License

MIT License
