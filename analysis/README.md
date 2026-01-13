# Job Profitability Analysis Dashboard

A structured Streamlit dashboard for analyzing job profitability from quote to execution. Follows a hierarchical drill-down approach (Category → Job → Task) to identify margin erosion, scope creep, and problematic projects.

## Analysis Structure

The dashboard follows the structured approach outlined in the analysis plan:

### Level 0: Overall KPIs
- Total revenue, cost, profit, and margin
- Jobs over budget / at loss counts
- Hours variance summary
- Profit lost to overruns

### Level 1: Category Analysis
*"Which areas of the business are most prone to profit leaks?"*
- Margin % by category (bar chart)
- Hours variance % by category
- Category comparison table

### Level 2: Job-Level Analysis
*"Quote vs Actual Summary per job — identify problematic projects"*
- Job profitability table with sorting
- Filter by overruns, losses, margin erosion
- Margin scatter plot (Quoted vs Actual)

### Level 3: Task Drill-Down
*"Which specific task or phase caused the project overrun?"*
- Task hours chart (Quoted vs Actual)
- Task details table with flags
- Unquoted tasks (scope creep) alerts
- Unbilled hours warnings

### Level 4: Synthesis
*"Why do jobs run over?"*
- Scope creep / unquoted work analysis
- Underestimation of effort metrics
- Billing issues (unbilled hours)
- Non-billable work costs
- Top overrun jobs list
- Loss-making jobs list

## Key Features

- **Fiscal Year Filtering**: Filter by `[Job] Start Date` using Australian FY (Jul-Jun)
- **Hierarchical Navigation**: Category → Job → Task drill-down
- **Margin Erosion Tracking**: Compare quoted vs actual margins
- **Scope Creep Detection**: Flag tasks with 0 quoted hours but actual work
- **Billing Efficiency**: Track unbilled hours
- **Visual Insights**: Interactive Altair charts

## Repository Structure

```
job-profitability-analysis/
├── data/
│   └── [Your Excel file here]
├── notebooks/
│   └── exploration.ipynb
├── app.py                 # Streamlit dashboard
├── analysis.py            # Data processing module
├── requirements.txt
└── README.md
```

## Data Requirements

### Required Columns

| Column | Description |
|--------|-------------|
| `[Job] Job No.` | Unique job identifier |
| `[Job] Name` | Job/project name |
| `[Job] Category` | Business category (Level 1 grouping) |
| `[Job] Client` | Client name |
| `[Job] Client Manager` | Account manager |
| `[Job] Start Date` | Job start date (for FY filtering) |
| `[Job] Status` | Job status |
| `[Job Task] Name` | Task name (Level 3) |
| `[Job Task] Quoted Time` | Estimated hours |
| `[Job Task] Quoted Amount` | Estimated revenue |
| `[Job Task] Actual Time (totalled)` | Actual hours logged |
| `[Job Task] Billable Amount` | Billable revenue |
| `[Job Task] Invoiced Amount` | Amount invoiced |
| `[Job Task] Invoiced Time` | Hours invoiced |
| `[Job Task] Billable` | Yes/No billable flag |
| `Task Category` | Task classification |
| `[Task] Base Rate` | Internal hourly rate |
| `[Task] Billable Rate` | Client billing rate |
| `Time+Material (Base)` | Actual cost |

### Exclusions

Entries where `[Job Task] Name` = **"Social Garden Invoice Allocation"** are automatically excluded as internal allocations.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/job-profitability-analysis.git
cd job-profitability-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add data
mkdir -p data
# Place your Excel file in data/

# Run app
streamlit run app.py
```

## Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Margin %** | (Billable - Cost) / Billable × 100 | Profit as % of revenue |
| **Quoted Margin %** | (Quoted - Cost) / Quoted × 100 | Expected margin from quote |
| **Margin Erosion** | Quoted Margin % - Actual Margin % | Lost margin |
| **Hours Variance** | Actual Hours - Quoted Hours | Overrun (+) or underrun (-) |
| **Hours Variance %** | Variance / Quoted × 100 | Relative overrun |
| **Unbilled Hours** | Actual Hours - Invoiced Hours | Work not billed |

## Fiscal Year Logic

Australian fiscal year convention:
- **FY26** = July 1, 2025 – June 30, 2026
- Jobs assigned to FY based on `[Job] Start Date`
- Month ≥ 7 → next calendar year's FY

## Analysis Workflow

1. **Select Period** — Choose fiscal year from sidebar
2. **Review Categories** — Identify low-margin or high-overrun categories
3. **Drill into Category** — Select category to filter job list
4. **Identify Problem Jobs** — Sort by margin, profit, or variance
5. **Analyze Tasks** — Select job to see task breakdown
6. **Review Synthesis** — Understand root causes of overruns

## Common Issues Detected

| Issue | Data Signal | Recommendation |
|-------|-------------|----------------|
| **Scope Creep** | Quoted Hours = 0, Actual > 0 | Improve initial scoping |
| **Underestimation** | Actual Hours >> Quoted | Use historical data for quotes |
| **Billing Gaps** | Unbilled Hours > 0 | Review invoicing process |
| **Margin Erosion** | Actual Margin << Quoted | Review pricing/cost control |
| **Losses** | Profit < 0 | Investigate specific jobs |

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `app.py`

### Local
```bash
streamlit run app.py
```

## References

- Analysis plan based on WorkflowMax and WorkGuru profitability best practices
- Margin calculations follow standard project accounting (Profit / Revenue)
- Australian FY convention (July–June)

