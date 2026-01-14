# Job Profitability Analysis Dashboard

A structured Streamlit dashboard for analyzing job profitability with **full metric traceability and reconciliation**.

## Key Features

- **Clear Metric Definitions**: Every metric has documented formula and source fields
- **Calculated Values**: Billable Value and Cost computed from rates × hours (not from data fields)
- **Reconciliation Panel**: Verify filtered data matches source totals
- **Configurable Filters**: Toggle inclusion/exclusion of allocation entries and non-billable tasks
- **Hierarchical Drill-Down**: Category → Job → Task

---

## Metric Definitions

| Metric | Formula | Source Fields |
|--------|---------|---------------|
| **Quoted Hours** | Direct | `[Job Task] Quoted Time` |
| **Quoted Amount** | Direct | `[Job Task] Quoted Amount` |
| **Actual Hours** | Direct | `[Job Task] Actual Time (totalled)` |
| **Billable Value** | Calculated | `Actual Hours × [Task] Billable Rate` |
| **Cost (T&M)** | Calculated | `Actual Hours × [Task] Base Rate` |
| **Profit** | Calculated | `Billable Value - Cost` |
| **Margin %** | Calculated | `(Profit / Billable Value) × 100` |
| **Quoted Margin %** | Calculated | `(Quoted Amount - Cost) / Quoted Amount × 100` |
| **Margin Erosion** | Calculated | `Quoted Margin % - Actual Margin %` |
| **Hours Variance** | Calculated | `Actual Hours - Quoted Hours` |
| **Unbilled Hours** | Calculated | `Actual Hours - Invoiced Hours` |

### Why Calculated Values?

- **Billable Value** uses `[Task] Billable Rate` × hours to show what *should* be billed at standard rates
- **Cost (T&M)** uses `[Task] Base Rate` × hours for true internal labor cost
- This ensures consistency and traceability vs using pre-aggregated data fields

---

## Filtering Logic

### Toggle Options (in sidebar)

| Filter | Default | Description |
|--------|---------|-------------|
| **Exclude SG Allocation** | ✓ ON | Removes "Social Garden Invoice Allocation" entries |
| **Billable Tasks Only** | ✓ ON | Keeps only tasks where `Base Rate > 0` AND `Billable Rate > 0` |
| **Fiscal Year** | All | Filter by `[Job] Start Date` (Australian FY: Jul-Jun) |

### Billable Task Definition

A task is considered "billable" for analysis if:
```
[Task] Base Rate > 0  AND  [Task] Billable Rate > 0
```

Tasks without proper rate assignments (rate = 0 or #N/A) are excluded by default.

---

## Reconciliation & Traceability

The dashboard includes a **Data Reconciliation** panel showing:

### Filter Summary
- Raw records count
- Records after each filter
- Exclusion breakdown by filter type

### Validation Totals
| Metric | What It Shows |
|--------|---------------|
| Sum of Quoted Hours | `SUM([Job Task] Quoted Time)` |
| Sum of Actual Hours | `SUM([Job Task] Actual Time (totalled))` |
| Sum of Quoted Amount | `SUM([Job Task] Quoted Amount)` |
| Sum of Billable Value | `SUM(Actual Hours × Billable Rate)` — calculated |
| Sum of Cost T&M | `SUM(Actual Hours × Base Rate)` — calculated |

### Field vs Calculated Comparison
Shows difference between:
- `[Job Task] Billable Amount` field (from data)
- Calculated Billable Value (Hours × Rate)

This helps identify discrepancies.

---

## Repository Structure

```
job-profitability-analysis/
├── data/
│   └── [Your Excel file]
├── notebooks/
│   └── exploration.ipynb
├── app.py                 # Streamlit dashboard
├── analysis.py            # Data processing with metric docs
├── requirements.txt
└── README.md
```

---

## Required Data Columns

### Job-Level
| Column | Used For |
|--------|----------|
| `[Job] Job No.` | Job identifier |
| `[Job] Name` | Job name |
| `[Job] Category` | Level 1 grouping |
| `[Job] Client` | Client name |
| `[Job] Start Date` | Fiscal year determination |

### Task-Level
| Column | Used For |
|--------|----------|
| `[Job Task] Name` | Task identifier |
| `[Job Task] Quoted Time` | Estimated hours |
| `[Job Task] Quoted Amount` | Estimated revenue |
| `[Job Task] Actual Time (totalled)` | Logged hours |
| `[Job Task] Invoiced Time` | Billed hours |
| `[Job Task] Invoiced Amount` | Billed amount |
| `Task Category` | Task classification |

### Rate Fields (Critical)
| Column | Used For |
|--------|----------|
| `[Task] Base Rate` | Internal cost rate ($/hr) |
| `[Task] Billable Rate` | Client billing rate ($/hr) |

---

## Installation

```bash
git clone https://github.com/yourusername/job-profitability-analysis.git
cd job-profitability-analysis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

mkdir -p data
# Add your Excel file to data/

streamlit run app.py
```

---

## Analysis Workflow

1. **Load Data** — Upload or place Excel in `data/`
2. **Configure Filters** — Set FY, toggle allocation/billable filters
3. **Verify Reconciliation** — Check totals match your source
4. **Analyze Categories** — Find problem areas
5. **Drill into Jobs** — Identify specific problem projects
6. **Review Tasks** — Find root cause (scope creep, underestimation)
7. **Synthesis** — Review overall patterns

---

## Fiscal Year Logic

Australian convention:
- **FY26** = July 1, 2025 → June 30, 2026
- Jobs assigned based on `[Job] Start Date`
- Month ≥ 7 → next year's FY

---

## Common Issues Detected

| Issue | Signal | Root Cause |
|-------|--------|------------|
| **Scope Creep** | Quoted Hrs = 0, Actual > 0 | Work not in original quote |
| **Underestimation** | Actual Hrs >> Quoted Hrs | Poor estimation |
| **Margin Erosion** | Actual Margin << Quoted | Overruns without additional billing |
| **Unbilled Work** | Actual Hrs > Invoiced Hrs | Billing gaps |

