# Job Profitability Analysis Dashboard — Trend Edition

An interactive Streamlit dashboard for analyzing job profitability with **month-on-month trend analysis**, **narrative insights**, and **full metric traceability**.

## Key Features

### 📈 Month-on-Month Trend Analysis
- Track margin evolution across the financial year
- Compare Quoted vs Billable vs Cost trends
- Visualize department performance over time
- Revenue realization tracking

### 💡 Narrative-Driven Insights
- Automated headline generation
- Margin driver identification
- Quoting accuracy analysis
- Actionable recommendations

### 🏢 Hierarchical Drill-Down
- Department → Product → Job → Task
- Full rollup of all metrics at each level
- Variance analysis at every layer

### ⚖️ Quoted vs Actual Reconciliation
- **Quoted Margin** = Quoted Amount – Base Cost
- **Actual Margin** = Billable Value – Base Cost
- **Margin Variance** = Actual Margin – Quoted Margin

---

## Metric Definitions

### Margin Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Quoted Margin** | `Quoted Amount - Base Cost` | Expected margin from quote |
| **Actual Margin** | `Billable Value - Base Cost` | Realized margin |
| **Margin Variance** | `Actual Margin - Quoted Margin` | Difference from expectations |
| **Quoted Margin %** | `(Quoted Margin / Quoted Amount) × 100` | Margin % if quoted was billed |
| **Actual Margin %** | `(Actual Margin / Billable Value) × 100` | Realized margin percentage |

### Rate Metrics (per hour)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Quoted Rate/Hr** | `Quoted Amount / Quoted Hours` | Implied hourly rate from quote |
| **Billable Rate/Hr** | `[Task] Billable Rate` | Standard client billing rate |
| **Cost Rate/Hr** | `[Task] Base Rate` | Internal T&M cost per hour |

### Value Metrics

| Metric | Formula | Source |
|--------|---------|--------|
| **Quoted Hours** | Direct | `[Job Task] Quoted Time` |
| **Quoted Amount** | Direct | `[Job Task] Quoted Amount` |
| **Actual Hours** | Direct | `[Job Task] Actual Time (totalled)` |
| **Billable Value** | Calculated | `Actual Hours × Billable Rate/Hr` |
| **Base Cost** | Calculated | `Actual Hours × Cost Rate/Hr` |

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Revenue Realization** | `(Billable Value / Quoted Amount) × 100` | % of quoted revenue realized |
| **Hours Variance** | `Actual Hours - Quoted Hours` | Hours over/under quote |
| **Margin Erosion** | `Quoted Margin % - Actual Margin %` | Margin lost vs expectation |

---

## Dashboard Tabs

### 📊 Executive Summary
- Key headlines and alerts
- Revenue & margin KPIs
- Rate analysis
- Margin bridge visualization (Quoted → Actual)

### 📈 Monthly Trends
- Selectable metric trends
- Quoted vs Billable vs Cost comparison
- Margin evolution (Quoted vs Actual)
- Department breakdown over time

### 🏢 Hierarchy Drill-Down
- Level 1: Department performance
- Level 2: Product analysis
- Level 3: Job profitability
- Level 4: Task breakdown

### 💡 Insights & Narratives
- Automated margin driver analysis
- Quoting accuracy insights
- Trend signals
- Action items
- Executive narrative summary

### 🔍 Reconciliation
- Filter summary
- Validation totals
- Metric definitions reference

---

## Hierarchy Structure

```
Department
└── Product
    └── Job
        └── Task
```

All metrics aggregate from Task → Job → Product → Department level.

---

## Time-Based Analysis

### Fiscal Year Logic (Australian)
- **FY26** = July 1, 2025 → June 30, 2026
- Jobs assigned based on `[Job] Start Date`
- Month ≥ 7 → next year's FY

### Monthly Aggregation
- Calendar month grouping (e.g., "Jul 2025")
- FY month ordering (1=Jul, 12=Jun)
- Trend analysis across selected FY

---

## Filtering Options

| Filter | Default | Description |
|--------|---------|-------------|
| **Fiscal Year** | Latest | Required for trend analysis |
| **Department** | All | Filter by business unit |
| **Exclude SG Allocation** | ✔ ON | Removes internal allocation entries |
| **Billable Tasks Only** | ✔ ON | Keeps tasks where Base Rate > 0 AND Billable Rate > 0 |

---

## Narrative Insights Generated

### Headline Alerts
- Revenue realization gaps
- Overall margin status
- Loss-making job counts

### Margin Drivers
- Worst/best performing departments
- Impact quantification

### Quoting Accuracy
- Underquoted jobs identification
- Scope creep (unquoted work) detection
- Excess hours analysis

### Action Items
- Top loss-making jobs to review
- Specific variance explanations

---

## Required Data Columns

### Job-Level
- `[Job] Job No.`, `[Job] Name`, `[Job] Client`
- `[Job] Start Date` (for FY and monthly grouping)
- `[Job] Status`, `[Job] Budget`

### Hierarchy
- `Department`, `Product`

### Task-Level
- `[Job Task] Name`, `Task Category`
- `[Job Task] Quoted Time`, `[Job Task] Quoted Amount`
- `[Job Task] Actual Time (totalled)`
- `[Job Task] Invoiced Time`, `[Job Task] Invoiced Amount`

### Rates
- `[Task] Base Rate` (Cost Rate/Hr)
- `[Task] Billable Rate` (Billable Rate/Hr)

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

1. **Select FY** — Choose fiscal year for trend analysis
2. **Review Headlines** — Check key alerts in Executive Summary
3. **Analyze Trends** — Explore monthly patterns
4. **Identify Problems** — Use hierarchy drill-down
5. **Read Insights** — Review automated narratives
6. **Take Action** — Follow recommended action items
7. **Verify Data** — Check reconciliation totals

---

## Common Issues Detected

| Issue | Signal | Root Cause |
|-------|--------|------------|
| **Scope Creep** | Unquoted tasks with actual hours | Work not in original quote |
| **Underestimation** | Actual Hrs >> Quoted Hrs | Poor estimation |
| **Margin Erosion** | Actual Margin << Quoted Margin | Revenue leakage |
| **Revenue Gap** | Realization < 100% | Discounts or billing gaps |
| **Rate Mismatch** | Billable Rate ≠ Quoted Rate/Hr | Pricing inconsistency |

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard with trend analysis |
| `analysis.py` | Data processing, monthly aggregations, insights |
| `requirements.txt` | Python dependencies |
| `README.md` | Documentation |