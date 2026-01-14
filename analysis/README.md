# Job Profitability Analysis Dashboard

A Streamlit dashboard for analyzing job profitability with **correct financial logic** and **root cause analysis**.

## ⚠️ Critical Financial Definitions

Understanding these definitions is essential for correct interpretation:

| Term | Definition | Source | Role |
|------|------------|--------|------|
| **Quoted Amount** | The client quote | `[Job Task] Quoted Amount` | **THIS IS REVENUE** |
| **Invoiced Amount** | What was actually billed | `[Job Task] Invoiced Amount` | Actual revenue captured |
| **Base Cost** | Internal labor cost | `Actual Hours × Cost Rate/Hr` | True cost |
| **Billable Value** | Hours × Billable Rate | Calculated | ⚠️ **Internal only, NOT revenue** |

### Why This Matters

**Wrong approach (what many dashboards do):**
> "Revenue = Billable Value (Hours × Billable Rate)"

**Correct approach (what this dashboard does):**
> "Revenue = Quoted Amount (client quote)"

The Quoted Amount is the committed number — it's what will be invoiced and recognized as revenue. Billable Rate is an internal control tool for margin management, not revenue recognition.

---

## Margin Calculations

### Quoted Margin (Expected)
```
Quoted Margin = Quoted Amount - Base Cost
```
*What we expected to make when we quoted the job.*

### Actual Margin (Realized)
```
Actual Margin = Invoiced Amount - Base Cost
```
*What we actually made based on what was billed.*

### Margin Variance
```
Margin Variance = Actual Margin - Quoted Margin
```
*Negative = margin eroded. Positive = margin improved.*

---

## Realization: The Key Metric

```
Realization % = (Invoiced Amount / Quoted Amount) × 100
```

| Realization | Meaning |
|-------------|---------|
| **100%** | Billed exactly what was quoted |
| **<100%** | Write-off, discount, or scope reduction |
| **>100%** | Change orders or additional billing |

**Target: 95%+**

---

## Why Did Margin Erode?

This dashboard diagnoses margin erosion by examining five root causes:

### 1. Was the Quote Too Low?
- **Signal:** Quoted Margin % below 35%
- **Signal:** Quoted Rate/Hr below Billable Rate (negative rate gap)
- **Fix:** Review quoting process, update rate cards

### 2. Was Scope Not Controlled?
- **Signal:** Unquoted tasks appear (scope creep)
- **Signal:** Actual Hours >> Quoted Hours
- **Fix:** Implement change order process, improve estimation

### 3. Were Base Rates Too High? (Wrong Resourcing)
- **Signal:** Cost Rate/Hr > Billable Rate/Hr
- **Signal:** Senior staff on junior tasks
- **Fix:** Resource allocation review

### 4. Was Revenue Not Captured?
- **Signal:** Realization < 100%
- **Signal:** Write-off amount > 0
- **Fix:** Billing process review, discount approval workflow

### 5. Was There Rate Mismatch?
- **Signal:** Large gap between Quoted Rate/Hr and Billable Rate/Hr
- **Fix:** Align quoting with standard rates

---

## Dashboard Features

### 📊 Executive Summary
- Quoted vs Invoiced revenue comparison
- Margin bridge: Quoted → Invoiced → Cost → Margin
- Realization % with status indicators
- Performance flags (losses, overruns, write-offs)

### 📈 Monthly Trends
- Realization % over time
- Quoted vs Actual margin trends
- Revenue capture patterns
- Department-level comparisons

### 🏢 Hierarchical Drill-Down
- Department → Product → Job → Task
- Filter by loss-making, low realization, write-offs
- Task-level scope creep identification

### 💡 Why Margins Erode
- Automated root cause analysis
- Quoting issues detection
- Scope creep quantification
- Write-off identification
- Action items generation

### 🔍 Job Diagnosis Tool
- Single-job deep dive
- Issue identification
- Root cause analysis
- Specific recommendations

### 📋 Reconciliation
- Data validation totals
- Filter summary
- Complete metric definitions

---

## Data Requirements

### Required Columns

**Job Level:**
- `[Job] Job No.`, `[Job] Name`, `[Job] Client`
- `[Job] Start Date`, `[Job] Status`
- `Department`, `Product`

**Task Level:**
- `[Job Task] Name`, `Task Category`
- `[Job Task] Quoted Time`, `[Job Task] Quoted Amount`
- `[Job Task] Actual Time (totalled)`
- `[Job Task] Invoiced Time`, `[Job Task] Invoiced Amount`
- `[Task] Base Rate`, `[Task] Billable Rate`

---

## Metric Definitions

### Revenue Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| Quoted Amount | Direct | Client quote = expected revenue |
| Invoiced Amount | Direct | Actual revenue billed |
| Write-Off | Quoted - Invoiced | Revenue not captured |

### Cost Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| Base Cost | Actual Hours × Cost Rate/Hr | Labor cost |
| Cost Rate/Hr | `[Task] Base Rate` | Internal cost per hour |

### Margin Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| Quoted Margin | Quoted Amount - Base Cost | Expected margin |
| Actual Margin | Invoiced Amount - Base Cost | Realized margin |
| Quoted Margin % | Quoted Margin / Quoted Amount × 100 | Expected margin rate |
| Actual Margin % | Actual Margin / Invoiced Amount × 100 | Realized margin rate |
| Margin Erosion | Quoted Margin % - Actual Margin % | Margin deterioration |

### Realization Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| Realization % | Invoiced / Quoted × 100 | Revenue capture rate |
| Write-Off % | Write-Off / Quoted × 100 | Revenue leakage rate |

### Rate Metrics (Internal Analysis)
| Metric | Formula | Description |
|--------|---------|-------------|
| Quoted Rate/Hr | Quoted Amount / Quoted Hours | Implied rate from quote |
| Actual Rate/Hr | Invoiced Amount / Actual Hours | Realized rate |
| Billable Rate/Hr | `[Task] Billable Rate` | Internal standard rate |
| Rate Gap | Quoted Rate/Hr - Billable Rate/Hr | Quoting vs standard |

---

## Common Use Cases

### "Why did Job X lose money?"
1. Go to **Job Diagnosis** tab
2. Select the job
3. Review the diagnosis:
   - Was quote too low? (check Quoted Margin %)
   - Scope creep? (check unquoted tasks)
   - Hour overrun? (check Hours Variance)
   - Write-off? (check Realization %)
   - Wrong resourcing? (check Cost Rate vs Billable Rate)

### "Which department is dragging margins?"
1. Go to **Drill-Down** tab
2. Review Department Performance chart
3. Sort by Actual Margin % or Realization %
4. Drill into problem departments

### "Are we quoting correctly?"
1. Go to **Why Margins Erode** tab
2. Review "Was the Quote Too Low?" section
3. Check rate gap analysis
4. Identify products/jobs with consistent underquoting

### "How much scope creep do we have?"
1. Go to **Why Margins Erode** tab
2. Review "Scope Creep" panel
3. See total unquoted cost
4. Drill into specific tasks

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

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `analysis.py` | Data processing, metrics, insights |
| `requirements.txt` | Dependencies |
| `README.md` | Documentation |

---

## Key Principles

1. **Quoted Amount = Revenue** — Never use Billable Value as revenue proxy
2. **Realization = Invoiced ÷ Quoted** — The real measure of revenue capture
3. **Diagnose, don't just report** — Explain WHY margins are good or bad
4. **Actionable insights** — Every metric should lead to a decision

