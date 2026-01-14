# Job Profitability Analysis Dashboard

A Streamlit dashboard for analyzing job profitability with **correct financial logic**.

## Financial Model

| Term | Source | Purpose |
|------|--------|---------|
| **Quoted Amount** | `[Job Task] Quoted Amount` | **= REVENUE** (what gets invoiced to client) |
| **Billable Value** | `Actual Hours × Billable Rate` | **= BENCHMARK** (what we *should* have quoted) |
| **Base Cost** | `Actual Hours × Cost Rate` | Internal labor cost |

### Why Billable Value Matters

Billable Value is **not revenue** — it's a sanity check:
- **Value Gap = Quoted Amount - Billable Value**
- **Positive** (+) = We quoted ABOVE our internal rates (premium pricing ✅)
- **Negative** (-) = We quoted BELOW our internal rates (discounting ⚠️)

---

## Key Metrics

### 1. Margin (Profitability)
```
Margin = Quoted Amount - Base Cost
Margin % = Margin / Quoted Amount × 100
```
**Target: 35%+**

### 2. Value Gap (Quoting Accuracy)
```
Value Gap = Quoted Amount - Billable Value
Value Gap % = Value Gap / Billable Value × 100
```
- **Positive** = Premium pricing (good)
- **Negative** = Underquoting (leaving money on table)

### 3. Effective Rate/Hr
```
Effective Rate = Quoted Amount / Actual Hours
```
If hours overrun, this drops — shows real revenue per hour worked.

---

## Dashboard Features

### 📊 Executive Summary
- Revenue & Margin overview
- Quoting Accuracy: Quoted vs Billable Value comparison
- Rate analysis (Quoted vs Billable vs Cost rates)
- Performance flags (losses, underquoted jobs, scope creep)

### 📈 Monthly Trends
- Margin % trends over time
- Value Gap % (quoting accuracy) trends
- Quoted vs Billable Value comparison by month
- Department breakdowns

### 🏢 Hierarchical Drill-Down
- Department → Product → Job → Task
- Filter by losses, underquoted, overruns
- Task-level scope creep identification

### 💡 Insights
- Quoting accuracy issues
- Scope & hour problems
- Rate issues
- Action items for problem jobs

### 🔍 Job Diagnosis
- Single-job deep dive
- Root cause analysis
- Specific recommendations

---

## Common Questions

### "Are we quoting correctly?"
Look at **Value Gap**:
- Overall positive = quoting above benchmark (good)
- Jobs with negative Value Gap = underquoted

### "Why is margin low on Job X?"
Check:
1. **Value Gap** — was it underquoted vs internal rates?
2. **Hours Variance** — did hours overrun?
3. **Unquoted Tasks** — scope creep?
4. **Effective Rate** — below cost rate?

### "What's scope creep costing us?"
Check **Unquoted Tasks** — tasks with:
- Quoted Hours = 0
- Actual Hours > 0

These represent work not in the original quote.

---

## Rate Definitions

| Rate | Formula | Meaning |
|------|---------|---------|
| **Quoted Rate/Hr** | Quoted Amount ÷ Quoted Hours | What we charged per quoted hour |
| **Billable Rate/Hr** | [Task] Billable Rate | Internal standard rate |
| **Effective Rate/Hr** | Quoted Amount ÷ Actual Hours | Actual revenue per hour worked |
| **Cost Rate/Hr** | [Task] Base Rate | What each hour costs us |

**Rate Gap = Quoted Rate - Billable Rate**
- Positive = quoting above standard (premium)
- Negative = discounting

---

## Files

| File | Purpose |
|------|---------|
| `analysis.py` | Data processing, metrics, insights |
| `app.py` | Streamlit dashboard |
| `README.md` | Documentation |

---

## Installation

```bash
pip install streamlit pandas numpy altair openpyxl

mkdir -p data
# Place your Excel file in data/

streamlit run app.py
```

---

## Key Principles

1. **Quoted Amount = Revenue** — this is what gets invoiced
2. **Billable Value = Benchmark** — use for sanity checking quotes
3. **Value Gap = Quoting Quality** — positive is good, negative means underquoting
4. **Margin = Quoted - Cost** — the real profit metric