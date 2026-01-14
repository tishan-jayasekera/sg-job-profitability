
# Job Profitability — Quote vs Ratecard vs Cost

A Streamlit dashboard to investigate **job profitability** from **quotation → execution → invoicing**, with a structured drilldown:

**Department → Product → Job → Task**

This is built for operational insight, not just reporting.

---

## Core business definitions (non‑negotiable)

**Quoted Amount (client quote)**  
- The commercial value agreed with the client (what we expect to invoice and recognise as revenue).  
- In the dataset: **`[Job Task] Quoted Amount`** (task-level allocation of the overall quote).

**Invoiced Amount**  
- Revenue actually invoiced to date.  
- In the dataset: **`[Job Task] Invoiced Amount`**.

**Ratecard Value (internal control)**  
- **NOT revenue**.  
- An internal “control” metric: `Actual Hours × Billable Rate`.  
- Used to test whether the quote is priced above/below internal ratecard expectations.

**Base Cost (time & materials cost)**  
- Internal delivery cost: `Actual Hours × Base Rate`.  
- Uses `Time+Material (Base)` where present, otherwise computes from Base Rate.

---

## What the dashboard answers

### 1) Was the quote sufficient for the work delivered?
- **Quote Margin ($)** = `Quoted Amount − Base Cost`
- **Quote Margin (%)** = `(Quoted Amount − Base Cost) / Quoted Amount`

### 2) Did we price above/below our internal ratecard for the hours delivered?
- **Pricing Adequacy (%)** = `Quoted Amount / Ratecard Value`
  - `< 100%` suggests the quote is **below** internal ratecard value for delivered hours (underpricing risk)

### 3) Did we actually realise the quote as revenue?
- **Realisation (%)** = `Invoiced Amount / Quoted Amount`

### 4) What drove margin erosion?
- Task overruns: `Actual Hours > Quoted Hours`
- Scope creep: **unquoted tasks** with meaningful cost

---

## Critical dataset rule (exclusion)

Exclude any rows where:

`[Job Task] Name == "Social Garden Invoice Allocation"`

This exclusion is enforced in code by default.

---

## Month-on-month trends (important limitation)

This dataset does **not** include a true timesheet entry date.  
So month-on-month analysis is **anchored** to a configurable date field (default: **Task Start Date**).

If you want “true MoM delivery cost by timesheet date”, the source export must include a timesheet entry date.

---

## Repo structure

```
.
├─ app.py                 # Streamlit dashboard entrypoint
├─ analysis.py             # Data parsing + metrics + rollups + insights
├─ requirements.txt
└─ data/
   └─ Quoted_Task_Report_FY26.xlsx   # optional local default (not committed)
```

---

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
streamlit run app.py
```

---

## Notes for teams

- This repo intentionally distinguishes **commercial** vs **internal** metrics.
- **Quoted Amount is the commercial benchmark. Ratecard Value is internal and must never be treated as revenue.**
- The “Insights” tab is designed to produce an operational story:
  - where pricing is structurally weak,
  - where scope control fails,
  - where estimating is systematically off.
