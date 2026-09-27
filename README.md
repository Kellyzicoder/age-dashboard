# ⛪ Age Dashboard

Interactive Streamlit dashboard showing how membership and the **age profile** of
**1,000 churches** changed from 2015 to 2026, built on **~512,000 member records**.

**Live:** https://kelly-age-dashboard.streamlit.app

## Attendance (check-in & follow-up)

| Page | What it does |
|---|---|
| Check-in | Ushers tick people as they arrive; ticks from every phone appear for everyone within ~5 seconds. Quick "add first-timer" form. |
| Follow-up | Priority list of who's been missing: 🟡 **yellow** = 3–4 services missed in a row, 🔴 **red** = 5+ in a row. Red first, then yellow; download as CSV. |
| Members | The register (members + first-timers), add people, and import your Google Sheets CSV exports. |
| SQL | Type any SQL query against the attendance database (read-only), with ready-made examples; download results. |

Data is stored in a **Postgres** database (Supabase or Neon) — never in this repo. Until `database_url` is set in the
app's Streamlit **Secrets**, the pages run on a SQLite demo database with invented names. Setup steps are in the app
under *Members → Setup*. Tables: `members`, `services`, `attendance` — query them from the SQL page, Supabase's SQL
editor, or Python (`pandas.read_sql`). CSV files and connection strings are blocked by `.gitignore`; don't commit them —
this repo is public.

## What's inside

Pages (sidebar navigation, filters shared across all pages):

| Page | Shows |
|---|---|
| Overview | KPI cards with sparklines, members by age group, age mix %, growth by age group, median age trend |
| Live activity | Real-time feed of members joining/leaving, refreshing every few seconds |
| Compare groups | Median age & indexed growth by region / denomination / setting, growth heatmap |
| Age pyramid | Male/female pyramid for any year, with the first year as an outline |
| All churches | Every church plotted age vs growth, top lists, searchable table, CSV download |
| Church profile | Drill into a single church vs the overall average |
| About the data | Data source and how to plug in your own |

Filters (sidebar): year range, region, denomination, urban/rural setting, gender.

Light and dark mode follow your device setting automatically.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free) on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**, choose this repo, branch `main`, file `app.py`.
3. Click **Deploy**. Every push to `main` redeploys automatically.

## Data

`generate_data.py` creates the synthetic dataset (`python generate_data.py`):

- `data/churches.csv.gz` — `church_id, church_name, region, denomination, setting, founded`
- `data/members.csv.gz` — `member_id, church_id, birth_year, join_year, gender, leave_year`

To use real data, put your own `data/churches.csv.gz` and `data/members.csv.gz` (same columns) in the repo and remove the `data/*.csv.gz` line from `.gitignore`. If the files are missing, the app generates the sample data automatically on first run.
