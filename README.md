# ⛪ FCC Attendance Tracker

Attendance tracking for **Favourite Child Church**: ushers tick people in on their phones, leaders see who has been
missing and follow up.

**Live:** https://fcc-attendance.streamlit.app

## Pages

| Page | What it does |
|---|---|
| Dashboard (home) | KPI tiles (last service, 4-service average, who needs a call, first-timers this month), where everyone stands (donut), people present over time, needs-follow-up list, and a side panel with notifications, latest check-ins and who to call next. Refreshes every 30 s. |
| Follow-up & Check-in | **Needs follow-up** tab: 🟡 yellow = 3–4 services missed in a row, 🔴 red = 5+; red first; CSV download. **Check-in** tab: ushers tick people as they arrive; ticks sync to every phone within ~5 s; quick "add first-timer" form. |
| Live | Real-time view of today's check-ins — count, first-timers, arrivals over time, latest arrivals. Refreshes itself; good on a screen during service. |
| Insights | Attendance per service (members vs first-timers, 4-service average), first-timers per month, first-timer return rate, attendance by group. |
| Members | The register (editable), sign-ups from the welcome form to approve, add people, and import the Google Sheets CSV exports. |
| SQL | Read-only SQL queries against the database, with ready-made examples; download results. |

Dark dashboard theme in the church colours (logo greens and gold). Chart colours are checked for colour-blind
separation and contrast. The sidebar has **Layout** controls (names per row on Check-in, panel stacking).

## Data

Stored in a **Postgres** database (Supabase) — never in this repo. Tables:

- `members` — id, full_name, phone, email, group_name, role, status, type (member / first_timer), date_joined, first_visit, invited_by, follow_up
- `services` — service_date, name
- `attendance` — service_date, member_id, checked_at (one row per person ticked per service)
- `registrations` — sign-ups from the [welcome form](https://github.com/Kellyzicoder/fcc-welcome), approved under *Members → Sign-ups*

Until `database_url` is set in the app's Streamlit **Secrets**, the pages run on a SQLite demo database with invented
names. Setup steps are in the app under *Members → Setup*. Query the data from the SQL page, Supabase's SQL editor,
or Python (`pandas.read_sql`).

CSV files and connection strings are blocked by `.gitignore`; don't commit them — this repo is public.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For real data locally, put `database_url` and `attendance_password` in `.streamlit/secrets.toml` (git-ignored).
