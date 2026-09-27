# ⛪ FCC Attendance Tracker

Attendance tracking for **Favourite Child Church**: ushers tick people in on their phones, leaders see who has been
missing and follow up.

**Live:** https://fcc-attendance.streamlit.app

## Pages

| Page | What it does |
|---|---|
| Check-in (home) | Ushers tick people as they arrive; ticks from every phone appear for everyone within ~5 seconds. Quick "add first-timer" form. |
| Follow-up | Priority list of who's been missing: 🟡 **yellow** = 3–4 services missed in a row, 🔴 **red** = 5+ in a row. Red first; download as CSV. |
| Live | Real-time view of today's check-ins — count, first-timers, arrivals over time, latest arrivals. Refreshes itself; good on a screen during service. |
| Insights | Attendance per service (members vs first-timers, 4-service average), first-timers per month, first-timer return rate, attendance by group. |
| Members | The register (members + first-timers), add people, and import the Google Sheets CSV exports. |
| SQL | Read-only SQL queries against the database, with ready-made examples; download results. |

The sidebar has **Layout** controls (side-by-side or stacked panels, names per row on Check-in).
Light and dark mode follow your device setting. Colours and logo come from the church branding.

## Data

Stored in a **Postgres** database (Supabase) — never in this repo. Tables:

- `members` — id, full_name, phone, email, group_name, role, status, type (member / first_timer), date_joined, first_visit, invited_by, follow_up
- `services` — service_date, name
- `attendance` — service_date, member_id, checked_at (one row per person ticked per service)
- `registrations` — sign-ups from the public form (coming soon), waiting for approval

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
