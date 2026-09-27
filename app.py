"""FCC Attendance Tracker — Streamlit app for Favourite Child Church.

Pages (sidebar): Follow-up (home) · Live · Check-in · Insights · Members · SQL.
All data lives in Postgres (Supabase) via `database_url` in Streamlit secrets; see attendance.py.
"""
from pathlib import Path

import streamlit as st

LOGO = Path(__file__).parent / "static" / "logo.png"
st.set_page_config(page_title="FCC Attendance", page_icon=str(LOGO) if LOGO.exists() else "⛪", layout="wide",
                   initial_sidebar_state="expanded")
if LOGO.exists():
    st.logo(str(LOGO), size="large")

# ---------- look & feel (colours from the church logo) ----------
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, [data-testid="stMetric"], [data-testid="stSidebar"] {font-family: 'Inter', system-ui, sans-serif;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {box-shadow: 0 1px 2px rgba(0,0,0,.05);}
[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8; font-weight: 500;}
[data-testid="stMetricValue"] {font-weight: 700; letter-spacing: -.02em;}
.hero {background: linear-gradient(120deg, #2c4b77 0%, #208088 55%, #2aa686 100%); color: #fff;
       border-radius: 1rem; padding: 1.3rem 1.8rem; margin-bottom: .4rem; display: flex; gap: 1.3rem; align-items: center;}
.hero .hero-logo {width: 64px; height: auto; flex: none; filter: drop-shadow(0 2px 6px rgba(0,0,0,.3));}
.hero .hero-text {min-width: 0;}
.hero .eyebrow {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #ffcf00;}
@media (max-width: 640px) { .hero {padding: 1.1rem 1.1rem; gap: .9rem;} .hero .hero-logo {width: 44px;} }
.hero h1 {font-size: 1.9rem; font-weight: 800; margin: .1rem 0 0; letter-spacing: -.02em; color: #fff; padding: 0;}
.hero p {margin: .35rem 0 .9rem; opacity: .88; font-size: .98rem;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
       border-radius: 999px; padding: .2rem .7rem; margin: 0 .35rem .35rem 0; font-size: .8rem; font-weight: 500;}
.live-dot {display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; background: #ffcf00;
           margin-right: .45rem; animation: pulse 1.6s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(255,207,0,.6);} 70% {box-shadow: 0 0 0 .5rem rgba(255,207,0,0);}
                  100% {box-shadow: 0 0 0 0 rgba(255,207,0,0);}}
</style>
""")

import attendance as A  # noqa: E402

pg = st.navigation({
    "Attendance": [
        st.Page(A.page_followup, title="Follow-up", icon=":material/notification_important:", url_path="followup",
                default=True),
        st.Page(A.page_live, title="Live", icon=":material/sensors:", url_path="live"),
        st.Page(A.page_checkin, title="Check-in", icon=":material/how_to_reg:", url_path="checkin"),
        st.Page(A.page_insights, title="Insights", icon=":material/insights:", url_path="insights"),
    ],
    "Admin": [
        st.Page(A.page_members, title="Members", icon=":material/badge:", url_path="members"),
        st.Page(A.page_sql, title="SQL", icon=":material/database:", url_path="sql"),
    ],
})
pg.run()
