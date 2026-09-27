"""FCC Attendance Tracker: live check-in, live arrivals, follow-up, insights, member register and SQL.

Storage is a plain SQL database, so you can query it directly:
  • Postgres (Supabase or Neon) when `database_url` is set in Streamlit secrets — the real data;
  • otherwise a local SQLite demo database filled with invented names (reset whenever the app restarts).
Real member data is never stored in this repository.

Tables (same in SQLite and Postgres):
  members(id, full_name, phone, email, group_name, role, status, type, date_joined, first_visit,
          invited_by, follow_up, created_at)
  services(service_date PRIMARY KEY, name)
  attendance(service_date, member_id, checked_at, PRIMARY KEY (service_date, member_id))
"""
from __future__ import annotations

import datetime as dt
import threading
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TZ = ZoneInfo("Pacific/Auckland")
BRAND = dict(navy="#2c4b77", teal="#208088", green="#2aa686", slate="#293641", gold="#ffcf00")  # from the FCC logo
YELLOW_AT, RED_AT = 3, 5          # services missed in a row
INACTIVE = {"inactive", "moved", "left", "deceased", "transferred"}
AMBER, CRIMSON = "#fab219", "#d03b3b"   # reserved status colours (always shown with icon + label)


def today() -> dt.date:
    return dt.datetime.now(TZ).date()


def now_iso() -> str:
    return dt.datetime.now(TZ).isoformat(timespec="seconds")


def norm(name: str) -> str:
    return " ".join(str(name).split()).lower()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------- storage (SQL: SQLite demo or Postgres)
MEMBER_COLS = ["full_name", "phone", "email", "group_name", "role", "status", "type", "date_joined", "first_visit",
               "invited_by", "follow_up", "created_at"]
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS members (
        id TEXT PRIMARY KEY, full_name TEXT NOT NULL, phone TEXT, email TEXT, group_name TEXT, role TEXT,
        status TEXT, type TEXT DEFAULT 'member', date_joined DATE, first_visit DATE, invited_by TEXT,
        follow_up TEXT, created_at TEXT)""",
    "CREATE TABLE IF NOT EXISTS services (service_date DATE PRIMARY KEY, name TEXT)",
    """CREATE TABLE IF NOT EXISTS attendance (
        service_date DATE NOT NULL REFERENCES services(service_date) ON DELETE CASCADE,
        member_id TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
        checked_at TEXT, PRIMARY KEY (service_date, member_id))""",
    "CREATE INDEX IF NOT EXISTS attendance_member ON attendance(member_id)",
]


def _txt(v) -> str:
    """Normalise DB values (Postgres returns date objects, SQLite returns strings) to plain strings."""
    if v is None:
        return ""
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


class SqlStore:
    """One store for both engines. Queries use standard SQL that runs unchanged on SQLite and Postgres."""

    def __init__(self, url: str | None):
        self.demo = not url
        self.url = url
        self.engine = "sqlite" if self.demo else "postgres"
        self.path = "/tmp/fcc_attendance_demo.db"
        self._lock = threading.RLock()
        self._conn = None
        self._cache = {}
        if self.demo:
            import os
            if os.path.exists(self.path):
                os.remove(self.path)  # fresh demo on every app start
        with self._lock:
            for stmt in SCHEMA:
                self._exec(stmt)
        if self.demo:
            _seed_demo(self)

    # -- connection handling
    def _connect(self):
        if self.demo:
            import sqlite3
            c = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            return c
        import psycopg
        return psycopg.connect(self.url, autocommit=True, connect_timeout=10)

    def _q(self, sql: str) -> str:
        return sql if self.demo else sql.replace("?", "%s")

    def _exec(self, sql, params=(), many=False, fetch=False):
        with self._lock:
            for attempt in (1, 2):  # one reconnect if the server dropped the connection
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    cur = self._conn.cursor()
                    if many:
                        cur.executemany(self._q(sql), params)
                    else:
                        cur.execute(self._q(sql), params)
                    if fetch:
                        cols = [d[0] for d in cur.description]
                        return [dict(zip(cols, r)) for r in cur.fetchall()]
                    return None
                except Exception:
                    if attempt == 2 or self.demo:
                        raise
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None

    def _cached(self, key, ttl, fn):
        now = dt.datetime.now().timestamp()
        hit = self._cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (now, val)
        return val

    # -- reads
    def list_members(self):
        def load():
            rows = self._exec("SELECT * FROM members", fetch=True)
            return [{k: _txt(v) for k, v in r.items()} | {"group": _txt(r.get("group_name"))} for r in rows]
        return self._cached("members", 60, load)

    def list_services(self):
        def load():
            svcs = {_txt(r["service_date"]): dict(date=_txt(r["service_date"]), name=r["name"] or "", present={})
                    for r in self._exec("SELECT service_date, name FROM services", fetch=True)}
            for r in self._exec("SELECT service_date, member_id, checked_at FROM attendance", fetch=True):
                s = svcs.get(_txt(r["service_date"]))
                if s is not None:
                    s["present"][r["member_id"]] = _txt(r["checked_at"])
            return list(svcs.values())
        return self._cached("services", 15, load)

    def get_service(self, date: str):  # live-poll read: small and always fresh
        rows = self._exec("SELECT a.member_id, a.checked_at, s.name FROM services s "
                          "LEFT JOIN attendance a ON a.service_date = s.service_date WHERE s.service_date = ?",
                          (date,), fetch=True)
        if not rows:
            return None
        return dict(date=date, name=rows[0]["name"] or "",
                    present={r["member_id"]: _txt(r["checked_at"]) for r in rows if r["member_id"]})

    # -- writes
    def ensure_service(self, date: str, name: str):
        self._exec("INSERT INTO services (service_date, name) VALUES (?, ?) "
                   "ON CONFLICT (service_date) DO UPDATE SET name = excluded.name", (date, name or "Service"))
        self._cache.pop("services", None)

    def set_present(self, date: str, mid: str, present: bool, name: str = "Sunday Service"):
        if present:
            self._exec("INSERT INTO services (service_date, name) VALUES (?, ?) ON CONFLICT (service_date) DO NOTHING",
                       (date, name))
            self._exec("INSERT INTO attendance (service_date, member_id, checked_at) VALUES (?, ?, ?) "
                       "ON CONFLICT (service_date, member_id) DO NOTHING", (date, mid, now_iso()))
        else:
            self._exec("DELETE FROM attendance WHERE service_date = ? AND member_id = ?", (date, mid))
        self._cache.pop("services", None)

    def upsert_members(self, rows: list[dict]):
        existing = {r["id"] for r in self._exec("SELECT id FROM members", fetch=True)}
        inserts, updates = [], []
        for r in rows:
            r = dict(r)
            if "group" in r:
                r["group_name"] = r.pop("group")
            mid = r.pop("id", None) or new_id()
            vals = [None if (r.get(c) == "" and c in ("date_joined", "first_visit")) else r.get(c) for c in MEMBER_COLS]
            (updates if mid in existing else inserts).append((mid, vals))
        if inserts:
            cols = ", ".join(["id"] + MEMBER_COLS)
            marks = ", ".join(["?"] * (len(MEMBER_COLS) + 1))
            self._exec(f"INSERT INTO members ({cols}) VALUES ({marks}) ON CONFLICT (id) DO NOTHING",
                       [[mid] + vals for mid, vals in inserts], many=True)
        if updates:  # only overwrite fields that were provided; COALESCE keeps what's already stored
            sets = ", ".join(f"{c} = COALESCE(?, {c})" for c in MEMBER_COLS)
            self._exec(f"UPDATE members SET {sets} WHERE id = ?", [vals + [mid] for mid, vals in updates], many=True)
        self._cache.pop("members", None)
        return len(rows)

    def update_members(self, changes: dict[str, dict]) -> int:
        """Set exact values (blanks allowed) for edited members — used by the editable register."""
        n = 0
        for mid, fields in changes.items():
            fields = dict(fields)
            if "group" in fields:
                fields["group_name"] = fields.pop("group")
            cols = [c for c in fields if c in MEMBER_COLS and c != "created_at"]
            if not cols:
                continue
            vals = [None if (fields[c] in ("", None) and c in ("date_joined", "first_visit")) else fields[c] for c in cols]
            self._exec(f"UPDATE members SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?", vals + [mid])
            n += 1
        self._cache.pop("members", None)
        return n

    # -- read-only SQL for the query page
    def run_query(self, sql: str, limit: int = 5000) -> pd.DataFrame:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise ValueError("Type a query first.")
        if ";" in sql:
            raise ValueError("Run one statement at a time.")
        if self.demo:
            import sqlite3
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)  # read-only at the file level
            try:
                cur = conn.execute(sql)
                cols = [d[0] for d in cur.description or []]
                return pd.DataFrame(cur.fetchmany(limit), columns=cols)
            finally:
                conn.close()
        import psycopg
        with psycopg.connect(self.readonly_url or self.url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")         # the database itself refuses any write
                cur.execute("SET LOCAL statement_timeout = '15s'")
                cur.execute(sql)
                cols = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(limit)
            conn.rollback()
        return pd.DataFrame(rows, columns=cols)

    readonly_url = None


def _seed_demo(store: "SqlStore"):
    """Invented people + 12 past Sundays of attendance, with a few people who recently stopped coming."""
    rng = np.random.default_rng(7)
    first = ["Ama", "Kwame", "Esi", "Kojo", "Abena", "Yaw", "Akosua", "Kofi", "Adwoa", "Kwabena", "Efua", "Kwaku",
             "Afia", "Kweku", "Aba", "Fiifi", "Naana", "Paa", "Serwaa", "Ekow", "Mia", "Leo", "Zoe", "Eli", "Ruth",
             "Noah", "Tia", "Sam", "Joy", "Ben"]
    last = ["Asante", "Owusu", "Boateng", "Appiah", "Darko", "Ofori", "Quaye", "Tetteh", "Addo", "Badu"]
    names = sorted({f"{rng.choice(first)} {rng.choice(last)}" for _ in range(80)})[:60]
    groups = ["Choir", "Ushering", "Youth", "Media", "Children", "Men", "Women", ""]
    people = [dict(id=new_id(), full_name=n, phone=f"021 {rng.integers(100, 999)} {rng.integers(1000, 9999)}",
                   email="", group=str(rng.choice(groups)), role="", status="", type="member", date_joined="",
                   first_visit="", invited_by="", follow_up="", created_at=now_iso()) for n in names]
    store.upsert_members([dict(p) for p in people])
    ids = [p["id"] for p in people]
    sundays = [today() - dt.timedelta(days=(today().weekday() + 1) % 7 + 7 * k) for k in range(12)][::-1]
    habit = {m: rng.beta(6, 2) for m in ids}
    stopped = {m: int(rng.integers(3, 9)) for m in rng.choice(ids, 9, replace=False)}
    svc, att = [], []
    for i, d in enumerate(sundays):
        svc.append((d.isoformat(), "Sunday Service"))
        for m, p in habit.items():
            if m in stopped and i >= len(sundays) - stopped[m]:
                continue
            if rng.random() < p:
                att.append((d.isoformat(), m, dt.datetime.combine(d, dt.time(10, int(rng.integers(0, 40))), TZ).isoformat()))
    store._exec("INSERT INTO services (service_date, name) VALUES (?, ?)", svc, many=True)
    store._exec("INSERT INTO attendance (service_date, member_id, checked_at) VALUES (?, ?, ?)", att, many=True)


@st.cache_resource(show_spinner="Connecting to the database…")
def get_store():
    try:
        url = st.secrets.get("database_url")
        ro = st.secrets.get("database_url_readonly")
    except Exception:
        url = ro = None
    store = SqlStore(url or None)
    store.readonly_url = ro or None
    return store


# ---------------------------------------------------------------- follow-up logic
def missed_streaks(members: list[dict], services: list[dict], upto: dt.date | None = None) -> pd.DataFrame:
    """Per person: services missed in a row (most recent first), last seen, and a yellow/red flag.

    Only services on or after a person's start (date joined / first visit) count against them.
    """
    upto = upto or today()
    svcs = sorted((s for s in services if s.get("date") and s["date"] <= upto.isoformat()), key=lambda s: s["date"])
    rows = []
    for m in members:
        if norm(m.get("status", "")) in INACTIVE:
            continue
        start = min([d for d in (m.get("date_joined"), m.get("first_visit")) if d] or ["0000"])
        mine = [s for s in svcs if s["date"] >= start]
        streak, last_seen = 0, None
        for s in reversed(mine):
            if m["id"] in (s.get("present") or {}):
                last_seen = s["date"]
                break
            streak += 1
        attended = sum(m["id"] in (s.get("present") or {}) for s in mine)
        level = "red" if streak >= RED_AT else "yellow" if streak >= YELLOW_AT else "ok"
        rows.append(dict(id=m["id"], name=m.get("full_name", ""), missed=streak, level=level, last_seen=last_seen,
                         attended=attended, eligible=len(mine), phone=m.get("phone", ""), group=m.get("group", ""),
                         type=m.get("type", "member"), invited_by=m.get("invited_by", ""),
                         follow_up=m.get("follow_up", "")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rate"] = np.where(df.eligible > 0, df.attended / df.eligible.clip(lower=1), np.nan)
    order = {"red": 0, "yellow": 1, "ok": 2}
    return df.sort_values(["level", "missed", "name"], key=lambda c: c.map(order) if c.name == "level"
                          else (-c if c.name == "missed" else c.str.lower())).reset_index(drop=True)


LEVEL_LABEL = {"red": f"🔴 Red · {RED_AT}+ missed", "yellow": f"🟡 Yellow · {YELLOW_AT}–{RED_AT - 1} missed",
               "ok": "🟢 On track"}


# ---------------------------------------------------------------- CSV import (your Google Sheets exports)
def _col(df: pd.DataFrame, *starts: str):
    for c in df.columns:
        if any(c.strip().lower().startswith(s) for s in starts):
            return c
    return None


def _date(v) -> str:
    v = "" if pd.isna(v) else str(v).strip()
    if not v:
        return ""
    d = pd.to_datetime(v, dayfirst=True, errors="coerce")
    return "" if pd.isna(d) else d.date().isoformat()


def _s(v) -> str:
    return "" if pd.isna(v) else " ".join(str(v).split())


def parse_registers(register: pd.DataFrame | None, first_timers: pd.DataFrame | None, existing: list[dict]):
    """Turn the two sheet exports into member records, merging people who appear in both lists
    and matching existing members by name so re-importing updates instead of duplicating."""
    by_name = {norm(m.get("full_name", "")): m["id"] for m in existing}
    out, notes = {}, []
    if register is not None:
        c = dict(name=_col(register, "full name"), phone=_col(register, "phone"), email=_col(register, "email"),
                 joined=_col(register, "date joined"), group=_col(register, "group"),
                 role=_col(register, "ministry", "role"), status=_col(register, "status"))
        reg = register[register[c["name"]].map(_s) != ""]
        dups = reg[c["name"]].map(norm).value_counts()
        for n, k in dups[dups > 1].items():
            notes.append(f"“{n.title()}” appears {k} times in the register — imported once; add the others by hand "
                         "if they are different people.")
        for _, r in reg.iterrows():
            key = norm(r[c["name"]])
            out[key] = dict(full_name=_s(r[c["name"]]), phone=_s(r.get(c["phone"])), email=_s(r.get(c["email"])),
                            date_joined=_date(r.get(c["joined"])), group=_s(r.get(c["group"])),
                            role=_s(r.get(c["role"])), status=_s(r.get(c["status"])), type="member")
    if first_timers is not None:
        c = dict(name=_col(first_timers, "full name"), date=_col(first_timers, "date of visit"),
                 phone=_col(first_timers, "phone"), email=_col(first_timers, "email"),
                 invited=_col(first_timers, "invited"), follow=_col(first_timers, "follow"))
        ft = first_timers[first_timers[c["name"]].map(_s) != ""]
        for _, r in ft.iterrows():
            key = norm(r[c["name"]])
            rec = dict(first_visit=_date(r.get(c["date"])), invited_by=_s(r.get(c["invited"])),
                       follow_up=_s(r.get(c["follow"])))
            if key in out:
                out[key].update({k: v for k, v in rec.items() if v})
                notes.append(f"“{out[key]['full_name']}” is in both lists — kept as a member with their first-visit details.")
            else:
                out[key] = dict(full_name=_s(r[c["name"]]), phone=_s(r.get(c["phone"])), email=_s(r.get(c["email"])),
                                date_joined="", group="", role="", status="", type="first_timer", **rec)
    rows = []
    for key, rec in out.items():
        if key in by_name:
            rec["id"] = by_name[key]
            rec = {k: v for k, v in rec.items() if v or k == "id"}  # never blank out data already in the database
        else:
            rec["created_at"] = now_iso()
        rows.append(rec)
    return rows, notes


# ---------------------------------------------------------------- page helpers
@st.cache_data(show_spinner=False)
def logo_img(css_class: str = "hero-logo") -> str:
    """The church logo as an inline <img> (empty string if the file is missing)."""
    import base64
    from pathlib import Path
    p = Path(__file__).parent / "static" / "logo.png"
    if not p.exists():
        return ""
    return f'<img class="{css_class}" alt="Favourite Child Church" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def hero_html(eyebrow: str, title: str, subtitle: str, chips: list[str], live: bool = False) -> str:
    dot = '<span class="live-dot"></span>' if live else ""
    return (f'<div class="hero">{logo_img()}<div class="hero-text"><div class="eyebrow">{eyebrow}</div>'
            f'<h1>{dot}{title}</h1><p>{subtitle}</p>' + "".join(f'<span class="chip">{c}</span>' for c in chips)
            + "</div></div>")


def header(title: str, subtitle: str, store, live: bool = False):
    badge = "Demo data (invented names · SQLite)" if store.demo else "Connected · Postgres database"
    st.html(hero_html("Favourite Child Church · Attendance", title, subtitle, [badge], live))


def gate(store) -> bool:
    """Real member data needs a password (set attendance_password in secrets). Demo data is open."""
    if store.demo:
        return True
    try:
        pw = st.secrets.get("attendance_password")
    except Exception:
        pw = None
    if not pw:
        st.error("Set `attendance_password` in the app's Secrets before real member data can be shown.")
        return False
    if st.session_state.get("att_ok"):
        return True
    with st.form("att_login"):
        entered = st.text_input("Attendance password", type="password")
        if st.form_submit_button("Unlock", type="primary"):
            if entered == pw:
                st.session_state.att_ok = True
                st.rerun()
            st.error("Wrong password.")
    return False


def layout_prefs(names: bool = False):
    """Sidebar layout controls. Streamlit can't drag-resize panels, so layout is chosen here instead."""
    with st.sidebar:
        st.markdown("**Layout**")
        mode = st.segmented_control("Panels", ["Side by side", "Stacked"], default="Side by side",
                                    key="lay_mode") or "Side by side"
        per_row = st.select_slider("Names per row", options=[1, 2, 3, 4], value=3, key="lay_names") if names else 3
    return mode, per_row


def demo_note(store):
    if store.demo:
        st.info("These pages are showing **invented demo people**. Connect your Postgres database (see *Members → Setup*) "
                "to use your real register.", icon=":material/science:")


# ---------------------------------------------------------------- pages
def page_checkin():
    store = get_store()
    header("Check-in", "Tick people as they arrive — every phone sees the same list within seconds", store, live=True)
    if not gate(store):
        return
    demo_note(store)
    _, per_row = layout_prefs(names=True)
    c1, c2, c3 = st.columns([1, 2, 2], vertical_alignment="bottom")
    day = c1.date_input("Service date", value=today(), format="DD/MM/YYYY")
    svc_name = c2.text_input("Service", value="Sunday Service")
    q = c3.text_input("Find a person", placeholder="Type a name…", key="ci_q")
    date = day.isoformat()

    members = sorted(store.list_members(), key=lambda m: norm(m.get("full_name", "")))
    members = [m for m in members if norm(m.get("status", "")) not in INACTIVE]

    def on_tick(mid):
        store.set_present(date, mid, bool(st.session_state[f"ci_{date}_{mid}"]), svc_name)

    @st.fragment(run_every=5)
    def live_list():
        s = store.get_service(date) or {}
        present = s.get("present") or {}
        shown = [m for m in members if not q or norm(q) in norm(m.get("full_name", ""))]
        k = st.columns(3)
        k[0].metric("Checked in", f"{len(present)}", border=True)
        k[1].metric("Not yet", f"{max(len(members) - len(present), 0)}", border=True)
        k[2].metric("First-timers today", f"{sum(1 for m in members if m['id'] in present and m.get('type') == 'first_timer')}",
                    border=True)
        with st.container(border=True):
            st.caption(f"Live · updated {dt.datetime.now(TZ):%H:%M:%S} · showing {len(shown)} of {len(members)}")
            cols = st.columns(per_row)
            for i, m in enumerate(shown):
                key = f"ci_{date}_{m['id']}"
                st.session_state[key] = m["id"] in present  # sync ticks made on other devices
                tag = " · first-timer" if m.get("type") == "first_timer" else ""
                cols[i % per_row].checkbox(f"{m['full_name']}{tag}", key=key, on_change=on_tick, args=(m["id"],))

    live_list()

    with st.expander("Add a first-timer and check them in", icon=":material/person_add:"):
        with st.form("ft_add", clear_on_submit=True):
            a, b, c = st.columns(3)
            name = a.text_input("Full name")
            phone = b.text_input("Phone")
            invited = c.text_input("Invited by / how they heard of us")
            if st.form_submit_button("Add & check in", type="primary") and name.strip():
                existing = {norm(m["full_name"]): m["id"] for m in store.list_members()}
                mid = existing.get(norm(name)) or new_id()
                store.upsert_members([dict(id=mid, full_name=" ".join(name.split()), phone=phone.strip(),
                                           invited_by=invited.strip(), type="first_timer", first_visit=date,
                                           status="", created_at=now_iso())])
                store.set_present(date, mid, True, svc_name)
                st.success(f"Welcome, {name.strip()}! Checked in.")
                st.rerun()


def page_followup():
    store = get_store()
    header("Follow-up", f"Who we haven't seen — {RED_AT}+ services missed in a row is red, "
                        f"{YELLOW_AT}–{RED_AT - 1} is yellow", store, live=True)
    if not gate(store):
        return
    demo_note(store)
    show = st.segmented_control("Show", ["Needs follow-up", "Red only", "Yellow only", "Everyone"],
                                default="Needs follow-up") or "Needs follow-up"

    @st.fragment(run_every=30)
    def live_followup():
        members, services = store.list_members(), store.list_services()
        df = missed_streaks(members, services)
        past = sorted([s for s in services if s.get("date", "") <= today().isoformat()], key=lambda s: s["date"])
        if df.empty or not past:
            st.info("No services recorded yet. Tick people on the **Check-in** page and this list fills itself in.")
            return
        red, yellow = int((df.level == "red").sum()), int((df.level == "yellow").sum())
        k = st.columns(4)
        k[0].metric("🔴 Red", red, f"{RED_AT}+ in a row", delta_color="off", border=True)
        k[1].metric("🟡 Yellow", yellow, f"{YELLOW_AT}–{RED_AT - 1} in a row", delta_color="off", border=True)
        k[2].metric("🟢 On track", int((df.level == "ok").sum()), border=True)
        last = past[-1]
        k[3].metric("Last service", f"{len(last.get('present') or {})} present",
                    dt.date.fromisoformat(last["date"]).strftime("%a %d %b"), delta_color="off", border=True)

        view = {"Needs follow-up": df[df.level != "ok"], "Red only": df[df.level == "red"],
                "Yellow only": df[df.level == "yellow"], "Everyone": df}[show]
        table = pd.DataFrame({
            "Status": view.level.map(LEVEL_LABEL), "Name": view.name, "Missed in a row": view.missed,
            "Last seen": pd.to_datetime(view.last_seen).dt.strftime("%d %b %Y").fillna("Not yet"),
            "Attendance": view.rate, "Phone": view.phone, "Group": view.group,
            "Type": view.type.map({"member": "Member", "first_timer": "First-timer"}).fillna(view.type),
            "Invited by": view.invited_by})

        def tint(col):
            return [f"background-color: {'rgba(208,59,59,.20)' if 'Red' in v else 'rgba(250,178,25,.25)' if 'Yellow' in v else ''}"
                    for v in col]
        with st.container(border=True):
            st.markdown(f"**Priority list** · {len(view)} people · most-missed first")
            st.dataframe(table.style.apply(tint, subset=["Status"]), hide_index=True, width="stretch",
                         height=min(38 * (len(table) + 1) + 4, 560),
                         column_config={"Attendance": st.column_config.ProgressColumn(
                             "Attendance", format="percent", min_value=0, max_value=1),
                             "Missed in a row": st.column_config.NumberColumn(format="%d")})
            st.download_button("Download follow-up list (CSV)", table.to_csv(index=False), "follow_up.csv", "text/csv",
                               icon=":material/download:")

        trend = pd.DataFrame([dict(date=s["date"], present=len(s.get("present") or {})) for s in past[-26:]])
        trend["date"] = pd.to_datetime(trend.date)
        fig = px.bar(trend, x="date", y="present", labels={"present": "People present", "date": ""})
        fig.update_traces(marker_color=BRAND["teal"], hovertemplate="%{x|%d %b %Y}: %{y} present<extra></extra>")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), bargap=0.25,
                          title=dict(text="Attendance per service", font=dict(size=15)))
        with st.container(border=True):
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"Live · recalculated {dt.datetime.now(TZ):%H:%M:%S} · only services since someone joined "
                   "or first visited count against them.")

    live_followup()


EDIT_COLS = ["full_name", "type", "phone", "email", "group", "role", "status", "date_joined", "first_visit",
             "invited_by", "follow_up"]
STATUSES = ["", "Active", "Inactive", "Moved", "Left", "Transferred", "Deceased"]


def _as_date(v):
    d = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(d) else d.date()


def register_editor(store):
    """Editable register: click a cell, type, then Save. Rows are never deleted here — set Status instead."""
    m = pd.DataFrame(store.list_members())
    if m.empty:
        st.info("No members yet — use **Import CSV** to load your register.")
        return
    for c in EDIT_COLS:
        if c not in m.columns:
            m[c] = ""
    m = m.set_index("id")[EDIT_COLS].fillna("")
    for c in ("date_joined", "first_visit"):
        m[c] = m[c].map(_as_date)
    extra = sorted({s for s in m.status.unique() if s and s not in STATUSES})
    q = st.text_input("Search", placeholder="Filter by name, group, phone…", key="reg_q")
    view = m if not q else m[m.apply(lambda r: q.lower() in " ".join(map(str, r)).lower(), axis=1)]
    view = view.sort_values("full_name", key=lambda s: s.str.lower())
    st.caption(f"{len(view)} of {len(m)} people · click a cell to edit, then **Save changes**. "
               "To take someone off the lists, set Status to Moved/Inactive (their history is kept).")
    edited = st.data_editor(
        view, key=f"reg_{q}", hide_index=True, width="stretch", height=520, num_rows="fixed",
        column_config={
            "full_name": st.column_config.TextColumn("Name", required=True, max_chars=100),
            "type": st.column_config.SelectboxColumn("Type", options=["member", "first_timer"], required=True),
            "phone": st.column_config.TextColumn("Phone", max_chars=30),
            "email": st.column_config.TextColumn("Email", max_chars=120),
            "group": st.column_config.TextColumn("Group"),
            "role": st.column_config.TextColumn("Ministry / role"),
            "status": st.column_config.SelectboxColumn("Status", options=STATUSES + extra),
            "date_joined": st.column_config.DateColumn("Joined", format="DD/MM/YYYY"),
            "first_visit": st.column_config.DateColumn("First visit", format="DD/MM/YYYY"),
            "invited_by": st.column_config.TextColumn("Invited by"),
            "follow_up": st.column_config.TextColumn("Follow-up notes"),
        })
    changes = {}
    for mid in edited.index:
        diff = {}
        for c in EDIT_COLS:
            old, new = view.at[mid, c], edited.at[mid, c]
            old = "" if old is None or (not isinstance(old, str) and pd.isna(old)) else old
            new = "" if new is None or (not isinstance(new, str) and pd.isna(new)) else new
            if hasattr(new, "isoformat"):
                new = new.isoformat()[:10]
            if hasattr(old, "isoformat"):
                old = old.isoformat()[:10]
            if str(old) != str(new):
                diff[c] = new.strip() if isinstance(new, str) else new
        if diff:
            changes[mid] = diff
    a, b = st.columns([1, 4], vertical_alignment="center")
    save = a.button(f"Save changes ({len(changes)})", type="primary", disabled=not changes, icon=":material/save:")
    if changes:
        b.caption("Unsaved: " + ", ".join(edited.at[mid, "full_name"] or "(no name)" for mid in list(changes)[:6])
                  + ("…" if len(changes) > 6 else ""))
    if save:
        if any(not str(edited.at[mid, "full_name"]).strip() for mid in changes):
            st.error("Every person needs a name.")
        else:
            n = store.update_members(changes)
            st.session_state.pop(f"reg_{q}", None)
            st.toast(f"Saved {n} {'person' if n == 1 else 'people'}.", icon=":material/check_circle:")
            st.rerun()


def page_members():
    store = get_store()
    header("Members", "Your register and first-timers, stored in the database", store)
    if not gate(store):
        return
    demo_note(store)
    tab_list, tab_add, tab_import, tab_setup = st.tabs(["Register", "Add person", "Import CSV", "Setup"])

    with tab_list:
        register_editor(store)

    with tab_add:
        with st.form("member_add", clear_on_submit=True):
            a, b = st.columns(2)
            name = a.text_input("Full name")
            kind = b.selectbox("Type", ["member", "first_timer"], format_func=lambda k: k.replace("_", "-").title())
            phone, email = a.text_input("Phone"), b.text_input("Email")
            group, role = a.text_input("Group"), b.text_input("Ministry / role")
            if st.form_submit_button("Add", type="primary") and name.strip():
                store.upsert_members([dict(full_name=" ".join(name.split()), type=kind, phone=phone.strip(),
                                           email=email.strip(), group=group.strip(), role=role.strip(), status="",
                                           date_joined=today().isoformat() if kind == "member" else "",
                                           first_visit=today().isoformat() if kind == "first_timer" else "",
                                           created_at=now_iso())])
                st.success(f"Added {name.strip()}.")

    with tab_import:
        st.markdown("Upload the CSV exports of your **General Church Register** and **First timers** sheets "
                    "(File → Download → CSV in Google Sheets). Re-importing updates people instead of duplicating them.")
        a, b = st.columns(2)
        reg_file = a.file_uploader("General Church Register (.csv)", type="csv")
        ft_file = b.file_uploader("First timers (.csv)", type="csv")
        if reg_file or ft_file:
            reg = pd.read_csv(reg_file, dtype=str) if reg_file else None
            ft = pd.read_csv(ft_file, dtype=str) if ft_file else None
            rows, notes = parse_registers(reg, ft, store.list_members())
            new = sum("id" not in r for r in rows)
            st.write(f"**{len(rows)}** people found — {new} new, {len(rows) - new} updates.")
            for n in notes:
                st.caption("• " + n)
            st.dataframe(pd.DataFrame(rows).drop(columns=["id", "created_at"], errors="ignore"),
                         hide_index=True, width="stretch", height=300)
            if st.button("Import into the database", type="primary", icon=":material/cloud_upload:"):
                n = store.upsert_members(rows)
                st.success(f"Imported {n} people." + (" (Demo store — this resets when the app restarts.)" if store.demo else ""))

    with tab_setup:
        st.markdown(SETUP_GUIDE)


def _parse_times(values) -> pd.Series:
    t = pd.to_datetime(pd.Series(list(values), dtype="object"), errors="coerce", utc=True)
    return t.dt.tz_convert(TZ)


def _plot(fig, height=320, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44 if title else 10, b=10),
                      title=dict(text=title, font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None))
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def page_live():
    store = get_store()
    header("Live", "Who has arrived — updates on its own every few seconds, great on a screen during service",
           store, live=True)
    if not gate(store):
        return
    demo_note(store)
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    day = c1.date_input("Service date", value=today(), format="DD/MM/YYYY", key="live_day")
    every = c2.select_slider("Refresh every", options=[3, 5, 10, 30], value=5, format_func=lambda s: f"{s}s")
    date = day.isoformat()

    @st.fragment(run_every=every)
    def live_body():
        members = {m["id"]: m for m in store.list_members()}
        s = store.get_service(date) or {}
        present = s.get("present") or {}
        past = sorted([x for x in store.list_services() if x.get("date", "") < date], key=lambda x: x["date"])
        prev = past[-1] if past else None
        prev_n = len(prev.get("present") or {}) if prev else None
        ft_today = [mid for mid in present if members.get(mid, {}).get("type") == "first_timer"]
        new_today = [mid for mid in present if members.get(mid, {}).get("first_visit") == date]
        active = [m for m in members.values() if norm(m.get("status", "")) not in INACTIVE]

        k = st.columns(4)
        k[0].metric("Checked in", len(present),
                    f"{len(present) - prev_n:+d} vs {dt.date.fromisoformat(prev['date']):%d %b}" if prev else None,
                    border=True)
        k[1].metric("First-timers here", len(ft_today), f"{len(new_today)} first visit today" if new_today else None,
                    delta_color="off", border=True)
        k[2].metric("Of the register", f"{len(present) / max(len(active), 1):.0%}", f"{len(active)} people",
                    delta_color="off", border=True)
        k[3].metric("Last service", prev_n if prev else "–",
                    dt.date.fromisoformat(prev["date"]).strftime("%a %d %b") if prev else None,
                    delta_color="off", border=True)

        if not present:
            st.info(f"No one is checked in for {day:%A %d %B} yet. Ticks on the **Check-in** page appear here "
                    "within seconds.", icon=":material/hourglass_top:")
        else:
            ev = pd.DataFrame([dict(id=mid, name=members.get(mid, {}).get("full_name", "(removed)"),
                                    type=members.get(mid, {}).get("type", "member"), at=at)
                               for mid, at in present.items()])
            ev["time"] = _parse_times(ev["at"]).set_axis(ev.index)  # keep NZ time zone
            ev = ev.sort_values("time")
            c1, c2 = st.columns([3, 2])
            with c1:
                arr = ev.dropna(subset=["time"]).assign(n=1)
                if len(arr):
                    arr["arrived"] = arr["n"].cumsum()
                    fig = px.line(arr, x="time", y="arrived", line_shape="hv", markers=True,
                                  labels={"arrived": "Checked in", "time": ""})
                    fig.update_traces(line=dict(width=2, color=BRAND["teal"]), marker=dict(size=6),
                                      hovertemplate="%{x|%H:%M}: %{y} checked in<extra></extra>")
                    _plot(fig, 330, "Arrivals so far")
            with c2.container(border=True):
                st.markdown("**Latest arrivals**")
                latest = ev.sort_values("time", ascending=False).head(12)
                st.dataframe(pd.DataFrame({
                    "Time": latest["time"].dt.strftime("%H:%M").fillna(""),
                    "Name": latest["name"],
                    "": latest["type"].map({"first_timer": "✨ First-timer"}).fillna("")}),
                    hide_index=True, width="stretch", height=min(38 * (len(latest) + 1) + 4, 480))
        st.caption(f"Live · updated {dt.datetime.now(TZ):%H:%M:%S} · refreshing every {every}s")

    live_body()


def page_insights():
    store = get_store()
    header("Insights", "How attendance is trending — services, first-timers and groups", store)
    if not gate(store):
        return
    demo_note(store)
    members, services = store.list_members(), store.list_services()
    mem = {m["id"]: m for m in members}
    past = sorted([s for s in services if s.get("date", "") <= today().isoformat()], key=lambda s: s["date"])
    if not past:
        st.info("No services recorded yet. After a few Sundays of ticking on **Check-in**, trends appear here.")
        return

    rows = []
    for s in past:
        p = s.get("present") or {}
        rows.append(dict(date=pd.to_datetime(s["date"]), present=len(p),
                         first_timers=sum(1 for mid in p if mem.get(mid, {}).get("type") == "first_timer")))
    per = pd.DataFrame(rows)
    per["members"] = per.present - per.first_timers
    per["avg4"] = per.present.rolling(4, min_periods=1).mean()
    last, prev = per.iloc[-1], (per.iloc[-2] if len(per) > 1 else None)
    recent_ids = set().union(*[set((s.get("present") or {}).keys()) for s in past[-4:]])

    k = st.columns(4)
    k[0].metric("Services recorded", len(per), border=True)
    k[1].metric("Last service", int(last.present),
                f"{int(last.present - prev.present):+d} vs previous" if prev is not None else None, border=True)
    k[2].metric("Average (last 4)", f"{per.present.tail(4).mean():.0f}", border=True)
    k[3].metric("Active people", len(recent_ids), "came at least once in the last 4 services",
                delta_color="off", border=True)

    long = per.melt(id_vars="date", value_vars=["members", "first_timers"], var_name="who", value_name="n")
    long["who"] = long.who.map({"members": "Members", "first_timers": "First-timers"})
    fig = px.bar(long, x="date", y="n", color="who", barmode="stack",
                 color_discrete_map={"Members": BRAND["teal"], "First-timers": BRAND["navy"]},
                 category_orders={"who": ["Members", "First-timers"]}, labels={"n": "People", "date": ""})
    fig.update_traces(marker_line_width=0, hovertemplate="%{x|%d %b %Y}: %{y}<extra>%{fullData.name}</extra>")
    fig.add_scatter(x=per.date, y=per.avg4, mode="lines", name="4-service average",
                    line=dict(color=BRAND["slate"], width=2, dash="dot"), hovertemplate="%{y:.1f}<extra>4-service avg</extra>")
    fig.update_layout(bargap=0.25, hovermode="x unified")
    _plot(fig, 360, "Attendance per service")

    c1, c2 = st.columns(2)
    with c1:
        ft = pd.DataFrame([m for m in members if m.get("first_visit")])
        if ft.empty:
            with st.container(border=True):
                st.markdown("**First-timers per month**")
                st.caption("No first-visit dates yet.")
        else:
            ft["month"] = pd.to_datetime(ft.first_visit).dt.to_period("M").dt.to_timestamp()
            by_m = ft.groupby("month").size().rename("n").reset_index()
            fig = px.bar(by_m, x="month", y="n", labels={"n": "First-timers", "month": ""})
            fig.update_traces(marker_color=BRAND["teal"], hovertemplate="%{x|%b %Y}: %{y}<extra></extra>")
            fig.update_layout(bargap=0.3)
            _plot(fig, 300, "First-timers per month")
    with c2:
        fts = [m for m in members if m.get("type") == "first_timer" or m.get("first_visit")]
        came_back = 0
        for m in fts:
            dates = sorted(s["date"] for s in past if m["id"] in (s.get("present") or {}))
            first = m.get("first_visit") or (dates[0] if dates else None)
            if first and any(d > first for d in dates):
                came_back += 1
        with st.container(border=True):
            st.markdown("**Did first-timers come back?**")
            a, b = st.columns(2)
            a.metric("First-timers", len(fts))
            b.metric("Came back at least once", came_back,
                     f"{came_back / len(fts):.0%}" if fts else None, delta_color="off")
            st.caption("Counts anyone with a first-visit date or marked first-timer who was ticked at a later service.")

    grp = {}
    for s in past[-8:]:
        for mid in (s.get("present") or {}):
            g = (mem.get(mid, {}).get("group") or "").strip() or "(no group)"
            grp[g] = grp.get(g, 0) + 1
    if grp:
        gdf = pd.DataFrame({"group": list(grp), "avg": [v / len(past[-8:]) for v in grp.values()]}).sort_values("avg")
        fig = px.bar(gdf, x="avg", y="group", orientation="h", labels={"avg": "Average per service", "group": ""})
        fig.update_traces(marker_color=BRAND["teal"], hovertemplate="%{y}: %{x:.1f} per service<extra></extra>")
        _plot(fig, max(220, 36 * len(gdf) + 90), f"Attendance by group (last {len(past[-8:])} services)")


SETUP_GUIDE = """
**Connect a Postgres database** — Supabase (recommended: login, table editor, SQL editor) or Neon. Both have free plans.

**Supabase**
1. Sign up at **supabase.com** → *New project* → choose a region near you (e.g. Sydney) and a database password.
2. Click **Connect** (top of the project) → *Connection string* → **Session pooler** → copy the URI
   (it looks like `postgresql://postgres.xxxx:[YOUR-PASSWORD]@aws-0-….pooler.supabase.com:5432/postgres`)
   and put your database password in place of `[YOUR-PASSWORD]`.
   Free Supabase projects pause after 7 days with no activity — weekly check-ins keep it awake; if it pauses,
   click *Restore* in Supabase.

**Neon** (alternative)
1. Sign up at **neon.tech** (or *Vercel → Storage → Neon*) → create a project → copy the connection string.

**Then, on share.streamlit.io** → your app → ⋮ → *Settings → Secrets*, paste:

```toml
attendance_password = "choose-a-strong-password"
database_url = "postgresql://…your connection string…?sslmode=require"
```
Save — the app restarts, creates its three tables, and switches from demo to your database.
Then open **Members → Import CSV** and upload your two sheets.

Optional, for the SQL page: create a read-only database user and add its URI as `database_url_readonly`.
The SQL page already runs every query in a read-only transaction, so it cannot change data either way.

You can also query the same tables in Supabase's own **SQL editor**, or from Python on your laptop:

```python
import pandas as pd, psycopg
with psycopg.connect("postgresql://…") as conn:
    df = pd.read_sql("SELECT * FROM attendance", conn)
```

Never commit connection strings or your CSVs to GitHub — the repo is public. Both are blocked in `.gitignore`.
"""


EXAMPLES = {
    "Attendance per service": """SELECT s.service_date, s.name, COUNT(a.member_id) AS present
FROM services s
LEFT JOIN attendance a ON a.service_date = s.service_date
GROUP BY s.service_date, s.name
ORDER BY s.service_date DESC""",
    "Missed in a row (who to call)": """WITH last_seen AS (
  SELECT m.id, m.full_name, m.phone, MAX(a.service_date) AS last_seen
  FROM members m
  LEFT JOIN attendance a ON a.member_id = m.id
  GROUP BY m.id, m.full_name, m.phone
)
SELECT l.full_name, l.phone, l.last_seen,
       (SELECT COUNT(*) FROM services s
        WHERE s.service_date > COALESCE(l.last_seen, '1900-01-01')
          AND s.service_date <= CURRENT_DATE) AS missed_in_a_row
FROM last_seen l
ORDER BY missed_in_a_row DESC, l.full_name""",
    "Attendance rate per person": """SELECT m.full_name, m.group_name,
       COUNT(a.member_id) AS attended,
       (SELECT COUNT(*) FROM services) AS services,
       ROUND(100.0 * COUNT(a.member_id) / NULLIF((SELECT COUNT(*) FROM services), 0), 1) AS rate_pct
FROM members m
LEFT JOIN attendance a ON a.member_id = m.id
GROUP BY m.id, m.full_name, m.group_name
ORDER BY rate_pct DESC""",
    "First-timers: did they come back?": """SELECT m.full_name, m.first_visit, m.invited_by,
       COUNT(a.member_id) AS visits, MAX(a.service_date) AS last_seen
FROM members m
LEFT JOIN attendance a ON a.member_id = m.id
WHERE m.type = 'first_timer'
GROUP BY m.id, m.full_name, m.first_visit, m.invited_by
ORDER BY m.first_visit DESC""",
    "Attendance by group": """SELECT COALESCE(NULLIF(m.group_name, ''), '(no group)') AS grp,
       COUNT(DISTINCT m.id) AS people, COUNT(a.member_id) AS check_ins
FROM members m
LEFT JOIN attendance a ON a.member_id = m.id
GROUP BY 1
ORDER BY check_ins DESC""",
    "Who invited the most first-timers": """SELECT invited_by, COUNT(*) AS first_timers
FROM members
WHERE invited_by IS NOT NULL AND invited_by <> ''
GROUP BY invited_by
ORDER BY first_timers DESC""",
}


def page_sql():
    store = get_store()
    header("SQL", "Ask the database anything — read-only, so nothing can be changed from here", store)
    if not gate(store):
        return
    demo_note(store)
    mode, _ = layout_prefs()
    left, right = (st.container(), st.container()) if mode == "Stacked" else st.columns([2, 5])
    with left.container(border=True):
        st.markdown("**Tables**")
        st.code("members\n  id, full_name, phone,\n  email, group_name, role,\n  status, type,\n  date_joined, first_visit,\n"
                "  invited_by, follow_up\n\nservices\n  service_date, name\n\nattendance\n"
                "  service_date,\n  member_id, checked_at", language=None)
        pick = st.selectbox("Example queries", list(EXAMPLES), index=None, placeholder="Pick an example…")
        if pick and st.session_state.get("sql_pick") != pick:
            st.session_state.sql_pick = pick
            st.session_state.sql_text = EXAMPLES[pick]
    with right:
        st.session_state.setdefault("sql_text", EXAMPLES["Missed in a row (who to call)"])
        sql = st.text_area("SQL", key="sql_text", height=230, label_visibility="collapsed")
        run = st.button("Run query", type="primary", icon=":material/play_arrow:")
        if run or "sql_result" not in st.session_state:
            try:
                t0 = dt.datetime.now()
                df = store.run_query(sql)
                st.session_state.sql_result = (df, (dt.datetime.now() - t0).total_seconds(), None)
            except Exception as e:  # show the database's own error message
                st.session_state.sql_result = (None, 0, str(e).strip().splitlines()[0][:400])
        df, secs, err = st.session_state.sql_result
        if err:
            st.error(err, icon=":material/error:")
        elif df is not None:
            with st.container(border=True):
                st.caption(f"{len(df):,} rows · {secs * 1000:.0f} ms" + (" · first 5,000 shown" if len(df) >= 5000 else ""))
                st.dataframe(df, hide_index=True, width="stretch", height=min(38 * (len(df) + 1) + 4, 520))
                st.download_button("Download results (CSV)", df.to_csv(index=False), "query_results.csv", "text/csv",
                                   icon=":material/download:")
