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
import re
import threading
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TZ = ZoneInfo("Pacific/Auckland")
BRAND = dict(navy="#2c4b77", teal="#208088", green="#2aa686", slate="#293641", gold="#ffcf00")  # from the FCC logo
YELLOW_AT, RED_AT = 3, 5          # services missed in a row
INACTIVE = {"inactive", "moved", "left", "deceased", "transferred"}
AMBER, CRIMSON = "#fab219", "#d03b3b"   # reserved status colours (always shown with icon + label)
# Chart colours, validated for colour-blind separation and contrast on the dark surface (#141c22).
SERIES = dict(members="#2aa686", first_timers="#5a8ef0")
STATUS = dict(ok="#2aa686", yellow=AMBER, red=CRIMSON)
INK = dict(primary="#e8eef2", secondary="#9fb0bd", muted="#6b7c89", grid="rgba(255,255,255,0.06)")


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
# Sign-ups from the public welcome form. On Postgres this table is created by the setup SQL (Members → Setup),
# because it needs Row Level Security and a UUID default; here it only exists for the SQLite demo.
DEMO_REGISTRATIONS = """CREATE TABLE IF NOT EXISTS registrations (
    id TEXT PRIMARY KEY, full_name TEXT, phone TEXT, email TEXT, invited_by TEXT, first_visit DATE, notes TEXT,
    wants_contact BOOLEAN DEFAULT TRUE, status TEXT DEFAULT 'pending', created_at TEXT, member_id TEXT)"""
REG_COLS = ["full_name", "phone", "email", "invited_by", "first_visit", "notes", "wants_contact", "status",
            "created_at", "member_id"]


def _txt(v) -> str:
    """Normalise DB values (Postgres returns date objects, SQLite returns strings) to plain strings."""
    if v is None:
        return ""
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


class DbUnavailable(Exception):
    """The database can't be reached right now. Carries a plain-English reason for the page to show."""


def _friendly(err: Exception) -> str:
    msg = str(err).lower()
    if "password authentication failed" in msg:
        return ("The database refused the password. Check `database_url` in the app's Secrets — "
                "it must contain the current Supabase database password.")
    if "circuit" in msg or "too many" in msg or "max client" in msg:
        return ("The database has paused logins for a moment after repeated failed attempts. "
                "Wait a few minutes, check the password in Secrets, then try again.")
    if "timeout" in msg or "timed out" in msg or "could not translate host" in msg or "resolve" in msg:
        return "Couldn't reach the database server. It may be paused or the internet connection dropped."
    return "The database isn't responding right now."


class SqlStore:
    """One store for both engines. Queries use standard SQL that runs unchanged on SQLite and Postgres.

    If Postgres can't be reached, the store backs off (30 s, doubling up to 5 min) and raises DbUnavailable
    straight away during that time — so auto-refreshing pages don't hammer the server and get locked out.
    """

    def __init__(self, url: str | None):
        self.demo = not url
        self.url = url
        self.engine = "sqlite" if self.demo else "postgres"
        self.path = "/tmp/fcc_attendance_demo.db"
        self._lock = threading.RLock()
        self._conn = None
        self._cache = {}
        self._schema_ready = False
        self._down_until = 0.0
        self._backoff = 0.0
        self.last_error = ""
        if self.demo:
            import os
            if os.path.exists(self.path):
                os.remove(self.path)  # fresh demo on every app start
            self._ensure_schema()
            _seed_demo(self)

    def _ensure_schema(self):
        if not self._schema_ready:
            for stmt in SCHEMA + ([DEMO_REGISTRATIONS] if self.demo else []):
                self._raw(stmt)
            self._schema_ready = True

    def retry_in(self) -> int:
        """Seconds until the next connection attempt is allowed (0 = now)."""
        return max(0, int(self._down_until - dt.datetime.now().timestamp()) + 1) if self._down_until else 0

    def retry_now(self):
        self._down_until = 0.0

    def _mark_down(self, err: Exception):
        self._backoff = min(max(self._backoff * 2, 30.0), 300.0)
        self._down_until = dt.datetime.now().timestamp() + self._backoff
        self.last_error = _friendly(err)
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _check_up(self):
        if self._down_until and dt.datetime.now().timestamp() < self._down_until:
            raise DbUnavailable(self.last_error)

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

    def _raw(self, sql, params=(), many=False, fetch=False):
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

    def _exec(self, sql, params=(), many=False, fetch=False):
        if self.demo:
            with self._lock:
                return self._raw(sql, params, many, fetch)
        import psycopg
        with self._lock:
            self._check_up()
            for attempt in (1, 2):  # one quiet reconnect if the server dropped an idle connection
                try:
                    self._ensure_schema()
                    out = self._raw(sql, params, many, fetch)
                    self._backoff, self._down_until = 0.0, 0.0
                    return out
                except psycopg.OperationalError as e:  # connection-level problem (not a bad query)
                    stale = self._conn is not None and attempt == 1  # an old connection went away: reconnect once
                    if not stale:
                        self._mark_down(e)
                        raise DbUnavailable(self.last_error) from e
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                except psycopg.InterfaceError as e:  # connection already closed
                    self._conn = None
                    if attempt == 2:
                        self._mark_down(e)
                        raise DbUnavailable(self.last_error) from e

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

    # -- sign-ups from the welcome form
    def list_registrations(self, status: str = "pending") -> list[dict] | None:
        """Sign-ups with this status, oldest first. None if the registrations table isn't set up yet."""
        cols = ", ".join(["id"] + REG_COLS)
        try:
            rows = self._exec(f"SELECT {cols} FROM registrations WHERE status = ? ORDER BY created_at", (status,),
                              fetch=True)
        except DbUnavailable:
            raise
        except Exception as e:  # table or a column missing → the setup SQL hasn't been run
            self.last_setup_error = str(e).strip().splitlines()[0][:300]
            return None
        out = []
        for r in rows:
            d = {k: _txt(v) for k, v in r.items()}
            d["wants_contact"] = bool(r.get("wants_contact")) if r.get("wants_contact") is not None else True
            out.append(d)
        return out

    def count_pending(self) -> int:
        regs = self._cached("pending", 30, lambda: self.list_registrations("pending"))
        return len(regs or [])

    def resolve_registration(self, reg_id: str, status: str, member_id: str | None = None):
        self._exec("UPDATE registrations SET status = ?, member_id = ? WHERE CAST(id AS TEXT) = ?",
                   (status, member_id, str(reg_id)))
        self._cache.pop("pending", None)

    def approve_registration(self, reg: dict, match_id: str | None = None, check_in: bool = True) -> str:
        """Add the sign-up to the register (or fill gaps on the matched person), optionally tick them present."""
        visit = reg.get("first_visit") or today().isoformat()
        fields = dict(full_name=" ".join(reg["full_name"].split()), phone=reg.get("phone", "").strip(),
                      email=reg.get("email", "").strip(), invited_by=reg.get("invited_by", "").strip())
        if match_id:  # existing person: only fill in what the form provided, never blank anything
            self.upsert_members([dict(id=match_id, **{k: v for k, v in fields.items() if v and k != "full_name"})])
            mid = match_id
        else:
            mid = new_id()
            note = reg.get("notes", "").strip()
            follow = "Welcome form" + ("" if reg.get("wants_contact", True) else " · prefers no contact")
            self.upsert_members([dict(id=mid, **fields, type="first_timer", status="", first_visit=visit,
                                      follow_up=follow + (f" · {note}" if note else ""), created_at=now_iso())])
        if check_in:
            self.set_present(visit, mid, True)
        self.resolve_registration(reg["id"], "approved", mid)
        return mid

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
        self._check_up()
        try:
            conn = psycopg.connect(self.readonly_url or self.url, connect_timeout=10)
        except psycopg.OperationalError as e:
            self._mark_down(e)
            raise DbUnavailable(self.last_error) from e
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")         # the database itself refuses any write
                cur.execute("SET LOCAL statement_timeout = '15s'")
                cur.execute(sql)
                cols = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(limit)
            conn.rollback()
        return pd.DataFrame(rows, columns=cols)

    readonly_url = None
    last_setup_error = ""


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
    regs = [("Grace Mensah", "022 481 2290", "", "Esi Boateng", "Loved the worship — would like to join the choir", True),
            ("Daniel Owusu", "", "daniel.o@example.com", "Instagram", "", False)]
    store._exec("INSERT INTO registrations (id, full_name, phone, email, invited_by, first_visit, notes, wants_contact, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                [(new_id(), n, ph, em, inv, sundays[-1].isoformat(), note, wc, now_iso()) for n, ph, em, inv, note, wc in regs],
                many=True)


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


def show_db_down(err: Exception, key: str = "page"):
    """Friendly 'can't reach the database' panel instead of a traceback, with a manual retry."""
    store = get_store()
    wait = store.retry_in()
    with st.container(border=True):
        st.error(f"**Can't connect to the database.** {err}", icon=":material/cloud_off:")
        st.caption("To avoid getting locked out, the app waits before trying again"
                   + (f" (next automatic try in about {wait} s)." if wait else ".")
                   + " Ticks and edits made while offline are not saved.")
        if st.button("Try again now", icon=":material/refresh:", key=f"db_retry_{key}"):
            store.retry_now()
            st.rerun()


def db_safe(fn):
    """Wrap a page or live fragment so a database outage shows a message, not a crash."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except DbUnavailable as e:
            show_db_down(e, fn.__name__)
    return run


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
    chips = ["Demo data — invented names"] if store.demo else []
    st.html(hero_html("Favourite Child Church · Attendance", title, subtitle, chips, live))


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
def checkin_panel(store):
    """Ushers tick people as they arrive; ticks from every phone sync within seconds."""
    _, per_row = layout_prefs(names=True)
    c1, c2, c3 = st.columns([1, 2, 2], vertical_alignment="bottom")
    day = c1.date_input("Service date", value=today(), format="DD/MM/YYYY")
    svc_name = c2.text_input("Service", value="Sunday Service")
    q = c3.text_input("Find a person", placeholder="Type a name…", key="ci_q")
    date = day.isoformat()

    members = sorted(store.list_members(), key=lambda m: norm(m.get("full_name", "")))
    members = [m for m in members if norm(m.get("status", "")) not in INACTIVE]

    def on_tick(mid):
        try:
            store.set_present(date, mid, bool(st.session_state[f"ci_{date}_{mid}"]), svc_name)
        except DbUnavailable:
            st.toast("Not saved — can't reach the database right now.", icon=":material/cloud_off:")

    @st.fragment(run_every=5)
    @db_safe
    def live_list():
        s = store.get_service(date) or {}
        present = s.get("present") or {}
        shown = [m for m in members if not q or norm(q) in norm(m.get("full_name", ""))]
        k = st.columns(3)
        k[0].metric("Checked in", f"{len(present)}", border=True)
        k[1].metric("Not yet", f"{max(len(members) - len(present), 0)}", border=True)
        k[2].metric("First-timers today", f"{sum(1 for m in members if m['id'] in present and m.get('type') == 'first_timer')}",
                    border=True)
        with card("ci_list"):
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


# ---------------------------------------------------------------- dashboard (home)
def _esc(v) -> str:
    import html
    return html.escape(str(v or ""))


def _delta(n: float | None, unit: str = "", suffix: str = "vs previous") -> str:
    """Up/down arrow + change. Arrow carries the colour; the words stay in text ink."""
    if n is None:
        return f'<span class="kpi-sub">{_esc(suffix)}</span>'
    arrow, cls = ("▲", "up") if n > 0 else ("▼", "down") if n < 0 else ("■", "flat")
    val = f"{abs(n):.0f}{unit}" if isinstance(n, (int, float)) else _esc(n)
    return f'<span class="kpi-delta {cls}"><i>{arrow}</i> {val}</span> <span class="kpi-sub">{_esc(suffix)}</span>'


def kpi_row(items: list[dict]) -> str:
    cells = "".join(
        f'<div class="kpi"><div class="kpi-top"><span class="kpi-icon">{it.get("icon", "")}</span>'
        f'<span class="kpi-label">{_esc(it["label"])}</span></div>'
        f'<div class="kpi-value">{it["value"]}</div><div class="kpi-foot">{it.get("foot", "")}</div></div>'
        for it in items)
    return f'<div class="kpi-grid">{cells}</div>'


def _ago(iso: str) -> str:
    t = pd.to_datetime(iso, errors="coerce", utc=True)
    if pd.isna(t):
        return ""
    mins = (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 60
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins:.0f} min ago"
    if mins < 60 * 24:
        return f"{mins / 60:.0f} h ago"
    return t.tz_convert(TZ).strftime("%a %d %b")


def feed(title: str, rows: list[tuple[str, str, str]], empty: str) -> str:
    """rows: (icon, main text (already escaped/HTML), small muted text)."""
    body = "".join(f'<li><span class="feed-ic">{ic}</span><div><div>{main}</div><small>{_esc(sub)}</small></div></li>'
                   for ic, main, sub in rows) or f'<li class="feed-empty">{_esc(empty)}</li>'
    return f'<div class="feed"><h4>{_esc(title)}</h4><ul>{body}</ul></div>'


@db_safe
def page_dashboard():
    store = get_store()
    header("Dashboard", "Attendance at a glance — updates itself every 30 seconds", store, live=True)
    if not gate(store):
        return
    demo_note(store)

    @st.fragment(run_every=30)
    @db_safe
    def body():
        members, services = store.list_members(), store.list_services()
        mem = {m["id"]: m for m in members}
        tday = today().isoformat()
        past = sorted([x for x in services if x.get("date", "") <= tday], key=lambda x: x["date"])
        df = missed_streaks(members, services)
        pending = store.count_pending()
        if not past:
            st.info("No services recorded yet — tick people on **Follow-up & Check-in → Check-in** to get started.")
            return

        counts = [len(x.get("present") or {}) for x in past]
        last, prev = past[-1], (past[-2] if len(past) > 1 else None)
        n_last, n_prev = counts[-1], (counts[-2] if len(counts) > 1 else None)
        avg4 = sum(counts[-4:]) / len(counts[-4:])
        prev4 = counts[-8:-4]
        red = int((df.level == "red").sum()) if not df.empty else 0
        yellow = int((df.level == "yellow").sum()) if not df.empty else 0
        ok = int((df.level == "ok").sum()) if not df.empty else 0
        month = tday[:7]
        ft_month = [m for m in members if (m.get("first_visit") or "").startswith(month)]
        last_day = dt.date.fromisoformat(last["date"])

        st.html(kpi_row([
            dict(icon="👥", label=f"Last service · {last_day:%a %d %b}", value=n_last,
                 foot=_delta(n_last - n_prev if n_prev is not None else None, "",
                             f"vs {dt.date.fromisoformat(prev['date']):%d %b}" if prev else "first service")),
            dict(icon="📈", label="Average · last 4 services", value=f"{avg4:.0f}",
                 foot=_delta(avg4 - sum(prev4) / len(prev4) if prev4 else None, "", "vs the 4 before")),
            dict(icon="🔔", label="Need a follow-up call", value=red + yellow,
                 foot=f'<span class="pill red">● {red} red</span> <span class="pill amber">● {yellow} yellow</span>'),
            dict(icon="✨", label=f"First-timers · {today():%B}", value=len(ft_month),
                 foot=(f'<span class="pill blue">{pending} sign-up{"s" if pending != 1 else ""} to approve</span>'
                       if pending else '<span class="kpi-sub">no sign-ups waiting</span>')),
        ]))

        left, right = st.columns([2.2, 1], gap="medium")
        with left:
            a, b = st.columns([1, 1.35], gap="medium")
            with a:  # donut: where everyone on the register stands
                fig = go.Figure(go.Pie(
                    labels=["On track", "Yellow", "Red"], values=[ok, yellow, red], hole=0.72, sort=False,
                    marker=dict(colors=[STATUS["ok"], STATUS["yellow"], STATUS["red"]],
                                line=dict(color="#141c22", width=2)),
                    textinfo="none", hovertemplate="%{label}: %{value} people (%{percent})<extra></extra>"))
                fig.add_annotation(text=f"<b style='font-size:30px;color:{INK['primary']}'>{ok + yellow + red}</b>"
                                        f"<br><span style='color:{INK['secondary']}'>on the register</span>",
                                   showarrow=False, x=0.5, y=0.5)
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
                _plot(fig, 330, "Where everyone stands", key="dash_donut")
            with b:  # trend: people present per service, members vs first-timers
                rows = []
                for x in past[-12:]:
                    p = x.get("present") or {}
                    ftn = sum(1 for mid in p if mem.get(mid, {}).get("type") == "first_timer")
                    rows.append(dict(date=pd.to_datetime(x["date"]), members=len(p) - ftn, first_timers=ftn))
                tr = pd.DataFrame(rows)
                fig = go.Figure()
                fig.add_scatter(x=tr.date, y=tr.members, name="Members", mode="lines", stackgroup="one",
                                line=dict(color=SERIES["members"], width=2), fillcolor="rgba(42,166,134,0.35)",
                                hovertemplate="%{y} members<extra></extra>")
                fig.add_scatter(x=tr.date, y=tr.first_timers, name="First-timers", mode="lines", stackgroup="one",
                                line=dict(color=SERIES["first_timers"], width=2), fillcolor="rgba(90,142,240,0.35)",
                                hovertemplate="%{y} first-timers<extra></extra>")
                fig.update_layout(hovermode="x unified")
                fig.update_xaxes(tickformat="%d %b")
                _plot(fig, 330, f"People present · last {len(tr)} services", key="dash_trend")

            with card("dash_followup"):
                st.markdown(f"**Needs a follow-up call** · {red + yellow} people")
                need = df[df.level != "ok"].head(8) if not df.empty else df
                if need.empty:
                    st.caption("Nobody has missed 3 or more services in a row. 🎉")
                else:
                    trs = "".join(
                        f'<tr><td><span class="dot {r.level}"></span>{_esc(r.name)}</td>'
                        f'<td><span class="pill {"red" if r.level == "red" else "amber"}">'
                        f'{"Red" if r.level == "red" else "Yellow"} · {r.missed} missed</span></td>'
                        f'<td>{_esc(pd.to_datetime(r.last_seen).strftime("%d %b") if r.last_seen else "Not yet")}</td>'
                        f'<td class="muted">{_esc(r.phone) or "—"}</td></tr>' for r in need.itertuples())
                    st.html(f'<table class="dash-table"><thead><tr><th>Name</th><th>Status</th><th>Last seen</th>'
                            f'<th>Phone</th></tr></thead><tbody>{trs}</tbody></table>')
                    if red + yellow > len(need):
                        st.caption(f"+ {red + yellow - len(need)} more on the Follow-up page.")

        with right, card("dash_side"):
            today_svc = store.get_service(tday) or {}
            here = today_svc.get("present") or {}
            notes = []
            if pending:
                notes.append(("📝", f"<b>{pending}</b> welcome-form sign-up{'s' if pending != 1 else ''} to approve",
                              "Members → Sign-ups"))
            if red:
                notes.append(("🔴", f"<b>{red}</b> {'person has' if red == 1 else 'people have'} missed {RED_AT}+ in a row",
                              "Follow-up"))
            notes.append(("✅", f"<b>{len(here)}</b> checked in today" if here else "No check-ins yet today",
                          f"{today():%A %d %B}"))
            src = here or (last.get("present") or {})
            arrivals = sorted(src.items(), key=lambda kv: kv[1] or "", reverse=True)[:6]
            act = [("✨" if mem.get(mid, {}).get("type") == "first_timer" else "🙋",
                    _esc(mem.get(mid, {}).get("full_name", "(removed)")), _ago(at)) for mid, at in arrivals]
            call = [] if df.empty else [
                ("📞", f"{_esc(r.name)}", f"{r.phone or 'no phone'} · {r.missed} missed")
                for r in df[df.level == "red"].head(5).itertuples()]
            st.html(feed("Notifications", notes, "All caught up")
                    + feed("Latest check-ins" + ("" if here else f" · {last_day:%d %b}"), act, "No check-ins yet")
                    + feed("Call next", call, "No one in red — great!"))
        st.caption(f"Updated {dt.datetime.now(TZ):%H:%M:%S}")

    body()


@db_safe
def page_followup():
    store = get_store()
    header("Follow-up & Check-in", f"Tick people in on the day, and see who we haven't seen — {RED_AT}+ services "
                                   f"missed in a row is red, {YELLOW_AT}–{RED_AT - 1} is yellow", store, live=True)
    if not gate(store):
        return
    demo_note(store)
    tab_fu, tab_ci = st.tabs([":material/notification_important: Needs follow-up", ":material/how_to_reg: Check-in"])
    with tab_fu:
        followup_panel(store)
    with tab_ci:
        checkin_panel(store)


def followup_panel(store):
    show = st.segmented_control("Show", ["Needs follow-up", "Red only", "Yellow only", "Everyone"],
                                default="Needs follow-up", key="fu_show") or "Needs follow-up"

    @st.fragment(run_every=30)
    @db_safe
    def live_followup():
        members, services = store.list_members(), store.list_services()
        df = missed_streaks(members, services)
        past = sorted([s for s in services if s.get("date", "") <= today().isoformat()], key=lambda s: s["date"])
        if df.empty or not past:
            st.info("No services recorded yet. Tick people on the **Check-in** tab and this list fills itself in.")
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
        with card("fu_list"):
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
        fig.update_traces(marker_color=SERIES["members"], hovertemplate="%{x|%d %b %Y}: %{y} present<extra></extra>")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), bargap=0.25,
                          title=dict(text="Attendance per service", font=dict(size=15)))
        with card("fu_trend"):
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"Live · recalculated {dt.datetime.now(TZ):%H:%M:%S} · only services since someone joined "
                   "or first visited count against them.")

    live_followup()


EDIT_COLS = ["full_name", "type", "phone", "email", "group", "role", "status", "date_joined", "first_visit",
             "invited_by", "follow_up"]
STATUSES = ["Active", "Inactive", "Moved", "Left", "Transferred", "Deceased"]


def _as_date(v) -> str:
    """Stored ISO date → 'DD/MM/YYYY' for the editor ('' when empty, so the cell shows blank, not None)."""
    d = pd.to_datetime(v, errors="coerce")
    return "" if pd.isna(d) else d.strftime("%d/%m/%Y")


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
    m["status"] = m["status"].map(lambda v: v or "Active")  # blank status means active
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
            "date_joined": st.column_config.TextColumn("Joined", help="DD/MM/YYYY", max_chars=10),
            "first_visit": st.column_config.TextColumn("First visit", help="DD/MM/YYYY", max_chars=10),
            "invited_by": st.column_config.TextColumn("Invited by"),
            "follow_up": st.column_config.TextColumn("Follow-up notes"),
        })
    changes, bad_dates = {}, []
    for mid in edited.index:
        diff = {}
        for c in EDIT_COLS:
            old, new = view.at[mid, c], edited.at[mid, c]
            old = "" if old is None or (not isinstance(old, str) and pd.isna(old)) else str(old).strip()
            new = "" if new is None or (not isinstance(new, str) and pd.isna(new)) else str(new).strip()
            if c == "status":
                old, new = ("" if v == "Active" else v for v in (old, new))
            if c in ("date_joined", "first_visit"):
                old = _date(old)
                parsed = _date(new)
                if new and not parsed:
                    bad_dates.append(f"{edited.at[mid, 'full_name']}: “{new}”")
                    continue
                new = parsed
            if old != new:
                diff[c] = new
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
        elif bad_dates:
            st.error("Dates need to look like 25/12/2025 — check: " + "; ".join(bad_dates[:5]))
        else:
            n = store.update_members(changes)
            st.session_state.pop(f"reg_{q}", None)
            st.toast(f"Saved {n} {'person' if n == 1 else 'people'}.", icon=":material/check_circle:")
            st.rerun()


@db_safe
def page_members():
    store = get_store()
    header("Members", "Your register and first-timers, stored in the database", store)
    if not gate(store):
        return
    demo_note(store)
    pending = store.count_pending()
    tab_list, tab_signups, tab_add, tab_import, tab_setup = st.tabs(
        ["Register", f"Sign-ups ({pending})" if pending else "Sign-ups", "Add person", "Import CSV", "Setup"])

    with tab_list:
        register_editor(store)

    with tab_signups:
        signups(store)

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


def signups(store):
    """Approve or reject sign-ups that came in from the public welcome form."""
    regs = store.list_registrations("pending")
    if regs is None:
        st.warning("The sign-ups table isn't set up yet (or is missing a column). Open Supabase → **SQL editor**, "
                   "paste the SQL below and click **Run**.", icon=":material/construction:")
        if store.last_setup_error:
            st.caption(f"Database said: {store.last_setup_error}")
        st.code(REGISTRATIONS_SQL, language="sql")
        return
    st.caption("People who filled in the welcome form. **Approve** adds them to the register as a first-timer "
               "(or updates the person with the same name) — nothing reaches the register until you do.")
    if not regs:
        st.success("No sign-ups waiting.", icon=":material/done_all:")
        return
    by_name = {norm(m["full_name"]): m for m in store.list_members()}
    for r in regs:
        match = by_name.get(norm(r["full_name"]))
        with card(f"signup_{r['id']}"):
            top = st.columns([3, 2], vertical_alignment="center")
            submitted = _parse_times([r["created_at"]]).iloc[0]
            top[0].markdown(f"**{r['full_name']}**" + (" · :orange[prefers no contact]" if not r["wants_contact"] else ""))
            top[1].caption(f"Sent {submitted:%a %d %b, %H:%M}" if pd.notna(submitted) else "")
            info = [("Phone", r["phone"]), ("Email", r["email"]), ("Invited by / heard via", r["invited_by"]),
                    ("First visit", r["first_visit"])]
            st.markdown("  \n".join(f"{k}: {v}" for k, v in info if v) or "_No contact details given._")
            if r["notes"]:
                st.info(r["notes"], icon=":material/chat:")
            if match:
                st.caption(f":material/link: Already on the register as **{match['full_name']}** "
                           f"({match.get('type', '').replace('_', '-') or 'member'}) — approving updates that person.")
            a, b, c = st.columns([2, 1, 1], vertical_alignment="center")
            tick = a.checkbox(f"Mark present on {r['first_visit'] or 'today'}", value=True, key=f"reg_ci_{r['id']}")
            if b.button("Approve", key=f"reg_ok_{r['id']}", type="primary", icon=":material/check:", width="stretch"):
                store.approve_registration(r, match["id"] if match else None, check_in=tick)
                st.toast(f"{r['full_name']} added to the register.", icon=":material/person_add:")
                st.rerun()
            if c.button("Reject", key=f"reg_no_{r['id']}", icon=":material/close:", width="stretch"):
                store.resolve_registration(r["id"], "rejected")
                st.toast(f"Sign-up from {r['full_name']} rejected.")
                st.rerun()


def _parse_times(values) -> pd.Series:
    t = pd.to_datetime(pd.Series(list(values), dtype="object"), errors="coerce", utc=True)
    return t.dt.tz_convert(TZ)


def _style(fig, height=320, title=None):
    """Dark card styling shared by every chart: transparent background, recessive grid, readable ink."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=44 if title else 8, b=8),
                      title=dict(text=title, font=dict(size=15, color=INK["primary"])) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK["secondary"]),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None),
                      hoverlabel=dict(bgcolor="#1d2830", bordercolor="#2c3b46", font=dict(color=INK["primary"])))
    fig.update_xaxes(showgrid=False, linecolor=INK["grid"], tickfont=dict(color=INK["secondary"]))
    fig.update_yaxes(gridcolor=INK["grid"], zeroline=False, tickfont=dict(color=INK["secondary"]))
    return fig


def card(name: str):
    """A rounded dark card (bordered container), styled in app.py via its st-key-card_* class."""
    return st.container(border=True, key="card_" + re.sub(r"\W+", "_", name.lower()).strip("_"))


def _plot(fig, height=320, title=None, key=None):
    _style(fig, height, title)
    with card(key or title or "chart"):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@db_safe
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
    @db_safe
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
                    fig.update_traces(line=dict(width=2, color=SERIES["members"]), marker=dict(size=8),
                                      hovertemplate="%{x|%H:%M}: %{y} checked in<extra></extra>")
                    _plot(fig, 330, "Arrivals so far")
            with c2, card("live_latest"):
                st.markdown("**Latest arrivals**")
                latest = ev.sort_values("time", ascending=False).head(12)
                st.dataframe(pd.DataFrame({
                    "Time": latest["time"].dt.strftime("%H:%M").fillna(""),
                    "Name": latest["name"],
                    "": latest["type"].map({"first_timer": "✨ First-timer"}).fillna("")}),
                    hide_index=True, width="stretch", height=min(38 * (len(latest) + 1) + 4, 480))
        st.caption(f"Live · updated {dt.datetime.now(TZ):%H:%M:%S} · refreshing every {every}s")

    live_body()


@db_safe
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
                 color_discrete_map={"Members": SERIES["members"], "First-timers": SERIES["first_timers"]},
                 category_orders={"who": ["Members", "First-timers"]}, labels={"n": "People", "date": ""})
    fig.update_traces(marker_line_width=0, hovertemplate="%{x|%d %b %Y}: %{y}<extra>%{fullData.name}</extra>")
    fig.add_scatter(x=per.date, y=per.avg4, mode="lines", name="4-service average",
                    line=dict(color=INK["secondary"], width=2, dash="dot"), hovertemplate="%{y:.1f}<extra>4-service avg</extra>")
    fig.update_layout(bargap=0.25, hovermode="x unified")
    _plot(fig, 360, "Attendance per service")

    c1, c2 = st.columns(2)
    with c1:
        ft = pd.DataFrame([m for m in members if m.get("first_visit")])
        if ft.empty:
            with card("ins_ft_empty"):
                st.markdown("**First-timers per month**")
                st.caption("No first-visit dates yet.")
        else:
            ft["month"] = pd.to_datetime(ft.first_visit).dt.to_period("M").dt.to_timestamp()
            by_m = ft.groupby("month").size().rename("n").reset_index()
            fig = px.bar(by_m, x="month", y="n", labels={"n": "First-timers", "month": ""})
            fig.update_traces(marker_color=SERIES["first_timers"], hovertemplate="%{x|%b %Y}: %{y}<extra></extra>")
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
        with card("ins_ft_back"):
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
        fig.update_traces(marker_color=SERIES["members"], hovertemplate="%{y}: %{x:.1f} per service<extra></extra>")
        _plot(fig, max(220, 36 * len(gdf) + 90), f"Attendance by group (last {len(past[-8:])} services)")


REGISTRATIONS_SQL = """-- Sign-ups from the FCC welcome form. Safe to run more than once.
create table if not exists registrations (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now()
);
alter table registrations
  add column if not exists full_name     text,
  add column if not exists phone         text,
  add column if not exists email         text,
  add column if not exists invited_by    text,
  add column if not exists first_visit   date,
  add column if not exists notes         text,
  add column if not exists wants_contact boolean default true,
  add column if not exists status        text default 'pending',
  add column if not exists member_id     text,
  add column if not exists created_at    timestamptz default now();

alter table registrations enable row level security;

-- The public form can only ADD pending sign-ups. It can't read, change or delete anything.
drop policy if exists "form can submit" on registrations;
create policy "form can submit" on registrations
  for insert to anon
  with check (status = 'pending' and length(trim(full_name)) between 2 and 120);
grant insert on registrations to anon;"""


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

**Welcome form sign-ups** — run the SQL in *Members → Sign-ups* once in Supabase's SQL editor. It creates the
`registrations` table the public form writes to, locked down so the form can only add new sign-ups.

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


@db_safe
def page_sql():
    store = get_store()
    header("SQL", "Ask the database anything — read-only, so nothing can be changed from here", store)
    if not gate(store):
        return
    demo_note(store)
    mode, _ = layout_prefs()
    left, right = (st.container(), st.container()) if mode == "Stacked" else st.columns([2, 5])
    with left, card("sql_tables"):
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
            with card("sql_result"):
                st.caption(f"{len(df):,} rows · {secs * 1000:.0f} ms" + (" · first 5,000 shown" if len(df) >= 5000 else ""))
                st.dataframe(df, hide_index=True, width="stretch", height=min(38 * (len(df) + 1) + 4, 520))
                st.download_button("Download results (CSV)", df.to_csv(index=False), "query_results.csv", "text/csv",
                                   icon=":material/download:")
