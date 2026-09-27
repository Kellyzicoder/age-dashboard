"""Attendance: live check-in, missed-service follow-up, and member register.

Data lives in Google Cloud Firestore when `gcp_service_account` is set in Streamlit secrets;
otherwise a shared in-memory demo store with invented names is used. Real member data is never
stored in this repository.

Firestore layout (small on purpose, so live polling stays within the free tier):
  members/{member_id}   full_name, phone, email, group, role, status, type ("member"|"first_timer"),
                        date_joined, first_visit, invited_by, follow_up, created_at
  services/{YYYY-MM-DD} name, date, present: {member_id: ISO timestamp}   ← one read per live refresh
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


# ---------------------------------------------------------------- storage backends
class DemoStore:
    """Shared in-memory store with invented people, so the pages work before Firestore is set up."""
    demo = True

    def __init__(self):
        self._lock = threading.Lock()
        rng = np.random.default_rng(7)
        first = ["Ama", "Kwame", "Esi", "Kojo", "Abena", "Yaw", "Akosua", "Kofi", "Adwoa", "Kwabena", "Efua", "Kwaku",
                 "Afia", "Kweku", "Aba", "Fiifi", "Naana", "Paa", "Serwaa", "Ekow", "Mia", "Leo", "Zoe", "Eli", "Ruth",
                 "Noah", "Tia", "Sam", "Joy", "Ben"]
        last = ["Asante", "Owusu", "Boateng", "Appiah", "Darko", "Ofori", "Quaye", "Tetteh", "Addo", "Badu"]
        names = sorted({f"{rng.choice(first)} {rng.choice(last)}" for _ in range(80)})[:60]
        groups = ["Choir", "Ushering", "Youth", "Media", "Children", "Men", "Women", ""]
        self.members = {}
        for n in names:
            mid = new_id()
            self.members[mid] = dict(full_name=n, phone=f"021 {rng.integers(100, 999)} {rng.integers(1000, 9999)}",
                                     email="", group=str(rng.choice(groups)), role="", status="", type="member",
                                     date_joined="", first_visit="", invited_by="", follow_up="", created_at=now_iso())
        # 12 past Sundays; each person has a personal attendance habit, a few have recently stopped coming
        sundays = [today() - dt.timedelta(days=(today().weekday() + 1) % 7 + 7 * k) for k in range(12)][::-1]
        habit = {m: rng.beta(6, 2) for m in self.members}
        stopped = {m: int(rng.integers(3, 9)) for m in rng.choice(list(self.members), 9, replace=False)}
        self.services = {}
        for i, d in enumerate(sundays):
            present = {}
            for m, p in habit.items():
                if m in stopped and i >= len(sundays) - stopped[m]:
                    continue
                if rng.random() < p:
                    present[m] = dt.datetime.combine(d, dt.time(10, int(rng.integers(0, 40))), TZ).isoformat()
            self.services[d.isoformat()] = dict(name="Sunday Service", date=d.isoformat(), present=present)

    def list_members(self):
        return [dict(v, id=k) for k, v in self.members.items()]

    def list_services(self):
        return [dict(v) for v in self.services.values()]

    def get_service(self, date: str):
        s = self.services.get(date)
        return dict(s, present=dict(s["present"])) if s else None

    def ensure_service(self, date: str, name: str):
        with self._lock:
            s = self.services.setdefault(date, dict(name=name, date=date, present={}))
            s["name"] = name or s["name"]

    def set_present(self, date: str, mid: str, present: bool, name: str = "Sunday Service"):
        with self._lock:
            s = self.services.setdefault(date, dict(name=name, date=date, present={}))
            if present:
                s["present"][mid] = now_iso()
            else:
                s["present"].pop(mid, None)

    def upsert_members(self, rows: list[dict]):
        with self._lock:
            for r in rows:
                mid = r.pop("id", None) or new_id()
                self.members[mid] = {**self.members.get(mid, {}), **r}
        return len(rows)


class FirestoreStore:
    """Google Cloud Firestore backend. Credentials come from st.secrets['gcp_service_account']."""
    demo = False

    def __init__(self, info: dict):
        from google.cloud import firestore
        from google.oauth2 import service_account
        self._fs = firestore
        creds = service_account.Credentials.from_service_account_info(dict(info))
        self.db = firestore.Client(project=info["project_id"], credentials=creds)
        self._cache = {}
        self._lock = threading.Lock()

    def _cached(self, key, ttl, fn):
        now = dt.datetime.now().timestamp()
        hit = self._cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (now, val)
        return val

    def _drop(self, *keys):
        for k in keys:
            self._cache.pop(k, None)

    def list_members(self):
        return self._cached("members", 120, lambda: [dict(d.to_dict(), id=d.id)
                                                     for d in self.db.collection("members").stream()])

    def list_services(self):
        return self._cached("services", 20, lambda: [d.to_dict() for d in self.db.collection("services").stream()])

    def get_service(self, date: str):  # always fresh: this is the live-poll read (1 document)
        snap = self.db.collection("services").document(date).get()
        return snap.to_dict() if snap.exists else None

    def ensure_service(self, date: str, name: str):
        self.db.collection("services").document(date).set({"name": name, "date": date}, merge=True)
        self._drop("services")

    def set_present(self, date: str, mid: str, present: bool, name: str = "Sunday Service"):
        ref = self.db.collection("services").document(date)
        if present:  # merge=True deep-merges the map, so concurrent ticks from several phones don't clash
            ref.set({"name": name, "date": date, "present": {mid: now_iso()}}, merge=True)
        else:
            ref.update({f"present.{mid}": self._fs.DELETE_FIELD})
        self._drop("services")

    def upsert_members(self, rows: list[dict]):
        n = 0
        for i in range(0, len(rows), 400):
            batch = self.db.batch()
            for r in rows[i:i + 400]:
                r = dict(r)
                mid = r.pop("id", None) or new_id()
                batch.set(self.db.collection("members").document(mid), r, merge=True)
                n += 1
            batch.commit()
        self._drop("members")
        return n


@st.cache_resource(show_spinner=False)
def get_store():
    try:
        info = st.secrets.get("gcp_service_account")
    except Exception:
        info = None
    return FirestoreStore(info) if info else DemoStore()


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
def header(title: str, subtitle: str, store, live: bool = False):
    badge = "Demo data (invented names)" if store.demo else "Connected · Google Cloud Firestore"
    dot = '<span class="live-dot"></span>' if live else ""
    st.html(f'<div class="hero"><div class="eyebrow">⛪ Attendance</div><h1>{dot}{title}</h1><p>{subtitle}</p>'
            f'<span class="chip">{badge}</span></div>')


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


def demo_note(store):
    if store.demo:
        st.info("These pages are showing **invented demo people**. Connect Firestore (see *Members → Setup*) "
                "to use your real register.", icon=":material/science:")


# ---------------------------------------------------------------- pages
def page_checkin():
    store = get_store()
    header("Check-in", "Tick people as they arrive — every phone sees the same list within seconds", store, live=True)
    if not gate(store):
        return
    demo_note(store)
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
            cols = st.columns(3)
            for i, m in enumerate(shown):
                key = f"ci_{date}_{m['id']}"
                st.session_state[key] = m["id"] in present  # sync ticks made on other devices
                tag = " · first-timer" if m.get("type") == "first_timer" else ""
                cols[i % 3].checkbox(f"{m['full_name']}{tag}", key=key, on_change=on_tick, args=(m["id"],))

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
        fig.update_traces(marker_color="#2a78d6", hovertemplate="%{x|%d %b %Y}: %{y} present<extra></extra>")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), bargap=0.25,
                          title=dict(text="Attendance per service", font=dict(size=15)))
        with st.container(border=True):
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"Live · recalculated {dt.datetime.now(TZ):%H:%M:%S} · only services since someone joined "
                   "or first visited count against them.")

    live_followup()


def page_members():
    store = get_store()
    header("Members", "Your register and first-timers, stored in the database", store)
    if not gate(store):
        return
    demo_note(store)
    tab_list, tab_add, tab_import, tab_setup = st.tabs(["Register", "Add person", "Import CSV", "Setup"])

    with tab_list:
        m = pd.DataFrame(store.list_members())
        if m.empty:
            st.info("No members yet — use **Import CSV** to load your register.")
        else:
            cols = [c for c in ["full_name", "type", "phone", "email", "group", "role", "status", "date_joined",
                                "first_visit", "invited_by"] if c in m.columns]
            st.dataframe(m[cols].sort_values("full_name", key=lambda s: s.str.lower()), hide_index=True,
                         width="stretch", height=520,
                         column_config={"full_name": "Name", "type": "Type", "date_joined": "Joined",
                                        "first_visit": "First visit", "invited_by": "Invited by"})

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


SETUP_GUIDE = """
**Connect Google Cloud Firestore** (free tier is plenty for this):

1. Go to **console.firebase.google.com** → *Add project* (or use an existing Google Cloud project).
2. *Build → Firestore Database → Create database* → **Production mode** → pick a region near you (e.g. `australia-southeast1`).
3. In **console.cloud.google.com** → *IAM & Admin → Service Accounts* → *Create service account*,
   give it the role **Cloud Datastore User**, then *Keys → Add key → JSON*. A `.json` file downloads.
4. On **share.streamlit.io** → your app → ⋮ → *Settings → Secrets*, paste:

```toml
attendance_password = "choose-a-strong-password"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "…"
private_key = "-----BEGIN PRIVATE KEY-----\\n…\\n-----END PRIVATE KEY-----\\n"
client_email = "…@your-project-id.iam.gserviceaccount.com"
client_id = "…"
token_uri = "https://oauth2.googleapis.com/token"
```
   (copy each value from the downloaded JSON file). Save — the app restarts and switches from demo to your database.
5. Open **Members → Import CSV** and upload your two sheets.

Never commit the JSON key or your CSVs to GitHub — the repo is public. Both are blocked in `.gitignore`.
"""
