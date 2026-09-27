"""Age Dashboard — Streamlit app.

Explores how the age profile of 1,000 churches changes over time (2015–2026)
using member-level records (join year, leave year, birth year).

Pages live in the sidebar (st.navigation); filters are shared across pages.
Light/dark follows the viewer's device setting.
The Live activity page refreshes itself every few seconds (st.fragment run_every).
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Age Dashboard", page_icon="⛪", layout="wide", initial_sidebar_state="expanded")

DATA = Path(__file__).parent / "data"
TZ = ZoneInfo("Pacific/Auckland")


# ---------- palette (validated reference palette) ----------
# Chart text, gridlines and backgrounds come from Streamlit's own Plotly theme, so they follow the
# viewer's light/dark setting automatically; only data colours are fixed here (they read on both).
T = dict(muted="#898781", mid="#a8a7a0", sep="rgba(128,128,128,0.35)",
         series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
         bands=["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"])

SERIES = T["series"]
# Age bands are ordered -> one sequential hue, light (young) to dark (old)
BANDS = ["0–12", "13–17", "18–24", "25–34", "35–49", "50–64", "65+"]
BAND_EDGES = [-1, 12, 17, 24, 34, 49, 64, 200]
BAND_MAP = dict(zip(BANDS, T["bands"]))
YOUTH_BANDS = BANDS[:3]  # under 25

# ---------- look & feel ----------
st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, [data-testid="stMetric"], [data-testid="stSidebar"] {{font-family: 'Inter', system-ui, sans-serif;}}
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}}
[data-testid="stMetric"] {{box-shadow: 0 1px 2px rgba(0,0,0,.05);}}
[data-testid="stMetricLabel"] p {{font-size: .82rem; opacity: .8; font-weight: 500;}}
[data-testid="stMetricValue"] {{font-weight: 700; letter-spacing: -.02em;}}
.hero {{background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); color: #fff;
        border-radius: 1rem; padding: 1.4rem 1.8rem; margin-bottom: .4rem;}}
.hero .eyebrow {{font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .75;}}
.hero h1 {{font-size: 1.9rem; font-weight: 800; margin: .1rem 0 0; letter-spacing: -.02em; color: #fff; padding: 0;}}
.hero p {{margin: .35rem 0 .9rem; opacity: .88; font-size: .98rem;}}
.chip {{display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px; padding: .2rem .7rem; margin: 0 .35rem .35rem 0; font-size: .8rem; font-weight: 500;}}
.live-dot {{display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; background: #0ca30c;
            margin-right: .45rem; animation: pulse 1.6s infinite;}}
@keyframes pulse {{0% {{box-shadow: 0 0 0 0 rgba(12,163,12,.6);}} 70% {{box-shadow: 0 0 0 .5rem rgba(12,163,12,0);}}
                   100% {{box-shadow: 0 0 0 0 rgba(12,163,12,0);}}}}
</style>
""")

PLOT_CFG = {"displayModeBar": False, "responsive": True}
LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, title=None),
)


def style(fig, height=380, title=None):
    fig.update_layout(**LAYOUT, height=height,
                      title=dict(text=title, font=dict(size=15)) if title else None)
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    if fig.layout.xaxis.title.text:  # leave room for the x-axis title above a bottom legend
        fig.update_layout(legend_y=-0.3)
    return fig


def card(where, fig):
    """Render a Plotly figure inside a bordered card."""
    with where.container(border=True):
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


# ---------- navigation (declared early so attendance pages skip the heavy age data) ----------
def _lazy(name):
    """Page wrapper that looks the page function up at run time (age pages are defined further down)."""
    def run():
        globals()[name]()
    run.__name__ = name
    return run


import attendance  # noqa: E402

pg = st.navigation({
    "Attendance": [
        st.Page(attendance.page_checkin, title="Check-in", icon=":material/how_to_reg:", url_path="checkin"),
        st.Page(attendance.page_followup, title="Follow-up", icon=":material/notification_important:", url_path="followup",
                default=True),
        st.Page(attendance.page_members, title="Members", icon=":material/badge:", url_path="members"),
    ],
    "Age dashboard": [
        st.Page(_lazy("page_overview"), title="Overview", icon=":material/dashboard:", url_path="overview"),
        st.Page(_lazy("page_live"), title="Live activity", icon=":material/sensors:", url_path="live"),
        st.Page(_lazy("page_compare"), title="Compare groups", icon=":material/compare_arrows:", url_path="compare"),
        st.Page(_lazy("page_pyramid"), title="Age pyramid", icon=":material/groups:", url_path="pyramid"),
        st.Page(_lazy("page_churches"), title="All churches", icon=":material/church:", url_path="churches"),
        st.Page(_lazy("page_profile"), title="Church profile", icon=":material/search:", url_path="profile"),
        st.Page(_lazy("page_about"), title="About the data", icon=":material/info:", url_path="about"),
    ],
})
if pg.title in ("Check-in", "Follow-up", "Members"):
    pg.run()
    st.stop()


# ---------- data (built once per server, shared by every viewer) ----------
# st.cache_resource hands back the same objects without copying, which matters for ~1.3M-row frames;
# nothing below mutates the cached frames.
@st.cache_resource(show_spinner="Preparing data (first load only)…")
def get_data():
    if (DATA / "members.csv.gz").exists():
        churches = pd.read_csv(DATA / "churches.csv.gz")
        members = pd.read_csv(DATA / "members.csv.gz", dtype={"leave_year": "Int64"})
    else:  # no CSVs in the repo (e.g. Streamlit Cloud): generate in memory, skipping the slow gzip step
        from generate_data import generate
        churches, members = generate()
    snap = build_snapshots(churches, members)
    cids = snap.church_id.to_numpy()
    starts = np.searchsorted(cids, churches.church_id.to_numpy(), side="left")
    ends = np.searchsorted(cids, churches.church_id.to_numpy(), side="right")
    bounds = dict(zip(churches.church_id, zip(starts, ends)))  # church_id -> (start, end) row slice
    return churches, snap, len(members), bounds


def build_snapshots(churches: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Active members per church × gender × single-year age × year, via one bincount per year."""
    y0, y1 = max(int(members.join_year.min()), 2015), int(members.join_year.max())
    ids = churches.church_id.to_numpy()
    cc = pd.Categorical(members.church_id, categories=ids).codes.astype(np.int64)
    gg = (members.gender.to_numpy() == "M").astype(np.int64)
    leave = members.leave_year.fillna(9999).astype(int).to_numpy()
    join = members.join_year.to_numpy()
    birth = members.birth_year.to_numpy()
    n_c, n_a = len(ids), 101
    out = []
    for y in range(y0, y1 + 1):
        act = (join <= y) & (leave > y)
        key = (cc[act] * 2 + gg[act]) * n_a + np.clip(y - birth[act], 0, 100)
        cnt = np.bincount(key, minlength=n_c * 2 * n_a)
        nz = np.flatnonzero(cnt)
        out.append(pd.DataFrame({"c": nz // (2 * n_a), "g": (nz // n_a) % 2, "age": nz % n_a, "n": cnt[nz], "year": y}))
    snap = pd.concat(out, ignore_index=True)
    snap = snap.sort_values(["c", "year"], kind="stable", ignore_index=True)  # church-major: each church is one slice
    meta = churches.set_index("church_id")
    snap = pd.DataFrame({
        "church_id": ids[snap.c], "gender": np.where(snap.g == 1, "M", "F"),
        "age": snap.age.astype("int16"), "n": snap.n.astype("int32"), "year": snap.year.astype("int16"),
        "band": pd.cut(snap.age, BAND_EDGES, labels=BANDS),
        "region": meta.region.to_numpy()[snap.c], "denomination": meta.denomination.to_numpy()[snap.c],
        "setting": meta.setting.to_numpy()[snap.c]})
    return snap


def median_by(df, keys):
    """Weighted median age per group, fully vectorised (no per-group Python)."""
    agg = df.groupby(keys + ["age"], observed=True, sort=True).n.sum().reset_index()
    g = agg.groupby(keys, observed=True, sort=False).n
    agg = agg[agg.n.groupby([agg[k] for k in keys], observed=True).cumsum() >= g.transform("sum") / 2]
    return agg.groupby(keys, observed=True).age.first().rename("median_age").reset_index()


def band_of(age: int) -> str:
    return BANDS[int(np.searchsorted(BAND_EDGES, age, side="left")) - 1]


churches, snap, N_MEMBERS, CHURCH_ROWS = get_data()
YEARS = sorted(snap.year.unique())

# ---------- shared sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    yr = st.slider("Year range", int(YEARS[0]), int(YEARS[-1]), (int(YEARS[0]), int(YEARS[-1])), key="yr")
    regions = st.multiselect("Region", sorted(churches.region.unique()), key="regions")
    denoms = st.multiselect("Denomination", sorted(churches.denomination.unique()), key="denoms")
    settings = st.multiselect("Setting", sorted(churches.setting.unique()), key="settings")
    gender = st.radio("Gender", ["All", "F", "M"], horizontal=True, key="gender")
    st.caption(f"Dataset: {len(churches):,} churches · {N_MEMBERS:,} member records")



@st.cache_resource(max_entries=64, show_spinner=False)
def filtered(yr, regions, denoms, settings, gender):
    """Filter once per filter combination; later page switches and reruns reuse the result."""
    geo = snap.year.between(*yr)  # all filters except gender (the pyramid splits by gender itself)
    if regions: geo &= snap.region.isin(regions)
    if denoms: geo &= snap.denomination.isin(denoms)
    if settings: geo &= snap.setting.isin(settings)
    snap_geo = snap[geo]
    f = snap_geo[snap_geo.gender == gender] if gender != "All" else snap_geo
    if f.empty:
        return None
    tot = f.groupby("year").n.sum()
    return dict(snap_geo=snap_geo, f=f, tot=tot, med=median_by(f, ["year"]).set_index("year").median_age,
                youth=f[f.band.isin(YOUTH_BANDS)].groupby("year").n.sum() / tot,
                old=f[f.band == "65+"].groupby("year").n.sum() / tot)


FILTER_KEY = (tuple(yr), tuple(regions), tuple(denoms), tuple(settings), gender)
R = filtered(*FILTER_KEY)
if R is None:
    st.warning("No data for these filters.")
    st.stop()
f, snap_geo, tot, med, youth, old = R["f"], R["snap_geo"], R["tot"], R["med"], R["youth"], R["old"]
first, last = int(f.year.min()), int(f.year.max())


def chip_label(selected, options, none_label):
    return none_label if not selected or len(selected) == len(options) else ", ".join(selected)


def hero(page_title: str, subtitle: str, live: bool = False):
    chips = [str(first) if first == last else f"{first}–{last}",
             chip_label(regions, churches.region.unique(), "All regions"),
             chip_label(denoms, churches.denomination.unique(), "All denominations"),
             chip_label(settings, churches.setting.unique(), "Urban, peri-urban & rural"),
             {"All": "All genders", "F": "Female", "M": "Male"}[gender]]
    dot = '<span class="live-dot"></span>' if live else ""
    st.html(
        f'<div class="hero"><div class="eyebrow">⛪ Age Dashboard</div><h1>{dot}{page_title}</h1><p>{subtitle}</p>'
        + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>")


# ---------- pages ----------
def page_overview():
    hero("Overview", f"How congregations are growing — and ageing — across {f.church_id.nunique():,} churches")
    prev = last - 1 if last - 1 in tot.index else last
    growth_idx = (tot / tot[first] - 1) * 100
    spark = dict(border=True, chart_type="line")
    k = st.columns(5)
    k[0].metric(f"Members ({last})", f"{tot[last]:,}", f"{(tot[last]/tot[prev]-1):+.1%} vs {prev}" if prev != last else None,
                chart_data=tot.tolist(), **spark)
    k[1].metric(f"Growth since {first}", f"{(tot[last]/tot[first]-1):+.1%}", f"{tot[last]-tot[first]:+,} members",
                delta_color="off", chart_data=growth_idx.round(1).tolist(), **spark)
    k[2].metric("Median age", f"{med[last]:.0f} yrs", f"{med[last]-med[first]:+.0f} since {first}", delta_color="inverse",
                chart_data=med.tolist(), **spark)
    k[3].metric("Under 25 share", f"{youth[last]:.1%}", f"{(youth[last]-youth[first])*100:+.1f} pts",
                chart_data=(youth * 100).round(1).tolist(), **spark)
    k[4].metric("65+ share", f"{old[last]:.1%}", f"{(old[last]-old[first])*100:+.1f} pts", delta_color="inverse",
                chart_data=(old * 100).round(1).tolist(), **spark)
    st.write("")

    by_band = f.groupby(["year", "band"], observed=True).n.sum().reset_index()
    c1, c2 = st.columns([3, 2])
    fig = px.area(by_band, x="year", y="n", color="band", category_orders={"band": BANDS},
                  color_discrete_map=BAND_MAP, labels={"n": "Members", "year": "", "band": "Age"})
    fig.for_each_trace(lambda t: t.update(fillcolor=BAND_MAP[t.name], line=dict(color=BAND_MAP[t.name], width=0.5)))
    card(c1, style(fig, title="Members by age group"))

    share = by_band.assign(pct=by_band.n / by_band.groupby("year").n.transform("sum") * 100)
    fig = px.bar(share, x="year", y="pct", color="band", category_orders={"band": BANDS},
                 color_discrete_map=BAND_MAP, labels={"pct": "% of members", "year": "", "band": "Age"})
    fig.update_traces(marker_line_width=0.5, marker_line_color=T["sep"], hovertemplate="%{y:.1f}%")
    fig.update_layout(bargap=0.15)
    card(c2, style(fig, title="Age mix (share of members)"))

    piv = by_band.pivot(index="year", columns="band", values="n")
    growth = (piv.iloc[-1] / piv.iloc[0] - 1).reindex(BANDS) * 100
    c3, c4 = st.columns([2, 3])
    fig = go.Figure(go.Bar(x=growth.values, y=BANDS, orientation="h", marker_color=[BAND_MAP[b] for b in BANDS],
                           text=[f"{v:+.0f}%" for v in growth.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[min(0, growth.min() * 1.25), growth.max() * 1.2]))
    card(c3, style(fig, title=f"Growth by age group, {first}→{last}"))

    fig = px.line(med.reset_index(), x="year", y="median_age", markers=True, labels={"median_age": "Median age", "year": ""})
    fig.update_traces(line=dict(width=2, color=SERIES[0]), marker=dict(size=8))
    card(c4, style(fig, title="Median member age"))


def page_live():
    hero("Live activity", "Members joining and leaving right now — updates automatically", live=True)
    c_on, c_speed, c_reset = st.columns([1, 2, 1], vertical_alignment="center")
    on = c_on.toggle("Live updates", value=True)
    every = c_speed.select_slider("Refresh every", options=[2, 3, 5, 10, 30], value=3, format_func=lambda s: f"{s}s")
    if c_reset.button("Reset feed", icon=":material/restart_alt:"):
        st.session_state.pop("live", None)

    # Simulation source: sample new events from the current (filtered) membership.
    # To go live for real, replace `next_events()` with a query against your members database.
    base_now = f[f.year == last]
    ch_w = base_now.groupby("church_id").n.sum()
    age_w = base_now.groupby("age").n.sum()
    cmeta = churches.set_index("church_id")

    if "live" not in st.session_state:
        st.session_state.live = dict(rng=np.random.default_rng(), members=int(tot[last]), events=[],
                                     history=[(datetime.now(TZ), int(tot[last]))], joined=0, left=0)

    def next_events(S):
        rng = S["rng"]
        n_join, n_leave = rng.poisson(max(1.0, len(ch_w) / 200)), rng.poisson(max(0.6, len(ch_w) / 330))
        now = datetime.now(TZ)
        evs = []
        for kind, n in (("Joined", n_join), ("Left", n_leave)):
            if n == 0:
                continue
            cids = rng.choice(ch_w.index, size=n, p=(ch_w / ch_w.sum()).values)
            if kind == "Joined":  # joiners skew young: children + young adults
                ages = np.where(rng.random(n) < 0.25, rng.integers(0, 3, n), np.clip(rng.normal(29, 10, n), 13, 85)).astype(int)
            else:
                ages = rng.choice(age_w.index, size=n, p=(age_w / age_w.sum()).values)
            for cid, a in zip(cids, ages):
                evs.append(dict(time=now, event=kind, church=cmeta.at[cid, "church_name"], region=cmeta.at[cid, "region"],
                                age=int(a), age_group=band_of(int(a))))
        return evs, n_join, n_leave

    def live_body():
        S = st.session_state.live
        if on:
            evs, nj, nl = next_events(S)
            S["events"] = (evs + S["events"])[:300]
            S["joined"] += nj; S["left"] += nl
            S["members"] += nj - nl
            S["history"] = (S["history"] + [(datetime.now(TZ), S["members"])])[-120:]
        ev = pd.DataFrame(S["events"])
        hist = pd.DataFrame(S["history"], columns=["time", "members"])

        k = st.columns(4)
        k[0].metric("Members right now", f"{S['members']:,}", f"{S['members'] - int(tot[last]):+,} this session", border=True)
        k[1].metric("Joined", f"{S['joined']:,}", border=True)
        k[2].metric("Left", f"{S['left']:,}", border=True)
        med_join = f"{ev[ev.event == 'Joined'].age.median():.0f} yrs" if len(ev) and (ev.event == "Joined").any() else "–"
        k[3].metric("Median age of joiners", med_join, border=True)

        c1, c2 = st.columns([3, 2])
        fig = px.line(hist, x="time", y="members", labels={"members": "Members", "time": ""})
        fig.update_traces(line=dict(width=2, color=SERIES[0]), mode="lines+markers", marker=dict(size=5))
        fig.update_layout(hovermode="x unified", showlegend=False)
        card(c1, style(fig, 340, "Membership — live"))

        if len(ev):
            by = ev.groupby(["age_group", "event"]).size().rename("n").reset_index()
            by["n"] = np.where(by.event == "Left", -by.n, by.n)
            fig = px.bar(by, x="n", y="age_group", color="event", orientation="h", barmode="relative",
                         category_orders={"age_group": BANDS, "event": ["Joined", "Left"]},
                         color_discrete_map={"Joined": SERIES[2], "Left": SERIES[1]},
                         labels={"n": "Members (left ← → joined)", "age_group": "", "event": ""})
            fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
            fig.update_traces(hovertemplate="%{y}: %{customdata}<extra>%{fullData.name}</extra>", customdata=None)
            for tr in fig.data:
                tr.customdata = np.abs(tr.x)
            card(c2, style(fig, 340, "This session by age group"))
        else:
            with c2.container(border=True):
                st.info("Waiting for the first events…", icon=":material/hourglass_top:")

        with st.container(border=True):
            st.markdown("**Latest activity**")
            if len(ev):
                show = ev.head(15).assign(time=lambda d: d.time.dt.strftime("%H:%M:%S"))
                st.dataframe(show, hide_index=True, width="stretch",
                             column_config={"event": st.column_config.TextColumn("Event"),
                                            "age_group": st.column_config.TextColumn("Age group")})
            st.caption(f"Last updated {datetime.now(TZ):%H:%M:%S} NZT · "
                       + (f"refreshing every {every}s" if on else "paused"))

    st.fragment(live_body, run_every=every if on else None)()


@st.cache_resource(max_entries=64, show_spinner=False)
def compare_data(key, dim):
    f = filtered(*key)["f"]
    first, last = int(f.year.min()), int(f.year.max())
    md = median_by(f, ["year", dim])
    t = f.groupby(["year", dim]).n.sum().reset_index()
    base = t[t.year == first].set_index(dim).n
    t["index"] = t.n / t[dim].map(base) * 100
    hb = f[f.year.isin([first, last])].groupby([dim, "band", "year"], observed=True).n.sum().unstack("year")
    hb = ((hb[last] / hb[first] - 1) * 100).unstack("band").reindex(columns=BANDS)
    return md, t, hb


def page_compare():
    hero("Compare groups", "Median age and membership growth side by side")
    dim = st.segmented_control("Compare by", ["region", "denomination", "setting"], default="denomination",
                               format_func=str.title) or "denomination"
    cmap = {g: SERIES[i % len(SERIES)] for i, g in enumerate(sorted(churches[dim].unique()))}
    md, t, hb = compare_data(FILTER_KEY, dim)
    c1, c2 = st.columns(2)
    fig = px.line(md, x="year", y="median_age", color=dim, color_discrete_map=cmap, markers=True,
                  labels={"median_age": "Median age", "year": "", dim: ""})
    fig.update_traces(line_width=2, marker_size=7)
    card(c1, style(fig, 420, "Median age over time"))

    fig = px.line(t, x="year", y="index", color=dim, color_discrete_map=cmap, markers=True,
                  labels={"index": f"Members (index, {first}=100)", "year": "", dim: ""})
    fig.update_traces(line_width=2, marker_size=7)
    fig.add_hline(y=100, line_color=T["muted"], line_width=1, line_dash="dot")
    card(c2, style(fig, 420, "Membership growth (indexed)"))

    # heatmap: growth % by group × age band (diverging red <-> blue, neutral midpoint)
    lim = float(np.nanpercentile(np.abs(hb.values), 95)) or 1
    fig = px.imshow(hb, text_auto=".0f", aspect="auto", zmin=-lim, zmax=lim,
                    color_continuous_scale=[[0, "#d03b3b"], [0.5, T["mid"]], [1, "#2a78d6"]],
                    labels=dict(color="Growth %", x="Age group", y=""))
    fig.update_layout(hovermode="closest")
    card(st, style(fig, 360, f"Growth % by age group, {first}→{last}"))


@st.cache_resource(max_entries=128, show_spinner=False)
def pyramid_data(key, py):
    R = filtered(*key)
    sg, first = R["snap_geo"], int(R["f"].year.min())
    p = sg[sg.year.isin([py, first])]
    p = p.assign(age5=(p.age // 5 * 5).clip(upper=85))
    cur = p[p.year == py].groupby(["age5", "gender"]).n.sum().unstack(fill_value=0)
    ref = p[p.year == first].groupby(["age5", "gender"]).n.sum().unstack(fill_value=0)
    return cur, ref


def page_pyramid():
    hero("Age pyramid", "Male and female members by 5-year age group")
    py = st.select_slider("Year", options=list(range(first, last + 1)), value=last)
    cur, ref = pyramid_data(FILTER_KEY, py)
    lab = lambda a: "85+" if a == 85 else f"{a}–{a+4}"
    ylab = [lab(a) for a in cur.index]
    fig = go.Figure()
    fig.add_bar(y=ylab, x=-cur.get("M", 0), orientation="h", name=f"Male {py}", marker_color=SERIES[0],
                customdata=cur.get("M", 0), hovertemplate="%{y}: %{customdata:,}<extra>Male</extra>")
    fig.add_bar(y=ylab, x=cur.get("F", 0), orientation="h", name=f"Female {py}", marker_color=SERIES[1],
                hovertemplate="%{y}: %{x:,}<extra>Female</extra>")
    if py != first:
        fig.add_scatter(y=[lab(a) for a in ref.index], x=-ref.get("M", 0), mode="lines", name=f"{first} outline",
                        line=dict(color=T["muted"], width=2, shape="hvh"), hoverinfo="skip")
        fig.add_scatter(y=[lab(a) for a in ref.index], x=ref.get("F", 0), mode="lines", showlegend=False,
                        line=dict(color=T["muted"], width=2, shape="hvh"), hoverinfo="skip")
    mx = max(cur.max().max(), ref.max().max()) * 1.1
    raw = mx / 3  # "nice" tick step: 1, 2 or 5 × 10^k, so both sides get even, readable labels
    mag = 10 ** np.floor(np.log10(raw))
    step = mag * min((m for m in (1, 2, 5, 10) if m * mag >= raw))
    ticks = np.arange(-np.ceil(mx / step), np.ceil(mx / step) + 1) * step
    fig.update_layout(barmode="overlay", bargap=0.1, hovermode="closest",
                      xaxis=dict(range=[ticks[0], ticks[-1]], title="Members", tickvals=ticks, ticktext=[f"{abs(v):,.0f}" for v in ticks]))
    card(st, style(fig, 600, f"Age pyramid {py}" + (f" vs {first}" if py != first else "")))


@st.cache_resource(max_entries=32, show_spinner=False)
def church_table(*key):
    f = filtered(*key)["f"]
    first, last = int(f.year.min()), int(f.year.max())
    cs = f[f.year.isin([first, last])].groupby(["church_id", "year"]).n.sum().unstack(fill_value=0)
    cm = median_by(f[f.year == last], ["church_id"]).set_index("church_id").median_age
    cm0 = median_by(f[f.year == first], ["church_id"]).set_index("church_id").median_age
    cy = f[(f.year == last) & f.band.isin(YOUTH_BANDS)].groupby("church_id").n.sum()
    tbl = churches.set_index("church_id").join(pd.DataFrame({
        f"members_{first}": cs.get(first), f"members_{last}": cs.get(last),
        "median_age": cm, "median_age_change": cm - cm0,
        "youth_share": cy / cs.get(last)})).dropna(subset=[f"members_{last}"])
    tbl["growth_pct"] = (tbl[f"members_{last}"] / tbl[f"members_{first}"] - 1) * 100
    tbl = tbl.reset_index()
    return tbl


def page_churches():
    hero("Churches", "Every church's age profile and growth — search, sort and download")
    tbl = church_table(*FILTER_KEY)
    c1, c2 = st.columns([3, 2])
    fig = px.scatter(tbl, x="median_age", y="growth_pct", size=f"members_{last}", color="denomination",
                     color_discrete_map={g: SERIES[i % 8] for i, g in enumerate(sorted(churches.denomination.unique()))},
                     hover_name="church_name", size_max=22, opacity=0.8,
                     labels={"median_age": f"Median age ({last})", "growth_pct": f"Growth % {first}→{last}", "denomination": ""})
    fig.update_traces(marker=dict(line=dict(width=0.5, color=T["sep"])))
    fig.add_hline(y=0, line_color=T["muted"], line_width=1, line_dash="dot")
    fig.update_layout(hovermode="closest")
    card(c1, style(fig, 480, "Every church: age vs growth"))

    with c2.container(border=True):
        st.markdown("**Fastest growing**")
        st.dataframe(tbl.nlargest(8, "growth_pct")[["church_name", "region", "growth_pct"]],
                     hide_index=True, width="stretch",
                     column_config={"growth_pct": st.column_config.NumberColumn("Growth", format="%+.0f%%")})
        st.markdown("**Ageing fastest** (median age rise)")
        st.dataframe(tbl.nlargest(8, "median_age_change")[["church_name", "region", "median_age_change"]],
                     hide_index=True, width="stretch",
                     column_config={"median_age_change": st.column_config.NumberColumn("Δ median age", format="%+.0f yrs")})

    with st.container(border=True):
        q = st.text_input("Search churches", placeholder="Name, region or denomination…")
        hay = (tbl.church_name + " " + tbl.region + " " + tbl.denomination).str.lower()
        show = tbl[hay.str.contains(q.lower(), regex=False)] if q else tbl
        st.dataframe(show.sort_values("growth_pct", ascending=False), hide_index=True, width="stretch", height=420,
                     column_config={
                         "growth_pct": st.column_config.NumberColumn("Growth %", format="%+.1f%%"),
                         "youth_share": st.column_config.ProgressColumn("Under 25", format="percent", min_value=0, max_value=1),
                         "median_age": st.column_config.NumberColumn("Median age", format="%.0f"),
                         "median_age_change": st.column_config.NumberColumn("Δ median age", format="%+.0f"),
                     })
        st.download_button("Download table (CSV)", show.to_csv(index=False), "church_age_summary.csv", "text/csv",
                           icon=":material/download:")


def page_profile():
    opts = churches.set_index("church_id").church_name
    cid = st.selectbox("Choose a church", opts.index, format_func=lambda i: f"{opts[i]} ({i})")
    info = churches.set_index("church_id").loc[cid]
    hero(opts[cid], f"{info.denomination} · {info.region} · {info.setting} · founded {info.founded}")
    s0, s1 = CHURCH_ROWS[cid]
    one = snap.iloc[s0:s1]
    one = one[one.year.between(*yr)]
    ob = one.groupby(["year", "band"], observed=True).n.sum().reset_index()
    c1, c2 = st.columns(2)
    fig = px.bar(ob, x="year", y="n", color="band", category_orders={"band": BANDS}, color_discrete_map=BAND_MAP,
                 labels={"n": "Members", "year": "", "band": "Age"})
    fig.update_traces(marker_line_width=0.5, marker_line_color=T["sep"])
    card(c1, style(fig, title="Members by age group"))
    om = median_by(one, ["year"])
    allm = med.reset_index().assign(who="All filtered churches")
    fig = px.line(pd.concat([om.assign(who=opts[cid]), allm]), x="year", y="median_age", color="who", markers=True,
                  color_discrete_sequence=[SERIES[1], T["muted"]], labels={"median_age": "Median age", "year": "", "who": ""})
    fig.update_traces(line_width=2, marker_size=7)
    card(c2, style(fig, title="Median age vs all churches"))


def page_about():
    hero("About the data", "Where the numbers come from and how to use your own")
    st.markdown(
        "Synthetic but realistic data generated by `generate_data.py`: 1,000 churches across 7 regions and "
        "7 denominations, with member-level join/leave years and birth years for 2015–2026.\n\n"
        "**Live activity** simulates new members joining and leaving, sampled from the current membership. "
        "To make it truly live, swap `next_events()` in `app.py` for a query against your members database "
        "(e.g. `st.connection('sql')` or a Google Sheet).\n\n"
        "To use real records, add `data/churches.csv.gz` and `data/members.csv.gz` (same columns) to the repo "
        "and remove the `data/*.csv.gz` line from `.gitignore`.\n\n"
        "**Theme:** the dashboard follows your device's light or dark mode automatically.")


pg.run()
