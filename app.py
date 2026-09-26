"""Age Dashboard — Streamlit app.

Explores how the age profile of 1,000 churches changes over time (2015–2026)
using member-level records (join year, leave year, birth year).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Age Dashboard", page_icon="⛪", layout="wide")

DATA = Path(__file__).parent / "data"

# ---------- look & feel ----------
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, [data-testid="stMetric"], [data-testid="stSidebar"] {font-family: 'Inter', system-ui, sans-serif;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {background: #ffffff; box-shadow: 0 1px 2px rgba(11,11,11,.04);}
[data-testid="stMetricLabel"] p {font-size: .82rem; color: #52514e; font-weight: 500;}
[data-testid="stMetricValue"] {font-weight: 700; letter-spacing: -.02em;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #ffffff;}
.stTabs [data-baseweb="tab-list"] {gap: .25rem;}
.stTabs [data-baseweb="tab"] {padding: .5rem 1rem; border-radius: .6rem .6rem 0 0; font-weight: 500;}
[data-testid="stSidebar"] {background: #f3f2ee;}
.hero {background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); color: #fff;
       border-radius: 1rem; padding: 1.6rem 1.8rem; margin-bottom: .4rem;}
.hero h1 {font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -.02em; color: #fff;}
.hero p {margin: .35rem 0 .9rem; opacity: .88; font-size: .98rem;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
       border-radius: 999px; padding: .2rem .7rem; margin: 0 .35rem .35rem 0; font-size: .8rem; font-weight: 500;}
</style>
""")

PLOT_CFG = {"displayModeBar": False, "responsive": True}


def card(where, fig):
    """Render a Plotly figure inside a bordered card."""
    with where.container(border=True):
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

# ---------- palette (validated reference palette; see README) ----------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Age bands are ordered -> one sequential hue, light (young) to dark (old)
BANDS = ["0–12", "13–17", "18–24", "25–34", "35–49", "50–64", "65+"]
BAND_EDGES = [-1, 12, 17, 24, 34, 49, 64, 200]
BAND_COLORS = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
BAND_MAP = dict(zip(BANDS, BAND_COLORS))
YOUTH_BANDS = BANDS[:3]  # under 25

LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, title=None),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, tickcolor=GRID, zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
)


def style(fig, height=380, title=None):
    fig.update_layout(**LAYOUT, height=height, title=dict(text=title, font=dict(size=15, color=INK)) if title else None)
    fig.update_xaxes(gridcolor=GRID, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, title_font_color=MUTED)
    return fig


# ---------- data ----------
@st.cache_data(show_spinner="Loading member records…")
def load():
    if not (DATA / "members.csv.gz").exists():  # first run (e.g. Streamlit Cloud): build the dataset
        import subprocess, sys
        DATA.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "generate_data.py"], cwd=Path(__file__).parent, check=True)
    churches = pd.read_csv(DATA / "churches.csv.gz")
    members = pd.read_csv(DATA / "members.csv.gz", dtype={"leave_year": "Int64"})
    return churches, members


@st.cache_data(show_spinner="Building yearly age snapshots…")
def snapshots(members: pd.DataFrame) -> pd.DataFrame:
    """Active members per church × year × single-year age × gender."""
    y0, y1 = int(members.join_year.min()), int(members.join_year.max())
    y0 = max(y0, 2015)
    leave = members.leave_year.fillna(9999).astype(int).to_numpy()
    join = members.join_year.to_numpy()
    birth = members.birth_year.to_numpy()
    out = []
    for y in range(y0, y1 + 1):
        act = (join <= y) & (leave > y)
        age = np.clip(y - birth[act], 0, 100)
        df = pd.DataFrame({"church_id": members.church_id.to_numpy()[act], "gender": members.gender.to_numpy()[act], "age": age})
        g = df.groupby(["church_id", "gender", "age"], observed=True).size().rename("n").reset_index()
        g["year"] = y
        out.append(g)
    snap = pd.concat(out, ignore_index=True)
    snap["band"] = pd.cut(snap.age, BAND_EDGES, labels=BANDS)
    snap["age"] = snap.age.astype("int16"); snap["n"] = snap.n.astype("int32"); snap["year"] = snap.year.astype("int16")
    return snap


def weighted_median(ages, weights):
    order = np.argsort(ages)
    a, w = np.asarray(ages)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    return float(a[np.searchsorted(c, c[-1] / 2)]) if len(c) and c[-1] > 0 else np.nan


def median_by(df, keys):
    agg = df.groupby(keys + ["age"], observed=True).n.sum().reset_index()
    return agg.groupby(keys).apply(lambda g: weighted_median(g.age, g.n), include_groups=False).rename("median_age").reset_index()


churches, members = load()
snap = snapshots(members)
snap = snap.merge(churches[["church_id", "region", "denomination", "setting"]], on="church_id")

YEARS = sorted(snap.year.unique())

# ---------- sidebar filters ----------
st.sidebar.title("⛪ Filters")
yr = st.sidebar.slider("Year range", int(YEARS[0]), int(YEARS[-1]), (int(YEARS[0]), int(YEARS[-1])))
regions = st.sidebar.multiselect("Region", sorted(churches.region.unique()))
denoms = st.sidebar.multiselect("Denomination", sorted(churches.denomination.unique()))
settings = st.sidebar.multiselect("Setting", sorted(churches.setting.unique()))
gender = st.sidebar.radio("Gender", ["All", "F", "M"], horizontal=True)
st.sidebar.caption(f"Dataset: {len(churches):,} churches · {len(members):,} member records")

mask = snap.year.between(*yr)
if regions: mask &= snap.region.isin(regions)
if denoms: mask &= snap.denomination.isin(denoms)
if settings: mask &= snap.setting.isin(settings)
if gender != "All": mask &= snap.gender == gender
f = snap[mask]
if f.empty:
    st.warning("No data for these filters."); st.stop()
first, last = int(f.year.min()), int(f.year.max())

# ---------- header + KPIs ----------
chips = [f"{first}–{last}", ", ".join(regions) or "All regions", ", ".join(denoms) or "All denominations",
         ", ".join(settings) or "Urban, peri-urban & rural", {"All": "All genders", "F": "Female", "M": "Male"}[gender]]
st.html(
    '<div class="hero"><h1>⛪ Age Dashboard</h1>'
    f'<p>How congregations are growing — and ageing — across {f.church_id.nunique():,} churches</p>'
    + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>")

tot = f.groupby("year").n.sum()
med = median_by(f, ["year"]).set_index("year").median_age
youth = f[f.band.isin(YOUTH_BANDS)].groupby("year").n.sum() / tot
old = f[f.band == "65+"].groupby("year").n.sum() / tot
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

tabs = st.tabs(["📈 Overview", "🧭 Compare groups", "👥 Age pyramid", "🏛️ Churches", "🔎 Church profile"])

# ---------- Overview ----------
with tabs[0]:
    by_band = f.groupby(["year", "band"], observed=True).n.sum().reset_index()
    c1, c2 = st.columns([3, 2])
    fig = px.area(by_band, x="year", y="n", color="band", category_orders={"band": BANDS},
                  color_discrete_map=BAND_MAP, labels={"n": "Members", "year": "", "band": "Age"})
    fig.for_each_trace(lambda t: t.update(fillcolor=BAND_MAP[t.name], line=dict(color=BAND_MAP[t.name], width=0.5)))
    card(c1, style(fig, title="Members by age group"))

    share = by_band.assign(pct=by_band.n / by_band.groupby("year").n.transform("sum") * 100)
    fig = px.bar(share, x="year", y="pct", color="band", category_orders={"band": BANDS},
                 color_discrete_map=BAND_MAP, labels={"pct": "% of members", "year": "", "band": "Age"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="#fcfcfb", hovertemplate="%{y:.1f}%")
    fig.update_layout(bargap=0.15)
    card(c2, style(fig, title="Age mix (share of members)"))

    # growth by band: indexed to first year
    piv = by_band.pivot(index="year", columns="band", values="n")
    growth = (piv.iloc[-1] / piv.iloc[0] - 1).reindex(BANDS) * 100
    c3, c4 = st.columns([2, 3])
    fig = go.Figure(go.Bar(x=growth.values, y=BANDS, orientation="h",
                           marker_color=[BAND_MAP[b] for b in BANDS],
                           text=[f"{v:+.0f}%" for v in growth.values], textposition="outside",
                           hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
    card(c3, style(fig, title=f"Growth by age group, {first}→{last}"))

    mdf = med.reset_index()
    fig = px.line(mdf, x="year", y="median_age", markers=True, labels={"median_age": "Median age", "year": ""})
    fig.update_traces(line=dict(width=2, color=SERIES[0]), marker=dict(size=8))
    card(c4, style(fig, title="Median member age"))

# ---------- Compare groups ----------
with tabs[1]:
    dim = st.segmented_control("Compare by", ["region", "denomination", "setting"], default="denomination",
                               format_func=str.title) or "denomination"
    groups = sorted(f[dim].unique())
    cmap = {g: SERIES[i % len(SERIES)] for i, g in enumerate(sorted(churches[dim].unique()))}
    c1, c2 = st.columns(2)
    md = median_by(f, ["year", dim])
    fig = px.line(md, x="year", y="median_age", color=dim, color_discrete_map=cmap, markers=True,
                  labels={"median_age": "Median age", "year": "", dim: ""})
    fig.update_traces(line_width=2, marker_size=7)
    card(c1, style(fig, 420, "Median age over time"))

    t = f.groupby(["year", dim]).n.sum().reset_index()
    base = t[t.year == first].set_index(dim).n
    t["index"] = t.apply(lambda r: r.n / base[r[dim]] * 100, axis=1)
    fig = px.line(t, x="year", y="index", color=dim, color_discrete_map=cmap, markers=True,
                  labels={"index": f"Members (index, {first}=100)", "year": "", dim: ""})
    fig.update_traces(line_width=2, marker_size=7)
    fig.add_hline(y=100, line_color=MUTED, line_width=1, line_dash="dot")
    card(c2, style(fig, 420, "Membership growth (indexed)"))

    # heatmap: growth % by group × age band (diverging blue<->red, grey midpoint)
    hb = f[f.year.isin([first, last])].groupby([dim, "band", "year"], observed=True).n.sum().unstack("year")
    hb = ((hb[last] / hb[first] - 1) * 100).unstack("band").reindex(columns=BANDS)
    lim = float(np.nanpercentile(np.abs(hb.values), 95)) or 1
    fig = px.imshow(hb, text_auto=".0f", aspect="auto", zmin=-lim, zmax=lim,
                    color_continuous_scale=[[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#1c5cab"]],
                    labels=dict(color="Growth %", x="Age group", y=""))
    fig.update_layout(hovermode="closest")
    card(st, style(fig, 360, f"Growth % by age group, {first}→{last}"))

# ---------- Age pyramid ----------
with tabs[2]:
    py = st.select_slider("Year", options=list(range(first, last + 1)), value=last)
    pm = snap.year.between(*yr)
    if regions: pm &= snap.region.isin(regions)
    if denoms: pm &= snap.denomination.isin(denoms)
    if settings: pm &= snap.setting.isin(settings)
    p = snap[pm].copy()
    p["age5"] = (p.age // 5 * 5).clip(upper=85)
    lab = lambda a: "85+" if a == 85 else f"{a}–{a+4}"
    cur = p[p.year == py].groupby(["age5", "gender"]).n.sum().unstack(fill_value=0)
    ref = p[p.year == first].groupby(["age5", "gender"]).n.sum().unstack(fill_value=0)
    ylab = [lab(a) for a in cur.index]
    fig = go.Figure()
    fig.add_bar(y=ylab, x=-cur.get("M", 0), orientation="h", name=f"Male {py}", marker_color=SERIES[0],
                customdata=cur.get("M", 0), hovertemplate="%{y}: %{customdata:,}<extra>Male</extra>")
    fig.add_bar(y=ylab, x=cur.get("F", 0), orientation="h", name=f"Female {py}", marker_color=SERIES[1],
                hovertemplate="%{y}: %{x:,}<extra>Female</extra>")
    if py != first:
        fig.add_scatter(y=[lab(a) for a in ref.index], x=-ref.get("M", 0), mode="lines", name=f"{first} outline",
                        line=dict(color=INK, width=1.5, shape="hvh"), hoverinfo="skip")
        fig.add_scatter(y=[lab(a) for a in ref.index], x=ref.get("F", 0), mode="lines", showlegend=False,
                        line=dict(color=INK, width=1.5, shape="hvh"), hoverinfo="skip")
    mx = max(cur.max().max(), ref.max().max()) * 1.1
    fig.update_layout(barmode="overlay", bargap=0.1, hovermode="closest",
                      xaxis=dict(range=[-mx, mx], tickformat=",", title="Members"))
    fig.update_xaxes(tickvals=np.linspace(-mx, mx, 7).round(-3), ticktext=[f"{abs(v):,.0f}" for v in np.linspace(-mx, mx, 7).round(-3)])
    card(st, style(fig, 560, f"Age pyramid {py}" + (f" vs {first}" if py != first else "")))

# ---------- Churches ----------
with tabs[3]:
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

    c1, c2 = st.columns([3, 2])
    fig = px.scatter(tbl, x="median_age", y="growth_pct", size=f"members_{last}", color="denomination",
                     color_discrete_map={g: SERIES[i % 8] for i, g in enumerate(sorted(churches.denomination.unique()))},
                     hover_name="church_name", size_max=22, opacity=0.75,
                     labels={"median_age": f"Median age ({last})", "growth_pct": f"Growth % {first}→{last}", "denomination": ""})
    fig.update_traces(marker=dict(line=dict(width=1, color="#fcfcfb")))
    fig.add_hline(y=0, line_color=MUTED, line_width=1, line_dash="dot")
    fig.update_layout(hovermode="closest")
    card(c1, style(fig, 460, "Every church: age vs growth"))

    with c2:
        st.markdown("**Fastest growing**")
        st.dataframe(tbl.nlargest(8, "growth_pct")[["church_name", "region", "growth_pct"]],
                     hide_index=True, width="stretch",
                     column_config={"growth_pct": st.column_config.NumberColumn("Growth", format="%+.0f%%")})
        st.markdown("**Ageing fastest** (median age rise)")
        st.dataframe(tbl.nlargest(8, "median_age_change")[["church_name", "region", "median_age_change"]],
                     hide_index=True, width="stretch",
                     column_config={"median_age_change": st.column_config.NumberColumn("Δ median age", format="%+.0f yrs")})

    q = st.text_input("Search churches", placeholder="Name, region or denomination…")
    show = tbl if not q else tbl[tbl.apply(lambda r: q.lower() in f"{r.church_name} {r.region} {r.denomination}".lower(), axis=1)]
    st.dataframe(show.sort_values("growth_pct", ascending=False), hide_index=True, width="stretch", height=420,
                 column_config={
                     "growth_pct": st.column_config.NumberColumn("Growth %", format="%+.1f%%"),
                     "youth_share": st.column_config.ProgressColumn("Under 25", format="percent", min_value=0, max_value=1),
                     "median_age": st.column_config.NumberColumn("Median age", format="%.0f"),
                     "median_age_change": st.column_config.NumberColumn("Δ median age", format="%+.0f"),
                 })
    st.download_button("⬇️ Download table (CSV)", show.to_csv(index=False), "church_age_summary.csv", "text/csv")

# ---------- Church profile ----------
with tabs[4]:
    opts = churches.set_index("church_id").church_name
    cid = st.selectbox("Choose a church", opts.index, format_func=lambda i: f"{opts[i]} ({i})")
    info = churches.set_index("church_id").loc[cid]
    st.caption(f"{info.denomination} · {info.region} · {info.setting} · founded {info.founded}")
    one = snap[(snap.church_id == cid) & snap.year.between(*yr)]
    ob = one.groupby(["year", "band"], observed=True).n.sum().reset_index()
    c1, c2 = st.columns(2)
    fig = px.bar(ob, x="year", y="n", color="band", category_orders={"band": BANDS}, color_discrete_map=BAND_MAP,
                 labels={"n": "Members", "year": "", "band": "Age"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="#fcfcfb")
    card(c1, style(fig, title="Members by age group"))
    om = median_by(one, ["year"])
    allm = med.reset_index().assign(who="All filtered churches")
    fig = px.line(pd.concat([om.assign(who=opts[cid]), allm]), x="year", y="median_age", color="who", markers=True,
                  color_discrete_sequence=[SERIES[1], MUTED], labels={"median_age": "Median age", "year": "", "who": ""})
    fig.update_traces(line_width=2, marker_size=7)
    card(c2, style(fig, title="Median age vs all churches"))

with st.expander("About this data"):
    st.markdown(
        "Synthetic but realistic data generated by `generate_data.py`: 1,000 churches across 7 regions and "
        "7 denominations, with member-level join/leave years and birth years for 2015–2026. "
        "Replace `data/churches.csv.gz` and `data/members.csv.gz` with your own records (same columns) "
        "and the dashboard updates automatically.")
