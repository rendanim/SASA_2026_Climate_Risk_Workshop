import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aifs_app.ui import (COLORS, REGIONS, bundle, check_yourself, explain, lead, map_fig, predict_first, show, style,
                         synthetic_banner, text, zone)

b, L, Z = bundle(), lead(), zone()
st.title("Data preliminaries")
st.caption("Read before Lab 1 · what the data are, how forecasts and observations are matched, and what to watch for")
synthetic_banner()
st.markdown(text("data.intro"))

# rainfall amounts per category, from the stored test pairs (weighted by each zone's valid test cells)
CATS = [(-np.inf, 0.05, "0"), (0.05, 1, "0.1–1"), (1, 5, "1–5"), (5, 10, "5–10"), (10, 20, "10–20"),
        (20, 50, "20–50"), (50, 100, "50–100"), (100, np.inf, "≥ 100")]


@st.cache_data(show_spinner=False)
def rain_tables(lead_: int):
    rows, shares = [], {}
    tot = {"obs": np.zeros(len(CATS)), "fc": np.zeros(len(CATS))}
    for z in b.zones_with_model(lead_):
        t = b.test(lead_, z)
        y, x, w = t.y * 24, t.x * 24, t.weight
        cy = np.array([((y >= lo) & (y < hi)).sum() for lo, hi, _ in CATS], float)
        cx = np.array([((x >= lo) & (x < hi)).sum() for lo, hi, _ in CATS], float)
        shares[z] = (cy / cy.sum(), cx / cx.sum())
        tot["obs"] += cy * w
        tot["fc"] += cx * w
        rows.append({"zone": b.zone_code(z), "climate": b.zone_name(z), "grid boxes": b.zone_pixels(z),
                     "rain days (≥ 1 mm)": float((y >= 1).mean()), "mean (mm/day)": float(y.mean()),
                     "95th pct (mm/day)": float(np.percentile(y, 95)),
                     "99th pct (mm/day)": float(np.percentile(y, 99)),
                     "AIFS mean (mm/day)": float(x.mean())})
    shares["all"] = (tot["obs"] / tot["obs"].sum(), tot["fc"] / tot["fc"].sum())
    return pd.DataFrame(rows).set_index("zone"), shares


ztab, shares = rain_tables(L)
dry_all = shares["all"][0][0] + shares["all"][0][1]          # below 1 mm
dry_zone = shares[Z][0][0] + shares[Z][0][1]
predict_first("data.predict", p_all=f"{dry_all:.0%}", p_zone=f"{dry_zone:.0%}", zone=b.zone_label(Z))

# --- the ingredients -----------------------------------------------------------------------------------
st.markdown("#### The four ingredients")
p = b.periods
src = pd.DataFrame([
    ["Forecast", "AIFS-Single (ECMWF machine-learning weather model): one 00 UTC run per day, 24 h rainfall",
     "0.25°, regridded to 0.1°", f"training: {p.get('train', '')} · test: {p.get('test', '')}"],
    ["Observations", "GPM IMERG V07: daily (00–24 UTC) rainfall estimated from satellites",
     "0.1°", "training: Final run (gauge-adjusted) · 2026 test: Late run where Final is not yet out"],
    ["Reference forecast", "IMERG climatology: every day within ±15 days of the date in 2000–2020 (about 650 values)",
     "0.1°", "scored on the 2026 test days"],
    ["Pooling groups", "Köppen–Geiger climate zones (Beck et al. 2023)", "0.1°",
     f"{len(b.zones)} zones, {b.n_land:,} land grid boxes"],
], columns=["role", "dataset", "grid", "period"])
st.dataframe(src, hide_index=True, width="stretch")
runs = b.meta.get("test_imerg_runs")
if runs:
    st.caption(f"2026 IMERG: {runs.get('late', 0)} days from the Late run and {runs.get('final', 0)} from the "
               "Final run. Training used the Final run throughout.")
explain("data.sources.explain")

st.markdown("#### About the datasets")
for tab, key in zip(st.tabs(["AIFS-Single", "GPM IMERG", "IMERG climatology", "Köppen–Geiger zones"]),
                    ("aifs", "imerg", "clim", "kg")):
    with tab:
        st.markdown(text(f"data.about.{key}"))

# --- the periods ----------------------------------------------------------------------------------------
st.divider()
st.markdown("#### When: training, test and case-study days")
tr, te = b.train_dates(L), b.test_dates(L)
fig = go.Figure()
for dates, name, col in ((tr, "training", COLORS["idr"]), (te, "test (never seen in training)", COLORS["raw"])):
    fig.add_trace(go.Scatter(x=pd.to_datetime(dates), y=[name] * len(dates), mode="markers", name=name,
                             marker=dict(symbol="line-ns-open", size=16, color=col, line=dict(width=2)),
                             hovertemplate="%{x|%d %b %Y}<extra>" + name + "</extra>"))
fig.add_trace(go.Scatter(x=pd.to_datetime(b.case_dates), y=["case-study days"] * len(b.case_dates), mode="markers",
                         name="case-study days", marker=dict(symbol="diamond", size=12, color=COLORS["ink"]),
                         hovertemplate="%{x|%d %b %Y}<extra>heavy-rain case</extra>"))
style(fig, "date the forecast is valid for (IMERG day)", "", height=260, showlegend=False)
fig.update_yaxes(categoryorder="array", categoryarray=["case-study days", "test (never seen in training)", "training"])
show(fig)
explain("data.timeline.explain", n_train=len(tr), n_test=len(te), lead=L)

# --- lead days --------------------------------------------------------------------------------------------
st.divider()
st.markdown("#### What \"lead day\" means")
lt = pd.DataFrame([{"lead day": d, "hours after the 00 UTC run": f"{24 * (d - 1)}–{24 * d} h",
                    "matched with IMERG for": "run date + 1 day" if d == 2 else f"run date + {d - 1} days",
                    "training days": len(b.train_dates(d)), "test days": len(b.test_dates(d))}
                   for d in b.lead_days])
st.dataframe(lt, hide_index=True, width="stretch")
explain("data.lead.explain")

# --- what rain looks like -----------------------------------------------------------------------------
st.divider()
st.markdown("#### Rain is mostly zero, and occasionally extreme")
scope = st.radio("Show", ["All zones", f"Zone {b.zone_label(Z)}"], horizontal=True, key="data_scope")
so, sf = shares["all"] if scope == "All zones" else shares[Z]
pct_label = lambda v: "0" if v == 0 else (f"{v:.1%}" if v >= 0.01 else f"{v:.2%}" if v >= 1e-4 else "< 0.01%")  # noqa: E731
fig = go.Figure()
labels = [c[2] for c in CATS]
fig.add_trace(go.Bar(x=labels, y=so, name="IMERG observed", marker_color=COLORS["obs"],
                     text=[pct_label(v) for v in so], textposition="outside"))
fig.add_trace(go.Bar(x=labels, y=sf, name="AIFS forecast", marker_color=COLORS["raw"],
                     text=[pct_label(v) for v in sf], textposition="outside"))
style(fig, "daily rain (mm/day)", "share of grid-box days", height=400, barmode="group")
fig.update_yaxes(type="log", tickvals=[1e-4, 1e-3, 1e-2, 0.1, 1], ticktext=["0.01%", "0.1%", "1%", "10%", "100%"])
show(fig)
st.caption(f"2026 test days at lead day {L}, from the stored sample of test pairs; zones weighted by their size.")
st.dataframe(ztab.sort_values("grid boxes", ascending=False).style.format(
    {"grid boxes": "{:,}", "rain days (≥ 1 mm)": "{:.0%}", "mean (mm/day)": "{:.1f}", "95th pct (mm/day)": "{:.1f}",
     "99th pct (mm/day)": "{:.1f}", "AIFS mean (mm/day)": "{:.1f}"}), width="stretch")
explain("data.rain.explain", "data.rain.maths")

# --- one day -------------------------------------------------------------------------------------------
st.divider()
st.markdown("#### Look at one day")
c1, c2 = st.columns(2)
case = c1.selectbox("Case-study day", range(len(b.case_dates)), format_func=lambda i: f"{b.case_dates[i]:%d %b %Y}",
                    key="data_case")
region = c2.selectbox("Zoom to", list(REGIONS), key="data_region")
m1, m2 = st.columns(2)
fc, ob = b.case_forecast(case, L), b.case_obs(case)
with m1:
    show(map_fig(b.coarsen(b.to_map(fc)), b.clat, b.clon, zmin=0, zmax=60, cbar="mm/day", region=region,
                 title=f"AIFS, lead day {L}, regridded to 0.1°", hover="%{z:.1f} mm/day", height=420))
with m2:
    show(map_fig(b.coarsen(b.to_map(ob)), b.clat, b.clon, zmin=0, zmax=60, cbar="mm/day", region=region,
                 title="IMERG observed", hover="%{z:.1f} mm/day", height=420))
explain("data.oneday.explain")

# --- conventions and caveats ---------------------------------------------------------------------------
st.divider()
st.markdown("#### Units and conventions")
st.markdown(text("data.units"))
st.markdown("#### Caveats to keep in mind")
st.markdown(text("data.caveats"))

check_yourself("data.check", n_train=len(tr), zone=b.zone_label(Z))
