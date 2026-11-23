import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aifs_app.core import idr
from aifs_app.ui import (COLORS, LOG_SHARE_TICKS, bundle, check_yourself, discrete_scale, explain, lead, map_fig, predict_first, show,
                         style, synthetic_banner, text, zone)

b, L, Z = bundle(), lead(), zone()
tr = b.train(L, Z)
x_mm, y_mm = idr.level_to_mm_day(tr.xi), idr.level_to_mm_day(tr.yi)

st.title("Lab 1 · Data for a data-scarce continent")
st.caption("Objective: uncertainty and decisions with limited or noisy data · Notebook §1–3")
synthetic_banner()
st.markdown(text("lab1.intro"))

dry_fc = x_mm < 0.05
p_rain_given_dry = float((y_mm[dry_fc] >= 1).mean()) if dry_fc.any() else float("nan")
predict_first("lab1.predict", zone=b.zone_label(Z), p=f"{p_rain_given_dry:.0%}", n=f"{int(dry_fc.sum()):,}")

# --- climate zones ---------------------------------------------------------------------------------
st.markdown("#### ② Play: one model per climate zone")
zm = b.zone_mode_map()
present = [z for z in sorted(b.zones) if (zm == z).any()]
idx = np.full(zm.shape, np.nan)
for i, z in enumerate(present):
    idx[zm == z] = i
c1, c2 = st.columns([3, 2])
with c1:
    fig = map_fig(idx, b.clat, b.clon, colorscale=discrete_scale([b.zone_color(z) for z in present]),
                  zmin=-0.5, zmax=len(present) - 0.5, cbar="", tickvals=list(range(len(present))),
                  ticktext=[b.zone_code(z) for z in present], hover="zone %{z}", height=520, agg="first",
                  title="Köppen–Geiger climate zones (Beck et al. 2023)")
    shown = np.array(fig.data[0].z, dtype=float)
    names = np.array([""] + [b.zone_label(z) for z in present], dtype=object)
    fig.data[0].customdata = names[np.where(np.isfinite(shown), shown + 1, 0).astype(int)]
    fig.data[0].hovertemplate = "%{y:.1f}°N, %{x:.1f}°E<br>%{customdata}<extra></extra>"
    show(fig)
with c2:
    n_days = len(b.train_dates(L))
    heavy = float((y_mm >= 50).mean())
    st.markdown(f"**The data budget for {b.zone_label(Z)}, lead day {L}**")
    m1, m2 = st.columns(2)
    m1.metric("Training days", n_days, help="Each AIFS run gives one forecast per grid box for this lead day.")
    m2.metric("Grid boxes in the zone", f"{b.zone_pixels(Z):,}")
    m1.metric("One grid box alone", f"{n_days} pairs", help="Its own record over the training period.")
    m2.metric("…with ≥ 50 mm/day", f"≈ {heavy * n_days:.1f} days",
              help="The share of heavy-rain days in the pooled sample times the number of training days.")
    m1.metric("Pooled across the zone", f"{tr.n:,} pairs",
              help=f"From {np.unique(tr.loc).size:,} grid boxes picked at random, thinned to at most "
                   f"{int(b.meta['num_data_pts']):,} pairs.")
    m2.metric("…with ≥ 50 mm/day", f"{int((y_mm >= 50).sum()):,} pairs")
    st.caption(text("lab1.budget.caption"))
explain("lab1.zones.explain", "lab1.zones.maths")

# --- forecast vs observation -------------------------------------------------------------------------
st.divider()
st.markdown("#### What does AIFS get right?")
c1, c2 = st.columns([3, 2])
ticks = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200])
tf = lambda v: np.log10(1 + np.asarray(v))  # noqa: E731
with c1:
    edges = tf(np.r_[0, np.geomspace(0.2, max(250, x_mm.max(), y_mm.max()), 45)])
    H, _, _ = np.histogram2d(tf(x_mm), tf(y_mm), bins=[edges, edges])
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Heatmap(
        z=np.log10(np.where(H.T > 0, H.T, np.nan)), x=centers, y=centers, colorscale="Blues",
        colorbar=dict(title=dict(text="pairs", side="right"), tickvals=[0, 1, 2, 3, 4],
                      ticktext=["1", "10", "100", "1k", "10k"], thickness=12),
        hovertemplate="%{z:.1f} (log10 pairs)<extra></extra>"))
    top = tf(250)
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect forecast (1:1)",
                             line=dict(color=COLORS["clim"], dash="dash")))
    style(fig, "AIFS forecast (mm/day)", "IMERG observed (mm/day)", height=470)
    fig.update_xaxes(tickvals=tf(ticks), ticktext=[str(t) for t in ticks], range=[0, top])
    fig.update_yaxes(tickvals=tf(ticks), ticktext=[str(t) for t in ticks], range=[0, top])
    show(fig)
with c2:
    thr = [1, 10, 50]
    fa = [float((x_mm >= t).mean()) for t in thr]
    fo = [float((y_mm >= t).mean()) for t in thr]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"≥ {t} mm" for t in thr], y=fa, name="AIFS", marker_color=COLORS["raw"],
                         text=[f"{v:.1%}" for v in fa], textposition="outside"))
    fig.add_trace(go.Bar(x=[f"≥ {t} mm" for t in thr], y=fo, name="IMERG", marker_color=COLORS["obs"],
                         text=[f"{v:.1%}" for v in fo], textposition="outside"))
    style(fig, "daily rain", "share of days", height=470, barmode="group")
    fig.update_yaxes(type="log", **LOG_SHARE_TICKS)
    show(fig)
r = float(np.corrcoef(x_mm, y_mm)[0, 1])
explain("lab1.scatter.explain", "lab1.scatter.maths", r=f"{r:.2f}", f1a=f"{fa[0]:.0%}", f1o=f"{fo[0]:.0%}",
        f50a=f"{fa[2]:.2%}", f50o=f"{fo[2]:.2%}")

# --- when are the training days? -----------------------------------------------------------------------
st.divider()
st.markdown("#### Which seasons does the model learn from?")
dates = b.train_dates(L)[tr.day]
months = np.array([d.year * 12 + d.month - 1 for d in dates])
um, cnt = np.unique(months, return_counts=True)
wet = np.array([float((y_mm[months == m] >= 1).mean()) for m in um])
labels = [f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m % 12]} {m // 12}" for m in um]
fig = go.Figure(go.Bar(x=labels, y=wet, marker_color=COLORS["idr"], customdata=cnt,
                       hovertemplate="%{x}: %{y:.0%} rain days (≥ 1 mm)<br>%{customdata:,} training pairs<extra></extra>"))
style(fig, "month of the training period", "share of rain days (IMERG ≥ 1 mm)", height=320)
fig.update_yaxes(tickformat=".0%")
show(fig)
explain("lab1.season.explain", n_months=len(um))

check_yourself("lab1.check", n_days=n_days, n_pairs=f"{tr.n:,}", heavy_one=f"{heavy * n_days:.1f}",
               heavy_pool=f"{int((y_mm >= 50).sum()):,}", zone=b.zone_label(Z))
