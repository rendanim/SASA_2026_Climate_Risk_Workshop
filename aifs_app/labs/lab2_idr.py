import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aifs_app.core import idr
from aifs_app.core.pava import expand, group_by_x, pava_steps_antitonic
from aifs_app.ui import (AMOUNT_BLUES, COLORS, bundle, check_yourself, explain, lead, model, predict_first, show,
                         style, synthetic_banner, text, zone)

b, L, Z = bundle(), lead(), zone()
tr, m = b.train(L, Z), model(L, Z)
x_mm = idr.level_to_mm_day(tr.xi)
tf = lambda v: np.log10(1 + np.asarray(v, float))  # noqa: E731
TICKS = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200])

st.title("Lab 2 · From one number to a distribution")
st.caption("Objective: statistical foundations of probabilistic forecasting · Notebook §4")
synthetic_banner()
st.markdown(text("lab2.intro"))
predict_first("lab2.predict")

# --- one threshold at a time ---------------------------------------------------------------------------
st.markdown("#### ② Play: one threshold at a time")
t_mm = st.select_slider("Rain threshold *t* (mm/day)", [0.5, 1, 2, 5, 10, 20, 50], value=10, key="lab2_t")
t_lvl = int(idr.mm_day_to_level(t_mm))
below = tr.yi <= t_lvl
x_top = float(np.quantile(x_mm, 0.995))
edges = np.r_[-0.01, 0.05, np.geomspace(0.2, max(x_top, 1), 28)]
k = np.digitize(x_mm, edges) - 1
share = np.array([below[k == i].mean() if (k == i).sum() >= 15 else np.nan for i in range(len(edges) - 1)])
count = np.array([(k == i).sum() for i in range(len(edges) - 1)])
mid = np.r_[0.0, np.sqrt(edges[1:-1] * edges[2:])]
grid_lvl = np.unique(np.r_[0, np.rint(np.geomspace(1, max(x_top, 1) * 10, 300))]).astype(int)
curve = idr.cdf_at(m, grid_lvl, t_lvl)

fig = go.Figure()
ok = np.isfinite(share)
fig.add_trace(go.Scatter(
    x=tf(mid[ok]), y=share[ok], mode="markers", name="Training days: share with rain ≤ t (binned by forecast)",
    marker=dict(size=np.clip(np.sqrt(count[ok]) / 2, 5, 22), color=COLORS["point"], opacity=0.6),
    customdata=np.c_[mid[ok], count[ok]],
    hovertemplate="AIFS ≈ %{customdata[0]:.1f} mm/day<br>share ≤ t: %{y:.2f}<br>%{customdata[1]:,} pairs<extra></extra>"))
fig.add_trace(go.Scatter(x=tf(grid_lvl / 10), y=curve, mode="lines", name="IDR: P(IMERG ≤ t | AIFS)",
                         line=dict(color=COLORS["idr"], width=3), customdata=grid_lvl / 10,
                         hovertemplate="AIFS %{customdata:.1f} mm/day<br>P(≤ t) = %{y:.2f}<extra>IDR</extra>"))
style(fig, "AIFS forecast (mm/day)", f"P(IMERG ≤ {t_mm:g} mm/day)", height=430)
fig.update_xaxes(tickvals=tf(TICKS), ticktext=[str(v) for v in TICKS], range=[0, tf(x_top)])
fig.update_yaxes(range=[-0.02, 1.02])
show(fig)
p0, p_hi = idr.cdf_at(m, [0, int(round(x_top * 10))], t_lvl)
c1, c2, c3 = st.columns(3)
c1.metric("All training days ≤ t (climatology)", f"{below.mean():.0%}")
c2.metric("IDR when AIFS says 0 mm", f"{p0:.0%}")
c3.metric(f"IDR when AIFS says {x_top:.0f} mm", f"{p_hi:.0%}")
st.markdown("#### ③ Explain simply")
explain("lab2.threshold.explain", "lab2.threshold.maths")

# --- PAVA ------------------------------------------------------------------------------------------
st.divider()
st.markdown("#### Step inside: pool adjacent violators (PAVA)")
st.markdown(text("lab2.pava.explain"))
rng = np.random.default_rng(1)
order = np.argsort(tr.xi + rng.random(tr.n) * 0.1)
pick = order[np.linspace(0, tr.n - 1, 30).astype(int)]
px, py = tr.xi[pick], tr.yi[pick]
ux, raw, w = group_by_x(px, py <= t_lvl)
steps = pava_steps_antitonic(raw, w)
check = idr.cdf_at(idr.fit_levels(px, py), ux, t_lvl)
assert np.allclose(expand(steps[-1], len(ux)), check, atol=1e-9)
if len(steps) <= 2:
    st.info("These 30 days are already in order at this threshold, so nothing needs pooling. "
            "Try another threshold.", icon=":material/info:")
kstep = st.slider("PAVA step", 0, len(steps) - 1, 0, key=f"lab2_pava_{t_mm}") if len(steps) > 1 else 0
step = steps[kstep]
pos = np.arange(len(ux)) + 1
fig = go.Figure()
fig.add_trace(go.Scatter(x=pos, y=raw, mode="markers", name="Raw share ≤ t at this AIFS value",
                         marker=dict(size=10, color=COLORS["point"], symbol="circle-open", line=dict(width=2)),
                         customdata=ux / 10, hovertemplate="AIFS %{customdata:.1f} mm/day<br>raw %{y:.2f}<extra></extra>"))
for blk in step.blocks:
    hl = step.merged is not None and (blk.start, blk.stop) == step.merged
    fig.add_trace(go.Scatter(x=[blk.start + 0.6, blk.stop + 0.4], y=[blk.value, blk.value], mode="lines",
                             showlegend=False, line=dict(color=COLORS["ink"] if hl else COLORS["idr"], width=6 if hl else 3),
                             hovertemplate=f"pooled value {blk.value:.2f} over {int(blk.weight)} day(s)<extra></extra>"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Current fit (blocks)", line=dict(color=COLORS["idr"], width=3)))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Block just pooled", line=dict(color=COLORS["ink"], width=6)))
style(fig, "30 training days, ordered from driest to wettest AIFS forecast", "share ≤ t", height=330)
fig.update_yaxes(range=[-0.1, 1.1])
st.markdown(f"**Step {kstep} of {len(steps) - 1}.** {step.note}")
show(fig)
st.caption("The last step matches the IDR fit on the same 30 days; the page checks this every time it runs.")

# --- predictive distributions ----------------------------------------------------------------------
st.divider()
st.markdown("#### The full answer: a predictive distribution for every forecast")
amounts = [0, 2, 5, 10, 20, 40]
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    z_top = float(np.quantile(idr.level_to_mm_day(tr.yi), 0.999))
    for a, col in zip(amounts, AMOUNT_BLUES):
        F = idr.predict_cdf(m, [a * 10])[0]
        zz = np.r_[0, m.z / 10]
        fig.add_trace(go.Scatter(x=tf(zz), y=np.r_[F[0] if m.z[0] == 0 else 0, F], mode="lines", line_shape="hv",
                                 name=f"AIFS = {a} mm/day", line=dict(color=col, width=2.5), customdata=zz,
                                 hovertemplate="P(rain ≤ %{customdata:.1f} mm) = %{y:.2f}<extra></extra>"))
    style(fig, "observed rain (mm/day)", "P(rain ≤ amount)", height=420)
    fig.update_xaxes(tickvals=tf(TICKS), ticktext=[str(v) for v in TICKS], range=[0, tf(max(z_top, 60))])
    fig.update_yaxes(range=[0, 1.02])
    show(fig)
with c2:
    xs = np.arange(0, int(min(60, max(x_top, 10)) * 10) + 1, 5)
    q = idr.quantiles(m, xs, (0.1, 0.5, 0.9)) / 10
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]] / 10, y=np.r_[q[:, 2], q[::-1, 0]], fill="toself",
                             fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), name="10–90% range",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs / 10, y=q[:, 1], mode="lines", name="median", line=dict(color=COLORS["idr"], width=2.5)))
    lim = xs[-1] / 10
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="raw AIFS taken at face value (1:1)",
                             line=dict(color=COLORS["raw"], dash="dash")))
    style(fig, "raw AIFS forecast (mm/day)", "calibrated rain (mm/day)", height=420)
    show(fig)
explain("lab2.cdf.explain", "lab2.cdf.maths")

st.markdown("##### Read one forecast")
a = st.slider("AIFS forecast for a grid box (mm/day)", 0.0, float(max(5, round(x_top))), 10.0, 0.5, key="lab2_x")
lvl = int(idr.mm_day_to_level(a))
q10, q50, q90 = idr.quantiles(m, [lvl], (0.1, 0.5, 0.9))[0] / 10
c = st.columns(5)
c[0].metric("P(dry, < 1 mm)", f"{1 - idr.prob_exceed(m, [lvl], 9)[0]:.0%}")
c[1].metric("Median", f"{q50:.1f} mm")
c[2].metric("10–90% range", f"{q10:.1f}–{q90:.1f} mm")
c[3].metric("P(> 10 mm)", f"{idr.prob_exceed(m, [lvl], 100)[0]:.0%}")
c[4].metric("P(> 50 mm)", f"{idr.prob_exceed(m, [lvl], 500)[0]:.1%}")

p_dry0 = 1 - idr.prob_exceed(m, [0], 9)[0]
check_yourself("lab2.check", p_dry0=f"{p_dry0:.0%}", p_wet0=f"{1 - p_dry0:.0%}", zone=b.zone_label(Z),
               q50_20=f"{idr.quantiles(m, [200], (0.5,))[0, 0] / 10:.1f}")
