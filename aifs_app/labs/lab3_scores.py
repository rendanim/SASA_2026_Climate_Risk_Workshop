import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aifs_app.core import idr
from aifs_app.core.scores import Decomposition
from aifs_app.ui import (COLORS, bundle, check_yourself, explain, lead, model, pct, predict_first, show, style,
                         synthetic_banner, text, zone, zone_pair_scores)

b, L, Z = bundle(), lead(), zone()
m = model(L, Z)
tf = lambda v: np.log10(1 + np.asarray(v, float))  # noqa: E731
TICKS = np.array([0, 1, 2, 5, 10, 20, 50, 100])

st.title("Lab 3 · Scoring probabilistic forecasts")
st.caption("Objective: statistical foundations of probabilistic forecasting · Notebook §5–6")
synthetic_banner()
st.markdown(text("lab3.intro"))
predict_first("lab3.predict")

# --- CRPS in a picture ----------------------------------------------------------------------------------
st.markdown("#### ② Play: what the CRPS measures")
c1, c2 = st.columns(2)
a = c1.slider("AIFS forecast (mm/day)", 0.0, 60.0, 10.0, 0.5, key="lab3_fc")
o = c2.slider("What IMERG observed (mm/day)", 0.0, 80.0, 2.0, 0.5, key="lab3_obs")
al, ol = int(idr.mm_day_to_level(a)), int(idr.mm_day_to_level(o))
F = idr.predict_cdf(m, [al])[0]
zz = np.r_[0, m.z / 10, max(80, m.z[-1] / 10)]
Fi = np.r_[F[0] if m.z[0] == 0 else 0, F, 1]
Fc = np.r_[m.clim[0] if m.z[0] == 0 else 0, m.clim, 1]
obs_step = (zz >= o).astype(float)
s_raw = abs(a - o)
s_idr = float(idr.crps(m, [al], [ol])[0] * 24)
s_clim = float(idr.crps_clim_training(m, np.array([ol]))[0] * 24)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tf(zz), y=obs_step, mode="lines", line_shape="hv", name=f"what happened ({o:g} mm)",
                         line=dict(color=COLORS["obs"], width=2)))
fig.add_trace(go.Scatter(x=tf(zz), y=Fi, mode="lines", line_shape="hv", name=f"calibrated (IDR): CRPS {s_idr:.2f}",
                         line=dict(color=COLORS["idr"], width=3), fill="tonexty", fillcolor="rgba(42,120,214,0.15)"))
fig.add_trace(go.Scatter(x=tf([0, a, a, zz[-1]]), y=[0, 0, 1, 1], mode="lines",
                         name=f"raw AIFS, one number: CRPS {s_raw:.2f}", line=dict(color=COLORS["raw"], width=2, dash="dot")))
fig.add_trace(go.Scatter(x=tf(zz), y=Fc, mode="lines", line_shape="hv", name=f"zone climatology: CRPS {s_clim:.2f}",
                         line=dict(color=COLORS["clim"], width=2, dash="dash")))
style(fig, "rain (mm/day)", "P(rain ≤ amount)", height=420)
fig.update_xaxes(tickvals=tf(TICKS), ticktext=[str(v) for v in TICKS], range=[0, tf(80)])
fig.update_yaxes(range=[-0.02, 1.02])
show(fig)
explain("lab3.crps.explain", "lab3.crps.maths")

# --- headline results ------------------------------------------------------------------------------------
st.divider()
st.markdown("#### The notebook's headline: every land grid box, every test day")
rows = []
for d in b.lead_days:
    t = b.totals(d)
    rows.append({"lead day": d, "CRPS raw AIFS": t["raw"], "CRPS calibrated": t["idr"], "CRPS climatology": t["clim"],
                 "CRPSS raw": 1 - t["raw"] / t["clim"], "CRPSS calibrated": 1 - t["idr"] / t["clim"]})
df = pd.DataFrame(rows).set_index("lead day")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    for col, k, dash in (("CRPS raw AIFS", "raw", "solid"), ("CRPS calibrated", "idr", "solid"),
                         ("CRPS climatology", "clim", "dash")):
        fig.add_trace(go.Scatter(x=df.index, y=df[col] * 24, mode="lines+markers", name=col.replace("CRPS ", ""),
                                 line=dict(color=COLORS[k], width=2.5, dash=dash), marker=dict(size=9)))
    style(fig, "lead day", "mean CRPS (mm/day), lower is better", height=380)
    fig.update_xaxes(tickvals=b.lead_days)
    fig.update_yaxes(rangemode="tozero")
    show(fig)
with c2:
    fig = go.Figure()
    for col, k in (("CRPSS raw", "raw"), ("CRPSS calibrated", "idr")):
        fig.add_trace(go.Scatter(x=df.index, y=df[col] * 100, mode="lines+markers", name=col.replace("CRPSS ", ""),
                                 line=dict(color=COLORS[k], width=2.5), marker=dict(size=9)))
    fig.add_hline(y=0, line=dict(color=COLORS["clim"], dash="dash"), annotation_text="climatology",
                  annotation_position="bottom right")
    style(fig, "lead day", "CRPSS vs IMERG climatology (%), higher is better", height=380)
    fig.update_xaxes(tickvals=b.lead_days)
    show(fig)
st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns[:3]} | {c: "{:+.1%}" for c in df.columns[3:]}),
             width="stretch")
st.caption("CRPS in mm/hr as in the notebook (the charts show mm/day). Raw AIFS is one number, so its CRPS is the "
           "absolute error.")

# --- decomposition ---------------------------------------------------------------------------------------
st.divider()
st.markdown("#### Why raw AIFS loses to climatology, and still carries information")
scope = st.radio("Scope", ["All of Africa", f"Zone {b.zone_label(Z)}"], horizontal=True, key="lab3_scope")
if scope == "All of Africa":
    t = b.totals(L)
    dec = Decomposition(t["raw"], t["idr"], t["clim"])
else:
    zs = b.zone_scores(L)[Z]
    dec = Decomposition(zs["raw"], zs["idr"], zs["clim"])
k = 24
fig = go.Figure(go.Waterfall(
    x=["UNC<br>(climatology)", "− DSC<br>(discrimination)", "+ MCB<br>(miscalibration)", "= raw AIFS"],
    measure=["absolute", "relative", "relative", "total"],
    y=[dec.unc * k, -dec.dsc * k, dec.mcb * k, 0],
    text=[f"{dec.unc * k:.2f}", f"−{dec.dsc * k:.2f}", f"+{dec.mcb * k:.2f}", f"{dec.raw * k:.2f}"],
    textposition="outside", connector=dict(line=dict(color=COLORS["grid"])),
    increasing=dict(marker=dict(color=COLORS["raw"])), decreasing=dict(marker=dict(color=COLORS["idr"])),
    totals=dict(marker=dict(color=COLORS["ink2"]))))
style(fig, "", "mean CRPS (mm/day)", height=400, showlegend=False)
c1, c2 = st.columns([3, 2])
with c1:
    show(fig)
with c2:
    st.metric("Raw AIFS vs climatology", pct(dec.skill_raw), help="CRPSS = 1 − CRPS / CRPS climatology")
    st.metric("Calibrated vs climatology", pct(dec.skill_cal))
    st.metric("Share of raw error removed by calibration", pct(dec.mcb / dec.raw, signed=False))
explain("lab3.decomp.explain", "lab3.decomp.maths", lead=L, mcb=f"{dec.mcb * k:.2f}", dsc=f"{dec.dsc * k:.2f}",
        unc=f"{dec.unc * k:.2f}")

with st.expander("Advanced: in-sample versus out-of-sample recalibration"):
    te = b.test(L, Z)
    s = zone_pair_scores(L, Z)
    m_in = idr.fit_levels(te.xi, te.yi)
    cal_in = float(idr.crps(m_in, te.xi, te.yi).mean())
    ref = float(s["clim"].mean())
    raw = float(s["raw"].mean())
    out_ = float(s["idr"].mean())
    tbl = pd.DataFrame({
        "recalibrated on": ["training period (the honest forecast)", "the test sample itself (diagnostic)"],
        "CRPS cal (mm/day)": [out_ * k, cal_in * k],
        "MCB (mm/day)": [(raw - out_) * k, (raw - cal_in) * k],
        "DSC (mm/day)": [(ref - out_) * k, (ref - cal_in) * k],
    })
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns[1:]}), hide_index=True, width="stretch")
    st.markdown(text("lab3.insample.explain", n=f"{te.n:,}", zone=b.zone_label(Z)))

# --- through the year --------------------------------------------------------------------------------------
st.divider()
st.markdown("#### Skill through the test period")
dates, d = b.daily(L)
s = pd.DataFrame(d, index=pd.DatetimeIndex(dates)).asfreq("D").rolling(7, center=True, min_periods=4).mean() * 24
fig = go.Figure()
for key, name, dash in (("raw", "raw AIFS", "solid"), ("idr", "calibrated (IDR)", "solid"), ("clim", "climatology", "dash")):
    fig.add_trace(go.Scatter(x=s.index, y=s[key], mode="lines", name=name, line=dict(color=COLORS[key], width=2, dash=dash)))
style(fig, "", "mean CRPS over Africa (mm/day), 7-day mean", height=340)
fig.update_yaxes(rangemode="tozero")
show(fig)
explain("lab3.daily.explain")

t2 = b.totals(L)
check_yourself("lab3.check", lead=L, raw_s=pct(1 - t2["raw"] / t2["clim"]), cal_s=pct(1 - t2["idr"] / t2["clim"]),
               mcb=f"{(t2['raw'] - t2['idr']) * 24:.2f}", dsc=f"{(t2['clim'] - t2['idr']) * 24:.2f}")
