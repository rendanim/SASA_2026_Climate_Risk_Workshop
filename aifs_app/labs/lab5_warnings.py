import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aifs_app.core import idr
from aifs_app.core.scores import brier, contingency, economic_value, reliability, roc_curve
from aifs_app.ui import (COLORS, LOG_SHARE_TICKS, REGIONS, WARN_COLORS, WARN_NAMES, bundle, check_yourself, discrete_scale,
                         event_probs, explain, lead, map_fig, model, pct, predict_first, show, style,
                         synthetic_banner, text, zone)

b, L, Z = bundle(), lead(), zone()
st.title("Lab 5 · Early warnings and risk indices")
st.caption("Objective: techniques for constructing climate risk indices and early-warning indicators · Notebook §9")
synthetic_banner()
st.markdown(text("lab5.intro"))
predict_first("lab5.predict")

c1, c2 = st.columns([2, 1])
T = c1.select_slider("Heavy-rain threshold (mm/day)", [1, 5, 10, 20, 30, 50], value=int(b.event_mm_day), key="lab5_T")
scope = c2.radio("Test pairs from", ["All zones", "This zone"], horizontal=True, key="lab5_scope")
zones = tuple(b.zones_with_model(L)) if scope == "All zones" else (Z,)
ev = event_probs(L, float(T), zones)
p, raw, clim, o, w = ev["p"], ev["raw"], ev["clim"], ev["obs"], ev["w"]
ref_name = "IMERG climatology (±15 days, 2000–2020)" if ev["imerg_clim"] else "training-period zone climatology"
if not ev["imerg_clim"]:
    st.caption(f"The IMERG climatology's probabilities were pre-computed only for {b.event_mm_day:g} mm/day, so at "
               f"{T} mm/day the reference is each zone's training-period climatology, a weaker yardstick.")

# --- A: reliability ------------------------------------------------------------------------------------
st.markdown("#### ② Play A: can the probabilities be trusted?")
bs = {k: brier(v, o, w) for k, v in (("raw", raw), ("idr", p), ("clim", clim))}
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=COLORS["clim"], dash="dot", width=1),
                             name="perfectly reliable", hoverinfo="skip"))
    for key, v, name, dash in (("idr", p, "calibrated (IDR)", "solid"), ("clim", clim, ref_name, "dash"),
                               ("raw", raw, "raw AIFS (yes/no)", "solid")):
        r = reliability(v, o, w)
        ok = r.count >= max(30 * w.mean(), 1)
        fig.add_trace(go.Scatter(x=r.mean_p[ok], y=r.freq[ok], mode="lines+markers" if key != "raw" else "markers",
                                 name=name, line=dict(color=COLORS[key], width=2.5, dash=dash), marker=dict(size=10),
                                 customdata=r.share[ok],
                                 hovertemplate="forecast %{x:.2f} → observed %{y:.2f}<br>%{customdata:.1%} of forecasts<extra></extra>"))
    style(fig, "forecast probability", f"observed frequency of rain > {T} mm/day", height=430)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    show(fig)
with c2:
    r_idr, r_cl = reliability(p, o, w), reliability(clim, o, w)
    fig = go.Figure()
    centers = (r_idr.edges[:-1] + r_idr.edges[1:]) / 2
    fig.add_trace(go.Bar(x=centers - 0.02, y=r_idr.share, width=0.035, name="calibrated", marker_color=COLORS["idr"]))
    fig.add_trace(go.Bar(x=centers + 0.02, y=r_cl.share, width=0.035, name="climatology", marker_color=COLORS["clim"]))
    style(fig, "forecast probability", "share of forecasts", height=260)
    fig.update_yaxes(type="log", **LOG_SHARE_TICKS)
    show(fig)
    m1, m2 = st.columns(2)
    m1.metric("BSS calibrated", pct(1 - bs["idr"] / bs["clim"]))
    m2.metric("BSS raw yes/no", pct(1 - bs["raw"] / bs["clim"]))
    st.caption(f"Event base rate: {np.average(o, weights=w):.1%} of grid-box days.")
explain("lab5.rel.explain", "lab5.rel.maths")

# --- B: deciding when to warn ----------------------------------------------------------------------------
st.divider()
st.markdown("#### ② Play B: when should a forecaster issue a warning?")
cut = st.slider("Warn when the calibrated probability is at least", 0.05, 0.95, 0.3, 0.05, key="lab5_cut")
ct = contingency(p, o, cut, w)
ct_raw = contingency(raw, o, 0.5, w)


@st.cache_data(show_spinner=False)
def curves(lead_: int, T_: float, zones_: tuple):
    e = event_probs(lead_, T_, zones_)
    al = np.linspace(0.01, 0.99, 99)
    return (roc_curve(e["p"], e["obs"], e["w"]), al, economic_value(e["p"], e["obs"], al, e["w"]),
            economic_value(e["raw"], e["obs"], al, e["w"], fixed_warn=e["raw"] > 0.5),
            economic_value(e["clim"], e["obs"], al, e["w"]))


(pofd, pod, cuts, auc), alphas, v_idr, v_raw, v_cl = curves(L, float(T), zones)
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=COLORS["clim"], dash="dot", width=1),
                             name="no skill", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pofd, y=pod, mode="lines", name=f"calibrated (AUC {auc:.2f})",
                             line=dict(color=COLORS["idr"], width=2.5), customdata=cuts,
                             hovertemplate="warn at p ≥ %{customdata:.2f}<br>false alarm rate %{x:.2f}<br>hit rate %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[ct["pofd"]], y=[ct["pod"]], mode="markers", name=f"your rule: p ≥ {cut:.2f}",
                             marker=dict(color=COLORS["idr"], size=15, line=dict(color="white", width=2))))
    fig.add_trace(go.Scatter(x=[ct_raw["pofd"]], y=[ct_raw["pod"]], mode="markers", name="raw AIFS says yes",
                             marker=dict(color=COLORS["raw"], size=14, symbol="diamond")))
    style(fig, "false alarm rate (share of non-events warned)", "hit rate (share of events warned)", height=430)
    show(fig)
with c2:
    tot = sum(ct[k] for k in ("hits", "misses", "false_alarms", "correct_negatives"))
    tbl = pd.DataFrame({"event happened": [ct["hits"] / tot, ct["misses"] / tot],
                        "no event": [ct["false_alarms"] / tot, ct["correct_negatives"] / tot]},
                       index=["warning issued", "no warning"])
    st.dataframe(tbl.style.format("{:.2%}"), width="stretch")
    m1, m2 = st.columns(2)
    m1.metric("Hit rate", pct(ct["pod"], signed=False), help="Share of heavy-rain grid-box days that got a warning.")
    m2.metric("False alarm ratio", pct(ct["far"], signed=False), help="Share of warnings with no heavy rain.")
    m1.metric("Raw AIFS hit rate", pct(ct_raw["pod"], signed=False))
    m2.metric("Raw AIFS false alarm ratio", pct(ct_raw["far"], signed=False))
explain("lab5.roc.explain")

# --- C: economic value -----------------------------------------------------------------------------------
st.divider()
st.markdown("#### ② Play C: who benefits? Cost and loss")
a_user = st.slider("Cost of protecting ÷ loss if unprotected (C/L)", 0.01, 0.99, 0.2, 0.01, key="lab5_cl")
fig = go.Figure()
for v, k, name in ((v_idr, "idr", "calibrated (act when p ≥ C/L)"), (v_raw, "raw", "raw AIFS (act when it says yes)"),
                   (v_cl, "clim", "climatological probability")):
    fig.add_trace(go.Scatter(x=alphas, y=v, mode="lines", name=name, line=dict(color=COLORS[k], width=2.5)))
fig.add_vline(x=a_user, line=dict(color=COLORS["ink2"], dash="dot"))
fig.add_hline(y=0, line=dict(color=COLORS["clim"], width=1))
style(fig, "cost/loss ratio C/L", "relative economic value", height=380)
fig.update_xaxes(type="log", tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
fig.update_yaxes(range=[-0.5, 1])
show(fig)
i = int(np.argmin(np.abs(alphas - a_user)))
explain("lab5.value.explain", "lab5.value.maths", cl=f"{a_user:.2f}", v_idr=pct(v_idr[i], signed=False),
        v_raw=pct(v_raw[i], signed=False))

# --- D: a warning index ----------------------------------------------------------------------------------
st.divider()
st.markdown("#### ② Play D: build a heavy-rain warning index")
st.markdown(text("lab5.index.intro"))
c1, c2, c3, c4 = st.columns(4)
opts = [5, 10, 20, 30, 40, 50, 75, 100, 150]
s1 = c1.selectbox("Moderate rain above (mm/day)", opts, index=1, key="lab5_s1")
s2 = c2.selectbox("High above", opts, index=3, key="lab5_s2")
s3 = c3.selectbox("Severe above", opts, index=5, key="lab5_s3")
if not s1 < s2 < s3:
    fixed = sorted({s1, s2, s3} | {v for v in opts if v > max(s1, s2, s3)})[:3]
    s1, s2, s3 = fixed if len(fixed) == 3 else (10, 30, 50)
    st.caption(f"Severity thresholds must increase, so the index uses {s1} / {s2} / {s3} mm/day.")
like = c4.select_slider("Likelihood: medium / high from", options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        value=(0.2, 0.5), key="lab5_like")
SEV = [s1, s2, s3]
MATRIX = np.array([[0, 1, 1],    # moderate rain: low / medium / high likelihood
                   [1, 1, 2],    # high
                   [1, 2, 3]])   # severe


def warning_level(probs: list[np.ndarray]) -> np.ndarray:
    level = np.zeros(probs[0].shape, int)
    for k, pk in enumerate(probs):
        band = np.where(pk >= like[1], 2, np.where(pk >= like[0], 1, -1))
        lvl = np.where(band >= 0, MATRIX[k][np.maximum(band, 0)], 0)
        level = np.maximum(level, lvl)
    return level


mat = go.Figure(go.Heatmap(z=MATRIX, x=[f"low<br>< {like[0]:.0%}", f"medium<br>{like[0]:.0%}–{like[1]:.0%}",
                                        f"high<br>≥ {like[1]:.0%}"],
                           y=[f"moderate > {s1}", f"high > {s2}", f"severe > {s3}"],
                           colorscale=discrete_scale([WARN_COLORS[k] for k in range(4)]), zmin=-0.5, zmax=3.5,
                           showscale=False, text=[[WARN_NAMES[v] for v in row] for row in MATRIX],
                           texttemplate="%{text}", hoverinfo="skip"))
style(mat, "likelihood (calibrated probability)", "severity (mm/day)", height=300)

cases = b.case_dates
c1, c2 = st.columns([1, 2])
with c1:
    show(mat)
    case = st.selectbox("Case-study day (heavy rain over large areas)", range(len(cases)),
                        format_func=lambda i: f"{cases[i]:%d %b %Y}", key="lab5_case")
    region = st.selectbox("Zoom to", list(REGIONS), key="lab5_region")


@st.cache_data(show_spinner="Calibrating every land grid box for this day…")
def case_fields(i: int, lead_: int, thresholds: tuple[int, ...]):
    fc = b.case_forecast(i, lead_)
    xi = idr.mm_day_to_level(np.nan_to_num(fc, nan=0.0))
    probs = [np.full(b.n_land, np.nan, np.float32) for _ in thresholds]
    for z in b.zones_with_model(lead_):
        idx = np.flatnonzero(b.pix_zone == z)
        mz = model(lead_, z)
        for k, t in enumerate(thresholds):
            probs[k][idx] = idr.prob_exceed(mz, xi[idx], int(idr.mm_day_to_level(t)))
    for k in range(len(probs)):
        probs[k][~np.isfinite(fc)] = np.nan
    return fc, probs


fc, probs = case_fields(case, L, tuple(SEV))
obs = b.case_obs(case)
level = warning_level([np.nan_to_num(q) for q in probs]).astype(float)
level[~np.isfinite(probs[0])] = np.nan
obs_sev = np.select([obs > s3, obs > s2, obs > s1], [3, 2, 1], 0).astype(float)
obs_sev[~np.isfinite(obs)] = np.nan
f = int(b.meta.get("coarsen", 1))


def coarse_max(v):
    m_ = b.to_map(v)
    if f == 1:
        return m_
    ny, nx = (m_.shape[0] // f) * f, (m_.shape[1] // f) * f
    blk = m_[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(invalid="ignore"):
        out = np.nanmax(np.where(np.isfinite(blk), blk, -1), axis=(1, 3))
    return np.where(out < 0, np.nan, out)


wscale = discrete_scale([WARN_COLORS[k] for k in range(4)])
with c2:
    view = st.radio("Show", ["Warning index vs what happened", "Raw AIFS vs calibrated probability"], horizontal=True,
                    key="lab5_view")
    m1, m2 = st.columns(2)
    if view.startswith("Warning"):
        with m1:
            show(map_fig(coarse_max(level), b.clat, b.clon, colorscale=wscale, zmin=-0.5, zmax=3.5, region=region,
                         title=f"Warning index, lead day {L}", colorbar=False, hover="level %{z:.0f}", height=420,
                         agg="max"))
        with m2:
            show(map_fig(coarse_max(obs_sev), b.clat, b.clon, colorscale=wscale, zmin=-0.5, zmax=3.5, region=region,
                         title="What IMERG observed (same severity colours)", colorbar=False,
                         hover="severity %{z:.0f}", height=420, agg="max"))
    else:
        with m1:
            show(map_fig(b.coarsen(b.to_map(fc)), b.clat, b.clon, zmin=0, zmax=60, cbar="mm/day", region=region,
                         title=f"Raw AIFS, lead day {L}", hover="%{z:.1f} mm/day", height=420))
        with m2:
            show(map_fig(b.coarsen(b.to_map(probs[1])), b.clat, b.clon, colorscale="Purples", zmin=0, zmax=1,
                         cbar="probability", region=region, title=f"Calibrated P(rain > {s2} mm/day)",
                         hover="%{z:.2f}", height=420))
    st.caption("Maps show the highest level in each 0.2° block. Severity colours on the right use the same thresholds: "
               f"yellow > {s1}, orange > {s2}, red > {s3} mm/day observed.")

# verification of the index on the held-out test pairs
ps = [event_probs(L, float(t), zones)["p"] for t in SEV]
lv = warning_level(ps)
y_mm = np.concatenate([b.test(L, z).y for z in zones]) * 24
ob = np.select([y_mm > s3, y_mm > s2, y_mm > s1], [3, 2, 1], 0)
wts = ev["w"]
tab = pd.DataFrame({f"observed {['below moderate', 'moderate', 'high', 'severe'][j]}":
                    [wts[(lv == i) & (ob == j)].sum() / max(wts[lv == i].sum(), 1e-12) for i in range(4)]
                    for j in range(4)}, index=[WARN_NAMES[i] for i in range(4)])
tab.insert(0, "share of grid-box days", [wts[lv == i].sum() / wts.sum() for i in range(4)])
st.markdown(f"**How the index performed on held-out 2026 days (lead day {L}, {scope.lower()}).** "
            "Each row: when this level was issued, what was observed.")
st.dataframe(tab.style.format("{:.1%}"), width="stretch")
hi_obs = ob >= 2
caught = wts[hi_obs & (lv >= 2)].sum() / max(wts[hi_obs].sum(), 1e-12)
explain("lab5.index.explain", caught=pct(caught, signed=False), s2=s2)

check_yourself("lab5.check", T=T, auc=f"{auc:.2f}", bss=pct(1 - bs["idr"] / bs["clim"]),
               bss_raw=pct(1 - bs["raw"] / bs["clim"]))
