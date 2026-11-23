import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aifs_app.core.experiments import MODES
from aifs_app.core.scores import day_block_bootstrap, pair_bootstrap
from aifs_app.ui import (COLORS, bundle, check_yourself, explain, lead, learning_curve, pct, pooling, predict_first,
                         show, shuffles, style, synthetic_banner, text, zone, zone_pair_scores)

b, L, Z = bundle(), lead(), zone()
st.title("Lab 4 · Uncertainty with limited data")
st.caption("Objective: uncertainty quantification and decisions with limited or noisy data")
synthetic_banner()
st.markdown(text("lab4.intro"))
predict_first("lab4.predict", zone=b.zone_label(Z))

# --- A: how much data? -------------------------------------------------------------------------------------
st.markdown("#### ② Play A: how much training data does calibration need?")
mode = st.radio("What is scarce?", list(MODES), format_func=MODES.get, horizontal=True, key="lab4_mode")
lc = learning_curve(L, Z, mode, 5)
x_label = {"time": "weeks of AIFS runs used for training", "space": "grid boxes pooled",
           "pairs": "training pairs"}[mode]
med = np.nanmedian(lc.skill, axis=1)
lo, hi = np.nanmin(lc.skill, axis=1), np.nanmax(lc.skill, axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.r_[lc.sizes, lc.sizes[::-1]], y=np.r_[hi, lo[::-1]] * 100, fill="toself", mode="lines",
                         fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="range over 5 random draws",
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=lc.sizes, y=med * 100, mode="lines+markers", name="calibrated (median of 5 draws)",
                         line=dict(color=COLORS["idr"], width=2.5), marker=dict(size=8),
                         customdata=np.median(lc.n_pairs, axis=1),
                         hovertemplate="%{x:,}: CRPSS %{y:.1f}%<br>≈ %{customdata:,.0f} training pairs<extra></extra>"))
fig.add_hline(y=lc.full_skill * 100, line=dict(color=COLORS["idr"], dash="dot"),
              annotation_text="all stored training pairs", annotation_position="top left")
fig.add_hline(y=0, line=dict(color=COLORS["clim"], dash="dash"), annotation_text="climatology",
              annotation_position="bottom right")
style(fig, x_label, "CRPSS vs climatology (%)", height=420)
fig.update_xaxes(type="log", tickvals=lc.sizes, ticktext=[f"{v:,}" for v in lc.sizes])
show(fig)
st.caption(f"Raw AIFS on the same test pairs: CRPSS {pct(lc.raw_skill)}. Each point refits IDR on a random subset "
           "of the zone's training pairs and scores it on 5,000 held-out 2026 pairs from the zone.")
explain(f"lab4.lc.{mode}.explain", "lab4.lc.maths")

# --- B: how widely to pool? --------------------------------------------------------------------------------
st.divider()
st.markdown("#### ② Play B: how widely should we pool?")
pr = pooling(L)
labels = [b.zone_code(z) for z in pr.zones]
fig = go.Figure()
fig.add_trace(go.Bar(x=labels, y=pr.local * 100, name=f"local: {pr.n_local_boxes} grid boxes only",
                     marker_color=COLORS["local"]))
fig.add_trace(go.Bar(x=labels, y=pr.zone * 100, name="one model per climate zone (notebook)", marker_color=COLORS["idr"]))
fig.add_trace(go.Bar(x=labels, y=pr.africa * 100, name="one model for all of Africa", marker_color=COLORS["compare"]))
fig.add_trace(go.Scatter(x=labels, y=pr.raw * 100, mode="markers", name="raw AIFS",
                         marker=dict(color=COLORS["raw"], symbol="line-ew-open", size=22, line=dict(width=3))))
fig.add_hline(y=0, line=dict(color=COLORS["clim"], dash="dash"))
style(fig, "climate zone (largest first)", "CRPSS vs climatology (%)", height=430, barmode="group")
show(fig)
best = {"local": int(np.nansum(pr.local >= np.fmax(pr.zone, pr.africa))),
        "zone": int(np.nansum(pr.zone >= np.fmax(pr.local, pr.africa))),
        "africa": int(np.nansum(pr.africa > np.fmax(pr.local, pr.zone)))}
explain("lab4.pool.explain", n_local=pr.n_local_boxes, n_local_pairs=f"{int(np.median(pr.n_local)):,}",
        n_africa=f"{pr.n_africa:,}", best_zone=best["zone"], best_africa=best["africa"], best_local=best["local"],
        n_zones=len(pr.zones))

# --- C: is the skill real? -----------------------------------------------------------------------------------
st.divider()
st.markdown("#### ② Play C: is the skill real, and how sure are we?")
s = zone_pair_scores(L, Z)
te = b.test(L, Z)
boot_day = day_block_bootstrap(s["idr"], s["clim"], te.day, n_boot=400)
boot_pair = pair_bootstrap(s["idr"], s["clim"], n_boot=400)
point = 1 - s["idr"].mean() / s["clim"].mean()
q_day, q_pair = np.percentile(boot_day, [5, 95]), np.percentile(boot_pair, [5, 95])
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=boot_pair * 100, name="resampling pairs (assumes independence)", opacity=0.6,
                               marker_color=COLORS["point"], nbinsx=40))
    fig.add_trace(go.Histogram(x=boot_day * 100, name="resampling whole days", opacity=0.6,
                               marker_color=COLORS["idr"], nbinsx=40))
    fig.add_vline(x=point * 100, line=dict(color=COLORS["ink"], width=2), annotation_text="estimate")
    style(fig, "CRPSS of the calibrated forecast (%)", "bootstrap samples", height=380, barmode="overlay")
    show(fig)
    st.markdown(f"90% interval, resampling **days**: **{pct(q_day[0])} to {pct(q_day[1])}**  \n"
                f"90% interval, resampling **pairs**: {pct(q_pair[0])} to {pct(q_pair[1])}  \n"
                f"{te.n:,} test pairs, but only **{np.unique(te.day).size} distinct days**.")
with c2:
    sh = shuffles(L, Z)
    fig = go.Figure()
    fig.add_trace(go.Box(x=sh["shuffled"] * 100, name="AIFS shuffled<br>before fitting", marker_color=COLORS["point"],
                         boxpoints="all", jitter=0.4, orientation="h"))
    fig.add_trace(go.Scatter(x=[sh["real"] * 100], y=["real AIFS"], mode="markers", name="real AIFS",
                             marker=dict(color=COLORS["idr"], size=14)))
    fig.add_trace(go.Scatter(x=[sh["train_clim"] * 100], y=["training-period<br>climatology"], mode="markers",
                             name="training-period climatology", marker=dict(color=COLORS["clim"], size=14, symbol="diamond")))
    fig.add_vline(x=0, line=dict(color=COLORS["clim"], dash="dash"))
    style(fig, "CRPSS vs IMERG climatology (%)", "", height=380, showlegend=False)
    show(fig)
explain("lab4.real.explain", "lab4.real.maths", n_days=np.unique(te.day).size,
        shuffled=pct(float(np.median(sh["shuffled"]))), real=pct(sh["real"]), train_clim=pct(sh["train_clim"]))

check_yourself("lab4.check", zone=b.zone_label(Z))
