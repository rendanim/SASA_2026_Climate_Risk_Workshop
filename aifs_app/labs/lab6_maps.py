import numpy as np
import pandas as pd
import streamlit as st

from aifs_app.ui import (REGIONS, SKILL_SCALE, bundle, check_yourself, explain, lead, map_fig, predict_first, show,
                         synthetic_banner, text)

b, L = bundle(), lead()
st.title("Lab 6 · Where does AIFS add value?")
st.caption("Objective: case studies across African National Meteorological Services · Notebook §7–8")
synthetic_banner()
st.markdown(text("lab6.intro"))
predict_first("lab6.predict")

MEASURES = {
    "Calibrated AIFS vs climatology": lambda r, i, c: 1 - i / c,
    "Raw AIFS vs climatology": lambda r, i, c: 1 - r / c,
    "Error removed by calibration": lambda r, i, c: 1 - i / r,
}

st.markdown("#### ② Play: skill maps")
c1, c2 = st.columns(2)
measure = c1.radio("Measure", list(MEASURES), key="lab6_measure")
region = c2.selectbox("Region", list(REGIONS), key="lab6_region")
fn = MEASURES[measure]
cols = st.columns(2)
for k, d in enumerate(b.lead_days):
    r, i, c = (b.score_map(d, key) for key in ("raw", "idr", "clim"))
    with np.errstate(invalid="ignore", divide="ignore"):
        field = 100 * fn(r, i, c)
    with cols[k % 2]:
        show(map_fig(field, b.clat, b.clon, colorscale=SKILL_SCALE, zmin=-60, zmax=60, zmid=0, cbar="%",
                     region=region, title=f"Lead day {d}", hover="%{z:.0f}%", height=400), key=f"lab6_map_{d}")
st.caption("Blue = better. Each map averages the notebook's scores over every 2026 test day, then over 0.2° blocks.")
explain("lab6.maps.explain")


@st.cache_data
def region_table() -> pd.DataFrame:
    rows = []
    for name, ((la0, la1), (lo0, lo1)) in REGIONS.items():
        iy = (b.clat >= la0) & (b.clat <= la1)
        ix = (b.clon >= lo0) & (b.clon <= lo1)
        row = {"region": name}
        for d in b.lead_days:
            r, i, c = (np.nanmean(b.score_map(d, key)[np.ix_(iy, ix)]) for key in ("raw", "idr", "clim"))
            row[f"day {d} raw"] = 1 - r / c
            row[f"day {d} calibrated"] = 1 - i / c
        rows.append(row)
    return pd.DataFrame(rows).set_index("region")


st.divider()
st.markdown("#### Skill by region: CRPSS against climatology")
rt = region_table()
st.dataframe(rt.style.format("{:+.0%}").background_gradient(cmap="RdBu", vmin=-0.4, vmax=0.4), width="stretch")
st.caption("Area means over land in each box (equal weight per 0.2° block). Boxes overlap and are approximate.")

st.divider()
st.markdown("#### Skill by climate zone")
show_what = st.radio("Show", ["Calibrated AIFS vs climatology (CRPSS)", "Error removed by calibration"],
                     horizontal=True, key="lab6_zone_what")
rows = []
for z in b.zones:
    row = {"zone": b.zone_code(z), "climate": b.zone_name(z), "grid boxes": b.zone_pixels(z)}
    for d in b.lead_days:
        s = b.zone_scores(d).get(z)
        if s is None:
            row[f"day {d}"] = np.nan
        elif show_what.startswith("Calibrated"):
            row[f"day {d}"] = 1 - s["idr"] / s["clim"]
        else:
            row[f"day {d}"] = 1 - s["idr"] / s["raw"]
    rows.append(row)
zt = pd.DataFrame(rows).set_index("zone")
lead_cols = [f"day {d}" for d in b.lead_days]
vmin, vmax = (-0.4, 0.4) if show_what.startswith("Calibrated") else (0, 0.6)
st.dataframe(zt.style.format("{:+.0%}", subset=lead_cols, na_rep="–").format("{:,}", subset=["grid boxes"])
             .background_gradient(cmap="RdBu" if vmin < 0 else "Blues", subset=lead_cols, vmin=vmin, vmax=vmax),
             width="stretch")
explain("lab6.zones.explain")

best = rt[f"day {L} calibrated"].drop("All of Africa").idxmax()
worst = rt[f"day {L} calibrated"].drop("All of Africa").idxmin()
check_yourself("lab6.check", lead=L, best=best, worst=worst,
               best_v=f"{rt.loc[best, f'day {L} calibrated']:+.0%}", worst_v=f"{rt.loc[worst, f'day {L} calibrated']:+.0%}")
