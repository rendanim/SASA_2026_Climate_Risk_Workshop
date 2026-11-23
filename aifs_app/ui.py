"""Streamlit helpers shared by the labs: caching, sidebar, content, lab rhythm, chart and map styling."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aifs_app.core import experiments, idr
from aifs_app.core.bundle import Bundle, BundleMissing

CONTENT_FILE = Path(__file__).parent / "content" / "text.md"

# Same roles and colours as the notebook: raw AIFS orange, calibrated blue, climatology grey.
COLORS = {
    "raw": "#eb6834",
    "idr": "#2a78d6",
    "clim": "#898781",
    "obs": "#0b0b0b",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e9e8e4",
    "point": "#8a8984",
    "compare": "#9b59b6",
    "local": "#1baf7a",
}
LOG_SHARE_TICKS = dict(tickvals=[1e-4, 1e-3, 1e-2, 0.1, 1], ticktext=["0.01%", "0.1%", "1%", "10%", "100%"])
AMOUNT_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
RAIN_SCALE = [[0, "#f7fbff"], [0.05, "#deebf7"], [0.15, "#9ecae1"], [0.3, "#4292c6"], [0.55, "#2171b5"],
              [0.8, "#08519c"], [1, "#08306b"]]
SKILL_SCALE = "RdBu"
WARN_COLORS = {0: "#e8f3e8", 1: "#f7e04b", 2: "#f39c34", 3: "#d7301f"}
WARN_NAMES = {0: "No warning", 1: "Yellow", 2: "Orange", 3: "Red"}

# Regions used by African climate centres, for zooming the maps (lat range, lon range).
REGIONS = {
    "All of Africa": ((-35.0, 38.0), (-18.0, 52.0)),
    "East Africa (ICPAC region)": ((-12.0, 15.0), (28.0, 52.0)),
    "West Africa and Sahel (ACMAD / AGRHYMET)": ((4.0, 20.0), (-18.0, 16.0)),
    "Southern Africa (SADC CSC)": ((-35.0, -8.0), (10.0, 41.0)),
    "Central Africa (ECCAS)": ((-10.0, 10.0), (8.0, 32.0)),
    "North Africa": ((20.0, 38.0), (-18.0, 35.0)),
}


# --- data (cached) ---------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the data bundle…")
def bundle() -> Bundle:
    return Bundle()


def require_bundle() -> Bundle:
    try:
        return bundle()
    except BundleMissing as e:
        st.error(str(e), icon=":material/folder_off:")
        st.stop()


@st.cache_resource(show_spinner="Fitting the zone's IDR model…")
def model(lead: int, zone: int) -> idr.IDRModel:
    return bundle().model(lead, zone)


@st.cache_resource(show_spinner="Re-fitting IDR on smaller training sets (a few seconds)…")
def learning_curve(lead: int, zone: int, mode: str, repeats: int):
    return experiments.learning_curve(bundle(), lead, zone, mode, repeats=repeats)


@st.cache_resource(show_spinner="Fitting one model for all of Africa and small local models…")
def pooling(lead: int):
    return experiments.pooling_comparison(bundle(), lead)


@st.cache_resource(show_spinner="Re-fitting IDR on shuffled forecasts…")
def shuffles(lead: int, zone: int):
    return experiments.shuffle_control(bundle(), lead, zone)


@st.cache_resource(show_spinner="Scoring the test sample…")
def zone_pair_scores(lead: int, zone: int):
    from aifs_app.core.scores import pair_scores
    return pair_scores(model(lead, zone), bundle().test(lead, zone))


@st.cache_resource(show_spinner="Computing event probabilities for every test pair…")
def event_probs(lead: int, t_mm_day: float, zones: tuple[int, ...]):
    """IDR, raw yes/no and reference probabilities of rain > t, with outcomes and weights, pooled over zones."""
    b = bundle()
    lvl = int(idr.mm_day_to_level(t_mm_day))
    use_imerg_clim = abs(t_mm_day - b.event_mm_day) < 1e-9
    P, R, C, O, W, D, Z = [], [], [], [], [], [], []
    for z in zones:
        t, m = b.test(lead, z), model(lead, z)
        P.append(idr.prob_exceed(m, t.xi, lvl))
        R.append((t.xi > lvl).astype(float))
        if use_imerg_clim and t.clim_p is not None:
            C.append(t.clim_p)
        else:
            C.append(np.full(t.n, idr.clim_prob_exceed(m, lvl)))
        O.append((t.yi > lvl).astype(float))
        W.append(np.full(t.n, t.weight))
        D.append(t.day)
        Z.append(np.full(t.n, z))
    cat = np.concatenate
    return dict(p=cat(P), raw=cat(R), clim=cat(C), obs=cat(O), w=cat(W), day=cat(D), zone=cat(Z),
                imerg_clim=bool(use_imerg_clim and all(b.test(lead, z).clim_p is not None for z in zones)))


# --- sidebar ---------------------------------------------------------------------------------------

def sidebar() -> None:
    b = require_bundle()
    with st.sidebar:
        st.markdown("### Forecast settings")
        st.selectbox("Lead day", b.lead_days, index=0, key="lead",
                     format_func=lambda d: f"Day {d}  ({24 * (d - 1)}–{24 * d} h after the 00 UTC run)",
                     help="Lead day 2 is the rain from 24 to 48 hours after the forecast starts.")
        zones = b.zones_with_model(st.session_state.get("lead", b.lead_days[0]))
        default = b.default_zone() if b.default_zone() in zones else zones[0]
        if st.session_state.get("zone") not in zones:
            st.session_state["zone"] = default
        st.selectbox("Climate zone", zones, key="zone", format_func=b.zone_label,
                     help="One IDR model is fitted per Köppen–Geiger zone, pooling grid boxes with similar rain climates.")
        if b.synthetic:
            st.warning("**Synthetic demo data.** Run `python aifs_app/prepare_data.py` to build the real "
                       "AIFS/IMERG bundle before the workshop.", icon=":material/science:")
        else:
            st.caption(f"Data: AIFS-Single and GPM IMERG over Africa. Training: {b.periods.get('train', '')}. "
                       f"Test: {b.periods.get('test', '')}.")


def lead() -> int:
    return int(st.session_state.get("lead", bundle().lead_days[0]))


def zone() -> int:
    b = bundle()
    z = st.session_state.get("zone")
    zones = b.zones_with_model(lead())
    return int(z) if z in zones else (b.default_zone() if b.default_zone() in zones else zones[0])


def synthetic_banner() -> None:
    if bundle().synthetic:
        st.warning("These numbers come from **synthetic demo data**, not real AIFS or IMERG.", icon=":material/science:")


# --- content -----------------------------------------------------------------------------------------

@st.cache_data
def _content(mtime: float) -> dict[str, str]:
    text_ = re.sub(r"<!--.*?-->", "", CONTENT_FILE.read_text(encoding="utf-8"), flags=re.S)
    blocks: dict[str, str] = {}
    for part in re.split(r"^## ", text_, flags=re.M)[1:]:
        key, _, body = part.partition("\n")
        blocks[key.strip()] = body.strip()
    return blocks


class _Keep(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def text(key: str, **values) -> str:
    raw = _content(CONTENT_FILE.stat().st_mtime).get(key, f"*[missing text: {key}]*")
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(_Keep(values)[m.group(1)]), raw)


# --- the lab rhythm ----------------------------------------------------------------------------------

def predict_first(key: str, **values) -> None:
    with st.container(border=True):
        st.markdown("**① Predict first**  \n" + text(f"{key}.q", **values))
        with st.expander("Reveal the answer"):
            st.markdown(text(f"{key}.a", **values))


def explain(key: str, maths_key: str | None = None, **values) -> None:
    st.markdown(text(key, **values))
    if maths_key:
        with st.expander("The maths"):
            st.markdown(text(maths_key, **values))


def _save_answer(key: str) -> None:
    st.session_state[f"_saved_{key}"] = st.session_state.get(key, "")


def answer_box(key: str, height: int = 90) -> str:
    store = f"_saved_{key}"
    if key not in st.session_state and store in st.session_state:
        st.session_state[key] = st.session_state[store]
    value = st.text_area("Your answer (kept while this browser tab is open)", key=key, height=height,
                         on_change=_save_answer, args=(key,))
    st.session_state[store] = value
    return value


def check_yourself(key: str, heading: bool = True, **values) -> None:
    if heading:
        st.divider()
        st.markdown("#### ④ Check yourself")
    with st.container(border=True):
        st.markdown(text(f"{key}.q", **values))
        answer_box(f"answer_{key}")
        with st.expander("Reveal an answer built from the numbers on your screen"):
            st.markdown(text(f"{key}.a", **values))


# --- formatting -----------------------------------------------------------------------------------

def pct(v: float, signed: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{100 * v:+.0f}%" if signed else f"{100 * v:.0f}%"


def mmday(v_mm_hr: float) -> str:
    return f"{v_mm_hr * 24:.2f} mm/day"


def versus_clim(v: float) -> str:
    if v > 0.005:
        return f"beats climatology (CRPSS {pct(v)})"
    if v < -0.005:
        return f"is worse than climatology (CRPSS {pct(v)})"
    return "does no better than climatology"


# --- charts ------------------------------------------------------------------------------------------

def style(fig: go.Figure, x_title: str, y_title: str, height: int = 420, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=70, r=20, t=40, b=60),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(color=COLORS["ink2"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_color=COLORS["ink"]),
        **layout,
    )
    axis = dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, linecolor="#c3c2b7",
                ticks="outside", tickcolor="#c3c2b7", automargin=True)
    fig.update_xaxes(title=x_title, **axis)
    fig.update_yaxes(title=y_title, **axis)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def map_fig(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, *, title: str = "", colorscale=RAIN_SCALE,
            zmin=None, zmax=None, cbar: str = "", region: str = "All of Africa", height: int = 430,
            hover: str = "%{z:.1f}", zmid=None, colorbar=True, tickvals=None, ticktext=None,
            agg: str = "mean", max_cells: int = 45_000) -> go.Figure:
    (la0, la1), (lo0, lo1) = REGIONS[region]
    iy = (lat >= la0 - 0.01) & (lat <= la1 + 0.01)
    ix = (lon >= lo0 - 0.01) & (lon <= lo1 + 0.01)
    f, la, lo = field[np.ix_(iy, ix)], lat[iy], lon[ix]
    k = int(np.ceil(np.sqrt(f.size / max_cells)))
    if k > 1:   # keep the browser payload small: block-aggregate k x k cells
        f, la, lo = _block(f, k, agg), _block_axis(la, k), _block_axis(lo, k)
    fig = go.Figure(go.Heatmap(
        z=f, x=lo, y=la, colorscale=colorscale, zmin=zmin, zmax=zmax, zmid=zmid,
        showscale=colorbar, hoverongaps=False,
        colorbar=dict(title=dict(text=cbar, side="right"), thickness=12, len=0.85, tickvals=tickvals, ticktext=ticktext),
        hovertemplate="%{y:.1f}°N, %{x:.1f}°E<br>" + hover + "<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=40, r=10, t=36 if title else 10, b=30),
                      title=dict(text=title, font=dict(size=14), x=0.01) if title else None,
                      paper_bgcolor="#fcfcfb", plot_bgcolor="#eef1f4", font=dict(color=COLORS["ink2"], size=12))
    fig.update_xaxes(showgrid=False, zeroline=False, ticksuffix="°", constrain="domain")
    fig.update_yaxes(showgrid=False, zeroline=False, ticksuffix="°", scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def _block(f: np.ndarray, k: int, agg: str) -> np.ndarray:
    ny, nx = (f.shape[0] // k) * k, (f.shape[1] // k) * k
    b = f[:ny, :nx].reshape(ny // k, k, nx // k, k).astype(np.float64)
    fin = np.isfinite(b)
    n = fin.sum(axis=(1, 3))
    if agg == "first":   # categories: keep one representative cell per block
        return f[:ny:k, :nx:k].astype(np.float64)
    if agg == "max":
        out = np.where(fin, b, -np.inf).max(axis=(1, 3))
    else:
        out = np.where(fin, b, 0).sum(axis=(1, 3)) / np.maximum(n, 1)
    return np.where(n > 0, out, np.nan)


def _block_axis(v: np.ndarray, k: int) -> np.ndarray:
    n = (v.size // k) * k
    return v[:n].reshape(-1, k).mean(axis=1)


def discrete_scale(colors: list[str]) -> list:
    """Plotly colorscale with one flat band per category 0..n-1."""
    n = len(colors)
    out = []
    for i, c in enumerate(colors):
        out += [[i / n, c], [(i + 1) / n, c]]
    return out
