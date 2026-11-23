"""Isotonic distributional regression on a rainfall grid, ported from the AIFS → IMERG notebook.

Everything works on integer *levels*: rainfall in mm/hr divided by RES (0.1 mm/day expressed in
mm/hr) and rounded, so a forecast of 12.3 mm/day is level 123. This is exactly the notebook's
`fit_idr` / `predict_cdf`; the CRPS here is the same step-function CRPS as the notebook's lookup
tables, computed pair by pair so that models can be refitted cheaply inside the app.

No Streamlit imports here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_DAY = 1 / 24              # 1 mm/day in mm/hr (the notebook scores everything in mm/hr)
RES = 0.1 * MM_DAY           # one level = 0.1 mm/day

try:  # numba makes PAVA over thousands of thresholds take well under a second
    from numba import njit, prange

    @njit(cache=True)
    def _antitonic(y, w):
        # Pool-adjacent-violators: weighted least-squares fit of y that never increases.
        n = y.size
        val, wt, cnt = np.empty(n), np.empty(n), np.empty(n, np.int64)
        k = -1
        for i in range(n):
            k += 1
            val[k], wt[k], cnt[k] = y[i], w[i], 1
            while k > 0 and val[k - 1] < val[k]:
                tw = wt[k - 1] + wt[k]
                val[k - 1] = (val[k - 1] * wt[k - 1] + val[k] * wt[k]) / tw
                wt[k - 1], cnt[k - 1] = tw, cnt[k - 1] + cnt[k]
                k -= 1
        out, i = np.empty(n), 0
        for b in range(k + 1):
            for _ in range(cnt[b]):
                out[i] = val[b]
                i += 1
        return out

    @njit(parallel=True, cache=True)
    def _antitonic_columns(F, w):
        out = np.empty_like(F)
        for t in prange(F.shape[1]):
            out[:, t] = _antitonic(F[:, t].copy(), w)
        return out

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - slow but correct fallback
    HAVE_NUMBA = False

    def _antitonic(y, w):
        vals, wts, cnts = [], [], []
        for yi, wi in zip(y, w):
            vals.append(float(yi)); wts.append(float(wi)); cnts.append(1)
            while len(vals) > 1 and vals[-2] < vals[-1]:
                tw = wts[-2] + wts[-1]
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
                c = cnts[-2] + cnts[-1]
                del vals[-1], wts[-1], cnts[-1]
                vals[-1], wts[-1], cnts[-1] = v, tw, c
        return np.repeat(vals, cnts)

    def _antitonic_columns(F, w):
        return np.column_stack([_antitonic(F[:, t], w) for t in range(F.shape[1])])


@dataclass(frozen=True)
class IDRModel:
    """One fitted IDR model (the notebook's dict(x, z, cdf, clim))."""

    x: np.ndarray      # (R,) distinct forecast levels in the training data, ascending
    z: np.ndarray      # (Z,) distinct observed levels, ascending
    cdf: np.ndarray    # (R, Z) P(obs <= z | forecast = x), antitonic in x
    clim: np.ndarray   # (Z,) training-sample climatology (ignores the forecast)
    n_train: int

    @property
    def max_forecast_level(self) -> int:
        return int(self.x[-1])


def to_level(v_mm_hr) -> np.ndarray:
    return np.rint(np.asarray(v_mm_hr, dtype=np.float64) / RES).astype(np.int64)


def mm_day_to_level(v_mm_day) -> np.ndarray:
    return np.rint(np.asarray(v_mm_day, dtype=np.float64) * 10).astype(np.int64)


def level_to_mm_day(level) -> np.ndarray:
    return np.asarray(level, dtype=np.float64) / 10


def fit_levels(xi: np.ndarray, yi: np.ndarray) -> IDRModel:
    """Fit IDR to integer forecast/observation levels (notebook `fit_idr`, after rounding)."""
    xi = np.asarray(xi, np.int64)
    yi = np.asarray(yi, np.int64)
    if xi.size == 0:
        raise ValueError("No training pairs")
    ux, gx, nx = np.unique(xi, return_inverse=True, return_counts=True)
    z, gy = np.unique(yi, return_inverse=True)
    counts = np.bincount(gx * z.size + gy, minlength=ux.size * z.size).reshape(ux.size, z.size)
    F = np.cumsum(counts, axis=1) / nx[:, None]
    F = _antitonic_columns(F.astype(np.float64), nx.astype(np.float64))
    clim = np.cumsum(counts.sum(axis=0)) / xi.size
    return IDRModel(x=ux, z=z, cdf=F, clim=clim, n_train=int(xi.size))


def fit_idr(x_mm_hr: np.ndarray, y_mm_hr: np.ndarray) -> IDRModel:
    """Notebook `fit_idr`: inputs in mm/hr."""
    return fit_levels(to_level(x_mm_hr), to_level(y_mm_hr))


def predict_cdf(model: IDRModel, xi) -> np.ndarray:
    """Predictive CDFs (rows) at the observed levels model.z, for forecast levels xi.

    Linear interpolation between the nearest training forecasts, constant beyond them.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    pos = np.interp(xi, model.x, np.arange(model.x.size))
    il = np.floor(pos).astype(np.int64)
    ir = np.minimum(il + 1, model.x.size - 1)
    w = (pos - il)[:, None]
    return (1 - w) * model.cdf[il] + w * model.cdf[ir]


def cdf_at(model: IDRModel, xi, t_level) -> np.ndarray:
    """P(obs <= t | forecast) for each forecast level, at threshold level t (scalar or per-row)."""
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    pos = np.interp(xi, model.x, np.arange(model.x.size))
    il = np.floor(pos).astype(np.int64)
    ir = np.minimum(il + 1, model.x.size - 1)
    w = pos - il
    t = np.broadcast_to(np.asarray(t_level, dtype=np.int64), xi.shape)
    j = np.searchsorted(model.z, t, side="right") - 1
    jj = np.maximum(j, 0)
    val = (1 - w) * model.cdf[il, jj] + w * model.cdf[ir, jj]
    return np.where(j >= 0, val, 0.0)


def prob_exceed(model: IDRModel, xi, t_level) -> np.ndarray:
    """P(obs > t | forecast)."""
    return 1.0 - cdf_at(model, xi, t_level)


def clim_prob_exceed(model: IDRModel, t_level: int) -> float:
    j = np.searchsorted(model.z, int(t_level), side="right") - 1
    return float(1.0 - (model.clim[j] if j >= 0 else 0.0))


def quantiles(model: IDRModel, xi, ps=(0.1, 0.5, 0.9)) -> np.ndarray:
    """(n, len(ps)) quantile levels: the first observed level where the CDF reaches p (notebook q10/q50/q90)."""
    F = predict_cdf(model, xi)
    out = np.empty((F.shape[0], len(ps)))
    for k, p in enumerate(ps):
        out[:, k] = model.z[np.argmax(F >= p - 1e-12, axis=1)]
    return out


def crps_levels(F: np.ndarray, z: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """Exact CRPS (in levels) of step CDFs F (n, Z) on levels z, for observations yi (n,).

    Same integral as the notebook's lookup tables: unit cells g = 0, 1, 2, ...; F is 0 below z[0],
    F[k] on [z[k], z[k+1]) and 1 from z[-1] on; cells below the observation add F², cells from the
    observation upwards add (1 - F)².
    """
    yi = np.asarray(yi, dtype=np.float64)[:, None]
    zl, zr = z[:-1].astype(np.float64)[None, :], z[1:].astype(np.float64)[None, :]
    Fk = F[:, :-1]
    below = np.clip(yi - zl, 0, zr - zl)            # cells in the segment that lie below y
    above = (zr - zl) - below
    seg = (Fk * Fk * below + (1 - Fk) ** 2 * above).sum(axis=1)
    y = yi[:, 0]
    lower_tail = np.maximum(z[0] - np.maximum(y, 0), 0)  # F = 0 there, cells >= y add 1
    upper_tail = np.maximum(y - z[-1], 0)              # F = 1 there, cells < y add 1
    return seg + lower_tail + upper_tail


def crps(model: IDRModel, xi: np.ndarray, yi: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """CRPS in mm/hr of the IDR forecast for forecast levels xi and observed levels yi."""
    xi, yi = np.asarray(xi), np.asarray(yi)
    out = np.empty(xi.size)
    for s in range(0, xi.size, chunk):
        F = predict_cdf(model, xi[s:s + chunk])
        out[s:s + chunk] = crps_levels(F, model.z, yi[s:s + chunk])
    return out * RES


def crps_clim_training(model: IDRModel, yi: np.ndarray) -> np.ndarray:
    """CRPS (mm/hr) of the model's training-sample climatology, which ignores the forecast."""
    F = np.broadcast_to(model.clim, (len(yi), model.z.size))
    return crps_levels(np.asarray(F), model.z, yi) * RES


def lookup_tables(model: IDRModel, max_obs_level: int, event_level: int) -> dict:
    """Notebook `lookup_tables`, kept for parity checks against the full-table method."""
    R = int(model.x[-1]) + 1
    G = max(int(max_obs_level), int(model.z[-1])) + 1
    rows = predict_cdf(model, np.arange(R))
    j = np.searchsorted(model.z, np.arange(G), side="right") - 1
    Fg = np.where(j >= 0, rows[..., np.maximum(j, 0)], 0.0).astype(np.float32)
    Fc = np.where(j >= 0, model.clim[np.maximum(j, 0)], 0.0)

    def _crps_from_cdf(F):
        sq = F * F
        below = np.cumsum(sq, axis=-1) - sq
        above = np.cumsum(((1 - F) ** 2)[..., ::-1], axis=-1)[..., ::-1]
        return (RES * (below + above)).astype(np.float32)

    ev = int(event_level)
    return dict(R=R, crps=_crps_from_cdf(Fg), crps_clim=_crps_from_cdf(Fc),
                p_event=1 - Fg[:, min(ev, G - 1)] if ev < G else np.zeros(R, np.float32))
