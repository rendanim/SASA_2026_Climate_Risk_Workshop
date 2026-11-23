"""Verification scores used by the labs. No Streamlit imports here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import idr
from .bundle import TestPairs


def crpss(score: float, ref: float) -> float:
    return float(1 - score / ref) if ref > 0 else float("nan")


@dataclass(frozen=True)
class Decomposition:
    """CRPS_raw = MCB - DSC + UNC with MCB = raw - cal, DSC = clim - cal, UNC = clim (the notebook's identity)."""

    raw: float
    cal: float
    clim: float

    @property
    def mcb(self) -> float:
        return self.raw - self.cal

    @property
    def dsc(self) -> float:
        return self.clim - self.cal

    @property
    def unc(self) -> float:
        return self.clim

    @property
    def skill_raw(self) -> float:
        return crpss(self.raw, self.clim)

    @property
    def skill_cal(self) -> float:
        return crpss(self.cal, self.clim)


# --- per-pair scores on a test sample ----------------------------------------------------------

def pair_scores(model: idr.IDRModel, t: TestPairs) -> dict[str, np.ndarray]:
    """CRPS (mm/hr) of raw AIFS (absolute error), the IDR forecast and the reference, for every pair."""
    raw = np.abs(t.x - t.y)
    cal = idr.crps(model, t.xi, t.yi)
    ref = t.clim_crps if t.clim_crps is not None else idr.crps_clim_training(model, t.yi)
    return dict(raw=raw, idr=cal, clim=ref)


def weighted_mean(values: list[np.ndarray], weights: list[float]) -> float:
    num = sum(float(v.sum()) * w for v, w in zip(values, weights))
    den = sum(v.size * w for v, w in zip(values, weights))
    return num / den if den else float("nan")


# --- binary events -----------------------------------------------------------------------------

def brier(p, o, w=None) -> float:
    p, o = np.asarray(p, float), np.asarray(o, float)
    return float(np.average((p - o) ** 2, weights=w))


@dataclass(frozen=True)
class Reliability:
    mean_p: np.ndarray
    freq: np.ndarray
    share: np.ndarray     # share of forecasts in each bin
    count: np.ndarray     # (weighted) number of forecasts in each bin
    edges: np.ndarray


def reliability(p, o, w=None, n_bins: int = 10) -> Reliability:
    p, o = np.asarray(p, float), np.asarray(o, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    n = np.bincount(b, weights=w, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.bincount(b, weights=w * p, minlength=n_bins) / n
        fr = np.bincount(b, weights=w * o, minlength=n_bins) / n
    return Reliability(mean_p=mp, freq=fr, share=n / n.sum(), count=n, edges=edges)


def contingency(p, o, cutoff: float, w=None) -> dict[str, float]:
    """Warn when p >= cutoff."""
    warn, o = np.asarray(p) >= cutoff, np.asarray(o).astype(bool)
    w = np.ones(o.shape) if w is None else np.asarray(w, float)
    hits, misses = float(w[warn & o].sum()), float(w[~warn & o].sum())
    fa, cn = float(w[warn & ~o].sum()), float(w[~warn & ~o].sum())
    pod = hits / (hits + misses) if hits + misses else float("nan")
    far = fa / (hits + fa) if hits + fa else float("nan")
    pofd = fa / (fa + cn) if fa + cn else float("nan")
    return dict(hits=hits, misses=misses, false_alarms=fa, correct_negatives=cn, pod=pod, far=far, pofd=pofd)


def roc_curve(p, o, w=None, n_cut: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cuts = np.r_[np.linspace(0, 1, n_cut), 1.0001]
    pts = [contingency(p, o, c, w) for c in cuts]
    pofd = np.array([q["pofd"] for q in pts])
    pod = np.array([q["pod"] for q in pts])
    order = np.argsort(pofd)
    auc = float(np.trapezoid(pod[order], pofd[order]))
    return pofd, pod, cuts, auc


def economic_value(p, o, alphas: np.ndarray, w=None, fixed_warn=None) -> np.ndarray:
    """Relative economic value for cost/loss ratios `alphas` (Richardson 2000).

    Users act when p >= alpha (the best rule for reliable probabilities), unless `fixed_warn` (a
    yes/no forecast) is given. V = (E_clim - E_fc) / (E_clim - E_perfect), per unit loss.
    """
    o = np.asarray(o, float)
    w = np.ones_like(o) if w is None else np.asarray(w, float)
    W = w.sum()
    obar = float((w * o).sum() / W)
    out = np.empty(len(alphas))
    for k, a in enumerate(alphas):
        warn = np.asarray(fixed_warn, bool) if fixed_warn is not None else (np.asarray(p) >= a)
        h = (w * (warn & (o > 0))).sum() / W
        f = (w * (warn & (o == 0))).sum() / W
        m = (w * (~warn & (o > 0))).sum() / W
        e_fc = a * (h + f) + m
        e_clim = min(a, obar)
        e_perf = a * obar
        out[k] = (e_clim - e_fc) / (e_clim - e_perf) if e_clim > e_perf else np.nan
    return out


# --- uncertainty of a skill score ------------------------------------------------------------------

def day_block_bootstrap(score: np.ndarray, ref: np.ndarray, day: np.ndarray, n_boot: int = 500,
                        seed: int = 0, w: np.ndarray | None = None) -> np.ndarray:
    """CRPSS resampled by whole test DAYS (all grid boxes on a day share the weather)."""
    rng = np.random.default_rng(seed)
    w = np.ones_like(score) if w is None else w
    ud, inv = np.unique(day, return_inverse=True)
    s_day = np.bincount(inv, weights=w * score)
    r_day = np.bincount(inv, weights=w * ref)
    out = np.empty(n_boot)
    for b in range(n_boot):
        k = np.bincount(rng.integers(0, ud.size, ud.size), minlength=ud.size)
        out[b] = 1 - (k * s_day).sum() / (k * r_day).sum()
    return out


def pair_bootstrap(score: np.ndarray, ref: np.ndarray, n_boot: int = 500, seed: int = 0,
                   w: np.ndarray | None = None) -> np.ndarray:
    """The naive version: resample pairs as if every grid box on every day were independent."""
    rng = np.random.default_rng(seed)
    w = np.ones_like(score) if w is None else w
    out = np.empty(n_boot)
    n = score.size
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        out[b] = 1 - (w[i] * score[i]).sum() / (w[i] * ref[i]).sum()
    return out
