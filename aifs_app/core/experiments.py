"""Live experiments for the data-scarcity and pooling labs. No Streamlit imports here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import idr
from .bundle import Bundle, TestPairs, TrainPairs
from .scores import crpss, pair_scores


def test_subset(t: TestPairs, n_max: int, seed: int = 0) -> TestPairs:
    if t.n <= n_max:
        return t
    idx = np.sort(np.random.default_rng(seed).choice(t.n, n_max, replace=False))
    return t.take(idx)


def score_model(model: idr.IDRModel, t: TestPairs) -> dict[str, float]:
    s = pair_scores(model, t)
    out = {k: float(v.mean()) for k, v in s.items()}
    out["crpss"] = crpss(out["idr"], out["clim"])
    out["crpss_raw"] = crpss(out["raw"], out["clim"])
    return out


def _fit_ok(tr: TrainPairs) -> bool:
    return tr.n >= 20 and np.unique(tr.yi).size >= 2


# --- A: how much training data does calibration need? ----------------------------------------------

MODES = {
    "time": "Time: only N weeks of AIFS runs",
    "space": "Space: pool only N grid boxes",
    "pairs": "Pairs: N random training pairs",
}


def default_sizes(mode: str, tr: TrainPairs) -> list[int]:
    if mode == "time":
        weeks = int(np.ceil((tr.day.max() + 1) / 7))
        return sorted({w for w in (1, 2, 4, 8, 13, 26, weeks) if w <= weeks})
    if mode == "space":
        nloc = int(np.unique(tr.loc).size)
        return sorted({n for n in (1, 3, 10, 30, 100, 300, nloc) if n <= nloc})
    return sorted({n for n in (100, 300, 1000, 3000, 10000, tr.n) if n <= tr.n})


def subsample(tr: TrainPairs, mode: str, size: int, rng: np.random.Generator) -> TrainPairs:
    if mode == "time":
        n_days = int(tr.day.max()) + 1
        span = min(size * 7, n_days)
        start = int(rng.integers(0, n_days - span + 1))
        return tr.subset((tr.day >= start) & (tr.day < start + span))
    if mode == "space":
        locs = np.unique(tr.loc)
        keep = rng.choice(locs, size=min(size, locs.size), replace=False)
        return tr.subset(np.isin(tr.loc, keep))
    idx = rng.choice(tr.n, size=min(size, tr.n), replace=False)
    m = np.zeros(tr.n, bool)
    m[idx] = True
    return tr.subset(m)


@dataclass(frozen=True)
class LearningCurve:
    mode: str
    sizes: np.ndarray        # (S,)
    n_pairs: np.ndarray      # (S, R) training pairs actually used
    skill: np.ndarray        # (S, R) CRPSS vs climatology (NaN if the fit was impossible)
    full_skill: float        # notebook model (all stored pairs)
    raw_skill: float         # raw AIFS on the same test pairs


def learning_curve(b: Bundle, lead: int, zone: int, mode: str, sizes=None, repeats: int = 5,
                   n_test: int = 5000, seed: int = 0) -> LearningCurve:
    tr = b.train(lead, zone)
    te = test_subset(b.test(lead, zone), n_test, seed)
    sizes = np.array(sizes if sizes is not None else default_sizes(mode, tr))
    rng = np.random.default_rng(seed)
    skill = np.full((sizes.size, repeats), np.nan)
    npairs = np.zeros((sizes.size, repeats), int)
    ref = te.clim_crps if te.clim_crps is not None else None
    for i, s in enumerate(sizes):
        for r in range(repeats):
            sub = subsample(tr, mode, int(s), rng)
            npairs[i, r] = sub.n
            if not _fit_ok(sub):
                continue
            m = idr.fit_levels(sub.xi, sub.yi)
            c = idr.crps(m, te.xi, te.yi).mean()
            rr = ref.mean() if ref is not None else idr.crps_clim_training(m, te.yi).mean()
            skill[i, r] = crpss(c, rr)
    full = score_model(b.model(lead, zone), te)
    return LearningCurve(mode, sizes, npairs, skill, full["crpss"], full["crpss_raw"])


# --- B: how widely to pool? ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolingResult:
    zones: list[int]
    africa: np.ndarray     # CRPSS per zone: one model for every land grid box
    zone: np.ndarray       # CRPSS per zone: one model per climate zone (the notebook)
    local: np.ndarray      # CRPSS per zone: a model from only `n_local_boxes` grid boxes
    raw: np.ndarray        # CRPSS of raw AIFS
    n_africa: int
    n_local: np.ndarray    # training pairs in each local model
    n_local_boxes: int


def pooling_comparison(b: Bundle, lead: int, n_test: int = 3000, n_local_boxes: int = 10,
                       seed: int = 0) -> PoolingResult:
    rng = np.random.default_rng(seed)
    zones = b.zones_with_model(lead)
    total = int(b.meta["num_data_pts"])
    area = np.array([b.zone_pixels(z) for z in zones], float)
    take = np.maximum(1, np.round(total * area / area.sum())).astype(int)
    xs, ys = [], []
    for z, k in zip(zones, take):
        tr = b.train(lead, z)
        i = rng.choice(tr.n, size=min(k, tr.n), replace=False)
        xs.append(tr.xi[i])
        ys.append(tr.yi[i])
    m_af = idr.fit_levels(np.concatenate(xs), np.concatenate(ys))
    af, zn, lo, rw, nl = [], [], [], [], []
    for z in zones:
        te = test_subset(b.test(lead, z), n_test, seed)
        ref = te.clim_crps.mean() if te.clim_crps is not None else None
        s_af = idr.crps(m_af, te.xi, te.yi).mean()
        s_zn = idr.crps(b.model(lead, z), te.xi, te.yi).mean()
        tr = b.train(lead, z)
        sub = subsample(tr, "space", n_local_boxes, rng)
        nl.append(sub.n)
        if _fit_ok(sub):
            m_lo = idr.fit_levels(sub.xi, sub.yi)
            s_lo = idr.crps(m_lo, te.xi, te.yi).mean()
        else:
            s_lo = np.nan
        r = ref if ref is not None else idr.crps_clim_training(b.model(lead, z), te.yi).mean()
        af.append(crpss(s_af, r))
        zn.append(crpss(s_zn, r))
        lo.append(crpss(s_lo, r))
        rw.append(crpss(np.abs(te.x - te.y).mean(), r))
    return PoolingResult(zones, np.array(af), np.array(zn), np.array(lo), np.array(rw), int(sum(map(len, xs))),
                         np.array(nl), n_local_boxes)


# --- C: can calibration create skill? -------------------------------------------------------------------

def shuffle_control(b: Bundle, lead: int, zone: int, repeats: int = 20, n_test: int = 5000,
                    seed: int = 0) -> dict[str, np.ndarray | float]:
    """Refit after shuffling the forecasts in the training pairs, breaking the forecast-observation link."""
    tr = b.train(lead, zone)
    te = test_subset(b.test(lead, zone), n_test, seed)
    ref = te.clim_crps.mean() if te.clim_crps is not None else idr.crps_clim_training(b.model(lead, zone), te.yi).mean()
    real = crpss(idr.crps(b.model(lead, zone), te.xi, te.yi).mean(), ref)
    out = []
    for r in range(repeats):
        xs = np.random.default_rng(seed + r).permutation(tr.xi)
        out.append(crpss(idr.crps(idr.fit_levels(xs, tr.yi), te.xi, te.yi).mean(), ref))
    m = b.model(lead, zone)
    train_clim = crpss(idr.crps_clim_training(m, te.yi).mean(), ref)
    return dict(real=real, shuffled=np.array(out), train_clim=train_clim)
