"""Numerical checks of the IDR engine and scores:  python -m pytest aifs_app/tests -q"""
import numpy as np
import pytest

from aifs_app.core import idr
from aifs_app.core.bundle import Bundle
from aifs_app.core.scores import Decomposition, contingency, day_block_bootstrap, economic_value, reliability


def _toy(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.gamma(0.6, 6, n) * (rng.random(n) < 0.6)
    y = np.maximum(0, x * rng.lognormal(0, 0.8, n) * (rng.random(n) < 0.7) + rng.gamma(0.3, 3, n) * (rng.random(n) < 0.2))
    return x * idr.MM_DAY, y * idr.MM_DAY


def test_crps_matches_notebook_lookup_tables():
    x, y = _toy()
    m = idr.fit_idr(x, y)
    xt, yt = _toy(3000, seed=2)
    xi, yi = idr.to_level(xt), idr.to_level(yt)
    tab = idr.lookup_tables(m, yi.max(), 100)
    ref = tab["crps"][np.clip(xi, 0, tab["R"] - 1), yi]
    assert np.allclose(idr.crps(m, xi, yi), ref, atol=1e-5)
    assert np.allclose(idr.prob_exceed(m, xi, 100), tab["p_event"][np.clip(xi, 0, tab["R"] - 1)], atol=1e-6)
    assert np.allclose(idr.crps_clim_training(m, yi), tab["crps_clim"][yi], atol=1e-5)


def test_crps_matches_brute_force_integral():
    x, y = _toy(5000)
    m = idr.fit_idr(x, y)
    xi = np.array([0, 37, 120, 900])
    yi = np.array([0, 5, 300, 2000])
    F = idr.predict_cdf(m, xi)
    top = max(int(m.z[-1]), int(yi.max())) + 2
    g = np.arange(top)
    j = np.searchsorted(m.z, g, side="right") - 1
    Fg = np.where(j >= 0, F[:, np.maximum(j, 0)], 0.0)
    brute = ((Fg - (g[None, :] >= yi[:, None])) ** 2).sum(axis=1) * idr.RES
    assert np.allclose(idr.crps(m, xi, yi), brute)


def test_point_forecast_crps_is_absolute_error():
    # a "model" whose CDF is a single step at the forecast reduces CRPS to |x - y|
    z = np.arange(0, 400)
    x = np.array([0, 50, 120])
    cdf = (z[None, :] >= x[:, None]).astype(float)
    m = idr.IDRModel(x=x, z=z, cdf=cdf, clim=cdf.mean(0), n_train=3)
    yi = np.array([10, 50, 0])
    assert np.allclose(idr.crps(m, x, yi), np.abs(x - yi) * idr.RES)


def test_monotone_in_forecast():
    x, y = _toy()
    m = idr.fit_idr(x, y)
    assert (np.diff(m.cdf, axis=0) <= 1e-12).all()      # wetter forecast, never a drier outlook
    assert (np.diff(m.cdf, axis=1) >= -1e-12).all()     # valid CDFs


def test_scores_sanity():
    rng = np.random.default_rng(0)
    o = (rng.random(5000) < 0.2).astype(float)
    a = np.linspace(0.02, 0.98, 49)
    assert np.allclose(economic_value(o, o, a), 1.0)
    assert np.allclose(np.nan_to_num(economic_value(np.full(o.size, o.mean()), o, a)), 0.0, atol=1e-12)
    c = contingency(o, o, 0.5)
    assert c["pod"] == 1 and c["far"] == 0
    r = reliability(np.full(o.size, 0.2), o)
    assert np.isclose(np.nansum(r.share), 1)
    d = Decomposition(raw=14, cal=9, clim=10)
    assert d.mcb - d.dsc + d.unc == 14 and np.isclose(d.skill_raw, -0.4) and np.isclose(d.skill_cal, 0.1)
    s = rng.random(1000) + 0.5
    bt = day_block_bootstrap(s, s * 1.25, np.repeat(np.arange(100), 10), n_boot=50)
    assert np.allclose(bt, 0.2)


def test_app_models_are_the_pipeline_models():
    """Refitting from the stored pairs reproduces the models the pipeline fitted (same rng draws)."""
    from aifs_app import prepare_data as p
    b = Bundle()
    if not b.synthetic:
        pytest.skip("parity check rebuilds the synthetic source")
    src = p.SyntheticSource()
    X, Y, _ = src.train(b.lead_days[0])
    models, _ = p.fit_zone_models(src.grid, X, Y, np.random.default_rng(p.SEED))
    for z, m in models.items():
        a = b.model(b.lead_days[0], z)
        assert np.array_equal(a.x, m.x) and np.array_equal(a.z, m.z) and np.allclose(a.cdf, m.cdf)


def test_bundle_consistency():
    b = Bundle()
    for L in b.lead_days:
        t = b.totals(L)
        assert t["clim"] > 0 and t["idr"] > 0
        for z in b.zones_with_model(L):
            te = b.test(L, z)
            assert te.n > 0 and np.isfinite(te.x).all() and np.isfinite(te.y).all()
            assert b.train(L, z).n > 0
    assert len(b.case_dates) >= 1
    assert b.case_obs(0).shape == (b.n_land,)
