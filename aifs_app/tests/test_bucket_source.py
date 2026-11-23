"""Exercise the real-data loader on fake in-memory datasets shaped like the bucket's files."""
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")


def _fake_bucket(region):
    rng = np.random.default_rng(0)
    lat = np.round(np.arange(region["lat"][0], region["lat"][1] + 1e-9, 0.1), 2)
    lon = np.round(np.arange(region["lon"][0], region["lon"][1] + 1e-9, 0.1), 2)
    kg = np.where((lat[:, None] > 0), 3, 6) * np.ones((1, lon.size), int)
    kg[:3, :3] = 0                                   # some sea
    t_tr = pd.date_range("2024-03-02", "2024-07-31")
    t_te = pd.date_range("2026-01-01", "2026-04-30").delete([40])   # one missing IMERG day, as in 2026
    def rain(n):
        return (rng.gamma(0.5, 8, (n, lat.size, lon.size)) * (rng.random((n, lat.size, lon.size)) < 0.4)).astype(np.float32)
    imerg_tr = xr.Dataset({"precipitation": (("time", "lat", "lon"), rain(t_tr.size), {"units": "mm/day"})},
                          coords=dict(time=t_tr, lat=lat, lon=lon))
    imerg_te = xr.Dataset({"precip": (("time", "lat", "lon"), rain(t_te.size), {"units": "mm/day"}),
                           "imerg_run": (("time",), np.full(t_te.size, 2))}, coords=dict(time=t_te, lat=lat, lon=lon))
    glat = np.arange(region["lat"][0] - 1, region["lat"][1] + 1.01, 0.25)
    glon = np.arange(region["lon"][0] - 1, region["lon"][1] + 1.01, 0.25)
    steps = np.array([24.0 * d for d in (1, 2, 3, 4, 5, 6)])
    def aifs(times):
        v = rng.gamma(0.6, 8, (times.size, steps.size, glat.size, glon.size)).astype(np.float32) - 0.01
        da = xr.DataArray(v, dims=("time", "step", "latitude", "longitude"),
                          coords=dict(time=times, step=steps, latitude=glat, longitude=glon), attrs={"units": "mm"})
        return xr.Dataset({"tp_24h": da})
    clim = xr.Dataset({"crps_clim": (("time", "lat", "lon"), rng.random((pd.date_range("2026-01-01", "2026-04-30").size, lat.size, lon.size)).astype(np.float32) * 2, {"units": "mm/day"}),
                       "p_clim_10mm_day": (("time", "lat", "lon"), rng.random((pd.date_range("2026-01-01", "2026-04-30").size, lat.size, lon.size)).astype(np.float32))},
                      coords=dict(time=pd.date_range("2026-01-01", "2026-04-30"), lat=lat, lon=lon))
    files = {"aifs_tp24h_africa.zarr": aifs(pd.date_range("2024-03-01", "2024-07-25")),
             "aifs_tp24h_africa_2026.zarr": aifs(pd.date_range("2026-01-01", "2026-04-20")),
             "imerg_daily_africa_2024_2025.zarr": imerg_tr, "imerg_daily_africa_2026.zarr": imerg_te,
             "imerg_clim_crps_daily_africa_2026.zarr": clim}
    return files, xr.DataArray(kg, dims=("lat", "lon"), coords=dict(lat=lat, lon=lon), name="kg_class")


def test_bucket_pipeline_end_to_end(tmp_path, monkeypatch):
    from aifs_app import prepare_data as p
    from aifs_app.core.bundle import Bundle
    region = dict(lat=(-1.5, 1.5), lon=(30.0, 33.0))
    monkeypatch.setattr(p, "REGION", region)
    monkeypatch.setattr(p, "ZONE_MIN_VALID", 10)
    files, kg = _fake_bucket(region)
    monkeypatch.setattr(p.BucketSource, "open_zarr", staticmethod(lambda name: files[name]))
    grid = p.Grid(kg.lat.values, kg.lon.values, kg.values)
    src = p.BucketSource(grid)
    p.build_bundle(tmp_path, grid, src, synthetic=False)
    b = Bundle(tmp_path)
    assert not b.synthetic and b.lead_days == [2, 3, 4, 5]
    assert set(b.zones) == {3, 6}
    for L in b.lead_days:
        t = b.totals(L)
        assert np.isfinite([t["raw"], t["idr"], t["clim"]]).all()
        te = b.test(L, 3)
        assert te.clim_crps is not None and te.clim_p is not None
        # lead L forecasts are valid L-1 days after the run
        assert b.test_dates(L)[0] == pd.Timestamp("2026-01-01").date() + pd.Timedelta(days=L - 1)
    assert b.case_forecast(0, 2).shape == (b.n_land,)
    assert "Late run" in b.periods["test_label"]
    assert b.meta["test_imerg_runs"] == {"final": 0, "late": 119}
