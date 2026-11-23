"""Build the data bundle for the AIFS → IMERG IDR workshop app.

Run this ONCE, on a machine with internet access (Google Colab works well), from the repository root:

    pip install "zarr>=3" gcsfs xarray pandas numba
    python aifs_app/prepare_data.py                # real data from gs://acmad-tutorial (≈ 10–15 min)
    python aifs_app/prepare_data.py --synthetic    # small synthetic bundle, for testing the app offline

It follows the notebook `aifs_imerg_calibration_tutorial.ipynb` step by step (same files, settings,
random seed and order of random draws, so the zone models and headline scores are the notebook's),
then saves what the app needs into `data_aifs/` (about 40–60 MB):

* the notebook's full verification results: scores by lead, zone and day, reliability counts, and
  skill maps (coarsened to 0.2° for display);
* the exact training pairs each zone model was fitted on, so the app can refit and re-score live;
* a random sample of held-out 2026 test pairs per zone and lead, with the IMERG climatology's CRPS
  and heavy-rain probability for the same cells;
* raw AIFS forecasts, IMERG observations and climatological probabilities for a few heavy-rain
  case-study days (all land grid boxes at 0.1°).

The app never touches the bucket, so the workshop can run without internet once this is done.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aifs_app.core import idr  # noqa: E402
from aifs_app.core.legend import KG_LEGEND  # noqa: E402

# --- notebook configuration ----------------------------------------------------------------------
BUCKET = "acmad-tutorial"
FILES = dict(
    aifs_train="aifs_tp24h_africa.zarr",
    aifs_test="aifs_tp24h_africa_2026.zarr",
    imerg_train="imerg_daily_africa_2024_2025.zarr",
    imerg_test="imerg_daily_africa_2026.zarr",
    koppen="koppen_geiger_africa_0p1.nc",
    clim_test="imerg_clim_crps_daily_africa_2026.zarr",
)
REGION = dict(lat=(-35.0, 38.0), lon=(-18.0, 52.0))
LEAD_DAYS = [2, 3, 4, 5]
EVENT_MM_DAY = 10.0
MM_DAY = idr.MM_DAY
EVENT = EVENT_MM_DAY * MM_DAY
RES = idr.RES
ZONE_SAMPLE_LOCS = 1_000
NUM_DATA_PTS = 50_000
ZONE_MIN_VALID = 100
SEED = 0

# --- app bundle settings ---------------------------------------------------------------------------
TEST_SAMPLE_PER_ZONE = 20_000   # held-out test pairs kept per zone and lead for live scoring
N_CASES = 4                     # heavy-rain case-study days
CASE_SPACING_DAYS = 21          # case days at least this far apart
CASE_HEAVY_MM_DAY = 50.0        # case days ranked by land area with more than this
COARSEN = 2                     # 0.1° maps -> 0.2° for display
TO_MM_HR = {"mm/hr": 1.0, "mm/h": 1.0, "mm/day": 1 / 24, "mm": 1 / 24}
PROB_BINS = np.linspace(0, 1, 11)


# ==================================================================================================
# Shared pipeline (works on numpy tables: days x land grid boxes, in mm/hr)
# ==================================================================================================

class Grid:
    def __init__(self, lat, lon, kg):
        self.lat, self.lon, self.kg = np.asarray(lat), np.asarray(lon), np.asarray(kg).astype(np.int16)
        self.land = self.kg > 0
        self.pix_zone = self.kg[self.land]
        self.npix = int(self.land.sum())
        self.zones = [int(z) for z in np.unique(self.pix_zone)]

    def to_map(self, values):
        out = np.full(self.land.shape, np.nan, dtype=np.float32)
        out[self.land] = values
        return out


def coarsen(field: np.ndarray, f: int) -> np.ndarray:
    """Block-average a 2-D field by f x f, ignoring NaN (sea)."""
    if f == 1:
        return field.astype(np.float32)
    ny, nx = (field.shape[0] // f) * f, (field.shape[1] // f) * f
    b = field[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-sea blocks
        return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def coarsen_axis(v: np.ndarray, f: int) -> np.ndarray:
    n = (v.size // f) * f
    return np.asarray(v[:n], dtype=np.float64).reshape(-1, f).mean(axis=1)


def fit_zone_models(grid: Grid, X, Y, rng):
    """Notebook cell 21, plus the record of exactly which pairs each model was fitted on."""
    models, samples = {}, {}
    for z in grid.zones:
        zp = np.flatnonzero(grid.pix_zone == z)
        locs = rng.choice(zp, size=min(zp.size, ZONE_SAMPLE_LOCS), replace=False)
        x, y = X[:, locs].ravel(), Y[:, locs].ravel()
        ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        if ok.size < ZONE_MIN_VALID or np.unique(y[ok]).size < 2:
            continue
        ok = rng.choice(ok, size=min(ok.size, NUM_DATA_PTS), replace=False)
        models[z] = idr.fit_idr(x[ok], y[ok])
        samples[z] = dict(
            x=np.clip(idr.to_level(x[ok]), 0, 65535).astype(np.uint16),
            y=np.clip(idr.to_level(y[ok]), 0, 65535).astype(np.uint16),
            day=(ok // locs.size).astype(np.uint16),
            loc=(ok % locs.size).astype(np.uint16),
            locs=locs.astype(np.int32),
        )
    return models, samples


def verify(grid: Grid, models, X, Y, clim=None):
    """Notebook cell 21 `verify`, unchanged in substance."""
    npix = grid.npix
    out = {k: np.zeros(npix) for k in ("raw", "idr", "clim", "n")}
    daily = {k: np.zeros(X.shape[0]) for k in ("raw", "idr", "clim", "n")}
    rel = {k: dict(n=np.zeros(10), obs=np.zeros(10), prob=np.zeros(10)) for k in ("raw", "idr", "clim")}
    bs = dict(raw=0.0, idr=0.0, clim=0.0)
    ev = int(round(EVENT / RES))
    for z, model in models.items():
        zp = np.flatnonzero(grid.pix_zone == z)
        x, y = X[:, zp], Y[:, zp]
        ok = np.isfinite(x) & np.isfinite(y)
        if clim is not None:
            ok &= np.isfinite(clim["crps"][:, zp])
        yi = np.where(ok, np.rint(y / RES), 0).astype(np.int64)
        tab = idr.lookup_tables(model, yi.max(), ev)
        xi = np.where(ok, np.clip(np.rint(x / RES), 0, tab["R"] - 1), 0).astype(np.int64)
        raw = np.abs(x - y)
        idr_c = tab["crps"][xi, yi]
        ref = tab["crps_clim"][yi] if clim is None else clim["crps"][:, zp]
        for key, v in (("raw", raw), ("idr", idr_c), ("clim", ref)):
            out[key][zp] = np.where(ok, v, 0).sum(axis=0)
            daily[key] += np.where(ok, v, 0).sum(axis=1)
        out["n"][zp] = ok.sum(axis=0)
        daily["n"] += ok.sum(axis=1)

        event = (y > EVENT)[ok].astype(np.float64)
        p_idr = tab["p_event"][xi][ok]
        bs["raw"] += (((x > EVENT)[ok] - event) ** 2).sum()
        bs["idr"] += ((p_idr - event) ** 2).sum()
        if clim is None or clim.get("p_event") is None:
            p_ref = idr.clim_prob_exceed(model, ev)
        else:
            p_ref = clim["p_event"][:, zp][ok]
        bs["clim"] += ((p_ref - event) ** 2).sum()
        p_raw = (x > EVENT)[ok].astype(np.float64)
        for key, pk in (("raw", p_raw), ("idr", p_idr), ("clim", np.broadcast_to(p_ref, event.shape))):
            b = np.clip(np.digitize(pk, PROB_BINS) - 1, 0, 9)
            rel[key]["n"] += np.bincount(b, minlength=10)
            rel[key]["obs"] += np.bincount(b, weights=event, minlength=10)
            rel[key]["prob"] += np.bincount(b, weights=pk, minlength=10)
    n = out.pop("n")
    maps = {k: np.where(n > 0, v / np.maximum(n, 1), np.nan) for k, v in out.items()}
    totals = {k: v.sum() / n.sum() for k, v in out.items()}
    totals.update({f"bs_{k}": v / n.sum() for k, v in bs.items()})
    zone_sums = {z: {k: out[k][grid.pix_zone == z].sum() / max(n[grid.pix_zone == z].sum(), 1)
                     for k in out} for z in models}
    zone_n = {z: int(n[grid.pix_zone == z].sum()) for z in models}
    nd = daily.pop("n")
    daily = {k: np.where(nd > 0, v / np.maximum(nd, 1), np.nan) for k, v in daily.items()}
    return dict(maps=maps, totals=totals, zones=zone_sums, zone_n=zone_n, rel=rel, n=n, daily=daily)


def sample_test_pairs(grid: Grid, models, X, Y, clim, rng):
    """A random sample of held-out (forecast, observation) cells per zone, with the reference forecast."""
    out = {}
    for z in models:
        zp = np.flatnonzero(grid.pix_zone == z)
        x, y = X[:, zp], Y[:, zp]
        ok = np.isfinite(x) & np.isfinite(y)
        if clim is not None:
            ok &= np.isfinite(clim["crps"][:, zp])
        idx = np.flatnonzero(ok.ravel())
        if idx.size == 0:
            continue
        idx = np.sort(rng.choice(idx, size=min(idx.size, TEST_SAMPLE_PER_ZONE), replace=False))
        day, col = idx // zp.size, idx % zp.size
        rec = dict(x=X[day, zp[col]].astype(np.float32), y=Y[day, zp[col]].astype(np.float32),
                   day=day.astype(np.uint16), pix=zp[col].astype(np.int32), n_valid=int(ok.sum()))
        if clim is not None:
            rec["clim_crps"] = clim["crps"][day, zp[col]].astype(np.float32)
            if clim.get("p_event") is not None:
                rec["clim_p_event"] = clim["p_event"][day, zp[col]].astype(np.float32)
        out[z] = rec
    return out


def choose_case_days(obs_test, obs_dates, n=N_CASES, spacing=CASE_SPACING_DAYS):
    """Test days with the most land area above CASE_HEAVY_MM_DAY, at least `spacing` days apart.

    Skips the first and last 5 days so every lead day has a forecast valid on each case day.
    """
    heavy = np.nansum(obs_test > CASE_HEAVY_MM_DAY * MM_DAY, axis=1).astype(float)
    heavy[:5] = -1
    heavy[-5:] = -1
    chosen = []
    for i in np.argsort(-heavy):
        if heavy[i] < 0:
            break
        if all(abs((obs_dates[i] - obs_dates[j]).days) >= spacing for j in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda i: obs_dates[i])


def build_bundle(out_dir: Path, grid: Grid, source, synthetic: bool, notes: str = "") -> None:
    """Run the notebook pipeline through `source` and save the app bundle."""
    t_all = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_test, obs_test_dates = source.obs_test()
    clim_full = source.clim_test(obs_test_dates)
    case_rows = choose_case_days(obs_test, obs_test_dates)
    case_dates = [obs_test_dates[i].date() for i in case_rows]
    print("Case-study days:", ", ".join(f"{d:%d %b %Y}" for d in case_dates))

    results, pairs, cases = {}, {}, {}
    for i, r in enumerate(case_rows):
        cases[f"case{i}_obs"] = (obs_test[r] / MM_DAY).astype(np.float16)
        if clim_full is not None and clim_full.get("p_event") is not None:
            cases[f"case{i}_pclim"] = clim_full["p_event"][r].astype(np.float16)

    meta_zones = {}
    rng = np.random.default_rng(SEED)              # same sequence of draws as the notebook
    for lead in LEAD_DAYS:
        t0 = time.time()
        X_tr, Y_tr, train_dates = source.train(lead)
        models, samples = fit_zone_models(grid, X_tr, Y_tr, rng)
        del X_tr, Y_tr
        X_te, Y_te, dates, rows = source.test(lead, obs_test, obs_test_dates)
        clim = None if clim_full is None else {k: (None if v is None else v[rows]) for k, v in clim_full.items()}
        res = verify(grid, models, X_te, Y_te, clim)
        test_s = sample_test_pairs(grid, models, X_te, Y_te, clim, np.random.default_rng(SEED + 1000 + lead))

        for i, d in enumerate(case_dates):
            hit = np.flatnonzero(dates == d)
            cases[f"case{i}_L{lead}"] = ((X_te[hit[0]] / MM_DAY).astype(np.float16) if hit.size
                                         else np.full(grid.npix, np.nan, np.float16))
        del X_te, Y_te

        p = f"L{lead}_"
        t = res["totals"]
        results[p + "totals"] = np.array([t["raw"], t["idr"], t["clim"], t["bs_raw"], t["bs_idr"], t["bs_clim"]])
        zs = sorted(res["zones"])
        results[p + "zones"] = np.array([[z, res["zones"][z]["raw"], res["zones"][z]["idr"], res["zones"][z]["clim"],
                                          res["zone_n"][z]] for z in zs], dtype=np.float64)
        for k in ("raw", "idr", "clim"):
            results[p + f"rel_{k}"] = np.vstack([res["rel"][k]["n"], res["rel"][k]["obs"], res["rel"][k]["prob"]])
            results[p + f"map_{k}"] = coarsen(grid.to_map(res["maps"][k]), COARSEN).astype(np.float16)
        results[p + "daily"] = np.vstack([res["daily"][k] for k in ("raw", "idr", "clim")]).astype(np.float32)
        results[p + "dates"] = np.array([d.toordinal() for d in dates], dtype=np.int32)

        pairs[p + "train_dates"] = np.array([d.toordinal() for d in train_dates], dtype=np.int32)
        pairs[p + "test_dates"] = results[p + "dates"]
        for z, s in samples.items():
            pairs[p + f"Z{z}_train"] = np.vstack([s["x"], s["y"], s["day"], s["loc"]])
            pairs[p + f"Z{z}_locs"] = s["locs"]
        for z, s in test_s.items():
            pairs[p + f"Z{z}_test_xy"] = np.vstack([s["x"], s["y"]])
            pairs[p + f"Z{z}_test_day"] = s["day"]
            pairs[p + f"Z{z}_test_pix"] = s["pix"]
            if "clim_crps" in s:
                pairs[p + f"Z{z}_test_clim"] = s["clim_crps"]
            if "clim_p_event" in s:
                pairs[p + f"Z{z}_test_pclim"] = s["clim_p_event"]
            meta_zones.setdefault(str(lead), {})[str(z)] = dict(n_valid_test=s["n_valid"],
                                                                n_train=int(samples[z]["x"].size))
        print(f"lead day {lead}: {len(models)} zone models | {len(dates)} test days | mean CRPS raw "
              f"{t['raw']:.3f}, IDR {t['idr']:.3f}, climatology {t['clim']:.3f} mm/hr | {time.time() - t0:.0f} s")

    clat, clon = coarsen_axis(grid.lat, COARSEN), coarsen_axis(grid.lon, COARSEN)
    np.savez_compressed(out_dir / "grid.npz", lat=grid.lat.astype(np.float32), lon=grid.lon.astype(np.float32),
                        kg=grid.kg.astype(np.uint8), clat=clat.astype(np.float32), clon=clon.astype(np.float32))
    np.savez_compressed(out_dir / "results.npz", **results)
    np.savez_compressed(out_dir / "pairs.npz", **pairs)
    np.savez_compressed(out_dir / "cases.npz", dates=np.array([d.toordinal() for d in case_dates], np.int32), **cases)

    meta = dict(
        synthetic=synthetic, created=dt.datetime.now().isoformat(timespec="minutes"), notes=notes,
        region=REGION, lead_days=LEAD_DAYS, event_mm_day=EVENT_MM_DAY, res_mm_day=0.1,
        zone_sample_locs=ZONE_SAMPLE_LOCS, num_data_pts=NUM_DATA_PTS, zone_min_valid=ZONE_MIN_VALID, seed=SEED,
        test_sample_per_zone=TEST_SAMPLE_PER_ZONE, coarsen=COARSEN, case_heavy_mm_day=CASE_HEAVY_MM_DAY,
        n_land=grid.npix,
        zones={str(z): dict(code=KG_LEGEND[z][0], name=KG_LEGEND[z][1], color=KG_LEGEND[z][2],
                            n_pixels=int((grid.pix_zone == z).sum())) for z in grid.zones},
        per_lead=meta_zones,
        periods=source.periods(),
        test_imerg_runs=source.test_runs() if hasattr(source, "test_runs") else None,
        files=FILES if not synthetic else {},
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"Saved bundle to {out_dir} ({size:.0f} MB) in {time.time() - t_all:.0f} s")


# ==================================================================================================
# Real data: the public bucket, exactly as in the notebook
# ==================================================================================================

class BucketSource:
    def __init__(self, grid: Grid):
        import pandas as pd  # noqa: F401
        self.grid = grid
        self._aifs = {}
        self._train_obs = None

    @staticmethod
    def open_zarr(name):
        import xarray as xr
        return xr.open_zarr(f"gs://{BUCKET}/{name}", decode_timedelta=False, storage_options={"token": "anon"})

    @staticmethod
    def in_mm_hr(da):
        units = da.attrs.get("units")
        if units not in TO_MM_HR:
            raise ValueError(f"{da.name}: unknown units {units!r}")
        return da * TO_MM_HR[units]

    @classmethod
    def load_grid(cls) -> Grid:
        import tempfile

        import gcsfs
        import xarray as xr
        local = Path(tempfile.gettempdir()) / FILES["koppen"]
        if not local.exists():
            gcsfs.GCSFileSystem(token="anon").get(f"{BUCKET}/{FILES['koppen']}", str(local))
        kg = xr.open_dataset(local)["kg_class"].load().sel(lat=slice(*REGION["lat"]), lon=slice(*REGION["lon"]))
        return Grid(kg.lat.values, kg.lon.values, kg.values)

    def to_pixels(self, da, regrid=False, block=32):
        g = self.grid
        out = np.empty((da.sizes["time"], g.npix), dtype=np.float32)
        for s in range(0, da.sizes["time"], block):
            blk = da.isel(time=slice(s, s + block))
            if regrid:
                blk = blk.interp(lat=g.lat, lon=g.lon, method="linear")
            out[s:s + block] = blk.values[:, g.land]
        return out

    def load_imerg(self, name):
        import pandas as pd
        ds = self.open_zarr(name)
        var = "precip" if "precip" in ds else "precipitation"
        da = self.in_mm_hr(ds[var]).sel(lat=slice(*REGION["lat"]), lon=slice(*REGION["lon"]))
        assert da.shape[1:] == self.grid.land.shape, f"{name} is not on the IMERG 0.1° grid"
        return self.to_pixels(da), pd.DatetimeIndex(da.time.values)

    def load_aifs(self, name):
        if name not in self._aifs:
            raw = self.open_zarr(name)["tp_24h"]
            ds = raw.rename(latitude="lat", longitude="lon")
            ds = ds.sel(step=[24.0 * d for d in LEAD_DAYS],
                        lat=slice(REGION["lat"][0] - 0.5, REGION["lat"][1] + 0.5),
                        lon=slice(REGION["lon"][0] - 0.5, REGION["lon"][1] + 0.5))
            self._aifs[name] = ds.clip(min=0).load() * TO_MM_HR[raw.attrs["units"]]
        return self._aifs[name]

    def paired(self, aifs, obs, obs_dates, lead):
        import pandas as pd
        fc = aifs.sel(step=24.0 * lead)
        valid = pd.DatetimeIndex(fc.time.values) + pd.Timedelta(days=lead - 1)
        pos = obs_dates.get_indexer(valid)
        keep = pos >= 0
        X = self.to_pixels(fc.isel(time=np.flatnonzero(keep)), regrid=True)
        return X, obs[pos[keep]], valid[keep], pos[keep]

    # --- interface used by build_bundle ---
    def obs_test(self):
        if not hasattr(self, "_obs_test"):
            self._obs_test = self.load_imerg(FILES["imerg_test"])
        return self._obs_test

    def clim_test(self, dates):
        ds = self.open_zarr(FILES["clim_test"]).sel(lat=slice(*REGION["lat"]), lon=slice(*REGION["lon"]))
        ds = ds.reindex(time=dates)
        p_name = f"p_clim_{EVENT_MM_DAY:g}mm_day"
        return dict(crps=self.to_pixels(self.in_mm_hr(ds["crps_clim"])),
                    p_event=self.to_pixels(ds[p_name]) if p_name in ds else None)

    def train(self, lead):
        if self._train_obs is None:
            self._train_obs = self.load_imerg(FILES["imerg_train"])
        obs, dates = self._train_obs
        X, Y, valid, _ = self.paired(self.load_aifs(FILES["aifs_train"]), obs, dates, lead)
        return X, Y, [d.date() for d in valid]

    def test(self, lead, obs, obs_dates):
        X, Y, valid, rows = self.paired(self.load_aifs(FILES["aifs_test"]), obs, obs_dates, lead)
        return X, Y, np.array([d.date() for d in valid]), rows

    def test_runs(self):
        # How many 2026 IMERG days come from the Final and Late runs (1 = Final, 2 = Late).
        ds = self.open_zarr(FILES["imerg_test"])
        if "imerg_run" not in ds:
            return None
        run = np.asarray(ds["imerg_run"].values)
        return dict(final=int((run == 1).sum()), late=int((run == 2).sum()))

    def periods(self):
        tr = self._train_obs[1] if self._train_obs is not None else None
        te = self._obs_test[1]
        return dict(train=f"{tr[0]:%d %b %Y} – {tr[-1]:%d %b %Y}" if tr is not None else "",
                    test=f"{te[0]:%d %b %Y} – {te[-1]:%d %b %Y}",
                    train_label="AIFS runs Mar 2024 – Jan 2025, IMERG V07B Final",
                    test_label="AIFS runs Jan – Oct 2026, IMERG V07 (Late run where Final is not yet out)",
                    clim_label="IMERG climatology: all days within ±15 days of the date in 2000–2020 (~650 values)")


# ==================================================================================================
# Synthetic data: same shapes and pipeline, for offline testing only
# ==================================================================================================

class SyntheticSource:
    """A coarse (0.5°) pretend Africa with a seasonal rain belt and a biased, noisy 'AIFS'.

    The numbers mean nothing physically. They exist so the app and its tests run without the bucket.
    """

    STEP = 0.5

    def __init__(self, seed=123):
        self.rng = np.random.default_rng(seed)
        lat = np.arange(REGION["lat"][0], REGION["lat"][1] + 1e-9, self.STEP)
        lon = np.arange(REGION["lon"][0], REGION["lon"][1] + 1e-9, self.STEP)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        # a crude continent: two overlapping ellipses plus the Horn
        land = (((LO - 18) / 30) ** 2 + ((LA - 14) / 22) ** 2 < 1) | (((LO - 24) / 16) ** 2 + ((LA + 12) / 23) ** 2 < 1) \
            | (((LO - 44) / 8) ** 2 + ((LA - 8) / 6) ** 2 < 1)
        kg = np.zeros(LA.shape, np.int16)
        a = np.abs(LA)
        kg[land & (a < 4)] = 1
        kg[land & (a >= 4) & (a < 7)] = 2
        kg[land & (a >= 7) & (a < 14)] = 3
        kg[land & (a >= 14) & (a < 18)] = 6
        kg[land & (LA >= 18) & (LA < 31)] = 4
        kg[land & (LA <= -18) & (LA > -26)] = 7
        kg[land & (LA <= -26)] = 9
        kg[land & (LA >= 31)] = 8
        kg[land & (LO > 32) & (LO < 40) & (LA > -10) & (LA < 8)] = 12
        kg[land & (LA <= -18) & (LA > -26) & (LO < 18)] = 4
        self.grid = Grid(lat, lon, kg)
        g = self.grid
        self.plat = LA[g.land]
        z = g.pix_zone
        wet = {1: 0.65, 2: 0.55, 3: 0.45, 6: 0.25, 4: 0.03, 7: 0.15, 9: 0.2, 8: 0.15, 12: 0.4}
        scale = {1: 9, 2: 10, 3: 9, 6: 7, 4: 4, 7: 6, 9: 5, 8: 6, 12: 7}
        self.p_wet = np.array([wet[int(v)] for v in z])
        self.scale = np.array([scale[int(v)] for v in z])
        self.train_dates = [dt.date(2024, 3, 2) + dt.timedelta(days=i) for i in range(330)]
        self.test_dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(290)]
        self._cache = {}

    def _season(self, dates):
        doy = np.array([d.timetuple().tm_yday for d in dates])[:, None]
        peak = 200 - 3.5 * (self.plat[None, :] + 10)       # rain belt follows the sun
        return 0.25 + 0.75 * np.clip(np.cos(2 * np.pi * (doy - peak) / 365), 0, 1) ** 1.5

    def _truth(self, dates, key):
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng({"train": 1, "test": 2}[key])
        s = self._season(dates)
        day_effect = rng.normal(0, 0.6, (len(dates), 1))   # shared weather: pairs are not independent
        latent = rng.normal(0, 1, (len(dates), self.grid.npix)) + day_effect
        from scipy.special import ndtr  # noqa: PLC0415
        u = ndtr(latent)
        p = np.clip(self.p_wet[None, :] * s * 1.6, 0, 0.95)
        amt = rng.gamma(0.8, self.scale[None, :] * (0.6 + s), size=latent.shape)
        y = np.where(u > 1 - p, amt * (0.5 + (u - (1 - p)) / np.maximum(p, 1e-6)), 0.0)
        self._cache[key] = (latent, y, s)
        return self._cache[key]

    def _forecast(self, latent, y, s, lead, rng):
        skill = {2: 0.75, 3: 0.68, 4: 0.6, 5: 0.52}[lead]
        lf = skill * latent + np.sqrt(1 - skill**2) * rng.normal(0, 1, latent.shape)
        from scipy.special import ndtr  # noqa: PLC0415
        u = ndtr(lf)
        p = np.clip(self.p_wet[None, :] * s * 2.4, 0, 0.97)   # AIFS drizzles too often
        x = np.where(u > 1 - p, 0.7 * self.scale[None, :] * (0.6 + s) * (0.3 + 2.2 * (u - (1 - p)) / np.maximum(p, 1e-6)), 0.0)
        return x

    def train(self, lead):
        latent, y, s = self._truth(self.train_dates, "train")
        x = self._forecast(latent, y, s, lead, np.random.default_rng(10 + lead))
        return (x * MM_DAY).astype(np.float32), (y * MM_DAY).astype(np.float32), self.train_dates

    def obs_test(self):
        import pandas as pd
        _, y, _ = self._truth(self.test_dates, "test")
        return (y * MM_DAY).astype(np.float32), pd.DatetimeIndex(self.test_dates)

    def clim_test(self, dates):
        # CRPS and P(> 10 mm/day) of a climatological ensemble drawn from the same seasonal climate
        rng = np.random.default_rng(7)
        s = self._season(list(dates.date))
        p = np.clip(self.p_wet[None, :] * s * 1.6, 0, 0.95)
        m = 40
        crps = np.empty(s.shape, np.float32)
        pe = np.empty(s.shape, np.float32)
        _, y, _ = self._truth(self.test_dates, "test")
        for d in range(s.shape[0]):
            wetm = rng.random((m, s.shape[1])) < p[d]
            ens = np.where(wetm, rng.gamma(0.8, self.scale * (0.6 + s[d]), (m, s.shape[1])) * 1.0, 0.0)
            ens.sort(axis=0)
            obs = y[d]
            t1 = np.abs(ens - obs[None, :]).mean(axis=0)
            i = np.arange(1, m + 1)[:, None]
            t2 = (2 * ((2 * i - m - 1) * ens).sum(axis=0)) / (m * m) / 2
            crps[d] = (t1 - t2) * MM_DAY
            pe[d] = (ens > EVENT_MM_DAY).mean(axis=0)
        return dict(crps=crps, p_event=pe)

    def test(self, lead, obs, obs_dates):
        latent, y, s = self._truth(self.test_dates, "test")
        x = self._forecast(latent, y, s, lead, np.random.default_rng(20 + lead))
        # forecast issued (lead - 1) days earlier: valid dates shift; keep it simple and aligned
        rows = np.arange(lead - 1, len(self.test_dates))
        X = (x[rows] * MM_DAY).astype(np.float32)
        return X, obs[rows], np.array([obs_dates[r].date() for r in rows]), rows

    def periods(self):
        return dict(train=f"{self.train_dates[0]:%d %b %Y} – {self.train_dates[-1]:%d %b %Y}",
                    test=f"{self.test_dates[0]:%d %b %Y} – {self.test_dates[-1]:%d %b %Y}",
                    train_label="SYNTHETIC stand-in for AIFS 2024–25 / IMERG",
                    test_label="SYNTHETIC stand-in for AIFS 2026 / IMERG",
                    clim_label="SYNTHETIC climatological ensemble")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="build a small synthetic bundle (no internet needed)")
    ap.add_argument("--out", default=None, help="output folder (default: data_aifs/, or data_aifs_synthetic/)")
    a = ap.parse_args(argv)
    if a.synthetic:
        src = SyntheticSource()
        out = Path(a.out) if a.out else ROOT / "data_aifs_synthetic"
        build_bundle(out, src.grid, src, synthetic=True,
                     notes="Synthetic data for testing the app. Not real AIFS or IMERG.")
    else:
        grid = BucketSource.load_grid()
        print(f"{grid.npix:,} land grid boxes at 0.1° in {len(grid.zones)} climate zones")
        src = BucketSource(grid)
        out = Path(a.out) if a.out else ROOT / "data_aifs"
        build_bundle(out, grid, src, synthetic=False)


if __name__ == "__main__":
    main()
