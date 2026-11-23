"""Load the data bundle written by prepare_data.py. No Streamlit imports here."""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from . import idr

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "data_aifs"
SYNTH_DIR = ROOT / "data_aifs_synthetic"


class BundleMissing(RuntimeError):
    pass


def find_bundle() -> Path:
    env = os.environ.get("AIFS_DATA")
    for d in ([Path(env)] if env else []) + [REAL_DIR, SYNTH_DIR]:
        if (d / "meta.json").exists():
            return d
    raise BundleMissing(
        "No data bundle found. From the repository root run `python aifs_app/prepare_data.py` "
        "(needs internet, about 15 minutes) or `python aifs_app/prepare_data.py --synthetic` "
        "(offline test data)."
    )


def _dates(ordinals) -> np.ndarray:
    return np.array([dt.date.fromordinal(int(o)) for o in ordinals])


@dataclass(frozen=True)
class TrainPairs:
    xi: np.ndarray     # forecast level (0.1 mm/day)
    yi: np.ndarray     # observed level
    day: np.ndarray    # index into dates
    loc: np.ndarray    # index into locs
    locs: np.ndarray   # land-pixel indices of the grid boxes sampled for this zone
    dates: np.ndarray  # valid dates of the training forecasts

    @property
    def n(self) -> int:
        return int(self.xi.size)

    def subset(self, mask: np.ndarray) -> "TrainPairs":
        return TrainPairs(self.xi[mask], self.yi[mask], self.day[mask], self.loc[mask], self.locs, self.dates)


@dataclass(frozen=True)
class TestPairs:
    x: np.ndarray          # raw AIFS, mm/hr
    y: np.ndarray          # IMERG, mm/hr
    day: np.ndarray        # index into dates
    pix: np.ndarray        # land-pixel index
    clim_crps: np.ndarray | None   # CRPS of IMERG climatology, mm/hr
    clim_p: np.ndarray | None      # IMERG climatology P(> event threshold)
    dates: np.ndarray
    weight: float          # (valid test cells in the zone) / (pairs in this sample)
    zone: int

    @property
    def n(self) -> int:
        return int(self.x.size)

    @cached_property
    def xi(self) -> np.ndarray:
        return idr.to_level(self.x)

    @cached_property
    def yi(self) -> np.ndarray:
        return idr.to_level(self.y)

    def take(self, idx) -> "TestPairs":
        g = lambda a: None if a is None else a[idx]  # noqa: E731
        return TestPairs(self.x[idx], self.y[idx], self.day[idx], self.pix[idx], g(self.clim_crps), g(self.clim_p),
                         self.dates, self.weight * self.n / max(len(np.atleast_1d(self.x[idx])), 1), self.zone)


class Bundle:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else find_bundle()
        self.meta = json.loads((self.path / "meta.json").read_text())
        self._grid = np.load(self.path / "grid.npz")
        self._res = np.load(self.path / "results.npz")
        self._pairs = np.load(self.path / "pairs.npz")
        self._cases = np.load(self.path / "cases.npz")

    # --- basics -------------------------------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic"))

    @property
    def lead_days(self) -> list[int]:
        return list(self.meta["lead_days"])

    @property
    def event_mm_day(self) -> float:
        return float(self.meta["event_mm_day"])

    @cached_property
    def zones(self) -> list[int]:
        """Zone ids, largest first."""
        z = self.meta["zones"]
        return sorted((int(k) for k in z), key=lambda k: -z[str(k)]["n_pixels"])

    def zone_code(self, z: int) -> str:
        return self.meta["zones"][str(z)]["code"]

    def zone_name(self, z: int) -> str:
        return self.meta["zones"][str(z)]["name"]

    def zone_color(self, z: int) -> str:
        return self.meta["zones"][str(z)]["color"]

    def zone_pixels(self, z: int) -> int:
        return int(self.meta["zones"][str(z)]["n_pixels"])

    def zone_label(self, z: int) -> str:
        return f"{self.zone_code(z)} · {self.zone_name(z)}"

    def zones_with_model(self, lead: int) -> list[int]:
        have = self.meta["per_lead"].get(str(lead), {})
        return [z for z in self.zones if str(z) in have]

    def default_zone(self) -> int:
        return 3 if 3 in self.zones else self.zones[0]

    # --- grid ---------------------------------------------------------------------------------
    @cached_property
    def lat(self):
        return self._grid["lat"]

    @cached_property
    def lon(self):
        return self._grid["lon"]

    @cached_property
    def kg(self):
        return self._grid["kg"].astype(np.int16)

    @cached_property
    def land(self):
        return self.kg > 0

    @cached_property
    def pix_zone(self):
        return self.kg[self.land]

    @property
    def n_land(self) -> int:
        return int(self.land.sum())

    @cached_property
    def clat(self):
        return self._grid["clat"]

    @cached_property
    def clon(self):
        return self._grid["clon"]

    def to_map(self, values) -> np.ndarray:
        out = np.full(self.land.shape, np.nan, dtype=np.float32)
        out[self.land] = values
        return out

    def coarsen(self, field: np.ndarray, f: int | None = None) -> np.ndarray:
        f = f or int(self.meta.get("coarsen", 1))
        if f == 1:
            return field
        ny, nx = (field.shape[0] // f) * f, (field.shape[1] // f) * f
        b = field[:ny, :nx].reshape(ny // f, f, nx // f, f)
        with np.errstate(invalid="ignore"):
            cnt = np.isfinite(b).sum(axis=(1, 3))
            s = np.nansum(b, axis=(1, 3))
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype(np.float32)

    @lru_cache(maxsize=1)
    def zone_mode_map(self) -> np.ndarray:
        """Coarsened zone map (most common zone in each block, if the block is at least half land), sea = 0."""
        f = int(self.meta.get("coarsen", 1))
        kg = self.kg
        if f == 1:
            return kg
        ny, nx = (kg.shape[0] // f) * f, (kg.shape[1] // f) * f
        blk = kg[:ny, :nx].reshape(ny // f, f, nx // f, f).transpose(0, 2, 1, 3).reshape(ny // f, nx // f, f * f)
        counts = (blk[..., None] == np.arange(1, 31)[None, None, None, :]).sum(axis=2)
        mode = counts.argmax(axis=2) + 1
        land_n = counts.sum(axis=2)
        return np.where(land_n * 2 >= f * f, mode, 0).astype(np.int16)

    # --- notebook results ---------------------------------------------------------------------
    def totals(self, lead: int) -> dict:
        t = self._res[f"L{lead}_totals"]
        return dict(raw=t[0], idr=t[1], clim=t[2], bs_raw=t[3], bs_idr=t[4], bs_clim=t[5])

    def zone_scores(self, lead: int) -> dict[int, dict]:
        a = self._res[f"L{lead}_zones"]
        return {int(r[0]): dict(raw=r[1], idr=r[2], clim=r[3], n=int(r[4])) for r in a}

    def reliability_counts(self, lead: int, key: str) -> dict:
        a = self._res[f"L{lead}_rel_{key}"]
        return dict(n=a[0], obs=a[1], prob=a[2])

    def daily(self, lead: int) -> tuple[np.ndarray, dict]:
        d = self._res[f"L{lead}_daily"]
        return _dates(self._res[f"L{lead}_dates"]), dict(raw=d[0], idr=d[1], clim=d[2])

    def score_map(self, lead: int, key: str) -> np.ndarray:
        return self._res[f"L{lead}_map_{key}"].astype(np.float32)

    # --- pairs --------------------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def train_dates(self, lead: int) -> np.ndarray:
        return _dates(self._pairs[f"L{lead}_train_dates"])

    @lru_cache(maxsize=None)
    def test_dates(self, lead: int) -> np.ndarray:
        return _dates(self._pairs[f"L{lead}_test_dates"])

    @lru_cache(maxsize=None)
    def train(self, lead: int, zone: int) -> TrainPairs:
        a = self._pairs[f"L{lead}_Z{zone}_train"].astype(np.int64)
        return TrainPairs(xi=a[0], yi=a[1], day=a[2], loc=a[3], locs=self._pairs[f"L{lead}_Z{zone}_locs"],
                          dates=self.train_dates(lead))

    @lru_cache(maxsize=None)
    def test(self, lead: int, zone: int) -> TestPairs:
        p = f"L{lead}_Z{zone}_"
        xy = self._pairs[p + "test_xy"].astype(np.float64)
        files = self._pairs.files
        n_valid = self.meta["per_lead"][str(lead)][str(zone)]["n_valid_test"]
        return TestPairs(
            x=xy[0], y=xy[1], day=self._pairs[p + "test_day"].astype(np.int64), pix=self._pairs[p + "test_pix"],
            clim_crps=self._pairs[p + "test_clim"].astype(np.float64) if p + "test_clim" in files else None,
            clim_p=self._pairs[p + "test_pclim"].astype(np.float64) if p + "test_pclim" in files else None,
            dates=self.test_dates(lead), weight=n_valid / xy.shape[1], zone=zone)

    @lru_cache(maxsize=None)
    def model(self, lead: int, zone: int) -> idr.IDRModel:
        """The notebook's zone model, refitted from the exact pairs it was trained on."""
        t = self.train(lead, zone)
        return idr.fit_levels(t.xi, t.yi)

    # --- case-study days ----------------------------------------------------------------------
    @cached_property
    def case_dates(self) -> np.ndarray:
        return _dates(self._cases["dates"])

    def case_obs(self, i: int) -> np.ndarray:
        return self._cases[f"case{i}_obs"].astype(np.float32)

    def case_forecast(self, i: int, lead: int) -> np.ndarray:
        return self._cases[f"case{i}_L{lead}"].astype(np.float32)

    def case_pclim(self, i: int) -> np.ndarray | None:
        k = f"case{i}_pclim"
        return self._cases[k].astype(np.float32) if k in self._cases.files else None

    # --- descriptions -------------------------------------------------------------------------
    @property
    def periods(self) -> dict:
        return self.meta.get("periods", {})
