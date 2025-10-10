#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConstellationOverlay — GitHub-ready generator (no previews)

Generates an overlay JSON that places each IAU constellation at the Earth
sub-point (directly-overhead point) for a given UTC snapshot, and now also
includes its ICRS centroid ("icrs": {ra_deg, dec_deg}) so clients can
compute hourly footprints on-device.

Usage examples:
  python constellationoverlay.py
  python constellationoverlay.py --snapshot-utc 2025-10-10T03:00:00Z --include-asterisms
  python constellationoverlay.py --ra-step 6 --dec-step 6 --out out/overlay.json

Deps:
  pip install astropy numpy
"""

import argparse, json, os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, get_constellation
from astropy.time import Time

# ---------------------------- Defaults ----------------------------

STAR_DOTS_BRIGHT = [
    {"latitude":  7.0, "longitude": -72.0},
    {"latitude": 21.0, "longitude": -33.0},
    {"latitude": -5.0, "longitude":  14.0},
]

# Asterism definitions (ICRS RA/Dec for key component stars)
ASTERISMS_DEF = [
    {"name":"Orion's Belt","parent":"Orion","kind":"line",
     "components_icrs_deg":[(83.0017,-0.2991),(84.0534,-1.2019),(85.1897,-1.9426)]},
    {"name":"Pleiades","parent":"Taurus","kind":"cluster",
     "components_icrs_deg":[(56.8711,24.1052),(56.75,24.2),(56.95,24.0)]},
    {"name":"Big Dipper","parent":"Ursa Major","kind":"line",
     "components_icrs_deg":[(165.460,61.751),(165.931,56.382),(178.457,53.694),
                            (183.856,57.032),(193.507,55.959),(200.981,54.925),(206.885,49.313)]},
    {"name":"Summer Triangle","parent":"Aquila/Lyra/Cygnus","kind":"triangle",
     "components_icrs_deg":[(279.2347,38.7837),(310.3579,45.2803),(297.6958,8.8683)]},
    {"name":"Northern Cross","parent":"Cygnus","kind":"line",
     "components_icrs_deg":[(310.3579,45.2803),(305.557,40.2567),(292.68,33.9670)]},
]

# Coarse sampling grid to approximate constellation centroids in ICRS
DEFAULT_RA_STEP_DEG  = 8.0
DEFAULT_DEC_STEP_DEG = 8.0
DEFAULT_DEC_MIN_DEG  = -80.0
DEFAULT_DEC_MAX_DEG  =  80.0

# Visual sizing (kept stable; the app controls look)
BASE_RADIUS_M       = 300_000.0
ALT_SCALE_M_PER_DEG =   5_000.0  # we use a constant below to keep stable ring size

# ---------------------------- Helpers -----------------------------

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_snapshot(s: Optional[str]) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    # accept ...Z or offset form
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def sky_grid_icrs(ra_step_deg: float, dec_step_deg: float,
                  dec_min_deg: float, dec_max_deg: float) -> SkyCoord:
    ras  = np.arange(0.0, 360.0 + 1e-9, ra_step_deg)
    decs = np.arange(dec_min_deg, dec_max_deg + 1e-9, dec_step_deg)
    RA, DEC = np.meshgrid(ras, decs)
    return SkyCoord(ra=RA.flatten()*u.deg, dec=DEC.flatten()*u.deg, frame="icrs")

def mean_radec_deg(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    # average on unit sphere to avoid 0/360 wrap issues
    ra = np.radians([p[0] for p in points])
    dec = np.radians([p[1] for p in points])
    x = np.cos(dec)*np.cos(ra); y = np.cos(dec)*np.sin(ra); z = np.sin(dec)
    x_m, y_m, z_m = np.mean(x), np.mean(y), np.mean(z)
    r = (x_m*x_m + y_m*y_m + z_m*z_m) ** 0.5
    x_m, y_m, z_m = x_m/r, y_m/r, z_m/r
    ra_c  = (np.degrees(np.arctan2(y_m, x_m)) + 360.0) % 360.0
    dec_c = np.degrees(np.arcsin(z_m))
    return float(ra_c), float(dec_c)

def constellation_centroids(ra_step_deg: float, dec_step_deg: float,
                            dec_min_deg: float, dec_max_deg: float) -> Dict[str, Tuple[float, float]]:
    grid = sky_grid_icrs(ra_step_deg, dec_step_deg, dec_min_deg, dec_max_deg)
    labels = [get_constellation(pt) for pt in grid]
    buckets: Dict[str, List[int]] = {}
    for i, nm in enumerate(labels):
        buckets.setdefault(nm.strip(), []).append(i)
    cents: Dict[str, Tuple[float, float]] = {}
    for nm, idxs in buckets.items():
        pts = [(float(grid.ra[i].degree), float(grid.dec[i].degree)) for i in idxs]
        cents[nm] = mean_radec_deg(pts)
    return cents

def subpoint_for_ra_dec(ra_deg: float, dec_deg: float, snapshot_utc: datetime) -> Tuple[float, float]:
    """Return (lat, lon) of the Earth sub-point for a sky position at snapshot_utc."""
    t = Time(snapshot_utc)
    gst_hours = float(t.sidereal_time("mean", "greenwich").hour)
    lon = ((ra_deg/15.0 - gst_hours) * 15.0 + 180.0) % 360.0 - 180.0
    lat = dec_deg
    return float(lat), float(lon)

# --------------------------- Generator ----------------------------

def build_overlay(snapshot_utc: datetime,
                  ra_step_deg: float, dec_step_deg: float,
                  dec_min_deg: float, dec_max_deg: float,
                  include_asterisms: bool) -> dict:
    """
    Build overlay with:
      - items[] for all 88 constellations:
          name, center{lat,lon}, radiusMeters, icrs{ra_deg, dec_deg}
      - optional asterisms[] similarly with icrs from their components
    """
    cents = constellation_centroids(ra_step_deg, dec_step_deg, dec_min_deg, dec_max_deg)
    items = []

    # constant-ish ring size (keep UI stable)
    ring_m = BASE_RADIUS_M + ALT_SCALE_M_PER_DEG * 60.0

    for name, (ra_deg, dec_deg) in cents.items():
        lat, lon = subpoint_for_ra_dec(ra_deg, dec_deg, snapshot_utc)
        item = {
            "name": name,
            "center": {"latitude": lat, "longitude": lon},
            "radiusMeters": ring_m,
            # NEW: expose ICRS centroid so clients can compute hourly footprints
            "icrs": {"ra_deg": round(ra_deg, 3), "dec_deg": round(dec_deg, 3)},
            # keep meta minimal/harmless (back-compat with earlier clients)
            "meta": {"visibilityTonight": {
                "visible": True,
                "max_alt_deg": 90.0,
                "max_alt_time_utc": iso_z(snapshot_utc),
                "zenith_sep_deg": 0.0,
                "hours_local": []
            }}
        }
        items.append(item)

    payload = {
        "version": "overlay.v1",
        "generated_at_utc": iso_z(snapshot_utc),
        "items": sorted(items, key=lambda it: it["center"]["longitude"]),
        "starDots": {"bright": STAR_DOTS_BRIGHT}
    }

    if include_asterisms:
        ast_items = []
        for a in ASTERISMS_DEF:
            ra_c, dec_c = mean_radec_deg(a["components_icrs_deg"])
            lat, lon = subpoint_for_ra_dec(ra_c, dec_c, snapshot_utc)
            ast_items.append({
                "name": a["name"],
                "parent": a["parent"],
                "kind": a["kind"],
                "center": {"latitude": lat, "longitude": lon},
                "icrs": {"ra_deg": round(ra_c, 3), "dec_deg": round(dec_c, 3)},
                "meta": {"visibilityTonight": {
                    "visible": True,
                    "hours_local": [],
                    "max_alt_deg": 90.0,
                    "max_alt_time_utc": iso_z(snapshot_utc),
                    "zenith_sep_deg": 0.0
                }}
            })
        payload["asterisms"] = sorted(ast_items, key=lambda it: it["center"]["longitude"])

    return payload

# ----------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate constellation overlay JSON (Earth sub-points + ICRS).")
    ap.add_argument("--snapshot-utc", type=str, default=None,
                    help="UTC like 2025-10-10T03:00:00Z (default: now)")
    ap.add_argument("--ra-step",  type=float, default=DEFAULT_RA_STEP_DEG,
                    help="sky sampling RA step (deg) for centroid approx")
    ap.add_argument("--dec-step", type=float, default=DEFAULT_DEC_STEP_DEG,
                    help="sky sampling Dec step (deg) for centroid approx")
    ap.add_argument("--dec-min",  type=float, default=DEFAULT_DEC_MIN_DEG)
    ap.add_argument("--dec-max",  type=float, default=DEFAULT_DEC_MAX_DEG)
    ap.add_argument("--include-asterisms", action="store_true")
    ap.add_argument("--out", type=str, default="out/overlay.json")
    args = ap.parse_args()

    snapshot_utc = parse_snapshot(args.snapshot_utc)

    payload = build_overlay(snapshot_utc,
                            args.ra_step, args.dec_step,
                            args.dec_min, args.dec_max,
                            args.include_asterisms)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ wrote {out_path} at {payload['generated_at_utc']}")

if __name__ == "__main__":
    main()
