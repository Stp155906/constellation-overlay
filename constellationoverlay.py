#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConstellationOverlay — generator with ICRS centroid and optional hourly Earth sub-point footprints.

What it does
------------
- Outputs a JSON overlay for constellations (and optional asterisms).
- For each item, includes:
    * "icrs": {"ra_deg": <float>, "dec_deg": <float>}
    * Optional "footprint": hourly sub-point (lat/lon) for selected UTC hours on snapshot date
    * "center": legacy key (sub-point at snapshot UTC hour, see --center-hour)
    * "radiusMeters": preserved for backward compatibility (configurable via --radius-meters)
    * "meta": preserved, includes name and type ("constellation" or "asterism")

CLI
---
Examples:
  python constellationoverlay.py
  python constellationoverlay.py --snapshot-utc 2025-10-10T03:00:00Z --footprints --hours "16,17,18,19,20,21,22,23,0,1,2,3"
  python constellationoverlay.py --include-asterisms --out out/overlay.json

Key flags:
  --footprints               Toggle hourly sub-point footprints (default: off)
  --hours "h1,h2,..."        Comma list of UTC hours to include in footprint (default: 0..23)
  --snapshot-utc ISO         Snapshot timestamp (default: now, rounded down to the hour)
  --center-hour H            Which UTC hour to use for the legacy "center" (default: first in hours list)
  --radius-meters N          Legacy radiusMeters value (default: 150000.0)
  --include-asterisms        Include asterisms from the catalog (default: false)
  --out PATH                 Output JSON file (default: ./overlay.json)

Catalog
-------
Replace/extend CATALOG with your true constellation/asterism centroids (ICRS RA/Dec in degrees).
"""

from __future__ import annotations
import json, math, argparse, os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

# --- Demo catalog -------------------------------------------------------------
# Replace this with your actual catalog or wire up a loader.
CATALOG: List[Dict[str, Any]] = [
    {"name": "Orion", "type": "constellation", "ra_deg": 83.0, "dec_deg": -5.0},
    {"name": "Ursa Major", "type": "constellation", "ra_deg": 165.0, "dec_deg": 55.0},
    {"name": "Big Dipper", "type": "asterism", "ra_deg": 165.0, "dec_deg": 55.0},
]

# --- Astro utilities ----------------------------------------------------------
def _wrap_angle_deg(x: float) -> float:
    """Wrap to [0, 360)."""
    return x % 360.0

def _wrap_lon_deg_180(x: float) -> float:
    """Wrap longitude to [-180, 180], east-positive."""
    x = ((x + 180.0) % 360.0) - 180.0
    return x

def _julian_date(dt: datetime) -> float:
    """UTC datetime -> Julian Date."""
    # Algorithm from "Astronomical Algorithms" (Meeus) simplified for Gregorian dates.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + (dt.minute + dt.second/60.0)/60.0)/24.0

    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + (A // 4)
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return JD

def _gmst_deg(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time (degrees) at UTC datetime."""
    JD = _julian_date(dt)
    T = (JD - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + 0.000387933 * T*T - (T**3)/38710000.0
    return _wrap_angle_deg(gmst)

def subpoint_for_icrs(ra_deg: float, dec_deg: float, dt: datetime) -> Dict[str, float]:
    """
    Compute Earth sub-point (lat, lon) for object with ICRS coordinates at UTC datetime dt.
    - Latitude equals declination (neglecting nutation/polar motion): lat = dec.
    - Longitude is where local sidereal time equals RA: lon = RA - GMST (wrapped to [-180, 180]).
    Returns: {"lat": ..., "lon": ...} with east-positive longitude degrees.
    """
    gmst = _gmst_deg(dt)  # in degrees
    lon = _wrap_lon_deg_180(ra_deg - gmst)
    lat = max(-90.0, min(90.0, dec_deg))
    return {"lat": lat, "lon": lon}

# --- Builder -----------------------------------------------------------------
def parse_hours(s: str | None) -> List[int]:
    if not s or s.strip() == "":
        return list(range(24))
    parts = [p.strip() for p in s.split(",")]
    hours: List[int] = []
    for p in parts:
        if p == "":
            continue
        try:
            h = int(p)
        except ValueError:
            raise SystemExit(f"Bad hour '{p}'. Use integers 0..23 separated by commas.")
        if not (0 <= h <= 23):
            raise SystemExit(f"Hour out of range: {h}. Must be 0..23.")
        hours.append(h)
    # preserve order but unique
    seen = set()
    ordered = []
    for h in hours:
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered

def hour_datetimes_on_snapshot(snapshot_utc: datetime, hours: List[int]) -> List[datetime]:
    # Use the snapshot's calendar date in UTC; replace hour with each requested hour.
    base = snapshot_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    date_only = base.date()
    dts = [datetime(date_only.year, date_only.month, date_only.day, h, tzinfo=timezone.utc) for h in hours]
    return dts

def build_items(snapshot_utc: datetime,
                include_asterisms: bool,
                add_footprints: bool,
                hours: List[int],
                center_hour: int,
                radius_meters: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    dts = hour_datetimes_on_snapshot(snapshot_utc, hours) if add_footprints else []
    # We compute center using center_hour on the same snapshot date:
    center_dt = hour_datetimes_on_snapshot(snapshot_utc, [center_hour])[0] if add_footprints or (center_hour in range(24)) else snapshot_utc

    for row in CATALOG:
        if row.get("type") == "asterism" and not include_asterisms:
            continue
        name = row["name"]
        typ = row.get("type", "constellation")
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])

        # ICRS block
        icrs = {"ra_deg": ra, "dec_deg": dec}

        # Center (legacy): sub-point at chosen hour
        center = subpoint_for_icrs(ra, dec, center_dt)

        # Optional hourly footprint
        footprint = None
        if add_footprints:
            points = [subpoint_for_icrs(ra, dec, dt) for dt in dts]
            footprint = {"hours_utc": hours, "points": points}

        item: Dict[str, Any] = {
            "name": name,
            "type": typ,
            "icrs": icrs,
            "center": center,
            "radiusMeters": float(radius_meters),
            "meta": {"source": "catalog_demo", "notes": "Replace CATALOG with your true data"},
        }
        if footprint:
            item["footprint"] = footprint

        items.append(item)
    return items

def build_overlay(snapshot_utc: datetime,
                  include_asterisms: bool,
                  add_footprints: bool,
                  hours: List[int],
                  center_hour: int,
                  radius_meters: float) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_utc": snapshot_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "footprints": bool(add_footprints),
        "hours_utc": hours if add_footprints else None,
        "items": build_items(snapshot_utc, include_asterisms, add_footprints, hours, center_hour, radius_meters),
    }
    if payload["hours_utc"] is None:
        del payload["hours_utc"]
    return payload

# --- CLI ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate constellation overlay JSON with ICRS and optional footprints.")
    parser.add_argument("--snapshot-utc", type=str, default=None,
                        help="Snapshot UTC timestamp (ISO 8601, e.g., 2025-10-10T03:00:00Z). Default: now (floored to hour).")
    parser.add_argument("--include-asterisms", action="store_true", help="Include asterisms from the catalog.")
    parser.add_argument("--footprints", action="store_true", help="Include hourly sub-point footprints.")
    parser.add_argument("--hours", type=str, default=None, help="Comma list of UTC hours, e.g. \"16,17,18,19,20,21,22,23,0,1,2,3\". Default: 0..23.")
    parser.add_argument("--center-hour", type=int, default=None, help="UTC hour to compute the legacy 'center'. Default: first hour from --hours (or 0).")
    parser.add_argument("--radius-meters", type=float, default=150000.0, help="Legacy radiusMeters value per item.")
    parser.add_argument("--out", type=str, default="overlay.json", help="Output JSON path.")

    args = parser.parse_args()

    # Parse snapshot time (UTC)
    if args.snapshot_utc:
        try:
            if args.snapshot_utc.endswith("Z"):
                snapshot = datetime.strptime(args.snapshot_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            else:
                snapshot = datetime.fromisoformat(args.snapshot_utc)
                if snapshot.tzinfo is None:
                    snapshot = snapshot.replace(tzinfo=timezone.utc)
                snapshot = snapshot.astimezone(timezone.utc)
        except Exception as e:
            raise SystemExit(f"Bad --snapshot-utc value: {e}")
    else:
        # Default to now, floored to the hour.
        now = datetime.now(timezone.utc)
        snapshot = now.replace(minute=0, second=0, microsecond=0)

    hours = parse_hours(args.hours)
    center_hour = args.center_hour if args.center_hour is not None else (hours[0] if hours else 0)

    payload = build_overlay(snapshot_utc=snapshot,
                            include_asterisms=args.include_asterisms,
                            add_footprints=args.footprints,
                            hours=hours,
                            center_hour=center_hour,
                            radius_meters=args.radius_meters)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Wrote {args.out} for snapshot {payload['snapshot_utc']} (generated {payload['generated_at_utc']}).")

if __name__ == "__main__":
    main()
