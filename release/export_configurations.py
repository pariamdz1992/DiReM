r"""Export the per-configuration files of the DiReM v1 release from the raw HTZ exports.

For every configuration listed in meta.csv (patch / carrier / cluster / antenna / power) this
reads the export folder <raw>\<patch>\<carrier>\<cluster>\<ant>\<power>\ and writes to <out>:

  emf\<config>.png           field strength E in dBuV/m (8-bit; 0 = no value) on HTZ's own
                             calculation grid. E(V/m) = 10**(E/20) * 1e-6.
  emf\<config>.csv           instead of the PNG, when the points do not form a regular grid:
                             lon, lat, E for every exported point
  coverage_rgb\<config>.png  coverage.BMP as exported (all transmitters of the configuration)
  layout\<config>.png        layout.BMP (OpenStreetMap basemap)
  object\<config>.png        object.BMP (base-station markers)
  configurations.csv         one row per configuration: grid type and geometry, reference carrier
  transmitters.csv           one row per map in meta.csv: the transmitter's record

Grids. HTZ calculates on a regular grid, usually 2 m in UTM (NAD83). The script finds each
configuration's grid by trying UTM, then longitude/latitude, then Web Mercator; `grid`, `epsg`,
`x0`, `y0` (centre of the top-left cell), `step_x` and `step_y` in configurations.csv are in the
units of that coordinate system. All variants of one tile cover the same extent.

<config> is <patch>_<carrier>_<cluster>_<ant>_<power>. Nothing under <raw> is modified.
Re-running skips configurations already listed in configurations.csv.

    py -m pip install pyproj
    py export_configurations.py --raw F:\ --meta F:\prepared\all\meta.csv --out C:\direm_v1_configs
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

KEY = ["patch", "freq_dir", "cluster", "ant", "power_dir"]
TX_FIELDS = ["location", "latitude", "longitude", "technology", "site_type", "tx_frequency",
             "tx_power", "tx_ant_azimuth", "tx_ant_elevation_angle", "structure_height",
             "tx_ant_height", "height", "tx_ant_gain", "tx_line_loss"]
CONF_FIELDS = ["config", "patch", "carrier", "cluster", "ant", "power", "grid", "epsg", "x0", "y0",
               "step_x", "step_y", "width", "height", "n_points", "ref_frequency_mhz"]


def grid_step(v, tol):
    """Spacing of the lattice the coordinates v lie on (None if there is only one position).

    lon/lat are written with 6 decimals, so points scatter slightly around their lattice position;
    values closer than `tol` are merged into one position before the gaps are measured.
    """
    u = np.unique(v)
    if len(u) < 2:
        return None
    groups = np.split(u, np.where(np.diff(u) > tol)[0] + 1)
    gaps = np.diff([grp.mean() for grp in groups])
    return float(np.median(gaps)) if len(gaps) else None


def fit_grid(x, y, extent, tol, default):
    """Place the points on their lattice, extended to the tile extent. None if they do not fit."""
    sx, sy = grid_step(x, tol) or default, grid_step(y, tol) or default
    ux0, ux1, uy0, uy1 = extent
    x0 = x.min() - np.floor((x.min() - ux0) / sx + 0.5) * sx
    y0 = y.max() + np.floor((uy1 - y.max()) / sy + 0.5) * sy
    col, row = (x - x0) / sx, (y0 - y) / sy
    ci, ri = np.round(col).astype(int), np.round(row).astype(int)
    if max(np.abs(col - ci).max(), np.abs(row - ri).max()) > 0.25:
        return None
    width = max(int(np.floor((ux1 - x0) / sx + 0.5)) + 1, ci.max() + 1)
    height = max(int(np.floor((y0 - uy0) / sy + 0.5)) + 1, ri.max() + 1)
    return x0, y0, sx, sy, width, height, ci, ri


def read_coverage(path):
    """lon, lat, E arrays and the reference carrier from a coverage.CSV (footer included)."""
    raw = pd.read_csv(path, sep=";", header=None, usecols=[0, 1, 2], names=["lon", "lat", "E"],
                      dtype=str, low_memory=False)
    lon, lat, e = (pd.to_numeric(raw[c], errors="coerce") for c in ("lon", "lat", "E"))
    ok = lon.notna() & lat.notna() & e.notna()
    freq = ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 400))
        m = re.search(rb"Frequency \(MHz\):\s*([\d.]+)", f.read())
        if m:
            freq = m.group(1).decode()
    return lon[ok].to_numpy(), lat[ok].to_numpy(), e[ok].to_numpy(), freq


def site_rows(folder, power_dir):
    """{site number: CSV row} for one export folder (sites sorted by number = CSV row order)."""
    rows = list(csv.DictReader(open(os.path.join(folder, f"{power_dir}_data.csv"),
                                    newline="", encoding="utf-8-sig", errors="replace")))
    nums = sorted(int(re.search(r"site(\d+)", os.path.basename(p), re.I).group(1))
                  for p in glob.glob(os.path.join(folder, "site*.BMP")))
    if len(nums) != len(rows):
        raise ValueError(f"{len(nums)} site images vs {len(rows)} CSV rows")
    return dict(zip(nums, rows))


def append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder holding the l## patch folders")
    ap.add_argument("--meta", required=True, help="meta.csv of the release")
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()
    for sub in ("emf", "coverage_rgb", "layout", "object"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    conf_csv = os.path.join(a.out, "configurations.csv")
    tx_csv = os.path.join(a.out, "transmitters.csv")
    done = set()
    if os.path.exists(conf_csv):                        # bring an older run's file to this layout
        old = pd.read_csv(conf_csv, dtype=str)
        if "step_m" in old.columns:
            old = old.rename(columns={"step_m": "step_x"})
            old["step_y"], old["grid"] = old["step_x"], "utm"
            old[CONF_FIELDS].to_csv(conf_csv, index=False)
        done = set(old.config)

    meta = pd.read_csv(a.meta)
    meta["site"] = meta["name"].str.extract(r"site(\d+)\.png$", flags=re.I)[0].astype(int)
    meta["config"] = meta[KEY].astype(str).agg("_".join, axis=1)
    problems = []
    for t, (tile, tm) in enumerate(meta.groupby(["patch", "freq_dir", "cluster"], sort=False), 1):
        confs = list(tm.groupby(KEY, sort=False))
        if all(g.config.iloc[0] in done for _, g in confs):
            continue
        data = {}
        for key, g in confs:
            folder = os.path.join(a.raw, *key)
            try:
                data[key] = (folder, g, *read_coverage(os.path.join(folder, "coverage.CSV")))
            except Exception as ex:
                problems.append(("_".join(key), f"coverage.CSV: {ex}"))
        if not data:
            continue
        zone = int((np.concatenate([d[2] for d in data.values()]).mean() + 180) // 6) + 1
        crs_list = [("utm", 26900 + zone, 0.5, 2.0),     # grid, EPSG, merge tolerance, default step
                    ("lonlat", 4269, 1e-5, 2e-5),
                    ("webmercator", 3857, 0.5, 2.0)]
        coords, extents = {}, {}

        def xy(grid, epsg, key):
            if (grid, key) not in coords:
                lon, lat = data[key][2], data[key][3]
                coords[grid, key] = (lon, lat) if epsg == 4269 else \
                    Transformer.from_crs("EPSG:4269", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
            return coords[grid, key]

        def extent(grid, epsg):
            if grid not in extents:
                pts = [xy(grid, epsg, k) for k in data]
                extents[grid] = (min(p[0].min() for p in pts), max(p[0].max() for p in pts),
                                 min(p[1].min() for p in pts), max(p[1].max() for p in pts))
            return extents[grid]

        for key, (folder, g, lon, lat, e, freq) in data.items():
            config = g.config.iloc[0]
            if config in done:
                continue
            try:
                rows = site_rows(folder, key[4])
                tx = []
                for r in g.itertuples():
                    rec = rows[r.site]
                    if abs(float(rec["tx_ant_azimuth"] or 0) - float(r.azimuth)) > 0.5:
                        raise ValueError(f"azimuth mismatch for {r.name}")
                    tx.append({"name": r.name, **{f: rec.get(f, "") for f in TX_FIELDS}})
            except Exception as ex:
                problems.append((config, str(ex)))
                continue
            fit = None
            for grid, epsg, tol, default in crs_list:
                x, y = xy(grid, epsg, key)
                fit = fit_grid(np.asarray(x), np.asarray(y), extent(grid, epsg), tol, default)
                if fit:
                    break
            if fit:
                x0, y0, sx, sy, width, height, ci, ri = fit
                img = np.zeros((height, width), np.uint8)
                img[ri, ci] = np.clip(np.round(e), 0, 255).astype(np.uint8)
                Image.fromarray(img, mode="L").save(os.path.join(a.out, "emf", config + ".png"))
            else:                                       # no regular grid: keep the points
                grid, epsg, x0 = "points", 4269, ""
                y0 = sx = sy = width = height = ""
                pd.DataFrame({"lon": lon, "lat": lat, "E": np.round(e).astype(int)}).to_csv(
                    os.path.join(a.out, "emf", config + ".csv"), index=False)
            for src, sub in (("coverage.BMP", "coverage_rgb"), ("layout.BMP", "layout"), ("object.BMP", "object")):
                path = os.path.join(folder, src)
                if os.path.exists(path):                # a missing image does not stop the rest
                    Image.open(path).save(os.path.join(a.out, sub, config + ".png"))
                else:
                    problems.append((config, f"no {src} (other files exported)"))
            r8 = lambda v: round(float(v), 8) if v != "" else ""
            append_csv(tx_csv, ["name"] + TX_FIELDS, tx)
            append_csv(conf_csv, CONF_FIELDS, [dict(config=config, patch=key[0], carrier=key[1],
                       cluster=key[2], ant=key[3], power=key[4], grid=grid, epsg=epsg, x0=r8(x0),
                       y0=r8(y0), step_x=r8(sx), step_y=r8(sy), width=width, height=height,
                       n_points=len(e), ref_frequency_mhz=freq)])
            done.add(config)
        if t % 20 == 0:
            print(f"  {t} tiles, {len(done)} configurations done")

    if os.path.exists(tx_csv):                          # a re-run may repeat a few rows
        pd.read_csv(tx_csv).drop_duplicates("name", keep="last").to_csv(tx_csv, index=False)
    conf = pd.read_csv(conf_csv)
    print(f"done: {len(done)} of {meta.config.nunique()} configurations; grids: {conf.grid.value_counts().to_dict()}")
    if problems:
        with open(os.path.join(a.out, "export_problems.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("config", "problem"), *problems])
        print(f"{len(problems)} problems -> export_problems.csv")


if __name__ == "__main__":
    main()
