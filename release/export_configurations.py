r"""Export the per-configuration files of the DiReM v1 release from the raw HTZ exports.

For every configuration listed in meta.csv (patch / carrier / cluster / antenna / power) this
reads the export folder <raw>\<patch>\<carrier>\<cluster>\<ant>\<power>\ and writes to <out>:

  emf\<config>.png           field strength E in dBuV/m (8-bit; 0 = no value) on HTZ's native
                             2 m calculation grid (UTM, NAD83). E(V/m) = 10**(E/20) * 1e-6.
  coverage_rgb\<config>.png  coverage.BMP as exported (all transmitters of the configuration)
  layout\<config>.png        layout.BMP (OpenStreetMap basemap)
  object\<config>.png        object.BMP (base-station markers)
  configurations.csv         one row per configuration: grid geometry and reference carrier
  transmitters.csv           one row per map in meta.csv: the transmitter's record

Each configuration keeps HTZ's own calculation grid (its spacing is measured from the points, 2 m
for most tiles). All variants of one tile (4 antennas x 3 powers) cover the same extent.
<config> is <patch>_<carrier>_<cluster>_<ant>_<power>. Nothing under <raw> is modified.
Re-running skips configurations already listed in configurations.csv.

    py -m pip install pyproj
    py export_configurations.py --raw F:\ --meta F:\prepared\all\meta.csv --out D:\direm_v1_configs
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
STEP = 2.0                                     # default HTZ calculation grid (m)


def grid_step(v):
    """Spacing (m) of the lattice the projected coordinates v lie on.

    lon/lat are written with 6 decimals, so projected points scatter by about 0.1 m around their
    lattice position. Values closer than 0.5 m are merged into one lattice position first.
    """
    u = np.unique(np.round(v, 2))
    if len(u) < 2:
        return STEP
    groups = np.split(u, np.where(np.diff(u) > 0.5)[0] + 1)
    centres = np.array([grp.mean() for grp in groups])
    gaps = np.diff(centres)
    return round(float(np.median(gaps)), 2) if len(gaps) else STEP
TX_FIELDS = ["location", "latitude", "longitude", "technology", "site_type", "tx_frequency",
             "tx_power", "tx_ant_azimuth", "tx_ant_elevation_angle", "structure_height",
             "tx_ant_height", "height", "tx_ant_gain", "tx_line_loss"]
CONF_FIELDS = ["config", "patch", "carrier", "cluster", "ant", "power", "epsg", "x0", "y0",
               "step_m", "width", "height", "n_points", "ref_frequency_mhz"]


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
    done = set(pd.read_csv(conf_csv).config) if os.path.exists(conf_csv) else set()

    meta = pd.read_csv(a.meta)
    meta["site"] = meta["name"].str.extract(r"site(\d+)\.png$", flags=re.I)[0].astype(int)
    meta["config"] = meta[KEY].astype(str).agg("_".join, axis=1)
    problems = []
    tiles = meta.groupby(["patch", "freq_dir", "cluster"], sort=False)
    for t, (tile, tm) in enumerate(tiles, 1):
        confs = [c for c in tm.groupby(KEY, sort=False)]
        if all(g.config.iloc[0] in done for _, g in confs):
            continue
        # read every variant of the tile, project to UTM, and build one shared grid
        data = {}
        for key, g in confs:
            folder = os.path.join(a.raw, *key)
            try:
                lon, lat, e, freq = read_coverage(os.path.join(folder, "coverage.CSV"))
                data[key] = (folder, g, lon, lat, e, freq)
            except Exception as ex:
                problems.append(("_".join(key), f"coverage.CSV: {ex}"))
        if not data:
            continue
        all_lon = np.concatenate([d[2] for d in data.values()])
        zone = int((all_lon.mean() + 180) // 6) + 1
        epsg = 26900 + zone                                        # NAD83 / UTM zone
        tr = Transformer.from_crs("EPSG:4269", f"EPSG:{epsg}", always_xy=True)
        xy = {k: tr.transform(d[2], d[3]) for k, d in data.items()}
        ux0 = min(v[0].min() for v in xy.values()); ux1 = max(v[0].max() for v in xy.values())
        uy0 = min(v[1].min() for v in xy.values()); uy1 = max(v[1].max() for v in xy.values())

        for key, (folder, g, lon, lat, e, freq) in data.items():
            config = g.config.iloc[0]
            if config in done:
                continue
            x, y = xy[key]
            step = min(grid_step(x), grid_step(y))
            # this configuration's own lattice, extended to the tile's full extent
            x0 = x.min() - np.floor((x.min() - ux0) / step + 0.5) * step
            y0 = y.max() + np.floor((uy1 - y.max()) / step + 0.5) * step
            col, row = (x - x0) / step, (y0 - y) / step
            ci, ri = np.round(col).astype(int), np.round(row).astype(int)
            resid = max(np.abs(col - ci).max(), np.abs(row - ri).max())
            if resid > 0.25:
                problems.append((config, f"points not on a regular grid (step {step} m, residual {resid:.2f} cells)"))
                continue
            width = max(int(np.floor((ux1 - x0) / step + 0.5)) + 1, ci.max() + 1)
            height = max(int(np.floor((y0 - uy0) / step + 0.5)) + 1, ri.max() + 1)
            grid = np.zeros((height, width), np.uint8)
            grid[ri, ci] = np.clip(np.round(e), 0, 255).astype(np.uint8)
            try:
                rows = site_rows(folder, key[4])
                tx = []
                for r in g.itertuples():
                    rec = rows[r.site]
                    if abs(float(rec["tx_ant_azimuth"] or 0) - float(r.azimuth)) > 0.5:
                        raise ValueError(f"azimuth mismatch for {r.name}")
                    tx.append({"name": r.name, **{f: rec.get(f, "") for f in TX_FIELDS}})
                Image.fromarray(grid, mode="L").save(os.path.join(a.out, "emf", config + ".png"))
            except Exception as ex:
                problems.append((config, str(ex)))
                continue
            for src, sub in (("coverage.BMP", "coverage_rgb"), ("layout.BMP", "layout"), ("object.BMP", "object")):
                path = os.path.join(folder, src)
                if os.path.exists(path):                       # a missing image does not stop the rest
                    Image.open(path).save(os.path.join(a.out, sub, config + ".png"))
                else:
                    problems.append((config, f"no {src} (other files exported)"))
            append_csv(tx_csv, ["name"] + TX_FIELDS, tx)
            append_csv(conf_csv, CONF_FIELDS, [dict(config=config, patch=key[0], carrier=key[1],
                       cluster=key[2], ant=key[3], power=key[4], epsg=epsg, x0=round(float(x0), 3),
                       y0=round(float(y0), 3), step_m=step, width=width, height=height,
                       n_points=len(e), ref_frequency_mhz=freq)])
            done.add(config)
        if t % 20 == 0:
            print(f"  {t} tiles, {len(done)} configurations done")

    if os.path.exists(tx_csv):                                   # a re-run may repeat a few rows
        pd.read_csv(tx_csv).drop_duplicates("name", keep="last").to_csv(tx_csv, index=False)
    n_conf = meta.config.nunique()
    print(f"done: {len(done)} of {n_conf} configurations")
    if problems:
        with open(os.path.join(a.out, "export_problems.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("config", "problem"), *problems])
        print(f"{len(problems)} problems -> export_problems.csv")


if __name__ == "__main__":
    main()
