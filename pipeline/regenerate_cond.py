#!/usr/bin/env python3
r"""
Regenerate ONLY the cond/ PNGs of an already-prepared dataset, with a better
Ch2 power/antenna encoding. Targets and meta.csv are untouched (no colormap
decoding), so this is much faster than a full re-prep.

SELF-CONTAINED: does not import prepare_dataset.py (avoids version-skew /
ImportError when only this file is copied to the Windows box).

tx_power in the CSVs is in dBm (confirmed 2026-07-15; p10/p50 = 10/50 dBm set
manually, pdef = real measured values, 0.2..159.3 dBm).

Modes (Ch0 = layout, Ch1 = Tx gaussian, unchanged in both):
  fspl (default): Ch2 = analytic free-space received power in dBm,
        P(px) = tx_power_dBm + G_ant(rel bearing, dB)
                - (20*log10(d_m) + 20*log10(f_MHz) - 27.55)
        normalized with a FIXED global range (--vmin/--vmax dBm) so power,
        frequency and azimuth are all physically encoded and comparable.
        This is the fix for "model blind to power": power enters ADDITIVELY
        in dB, so 105 vs 159 dBm give different fields (no clip-to-1.0).
  db:   old lobe shape (pattern x exponential decay) but power factor =
        tx dBm normalized over [--plo,--phi] with a 0.05 floor.

--tx-sigma is the Tx gaussian sigma in FULL-RES px. The old prep used 6.0 ->
~2px at 256 -> ~0.5px at the Swin's 64x64 cond features (invisible!). Default
here is 20 -> ~7px at 256, ~1.7px at 64: actually visible to the model.

Output names match prepare_dataset.py, so <out>/cond PNGs are overwritten
in place.

Usage (Windows; raw F:\ must be attached):
  Preview one cluster (no --root needed):
    py regenerate_cond.py --preview --cluster F:\l62\f1\cluster_1 --mode fspl ^
       --tx-sigma 20 --ant1 A1.msi --ant2 A2.msi --ant3 A3.msi --ant4 A4.msi
  Full run:
    py regenerate_cond.py --root F:\ --out F:\prepared\all --mode fspl ^
       --tx-sigma 20 --ant1 A1.msi --ant2 A2.msi --ant3 A3.msi --ant4 A4.msi
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image

OUT_SIZE = 256
M_PER_DEG = 111320.0

PART = re.compile(r"^l\d+", re.I)
FREQ = re.compile(r"^(f\d+|fdef)$", re.I)
CLUS = re.compile(r"^cluster_\d+$", re.I)
ANT = re.compile(r"^Ant\d+$", re.I)


# ---------- helpers (inlined from prepare_dataset.py) ----------
def natural_key(s):
    n = re.search(r"site(\d+)", os.path.basename(s))
    return int(n.group(1)) if n else -1


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def parse_msi(path):
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    i = next(k for k, l in enumerate(lines) if l.strip().upper().startswith("HORIZONTAL"))
    count = int(re.findall(r"\d+", lines[i])[0])
    atten = np.zeros(360)
    for j in range(count):
        parts = lines[i + 1 + j].split()
        if len(parts) >= 2:
            atten[int(round(float(parts[0]))) % 360] = float(parts[1])
    return (10.0 ** (-atten / 10.0)).astype(np.float32)


def parse_path(root, folder):
    # parse the ABSOLUTE path (robust to whatever --root resolves to, incl.
    # PowerShell's drive-relative "F:"). Split on both separators.
    parts = re.split(r"[\\/]+", os.path.abspath(folder))
    g = lambda rx: next((p for p in parts if rx.match(p)), "NA")
    return g(PART), g(FREQ), g(CLUS), g(ANT), os.path.basename(folder)


def resize01(arr):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0


def build_cond(layout, txx, txy, azimuth, patt_lin, pfactor, tx_sigma=20.0, tau_frac=0.35):
    """Old-style Ch2 = pattern x exponential decay x power factor."""
    H, W = layout.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ch0 = layout.astype(np.float32) / 255.0
    ch1 = np.exp(-((xx - txx) ** 2 + (yy - txy) ** 2) / (2 * tx_sigma ** 2)).astype(np.float32)
    east, north = xx - txx, txy - yy
    bearing = np.degrees(np.arctan2(east, north)) % 360
    rel = np.mod(np.round(bearing - azimuth).astype(int), 360)
    decay = np.exp(-np.hypot(east, north) / (tau_frac * max(W, H)))
    ch2 = np.clip(pfactor * patt_lin[rel] * decay, 0, 1).astype(np.float32)
    return np.stack([ch0, ch1, ch2], axis=2)


# ---------- power / fspl ----------
def to_dbm(v, power_units="dbm"):
    """power_units='dbm' (default): trust the source label, use as-is.
    power_units='mixed': treat values >60 as WATTS and convert (only if
    power_vs_target.py shows the >60 group shines like ~50 dBm; the 2026-07-15
    test said DON'T -> keep 'dbm')."""
    v = float(v)
    if power_units == "mixed" and v > 60.0:
        return 10 * np.log10(v) + 30
    return v


def db_power_factor(tx_dbm, lo=0.0, hi=60.0):
    return float(np.clip((float(tx_dbm) - lo) / (hi - lo), 0.05, 1.0))


def geo_setup(folder):
    """layout, geo->pixel fn, and (mx, my) meters-per-pixel from coverage.CSV."""
    layout = load_gray(os.path.join(folder, "layout.BMP"))
    H, W = layout.shape
    raw = pd.read_csv(os.path.join(folder, "coverage.CSV"), sep=";", header=None,
                      usecols=[0, 1], names=["lon", "lat"], dtype=str)
    lon, lat = _to_num(raw["lon"]), _to_num(raw["lat"])
    m = lon.notna() & lat.notna()
    lon, lat = lon[m].values, lat[m].values
    lon0, lon1, lat0, lat1 = lon.min(), lon.max(), lat.min(), lat.max()

    def g2p(la, lo):
        return ((lo - lon0) / (lon1 - lon0) * (W - 1),
                (lat1 - la) / (lat1 - lat0) * (H - 1))

    my = (lat1 - lat0) * M_PER_DEG / (H - 1)
    mx = (lon1 - lon0) * M_PER_DEG * np.cos(np.radians((lat0 + lat1) / 2)) / (W - 1)
    return layout, g2p, (mx, my)


def build_cond_fspl(layout, txx, txy, azimuth, patt_lin, tx_dbm, f_mhz, mpx,
                    tx_sigma=20.0, vmin=-120.0, vmax=30.0):
    H, W = layout.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ch0 = layout.astype(np.float32) / 255.0
    ch1 = np.exp(-((xx - txx) ** 2 + (yy - txy) ** 2) / (2 * tx_sigma ** 2)).astype(np.float32)
    east = (xx - txx) * mpx[0]
    north = (txy - yy) * mpx[1]
    bearing = np.degrees(np.arctan2(east, north)) % 360
    rel = np.mod(np.round(bearing - azimuth).astype(int), 360)
    gain_db = 10 * np.log10(np.maximum(patt_lin, 1e-6))          # = -atten_dB
    d = np.maximum(np.hypot(east, north), min(mpx))              # >= ~1 px, no log(0)
    p_dbm = (float(tx_dbm) + gain_db[rel]
             - (20 * np.log10(d) + 20 * np.log10(float(f_mhz)) - 27.55))
    ch2 = np.clip((p_dbm - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
    return np.stack([ch0, ch1, ch2], axis=2), p_dbm


def process_folder(folder, patt_lin, tag, args, stats, existing, preview=False, n_preview=6):
    layout, g2p, mpx = geo_setup(folder)
    df = pd.read_csv([c for c in glob.glob(os.path.join(folder, "p*_data.csv"))
                      if "noheader" not in c][0])
    sites = sorted(glob.glob(os.path.join(folder, "site*.BMP")), key=natural_key)
    if len(df) != len(sites):
        print(f"  !! {tag}: {len(sites)} sites vs {len(df)} rows -- skipping")
        return
    f_med = pd.to_numeric(df.get("tx_frequency"), errors="coerce").median()

    for i, sp in enumerate(sites):
        r = df.iloc[i]
        txx, txy = g2p(r["latitude"], r["longitude"])
        tx_dbm = to_dbm(r["tx_power"], args.power_units)
        f_mhz = float(pd.to_numeric(pd.Series([r.get("tx_frequency")]),
                                    errors="coerce").fillna(f_med).iloc[0])
        if args.mode == "fspl":
            cond_hw, p_dbm = build_cond_fspl(layout, txx, txy, r["tx_ant_azimuth"],
                                             patt_lin, tx_dbm, f_mhz, mpx,
                                             tx_sigma=args.tx_sigma,
                                             vmin=args.vmin, vmax=args.vmax)
            stats["dbm_min"] = min(stats["dbm_min"], float(p_dbm.min()))
            stats["dbm_max"] = max(stats["dbm_max"], float(p_dbm.max()))
        elif args.mode == "legacy":
            pf = float(np.clip(tx_dbm / 100.0, 0.0, 1.0))   # original power_factor
            cond_hw = build_cond(layout, txx, txy, r["tx_ant_azimuth"],
                                 patt_lin, pf, tx_sigma=args.tx_sigma)
        else:
            cond_hw = build_cond(layout, txx, txy, r["tx_ant_azimuth"],
                                 patt_lin, db_power_factor(tx_dbm, args.plo, args.phi),
                                 tx_sigma=args.tx_sigma)
        cond = resize01(cond_hw)
        name = f"{tag}_{os.path.basename(sp)[:-4]}.png"

        if preview:
            if i < n_preview:
                os.makedirs("regen_preview", exist_ok=True)
                Image.fromarray((cond * 255).astype(np.uint8)).save(
                    os.path.join("regen_preview", name))
            continue

        out_path = os.path.join(args.out, "cond", name)
        stats["overwritten" if name in existing else "new"] += 1
        Image.fromarray((cond * 255).astype(np.uint8)).save(out_path)
    print(f"  {tag}: {len(sites)} conds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="raw dataset root (required for full run)")
    ap.add_argument("--out", default=None, help="prepared dir containing cond/ (full run)")
    ap.add_argument("--mode", choices=["fspl", "db", "legacy"], default="fspl",
                    help="legacy = EXACT baseline cond (clip(dBm/100) power factor); "
                         "use --tx-sigma 6 to reproduce the original 0.162-NMSE cond bit-for-bit")
    ap.add_argument("--vmin", type=float, default=-120.0, help="fspl: received dBm mapped to 0")
    ap.add_argument("--vmax", type=float, default=30.0, help="fspl: received dBm mapped to 1")
    ap.add_argument("--plo", type=float, default=0.0, help="db mode: tx dBm mapped to floor")
    ap.add_argument("--phi", type=float, default=60.0, help="db mode: tx dBm mapped to 1")
    ap.add_argument("--power-units", choices=["dbm", "mixed"], default="dbm",
                    help="dbm=trust label as-is (default); mixed=convert >60 as watts")
    ap.add_argument("--tx-sigma", type=float, default=20.0,
                    help="Tx gaussian sigma in FULL-RES px (old prep=6.0 -> invisible at 64x64)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--cluster", default=None, help="preview: a single cluster dir path")
    ap.add_argument("--ant1", required=True)
    ap.add_argument("--ant2", required=True)
    ap.add_argument("--ant3", required=True)
    ap.add_argument("--ant4", required=True)
    args = ap.parse_args()
    patt = {"Ant1": parse_msi(args.ant1), "Ant2": parse_msi(args.ant2),
            "Ant3": parse_msi(args.ant3), "Ant4": parse_msi(args.ant4)}
    stats = {"overwritten": 0, "new": 0, "dbm_min": np.inf, "dbm_max": -np.inf}

    if args.preview:
        base = args.cluster or args.root
        if not base:
            ap.error("--preview needs --cluster (or --root) pointing at a cluster dir")
        for ant in ["Ant1", "Ant2", "Ant3", "Ant4"]:
            f = os.path.join(base, ant, "p10")
            if os.path.isdir(f):
                print(f"[preview {ant}/p10 mode={args.mode} tx_sigma={args.tx_sigma}]")
                process_folder(f, patt[ant], f"{ant}_p10", args, stats, set(), preview=True)
        if args.mode == "fspl" and np.isfinite(stats["dbm_min"]):
            print(f"fspl dBm range seen: {stats['dbm_min']:.1f} .. {stats['dbm_max']:.1f} "
                  f"(mapped over [{args.vmin}, {args.vmax}]) -> montages in regen_preview/")
        return

    # ---- full run ----
    if not args.root or not args.out:
        ap.error("full run needs both --root (raw tree) and --out (prepared dir with cond/)")
    existing = set(os.listdir(os.path.join(args.out, "cond")))
    leaves = []
    for dirpath, _, files in os.walk(args.root):
        low = [x.lower() for x in files]
        if "coverage.bmp" in low and any(x.startswith("site") and x.endswith(".bmp") for x in low):
            leaves.append(dirpath)
    print(f"discovered {len(leaves)} leaf folders; existing cond PNGs: {len(existing)}")
    if leaves:
        _p = parse_path(args.root, leaves[0])
        print(f"  sample parsed tag: {'_'.join(_p[:5])}  (from {leaves[0]})")
        if len(leaves) < 1000:
            print(f"  !! only {len(leaves)} leaves found -- --root is probably too deep. "
                  f"Point it at the DRIVE ROOT (use forward slash: --root F:/ ). Aborting.")
            return

    skipped_na = 0
    for lf in leaves:
        patch, freq, clus, ant, power = parse_path(args.root, lf)
        if ant not in patt:
            continue
        if patch == "NA" or freq == "NA":
            skipped_na += 1
            continue
        tag = f"{patch}_{freq}_{clus}_{ant}_{power}"
        try:
            process_folder(lf, patt[ant], tag, args, stats, existing)
        except Exception as e:
            print(f"  !! error {tag}: {e}")

    print(f"\nDONE mode={args.mode}: {stats['overwritten']} cond PNGs overwritten, "
          f"{stats['new']} NEW names (should be 0 -- investigate if not), "
          f"{skipped_na} skipped (unparsable patch/freq)")
    if stats["overwritten"] + stats["new"] != len(existing):
        print(f"!! {len(existing) - stats['overwritten'] - stats['new']} existing cond files "
              f"were NOT regenerated -- counts should match, investigate before re-upload.")
    if args.mode == "fspl" and np.isfinite(stats["dbm_min"]):
        print(f"fspl dBm range seen: {stats['dbm_min']:.1f} .. {stats['dbm_max']:.1f} "
              f"(mapped over [{args.vmin}, {args.vmax}]) -- if badly clipped, adjust --vmin/--vmax.")


if __name__ == "__main__":
    main()
