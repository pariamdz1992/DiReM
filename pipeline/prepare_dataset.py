#!/usr/bin/env python3
r"""
Build (cond, target) pairs for the FULL HTZ dataset (all patches l*, all
frequencies f*, all clusters/antennas/powers), with per-folder colormap
calibration so targets are the true absolute signal values.

Per site (= one transmitter):
  cond (3ch, region-absolute, 256x256):
    Ch0 = layout (buildings)
    Ch1 = Gaussian at the transmitter's TRUE pixel (CSV lat/lon)
    Ch2 = .msi antenna pattern as a lobe at the Tx, rotated to tx_ant_azimuth,
          scaled by per-transmitter power
  target (1ch, 256x256) = site coverage decoded via the folder's colormap
          (value 0..255, absolute / dBm-based)

Also writes meta.csv (name, patch, freq_dir, frequency, power, cluster, ant,
azimuth) for the later frequency+power scalar conditioning.

MSI order assumed Ant1..4 = B5 / N5-45x2 / R2-5 / ePMP (override with --antN).

Example:
  py prepare_dataset.py --root F:\  --out F:\prepared\all ^
     --ant1 ...Ant1.msi --ant2 ...Ant2.msi --ant3 ...Ant3.msi --ant4 ...Ant4.msi
  (--preview --cluster F:\l62\f1\cluster_1  processes just that cluster to prep_preview/)
"""
import os
import re
import csv
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image

OUT_SIZE = 256
REF_POWER = 100.0


def natural_key(s):
    n = re.search(r"site(\d+)", os.path.basename(s))
    return int(n.group(1)) if n else -1


def power_factor(tx_power):
    return float(np.clip(float(tx_power) / REF_POWER, 0.0, 1.0))


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


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


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def folder_setup(folder):
    """Return (layout_gray, geo->pixel fn, colormap refs) calibrated from THIS
    folder's coverage.BMP <-> coverage.CSV (per-folder auto-scaled colormap)."""
    layout = load_gray(os.path.join(folder, "layout.BMP"))
    H, W = layout.shape
    cov_rgb = np.asarray(Image.open(os.path.join(folder, "coverage.BMP")).convert("RGB"))
    Hc, Wc, _ = cov_rgb.shape
    raw = pd.read_csv(os.path.join(folder, "coverage.CSV"), sep=";", header=None,
                      usecols=[0, 1, 2], names=["lon", "lat", "val"], dtype=str)
    lon, lat, val = _to_num(raw["lon"]), _to_num(raw["lat"]), _to_num(raw["val"])
    m = lon.notna() & lat.notna() & val.notna()
    lon, lat, val = lon[m].values, lat[m].values, val[m].values
    lon0, lon1, lat0, lat1 = lon.min(), lon.max(), lat.min(), lat.max()

    def g2p(la, lo):
        return ((lo - lon0) / (lon1 - lon0) * (W - 1),
                (lat1 - la) / (lat1 - lat0) * (H - 1))

    # colormap inverse from coverage (color -> value)
    px = np.clip(((lon - lon0) / (lon1 - lon0) * (Wc - 1)).round().astype(int), 0, Wc - 1)
    py = np.clip(((lat1 - lat) / (lat1 - lat0) * (Hc - 1)).round().astype(int), 0, Hc - 1)
    pc = cov_rgb[py, px].astype(np.int64)
    key = pc[:, 0] * 65536 + pc[:, 1] * 256 + pc[:, 2]
    u, inv = np.unique(key, return_inverse=True)
    mv = np.bincount(inv, weights=val) / np.bincount(inv)
    refs = np.stack([u // 65536, (u // 256) % 256, u % 256, mv], 1).astype(np.float32)
    return layout, g2p, refs


def decode_site(path, refs):
    """Decode a jet-colormapped site BMP into value (0..255) via the folder refs."""
    im = Image.open(path)
    if im.mode == "P":
        pal = np.asarray(im.getpalette()[:768], dtype=np.float32).reshape(256, 3)
        d = ((pal[:, None, :] - refs[None, :, :3]) ** 2).sum(2)
        return refs[d.argmin(1), 3][np.asarray(im)]
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    d = ((rgb[:, None, :] - refs[None, :, :3]) ** 2).sum(2)
    return refs[d.argmin(1), 3].reshape(im.size[1], im.size[0])


def build_cond(layout, txx, txy, azimuth, patt_lin, pfactor, tx_sigma=6.0, tau_frac=0.35):
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


def resize01(arr):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0


def resize_u8(arr255):
    im = Image.fromarray(np.clip(arr255, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR))


def process_folder(folder, patt_lin, tag, meta, out_dir=None, preview=False, n_preview=6):
    layout, g2p, refs = folder_setup(folder)
    df = pd.read_csv([c for c in glob.glob(os.path.join(folder, "p*_data.csv"))
                      if "noheader" not in c][0])
    sites = sorted(glob.glob(os.path.join(folder, "site*.BMP")), key=natural_key)
    if len(df) != len(sites):
        print(f"  !! {tag}: {len(sites)} sites vs {len(df)} rows -- skipping")
        return
    if not preview:
        os.makedirs(os.path.join(out_dir, "cond"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "target"), exist_ok=True)
    else:
        os.makedirs("prep_preview", exist_ok=True)

    for i, sp in enumerate(sites):
        r = df.iloc[i]
        pf = power_factor(r["tx_power"])
        txx, txy = g2p(r["latitude"], r["longitude"])
        cond = resize01(build_cond(layout, txx, txy, r["tx_ant_azimuth"], patt_lin, pf))
        tgt = resize_u8(decode_site(sp, refs))                  # value 0..255 (absolute)
        name = f"{tag}_{os.path.basename(sp)[:-4]}.png"

        if preview:
            if i < n_preview:
                mont = np.concatenate([(cond * 255).astype(np.uint8),
                                       np.stack([tgt] * 3, 2)], axis=1)
                Image.fromarray(mont).save(os.path.join("prep_preview", name))
            continue

        Image.fromarray((cond * 255).astype(np.uint8)).save(os.path.join(out_dir, "cond", name))
        Image.fromarray(tgt.astype(np.uint8)).save(os.path.join(out_dir, "target", name))
        meta.append([name, r.get("tx_frequency", ""), r["tx_power"], r["tx_ant_azimuth"]])

    print(f"  {tag}: {len(sites)} pairs"
          + (f" (target range {int(refs[:,3].min())}..{int(refs[:,3].max())})" if not preview else ""))


PART = re.compile(r"^l\d+", re.I)
FREQ = re.compile(r"^(f\d+|fdef)$", re.I)
CLUS = re.compile(r"^cluster_\d+$", re.I)
ANT = re.compile(r"^Ant\d+$", re.I)


def parse_path(root, folder):
    parts = os.path.relpath(folder, root).split(os.sep)
    g = lambda rx: next((p for p in parts if rx.match(p)), "NA")
    return g(PART), g(FREQ), g(CLUS), g(ANT), os.path.basename(folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing l* patches (walks the whole tree)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--cluster", default=None, help="preview: a single cluster dir path")
    ap.add_argument("--ant1", required=True)
    ap.add_argument("--ant2", required=True)
    ap.add_argument("--ant3", required=True)
    ap.add_argument("--ant4", required=True)
    args = ap.parse_args()
    patt = {"Ant1": parse_msi(args.ant1), "Ant2": parse_msi(args.ant2),
            "Ant3": parse_msi(args.ant3), "Ant4": parse_msi(args.ant4)}

    if args.preview:
        base = args.cluster or args.root
        for ant in ["Ant1", "Ant2", "Ant3", "Ant4"]:
            f = os.path.join(base, ant, "p10")
            if os.path.isdir(f):
                print(f"[preview {ant}/p10]")
                process_folder(f, patt[ant], f"{ant}_p10", [], preview=True)
        return

    # ---- 1) discover every leaf folder (coverage.BMP + site*.BMP) ----
    leaves = []
    for dirpath, _, files in os.walk(args.root):
        low = [x.lower() for x in files]
        if "coverage.bmp" in low and any(x.startswith("site") and x.endswith(".bmp") for x in low):
            leaves.append(dirpath)
    from collections import Counter
    parsed = [(lf,) + parse_path(args.root, lf) for lf in leaves]
    patches = sorted({p[1] for p in parsed})
    freqs = sorted({p[2] for p in parsed})
    clusters = sorted({(p[1], p[2], p[3]) for p in parsed})
    print(f"discovered {len(leaves)} leaf folders under {args.root}")
    print(f"  patches: {patches}")
    print(f"  freqs:   {freqs}")
    print(f"  #(patch,freq,cluster) groups: {len(clusters)}")
    if len(patches) <= 1 and len(clusters) <= 1:
        print("  WARNING: only one cluster found -- is --root pointing at a single cluster "
              "instead of the folder that CONTAINS the l* patches?")

    # ---- 2) process each leaf ----
    meta = []
    for lf, patch, freq, clus, ant, power in parsed:
        if ant not in patt:
            print(f"  skip {lf} (ant='{ant}' not in Ant1..4)")
            continue
        tag = f"{patch}_{freq}_{clus}_{ant}_{power}"
        rows0 = len(meta)
        try:
            process_folder(lf, patt[ant], tag, meta, out_dir=args.out)
        except Exception as e:
            print(f"  !! error {tag}: {e}")
            continue
        for k in range(rows0, len(meta)):
            meta[k] = [meta[k][0], patch, freq, meta[k][1], power, clus, ant, meta[k][3]]

    with open(os.path.join(args.out, "meta.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "patch", "freq_dir", "frequency", "power", "cluster", "ant", "azimuth"])
        w.writerows(meta)
    print(f"\nDONE: {len(meta)} pairs -> {args.out}  (+ meta.csv)")


if __name__ == "__main__":
    main()
