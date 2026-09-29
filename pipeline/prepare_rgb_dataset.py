#!/usr/bin/env python3
r"""
Build the COMPLETE RGB dataset from the RAW HTZ store, in one pass.

    <out>/cond/<name>.png         3ch conditioning, REGENERATED from raw
    <out>/target_rgb/<name>.png   the site BMP as RGB (256px, NEAREST) -- the
                                  target, with NO colormap decoding and NO
                                  grayscale conversion
    <out>/meta.csv                name,patch,freq_dir,frequency,power,power_dir,
                                  cluster,ant,azimuth   (the repaired schema)
    <out>/colormap_refs.npz       per-folder color->value tables, needed at EVAL
                                  to score an RGB prediction in the same space
                                  the grayscale rows were scored in
    <out>/target_rgb_names.txt    manifest

Scope: ONLY the l<NN> patch folders directly under --raw-root. Everything else on
the drive (HTZ\, test\, output_sites\, $RECYCLE.BIN\ ...) is not the dataset --
it has no l<NN> component, so prepare_dataset.py tagged it "NA" and it was never
part of the prepared set.

Nothing is deduplicated or dropped. The only folders that produce no output are
the two prepare_dataset.py itself passed over, reproduced verbatim:
  * antenna dir not in Ant1..Ant4
  * len(p*_data.csv rows) != number of site*.BMP
Both are counted and printed.

THE CONDITIONING MATH IS IMPORTED, NOT REWRITTEN. `regenerate_cond.py` (the
script that actually produced the grayscale cond/) must sit next to this file or
be pointed at with --regen. A second implementation of build_cond would drift,
and a drifted conditioning invalidates the whole RGB-vs-grayscale comparison.

STEP 1 -- find out which settings produced the existing cond/, so the RGB track
gets the SAME inputs (this only READS the grayscale cond, and writes nothing):

    py prepare_rgb_dataset.py --raw-root F:\ --verify-cond F:\prepared\all\cond ^
       --ant1 A1.msi --ant2 A2.msi --ant3 A3.msi --ant4 A4.msi

STEP 2 -- build, using the winning mode/sigma it reports:

    py prepare_rgb_dataset.py --raw-root F:\ --out F:\prepared_rgb ^
       --mode legacy --tx-sigma 20 ^
       --ant1 A1.msi --ant2 A2.msi --ant3 A3.msi --ant4 A4.msi
"""
from __future__ import annotations

import os
import re
import csv
import sys
import glob
import argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colormap_codec import folder_refs, save_refs                        # noqa: E402

OUT_SIZE = 256
META_COLS = ["name", "patch", "freq_dir", "frequency", "power", "power_dir",
             "cluster", "ant", "azimuth"]
PART = re.compile(r"^l\d+$", re.I)
FREQ = re.compile(r"^(f\d+|fdef)$", re.I)
CLUS = re.compile(r"^cluster_\d+$", re.I)
ANT = re.compile(r"^Ant\d+$", re.I)
ANT_OK = re.compile(r"^Ant[1-4]$", re.I)
CANDIDATES = [("legacy", 6.0), ("legacy", 20.0), ("db", 20.0), ("fspl", 20.0),
              ("db", 6.0), ("fspl", 6.0)]


def load_regen(path_hint=""):
    """Import regenerate_cond.py -- the script that built the grayscale cond."""
    import importlib.util
    for cand in (path_hint, os.path.join(HERE, "regenerate_cond.py"),
                 os.path.join(os.path.dirname(HERE), "regenerate_cond.py"),
                 "regenerate_cond.py"):
        if cand and os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("regenerate_cond", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"conditioning math imported from {cand}")
            return mod
    raise SystemExit(
        "regenerate_cond.py not found. Put it next to this script (or pass --regen "
        "PATH).\nIt is the script that produced the grayscale cond/, and reusing it "
        "is what guarantees\nthe RGB track gets byte-identical conditioning.")


def patch_dirs(raw_root):
    subs = sorted(d for d in os.listdir(raw_root)
                  if PART.match(d) and os.path.isdir(os.path.join(raw_root, d)))
    if not subs:
        raise SystemExit(f"no l<NN> patch folders directly under {raw_root!r} -- "
                         f"point --raw-root at the drive/dir that CONTAINS l33, l62, ...")
    return subs


def find_leaves(raw_root, patches):
    leaves = []
    for p in patches:
        for dirpath, _, files in os.walk(os.path.join(raw_root, p)):
            low = [x.lower() for x in files]
            if "coverage.bmp" in low and any(x.startswith("site") and x.endswith(".bmp")
                                             for x in low):
                leaves.append(dirpath)
    return sorted(leaves)


def parse_tag(raw_root, folder):
    parts = os.path.relpath(folder, raw_root).split(os.sep)
    g = lambda rx: next((p for p in parts if rx.match(p)), "NA")
    patch, freq, clus, ant = g(PART), g(FREQ), g(CLUS), g(ANT)
    power_dir = os.path.basename(folder)
    return patch, freq, clus, ant, power_dir, f"{patch}_{freq}_{clus}_{ant}_{power_dir}"


def folder_rows(folder):
    """The p*_data.csv rows, positionally paired with the sorted site files --
    the same join prepare_dataset.py and regenerate_cond.py both use."""
    import pandas as pd
    cands = [c for c in glob.glob(os.path.join(folder, "p*_data.csv")) if "noheader" not in c]
    if not cands:
        return None, None
    df = pd.read_csv(cands[0])
    natural = lambda s: int(re.search(r"site(\d+)", os.path.basename(s)).group(1))
    sites = sorted(glob.glob(os.path.join(folder, "site*.BMP")), key=natural)
    return df, sites


def make_cond(R, folder_cache, r, mode, tx_sigma, args):
    """One cond image, via regenerate_cond.py's own functions."""
    import pandas as pd
    layout, g2p, mpx, f_med = folder_cache
    txx, txy = g2p(r["latitude"], r["longitude"])
    tx_dbm = R.to_dbm(r["tx_power"], args.power_units)
    f_mhz = float(pd.to_numeric(pd.Series([r.get("tx_frequency")]),
                                errors="coerce").fillna(f_med).iloc[0])
    patt = args._patt[args._ant]
    if mode == "fspl":
        cond_hw, _ = R.build_cond_fspl(layout, txx, txy, r["tx_ant_azimuth"], patt,
                                       tx_dbm, f_mhz, mpx, tx_sigma=tx_sigma,
                                       vmin=args.vmin, vmax=args.vmax)
    elif mode == "legacy":
        pf = float(np.clip(tx_dbm / 100.0, 0.0, 1.0))
        cond_hw = R.build_cond(layout, txx, txy, r["tx_ant_azimuth"], patt, pf,
                               tx_sigma=tx_sigma)
    else:
        cond_hw = R.build_cond(layout, txx, txy, r["tx_ant_azimuth"], patt,
                               R.db_power_factor(tx_dbm, args.plo, args.phi),
                               tx_sigma=tx_sigma)
    return (R.resize01(cond_hw) * 255).astype(np.uint8), f_mhz


def setup_folder(R, folder):
    import pandas as pd
    layout, g2p, mpx = R.geo_setup(folder)
    df, _ = folder_rows(folder)
    f_med = pd.to_numeric(df.get("tx_frequency"), errors="coerce").median()
    return layout, g2p, mpx, f_med


# --------------------------------------------------------------------------- #
# STEP 1: which settings produced the existing cond?
# --------------------------------------------------------------------------- #
def verify_cond(R, raw_root, cond_dir, args, n=12):
    patches = patch_dirs(raw_root)
    leaves = find_leaves(raw_root, patches)
    print(f"patches: {', '.join(patches)}   leaf folders: {len(leaves)}")
    existing = set(os.listdir(cond_dir))
    print(f"existing cond/: {len(existing)} files\n")

    results = {c: [] for c in CANDIDATES}
    checked = 0
    for folder in leaves:
        patch, freq, clus, ant, power_dir, tag = parse_tag(raw_root, folder)
        if not ANT_OK.match(ant):
            continue
        df, sites = folder_rows(folder)
        if df is None or len(df) != len(sites):
            continue
        pairs = [(i, sp, f"{tag}_{os.path.basename(sp)[:-4]}.png")
                 for i, sp in enumerate(sites)]
        pairs = [p for p in pairs if p[2] in existing]
        if not pairs:
            continue
        cache = setup_folder(R, folder)
        args._ant = ant
        for i, sp, name in pairs[:2]:
            ref = np.asarray(Image.open(os.path.join(cond_dir, name)).convert("RGB"))
            for mode, sigma in CANDIDATES:
                got, _ = make_cond(R, cache, df.iloc[i], mode, sigma, args)
                d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
                results[(mode, sigma)].append((float(d.mean()), float(d.max()),
                                               float((d == 0).mean())))
            checked += 1
            if checked >= n:
                break
        if checked >= n:
            break

    if not checked:
        raise SystemExit("no sample matched between the raw tree and that cond/ dir")
    print(f"compared {checked} images against {cond_dir}\n")
    print(f"  {'mode':8s} {'tx_sigma':>8}   {'mean|diff|':>10} {'max|diff|':>9} {'% identical px':>14}")
    best = None
    for c in CANDIDATES:
        a = np.array(results[c])
        row = (a[:, 0].mean(), a[:, 1].max(), 100 * a[:, 2].mean())
        print(f"  {c[0]:8s} {c[1]:8.1f}   {row[0]:10.3f} {row[1]:9.0f} {row[2]:13.1f}%")
        if best is None or row[0] < best[1][0]:
            best = (c, row)
    print(f"\n==> BEST MATCH: --mode {best[0][0]} --tx-sigma {best[0][1]:g}  "
          f"(mean|diff| {best[1][0]:.3f}, {best[1][2]:.1f}% identical pixels)")
    if best[1][0] < 0.01:
        print("    EXACT: the RGB track will get byte-identical conditioning.")
    else:
        print("    NOT exact. Do not build until this is understood -- different\n"
              "    conditioning means the RGB and grayscale rows are not comparable.\n"
              "    (Check the .msi files and their Ant1..4 order first.)")


# --------------------------------------------------------------------------- #
# STEP 2: build
# --------------------------------------------------------------------------- #
def build(R, raw_root, out, args):
    patches = patch_dirs(raw_root)
    leaves = find_leaves(raw_root, patches)
    print(f"patches: {', '.join(patches)}   leaf folders: {len(leaves)}")
    print(f"mode={args.mode} tx_sigma={args.tx_sigma} power_units={args.power_units} "
          f"resize={args.resize}")
    cond_dir = os.path.join(out, "cond")
    tgt_dir = os.path.join(out, "target_rgb")
    os.makedirs(cond_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    resample = Image.NEAREST if args.resize == "nearest" else Image.BILINEAR

    meta, refs_by_tag, written = [], {}, []
    skip_ant, skip_count, skip_refs = [], [], []
    for k, folder in enumerate(leaves):
        patch, freq, clus, ant, power_dir, tag = parse_tag(raw_root, folder)
        if not ANT_OK.match(ant):
            skip_ant.append(tag)
            continue
        df, sites = folder_rows(folder)
        if df is None or len(df) != len(sites):
            skip_count.append(f"{tag} ({len(sites) if sites else 0} sites vs "
                              f"{0 if df is None else len(df)} rows)")
            continue
        try:
            refs_by_tag[tag] = folder_refs(folder)
            cache = setup_folder(R, folder)
        except Exception as e:                                   # noqa: BLE001
            skip_refs.append(f"{tag} ({e})")
            continue

        args._ant = ant
        for i, sp in enumerate(sites):
            r = df.iloc[i]
            name = f"{tag}_{os.path.basename(sp)[:-4]}.png"
            # cond is the slow half (full-res mgrid trig per site). --no-cond skips
            # it for the case where a byte-identical cond/ already exists and has
            # been PROVEN identical with --verify-cond.
            cond, f_mhz = make_cond(R, cache, r, args.mode, args.tx_sigma, args)
            if not args.no_cond:
                Image.fromarray(cond).save(os.path.join(cond_dir, name))
            src = Image.open(sp).convert("RGB")
            tgt = src.resize((OUT_SIZE, OUT_SIZE), resample)
            tgt.save(os.path.join(tgt_dir, name))

            if args.qc and len(written) < args.qc:
                # source (smoothly downscaled, for the eye only) | target | cond
                qc_dir = os.path.join(out, "qc")
                os.makedirs(qc_dir, exist_ok=True)
                ref = src.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
                mont = np.concatenate([np.asarray(ref), np.asarray(tgt), cond], axis=1)
                Image.fromarray(mont).save(os.path.join(qc_dir, name))
            meta.append([name, patch, freq, f_mhz, float(r["tx_power"]), power_dir,
                         clus, ant, r["tx_ant_azimuth"]])
            written.append(name)
            if args.limit and len(written) >= args.limit:
                break
        if (k + 1) % 50 == 0:
            print(f"  [{k + 1}/{len(leaves)}] {len(written)} pairs, {len(refs_by_tag)} palettes")
        if args.limit and len(written) >= args.limit:
            break

    with open(os.path.join(out, "meta.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(META_COLS)
        w.writerows(meta)
    save_refs(os.path.join(out, "colormap_refs.npz"), refs_by_tag)
    with open(os.path.join(out, "target_rgb_names.txt"), "w") as fh:
        fh.write("\n".join(written))

    for label, lst in (("antenna dir not Ant1..4", skip_ant),
                       ("site/row count mismatch", skip_count),
                       ("folder setup failed", skip_refs)):
        if lst:
            print(f"\n  SKIPPED ({label}): {len(lst)} folders")
            for s in lst[:5]:
                print(f"    {s}")
            if len(lst) > 5:
                print(f"    ... and {len(lst) - 5} more")

    print(f"\nDONE: {len(written)} pairs -> {out}")
    print(f"  cond/ {'SKIPPED (--no-cond)' if args.no_cond else len(written)}   "
          f"target_rgb/ {len(written)}   meta.csv {len(meta)} rows   "
          f"colormap_refs.npz {len(refs_by_tag)} palettes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="dir CONTAINING the l<NN> patch folders")
    ap.add_argument("--out", default="", help="output dir (required unless --verify-cond)")
    ap.add_argument("--verify-cond", default="", help="existing cond/ dir: report which "
                                                      "mode/tx-sigma reproduces it, then exit")
    ap.add_argument("--verify-n", type=int, default=12)
    ap.add_argument("--regen", default="", help="path to regenerate_cond.py")
    ap.add_argument("--mode", choices=["fspl", "db", "legacy"], default="legacy")
    ap.add_argument("--tx-sigma", type=float, default=20.0)
    ap.add_argument("--power-units", choices=["dbm", "mixed"], default="dbm")
    ap.add_argument("--vmin", type=float, default=-120.0)
    ap.add_argument("--vmax", type=float, default=30.0)
    ap.add_argument("--plo", type=float, default=0.0)
    ap.add_argument("--phi", type=float, default=60.0)
    ap.add_argument("--resize", default="nearest", choices=["nearest", "bilinear"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cond", action="store_true",
                    help="write target_rgb/ + meta.csv + refs but NOT cond/ — only when a "
                         "cond/ proven identical by --verify-cond will be copied in")
    ap.add_argument("--qc", type=int, default=0,
                    help="write N side-by-side montages to <out>/qc/: "
                         "source | target_rgb | cond. Look at these before a full run.")
    ap.add_argument("--ant1", required=True)
    ap.add_argument("--ant2", required=True)
    ap.add_argument("--ant3", required=True)
    ap.add_argument("--ant4", required=True)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    R = load_regen(args.regen)
    args._patt = {"Ant1": R.parse_msi(args.ant1), "Ant2": R.parse_msi(args.ant2),
                  "Ant3": R.parse_msi(args.ant3), "Ant4": R.parse_msi(args.ant4)}
    args._ant = "Ant1"

    if args.verify_cond:
        verify_cond(R, args.raw_root, args.verify_cond, args, n=args.verify_n)
        return
    if not args.out:
        raise SystemExit("--out is required (or use --verify-cond)")
    build(R, args.raw_root, args.out, args)


if __name__ == "__main__":
    main()
