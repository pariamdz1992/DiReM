#!/usr/bin/env python3
r"""
Repair meta.csv: the 'power' column contains the folder tag (p10/p50/pdef)
instead of the numeric per-transmitter tx_power (prepare_dataset.py dropped
the CSV value when re-assembling rows). This re-joins tx_power from the raw
p*_data.csv files -- no re-prep needed.

Join logic: within one leaf folder, CSV row i corresponds to the i-th site
in natural (site-number) order. meta rows are grouped per leaf folder, ranked
by site number, and matched positionally. Every join is CROSS-CHECKED by
requiring the CSV's tx_ant_azimuth to equal the azimuth already in meta
(and tx_frequency to match where present); any mismatch aborts that group
loudly instead of writing wrong powers.

Writes:
  meta_orig.csv  (backup of the current meta.csv)
  meta.csv       with columns: name, patch, freq_dir, frequency,
                 power (NUMERIC tx_power), power_dir (p10/p50/pdef),
                 cluster, ant, azimuth

Usage (Windows):
  py fix_meta_power.py --root F:\ --prepared F:\prepared\all
"""
import os
import re
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

SITE_RE = re.compile(r"site(\d+)\.png$", re.I)


def leaf_dir(root, patch, freq_dir, cluster, ant, power_dir):
    p = os.path.join(root, patch, freq_dir, cluster, ant, power_dir)
    if os.path.isdir(p):
        return p
    hits = glob.glob(os.path.join(root, patch, "**", freq_dir, cluster, ant, power_dir),
                     recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r"raw dataset root, e.g. F:\ ")
    ap.add_argument("--prepared", required=True, help=r"prepared dir with meta.csv")
    args = ap.parse_args()

    meta_path = os.path.join(args.prepared, "meta.csv")
    meta = pd.read_csv(meta_path)
    if "power_dir" in meta.columns:
        print("meta.csv already has power_dir -- looks repaired. Nothing to do.")
        return

    meta = meta.rename(columns={"power": "power_dir"})
    meta["power"] = np.nan
    meta["site_no"] = meta["name"].str.extract(SITE_RE.pattern, flags=re.I)[0].astype(int)

    keys = ["patch", "freq_dir", "cluster", "ant", "power_dir"]
    n_groups = bad = 0
    for gkey, grp in meta.groupby(keys, sort=False):
        n_groups += 1
        folder = leaf_dir(args.root, *gkey)
        tag = "/".join(map(str, gkey))
        if folder is None:
            print(f"  !! {tag}: raw folder not found -- powers left NaN")
            bad += 1
            continue
        csvs = [c for c in glob.glob(os.path.join(folder, "p*_data.csv"))
                if "noheader" not in c]
        if not csvs:
            print(f"  !! {tag}: no p*_data.csv -- powers left NaN")
            bad += 1
            continue
        df = pd.read_csv(csvs[0])
        if len(df) != len(grp):
            print(f"  !! {tag}: {len(df)} CSV rows vs {len(grp)} meta rows -- skipped")
            bad += 1
            continue

        order = grp.sort_values("site_no").index          # positional rank = CSV row
        az_csv = pd.to_numeric(df["tx_ant_azimuth"], errors="coerce").values
        az_meta = pd.to_numeric(meta.loc[order, "azimuth"], errors="coerce").values
        if np.nanmax(np.abs(az_csv - az_meta)) > 0.5:
            print(f"  !! {tag}: azimuth cross-check FAILED "
                  f"(max diff {np.nanmax(np.abs(az_csv - az_meta)):.1f} deg) -- skipped")
            bad += 1
            continue
        if "tx_frequency" in df.columns:
            fq_csv = pd.to_numeric(df["tx_frequency"], errors="coerce").values
            fq_meta = pd.to_numeric(meta.loc[order, "frequency"], errors="coerce").values
            if np.nanmax(np.abs(fq_csv - fq_meta)) > 0.01:
                print(f"  !! {tag}: frequency cross-check FAILED -- skipped")
                bad += 1
                continue

        meta.loc[order, "power"] = pd.to_numeric(df["tx_power"], errors="coerce").values

    filled = meta["power"].notna().sum()
    print(f"\ngroups: {n_groups} total, {bad} skipped/failed")
    print(f"power filled for {filled}/{len(meta)} rows")
    pw = meta["power"].dropna()
    if len(pw):
        print(f"tx_power range {pw.min()}..{pw.max()} | "
              f"uniques ({pw.nunique()}): {sorted(pw.unique())[:15]}"
              f"{' ...' if pw.nunique() > 15 else ''}")
        print("per power_dir:")
        print(meta.groupby("power_dir")["power"].agg(["min", "max", "nunique"]))
    if filled < len(meta):
        print("!! some rows unfilled -- FIX THE WARNINGS ABOVE before uploading.")

    cols = ["name", "patch", "freq_dir", "frequency", "power", "power_dir",
            "cluster", "ant", "azimuth"]
    shutil.copy2(meta_path, os.path.join(args.prepared, "meta_orig.csv"))
    meta[cols].to_csv(meta_path, index=False)
    print(f"\nbackup -> meta_orig.csv ; repaired -> {meta_path}")


if __name__ == "__main__":
    main()
