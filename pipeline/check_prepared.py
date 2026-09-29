#!/usr/bin/env python3
r"""
QC the prepared dataset BEFORE transferring it to the cluster.

Checks (no matplotlib needed — text report + montage PNGs):
  1. cond/ , target/ , meta.csv counts all agree; no orphan files.
  2. meta.csv sanity: blank frequency cells, tx_power range/units,
     azimuth range, samples per (patch, freq_dir).
  3. Target value stats per freq_dir (mean/std/min/max, % saturated) --
     catches per-folder decode failures or a broken absolute scale.
  4. Flags near-constant targets (std < 2) = likely decode failures.
  5. Saves random [cond | target] montages to qc_montages/ for eyeballing
     (blob on the hotspot? lobe pointing at the hot side?).

Usage (Windows):
  py check_prepared.py F:\prepared\all --n-montage 24
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help=r"prepared dir, e.g. F:\prepared\all")
    ap.add_argument("--n-montage", type=int, default=24)
    ap.add_argument("--n-stats", type=int, default=400,
                    help="targets sampled per freq_dir for pixel stats")
    args = ap.parse_args()

    cond_dir, tgt_dir = os.path.join(args.root, "cond"), os.path.join(args.root, "target")
    meta = pd.read_csv(os.path.join(args.root, "meta.csv"))
    cond_files = set(os.listdir(cond_dir))
    tgt_files = set(os.listdir(tgt_dir))

    # ---- 1. counts ----
    print(f"meta.csv rows : {len(meta)}")
    print(f"cond PNGs     : {len(cond_files)}")
    print(f"target PNGs   : {len(tgt_files)}")
    names = set(meta["name"])
    for label, missing in [("cond missing for meta rows", names - cond_files),
                           ("target missing for meta rows", names - tgt_files),
                           ("cond orphans (not in meta)", cond_files - names),
                           ("target orphans (not in meta)", tgt_files - names)]:
        print(f"  {label}: {len(missing)}"
              + (f"  e.g. {sorted(missing)[:3]}" if missing else ""))
    if meta["name"].duplicated().any():
        print(f"  !! duplicated names in meta: {meta['name'].duplicated().sum()}")

    # ---- 2. meta sanity ----
    freq_num = pd.to_numeric(meta["frequency"], errors="coerce")
    print(f"\nfrequency: {freq_num.isna().sum()} blank/non-numeric of {len(meta)}"
          f" | range {freq_num.min()}..{freq_num.max()}")
    print("  unique frequencies:", sorted(freq_num.dropna().unique())[:20])
    pw = pd.to_numeric(meta["power"], errors="coerce")
    print(f"tx_power : {pw.isna().sum()} blank | range {pw.min()}..{pw.max()}"
          f" | uniques: {sorted(pw.dropna().unique())[:15]}")
    if pw.max() > 100:
        print("  NOTE: powers >100 exist -> power_factor clipped them to 1.0 in cond Ch2;"
              " raw value is still in meta for FiLM.")
    az = pd.to_numeric(meta["azimuth"], errors="coerce")
    print(f"azimuth  : range {az.min()}..{az.max()}")
    print("\nsamples per (patch, freq_dir):")
    print(meta.groupby(["patch", "freq_dir"]).size().unstack(fill_value=0))
    print("\nsamples per ant / power:")
    print(meta.groupby(["ant", "power"]).size().unstack(fill_value=0))

    # ---- 3+4. target stats per freq_dir ----
    print("\ntarget pixel stats (sampled):")
    print(f"{'freq_dir':>8} {'n':>5} {'mean':>7} {'std':>6} {'min':>4} {'max':>4} {'%=0':>6} {'%=255':>6} {'flat':>5}")
    rng = random.Random(0)
    flat_examples = []
    for fd, grp in meta.groupby("freq_dir"):
        pick = grp["name"].tolist()
        rng.shuffle(pick)
        pick = pick[:args.n_stats]
        vals, flat = [], 0
        for nm in pick:
            a = np.asarray(Image.open(os.path.join(tgt_dir, nm)), dtype=np.float32)
            if a.std() < 2:
                flat += 1
                if len(flat_examples) < 5:
                    flat_examples.append(nm)
            vals.append(a)
        v = np.stack(vals)
        print(f"{fd:>8} {len(pick):>5} {v.mean():>7.1f} {v.std():>6.1f} "
              f"{v.min():>4.0f} {v.max():>4.0f} {(v == 0).mean()*100:>5.1f}% "
              f"{(v == 255).mean()*100:>5.1f}% {flat:>5}")
    if flat_examples:
        print(f"  !! near-constant targets, inspect: {flat_examples}")

    # ---- 5. montages ----
    out = os.path.join(args.root, "qc_montages")
    os.makedirs(out, exist_ok=True)
    pick = meta.sample(min(args.n_montage, len(meta)), random_state=1)
    for _, r in pick.iterrows():
        c = np.asarray(Image.open(os.path.join(cond_dir, r["name"])).convert("RGB"))
        t = np.asarray(Image.open(os.path.join(tgt_dir, r["name"])).convert("L"))
        mont = np.concatenate([c, np.stack([t] * 3, 2)], axis=1)
        Image.fromarray(mont).save(os.path.join(out, r["name"]))
    print(f"\nwrote {len(pick)} montages -> {out}")
    print("Eyeball them: green-ish blob (Ch1) on the target hotspot, "
          "Ch2 lobe pointing at the hot side for directional antennas (Ant2).")


if __name__ == "__main__":
    main()
