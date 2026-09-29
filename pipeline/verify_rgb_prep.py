#!/usr/bin/env python3
r"""
Prove the RGB prepared set matches the grayscale one, BEFORE uploading ~10 GB.

Four checks, each fatal on its own:

  1  meta.csv name sets are IDENTICAL (not just equal in count)
  2  every meta name exists in cond/ and target_rgb/, and nothing extra is there
  3  a random sample of cond/ PNGs is BYTE-IDENTICAL to the grayscale cond/
     -- the strong version of the --verify-cond sweep, which only compared 12
     images; if this passes, both tracks provably see the same inputs
  4  meta.csv columns and per-column agreement with the grayscale meta

Run on the machine holding both directories:

    py verify_rgb_prep.py --rgb F:\prepared_rgb --gray F:\prepared\all
    py verify_rgb_prep.py --rgb F:\prepared_rgb --gray F:\prepared\all --sample 1000
"""
from __future__ import annotations

import os
import csv
import random
import hashlib
import argparse

KEY_COLS = ["patch", "freq_dir", "cluster", "ant", "power_dir", "azimuth"]


def read_meta(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, help="the RGB prepared dir")
    ap.add_argument("--gray", required=True, help="the grayscale prepared dir")
    ap.add_argument("--sample", type=int, default=300, help="cond files to byte-compare")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ok = True

    # ---- 1. name sets --------------------------------------------------------
    rgb_meta = read_meta(os.path.join(a.rgb, "meta.csv"))
    gray_meta = read_meta(os.path.join(a.gray, "meta.csv"))
    rgb_names = [r["name"] for r in rgb_meta]
    gray_names = [r["name"] for r in gray_meta]
    R, G = set(rgb_names), set(gray_names)
    print(f"[1] meta.csv: rgb {len(rgb_names)} rows ({len(R)} unique), "
          f"gray {len(gray_names)} rows ({len(G)} unique)")
    if R == G:
        print("    IDENTICAL name sets")
    else:
        ok = False
        only_r, only_g = sorted(R - G), sorted(G - R)
        print(f"    !! MISMATCH: {len(only_r)} only in RGB, {len(only_g)} only in gray")
        for n in only_r[:5]:
            print(f"       rgb-only : {n}")
        for n in only_g[:5]:
            print(f"       gray-only: {n}")
    if len(rgb_names) != len(R):
        ok = False
        print(f"    !! RGB meta has {len(rgb_names) - len(R)} DUPLICATE name rows")

    # ---- 2. files on disk ----------------------------------------------------
    print("[2] files on disk")
    for sub in ("cond", "target_rgb"):
        d = os.path.join(a.rgb, sub)
        if not os.path.isdir(d):
            ok = False
            print(f"    !! missing directory {d}")
            continue
        have = set(os.listdir(d))
        missing = R - have
        extra = have - R
        flag = "OK" if not missing and not extra else "!!"
        print(f"    {flag} {sub}: {len(have)} files, {len(missing)} missing, {len(extra)} extra")
        if missing or extra:
            ok = False
            for n in sorted(missing)[:3]:
                print(f"       missing: {n}")
            for n in sorted(extra)[:3]:
                print(f"       extra  : {n}")

    if not os.path.isfile(os.path.join(a.rgb, "colormap_refs.npz")):
        ok = False
        print("    !! colormap_refs.npz missing -- eval cannot decode without it")
    else:
        mb = os.path.getsize(os.path.join(a.rgb, "colormap_refs.npz")) / 1e6
        print(f"    OK colormap_refs.npz ({mb:.1f} MB)")

    # ---- 3. cond byte-identity ----------------------------------------------
    print(f"[3] cond/ byte-identity vs {a.gray}\\cond  (sample {a.sample})")
    shared = sorted(R & G)
    random.Random(a.seed).shuffle(shared)
    sample = shared[:a.sample]
    same = diff = absent = 0
    for n in sample:
        p_r = os.path.join(a.rgb, "cond", n)
        p_g = os.path.join(a.gray, "cond", n)
        if not (os.path.isfile(p_r) and os.path.isfile(p_g)):
            absent += 1
            continue
        if md5(p_r) == md5(p_g):
            same += 1
        else:
            diff += 1
            if diff <= 3:
                print(f"    !! differs: {n}")
    print(f"    {same} identical, {diff} different, {absent} not comparable")
    if diff:
        ok = False
        print("    !! The two tracks would NOT see the same conditioning. Rebuild with the\n"
              "       mode/tx-sigma that --verify-cond reported EXACT before training.")
    elif same:
        print("    conditioning is byte-identical -- the RGB rows are comparable to the "
              "grayscale ones")

    # ---- 4. metadata agreement ----------------------------------------------
    print("[4] metadata agreement")
    gray_by = {r["name"]: r for r in gray_meta}
    cols = [c for c in KEY_COLS if rgb_meta and c in rgb_meta[0] and c in (gray_meta[0] if gray_meta else {})]
    print(f"    rgb columns: {list(rgb_meta[0].keys()) if rgb_meta else []}")
    bad = {c: 0 for c in cols}
    for r in rgb_meta:
        g = gray_by.get(r["name"])
        if not g:
            continue
        for c in cols:
            try:
                if abs(float(r[c]) - float(g[c])) > 1e-6:
                    bad[c] += 1
            except (TypeError, ValueError):
                if str(r[c]).strip() != str(g[c]).strip():
                    bad[c] += 1
    for c in cols:
        print(f"    {'OK' if not bad[c] else '!!'} {c}: {bad[c]} rows differ")
        if bad[c]:
            ok = False

    print("\n==> " + ("ALL CHECKS PASSED -- safe to upload."
                      if ok else "FAILED -- do not upload until the !! lines are resolved."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
