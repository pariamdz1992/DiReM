#!/usr/bin/env bash
# Package the frozen DiReM v1 record (the 122,021 training maps) on the cluster.
#
#   bash package_v1.sh <prepared_dir> <prepared_rgb_dir> <out_dir>
#
#   <prepared_dir>      holds cond/, target/ and meta.csv
#   <prepared_rgb_dir>  holds target_rgb/ and colormap_refs.npz
#   <out_dir>           where the record's files are written (put it on $SCRATCH: it needs
#                       about as much space as cond/ + target/ + target_rgb/)
#
# Writes cond.tar, target.tar, target_rgb.tar, meta.csv, colormap_refs.npz,
# split_train.txt, split_val.txt, split_test.txt and SHA256SUMS.txt. Reads the inputs only.
set -euo pipefail
PREP=${1:?prepared dir}; RGB=${2:?prepared_rgb dir}; OUT=${3:?output dir}
mkdir -p "$OUT"

# 1. every name in meta.csv has exactly one file in cond/, target/ and target_rgb/
python3 - "$PREP" "$RGB" <<'EOF'
import os, sys
import pandas as pd
prep, rgb = sys.argv[1:3]
names = set(pd.read_csv(os.path.join(prep, "meta.csv"))["name"])
for d in (os.path.join(prep, "cond"), os.path.join(prep, "target"), os.path.join(rgb, "target_rgb")):
    files = {f for f in os.listdir(d) if f.endswith(".png")}
    print(f"{d}: {len(files)} files, {len(names - files)} missing, {len(files - names)} extra")
    if files != names:
        sys.exit("file names do not match meta.csv -- stopping")
print(f"meta.csv: {len(names)} maps -- all three folders match")
EOF

# 2. the paper's split, as explicit lists (same rule as pipeline/loaders_prepared.py)
python3 - "$PREP/meta.csv" "$OUT" <<'EOF'
import os, sys
import numpy as np
import pandas as pd
meta = pd.read_csv(sys.argv[1])
meta["group"] = meta["patch"].astype(str) + "/" + meta["cluster"].astype(str)
groups = np.array(sorted(meta["group"].unique()))
np.random.RandomState(42).shuffle(groups)
n = len(groups); tr, va = int(round(0.70 * n)), int(round(0.85 * n))
sizes = []
for split, part in (("train", groups[:tr]), ("val", groups[tr:va]), ("test", groups[va:])):
    names = meta.loc[meta["group"].isin(set(part)), "name"]
    names.to_csv(os.path.join(sys.argv[2], f"split_{split}.txt"), index=False, header=False)
    sizes.append(len(names))
    print(f"{split}: {len(part)} groups, {len(names)} maps")
if len(meta) == 122021 and sizes != [81297, 25613, 15111]:
    sys.exit("split sizes differ from the paper -- stopping")
EOF

# 3. archives (PNGs are already compressed, so plain tar)
tar -chf "$OUT/cond.tar" -C "$PREP" cond
tar -chf "$OUT/target.tar" -C "$PREP" target
tar -chf "$OUT/target_rgb.tar" -C "$RGB" target_rgb
cp "$PREP/meta.csv" "$RGB/colormap_refs.npz" "$OUT/"

# 4. checksums
(cd "$OUT" && sha256sum cond.tar target.tar target_rgb.tar meta.csv colormap_refs.npz split_*.txt > SHA256SUMS.txt)
du -sh "$OUT"/* | sort -h
echo "done: $OUT"
