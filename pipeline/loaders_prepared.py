from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")


class RadioPrepared(Dataset):
    """
    Loader for the corrected, colormap-decoded HTZ pairs produced by
    prepare_dataset.py:

        <data_root>/cond/<name>.png     # 3ch: [buildings, true-Tx, .msi beam x power]
        <data_root>/target/<name>.png   # 1ch: absolute signal (colormap-decoded)
        <data_root>/meta.csv            # name, patch, freq_dir, frequency, power, cluster, ant, azimuth

    Output dict (RadioDiff contract):
        image     : (1,256,256) target, normalized to [-1,1]
        cond      : (3,256,256) condition, normalized to [-1,1]
        frequency : scalar (MHz)   -- for later frequency conditioning
        power     : scalar         -- (also already baked into cond Ch2)
        img_name  : file name

    Split is by group = (patch, cluster number); all antenna and power
    variants of a group stay in one split.
    Deterministic shuffle (seed=42), ~70/15/15.

    Args:
        data_root : the prepared dir (contains cond/, target/, meta.csv)
        phase     : "train" | "val" | "test" | "all"
        freq      : optional freq_dir filter, e.g. "f1" (None/"" = all bands)
        transform : ToTensor
    """

    def __init__(self, data_root, phase="train", freq=None,
                 transform=transforms.ToTensor()):
        self.data_root = data_root
        self.cond_dir = os.path.join(data_root, "cond")
        self.tgt_dir = os.path.join(data_root, "target")
        self.transform = transform
        self.norm3 = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.norm1 = transforms.Normalize(mean=[0.5], std=[0.5])

        meta = pd.read_csv(os.path.join(data_root, "meta.csv"))
        if freq:
            meta = meta[meta["freq_dir"].astype(str) == str(freq)].reset_index(drop=True)

        # ---- region-level split: (patch, cluster) ----
        meta["region"] = meta["patch"].astype(str) + "/" + meta["cluster"].astype(str)
        regions = np.array(sorted(meta["region"].unique()))
        rng = np.random.RandomState(42)
        rng.shuffle(regions)
        n = len(regions)
        tr, va = int(round(0.70 * n)), int(round(0.85 * n))
        split = {"train": set(regions[:tr]),
                 "val": set(regions[tr:va]),
                 "test": set(regions[va:])}
        if phase == "all":
            keep = meta
        else:
            keep = meta[meta["region"].isin(split[phase])]
        self.samples = keep.reset_index(drop=True)

        print(f"RadioPrepared [{phase}{'/' + freq if freq else ''}]: "
              f"{len(self.samples)} samples from {self.samples['region'].nunique()} regions "
              f"(of {n} total regions)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        name = row["name"]

        cond = np.asarray(io.imread(os.path.join(self.cond_dir, name)))[..., :3]   # HxWx3 uint8
        tgt = np.asarray(io.imread(os.path.join(self.tgt_dir, name)))              # HxW uint8
        tgt = np.expand_dims(tgt, 2).astype(np.float32) / 255.0                    # HxWx1 [0,1]

        if self.transform:
            cond = self.transform(cond).type(torch.float32)      # 3xHxW [0,1]
            tgt = self.transform(tgt).type(torch.float32)        # 1xHxW [0,1]

        out = {}
        out["image"] = self.norm1(tgt)                           # [-1,1]
        out["cond"] = self.norm3(cond)                           # [-1,1]
        out["frequency"] = torch.tensor(float(row.get("frequency", 0) or 0), dtype=torch.float32)
        out["power"] = torch.tensor(float(row.get("power", 0) or 0), dtype=torch.float32)
        out["img_name"] = name
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("--freq", default=None)
    args = ap.parse_args()
    for ph in ("train", "val", "test"):
        ds = RadioPrepared(args.data_root, phase=ph, freq=args.freq)
        if len(ds):
            s = ds[0]
            print(f"  {ph}: image {tuple(s['image'].shape)} "
                  f"[{s['image'].min():.2f},{s['image'].max():.2f}]  "
                  f"cond {tuple(s['cond'].shape)} "
                  f"[{s['cond'].min():.2f},{s['cond'].max():.2f}]  "
                  f"freq={s['frequency'].item():.0f} power={s['power'].item():.0f}  {s['img_name']}")
