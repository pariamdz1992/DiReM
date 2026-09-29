#!/usr/bin/env python3
r"""
Colormap codec for the RGB track: value <-> RGB, and the analysis that says how
much a colormapped *representation* costs before any model is trained.

Why this file exists
--------------------
The grayscale track predicts the SCALAR signal (value 0..255, absolute, ~dBm+143.5)
because prepare_dataset.py inverts the jet colormap of the raw HTZ site BMPs.
The RGB track predicts the COLORMAPPED IMAGE instead. To put the two in the same
table, an RGB prediction must be decoded back to a value and scored in the exact
[0,1] value space the grayscale rows were scored in. That decode lives here, and
so does its inverse (used to build the RGB targets).

Two palettes are supported, and they answer different questions:

  "global" (recommended primary): ONE jet LUT for the whole dataset. The RGB
      target is the canonical grayscale target rendered through it, so the two
      tracks carry byte-identical signal content and differ ONLY in output
      representation -- which is exactly the thing being ablated. Needs no raw
      data: it is generated from data/prepared/all/target/ alone.

  "folder": each folder's own auto-scaled palette, recovered from
      coverage.BMP <-> coverage.CSV exactly as prepare_dataset.py::folder_setup
      does. This is the "as HTZ delivered it" variant. It needs the raw tree, and
      it adds a confound: the color->value mapping changes per folder, so the
      model is asked to predict a quantity whose encoding it cannot observe.

Self-test (no dataset needed):
    python3 colormap_codec.py --selftest
"""
from __future__ import annotations

import os
import numpy as np

__all__ = ["build_jet_lut", "encode_value", "decode_rgb", "decode_torch",
           "folder_refs", "refs_to_lut", "save_refs", "load_refs", "tag_of"]


# --------------------------------------------------------------------------- #
# global jet palette
# --------------------------------------------------------------------------- #
def build_jet_lut(n: int = 256) -> np.ndarray:
    """(n,3) uint8 jet LUT, computed analytically.

    Analytic on purpose: matplotlib is not in env_radiodiff / env_uvm, and a LUT
    that silently differs between the machine that WROTE the targets and the one
    that DECODES the predictions would corrupt every RGB number. This is the
    MATLAB-style jet; matplotlib's 'jet' uses slightly different breakpoints
    (--selftest prints the deviation, up to ~34/255 on a channel). That does NOT
    matter -- nothing here has to match matplotlib, only itself -- but never swap
    one for the other between prep and eval.
    """
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    r = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0, 1)
    return np.round(np.stack([r, g, b], 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# encode / decode
# --------------------------------------------------------------------------- #
def encode_value(val_u8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """value 0..255 (H,W) -> RGB (H,W,3) uint8 by LUT lookup."""
    idx = np.clip(np.round(np.asarray(val_u8)), 0, len(lut) - 1).astype(np.int64)
    return lut[idx]


def decode_rgb(rgb: np.ndarray, lut: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    """RGB (H,W,3) uint8/float -> value 0..255 float32, by NEAREST LUT color.

    Nearest-color is the only decode available for a model's output, because a
    prediction is not on-palette: blurred edges land between LUT entries. How
    badly that hurts is measured by --selftest, and it is the RGB track's
    intrinsic error floor.
    """
    a = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    L = lut.astype(np.float32)
    scale = 255.0 / (len(lut) - 1)
    out = np.empty(a.shape[0], dtype=np.float32)
    for s in range(0, a.shape[0], chunk):
        d = ((a[s:s + chunk, None, :] - L[None, :, :]) ** 2).sum(2)
        out[s:s + chunk] = d.argmin(1) * scale
    return out.reshape(np.asarray(rgb).shape[:2])


def decode_torch(pred, lut_t):
    """[B,3,H,W] in [0,1] -> [B,1,H,W] in [0,1]. `lut_t` is (K,3) float on the
    same device, in [0,1]. Used by the evaluator so RGB rows are scored in the
    grayscale rows' space."""
    import torch
    B, C, H, W = pred.shape
    assert C == 3, f"decode_torch expects 3 channels, got {C}"
    flat = pred.permute(0, 2, 3, 1).reshape(-1, 3)                     # N,3
    d = (flat * flat).sum(1, keepdim=True) - 2 * flat @ lut_t.t() + (lut_t * lut_t).sum(1)
    idx = d.argmin(1).to(torch.float32) / (lut_t.shape[0] - 1)
    return idx.reshape(B, H, W, 1).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- #
# per-folder palette (the "as delivered" variant)
# --------------------------------------------------------------------------- #
def folder_refs(folder: str) -> np.ndarray:
    """(K,4) float32 [R,G,B,value] for ONE raw HTZ folder.

    Verbatim in spirit from prepare_dataset.py::folder_setup -- the calibration
    that made the grayscale targets absolute. Kept here so the RGB track uses the
    SAME color<->value correspondence, never a second implementation.
    """
    from PIL import Image
    cov = np.asarray(Image.open(os.path.join(folder, "coverage.BMP")).convert("RGB"))
    Hc, Wc, _ = cov.shape
    # numpy, not pandas: this module is imported by the evaluator inside three
    # different venvs, and one of them has a pandas that does not load.
    import warnings
    with warnings.catch_warnings():
        # coverage.CSV has trailing/short lines; they are skipped, not fatal.
        warnings.simplefilter("ignore")
        arr = np.genfromtxt(os.path.join(folder, "coverage.CSV"), delimiter=";",
                            usecols=(0, 1, 2), dtype=np.float64, invalid_raise=False)
    arr = arr[np.isfinite(arr).all(1)]
    lon, lat, val = arr[:, 0], arr[:, 1], arr[:, 2]
    lon0, lon1, lat0, lat1 = lon.min(), lon.max(), lat.min(), lat.max()
    px = np.clip(((lon - lon0) / (lon1 - lon0) * (Wc - 1)).round().astype(int), 0, Wc - 1)
    py = np.clip(((lat1 - lat) / (lat1 - lat0) * (Hc - 1)).round().astype(int), 0, Hc - 1)
    pc = cov[py, px].astype(np.int64)
    key = pc[:, 0] * 65536 + pc[:, 1] * 256 + pc[:, 2]
    u, inv = np.unique(key, return_inverse=True)
    mv = np.bincount(inv, weights=val) / np.bincount(inv)
    return np.stack([u // 65536, (u // 256) % 256, u % 256, mv], 1).astype(np.float32)


def refs_to_lut(refs: np.ndarray, n: int = 256) -> np.ndarray:
    """(K,4) folder refs -> (n,3) uint8 LUT indexed by value, so encode/decode
    use one code path for both palettes. Values with no observed color take the
    nearest observed value's color."""
    v = refs[:, 3]
    grid = np.arange(n, dtype=np.float32) * (255.0 / (n - 1))
    j = np.abs(grid[:, None] - v[None, :]).argmin(1)
    return np.clip(refs[j, :3], 0, 255).astype(np.uint8)


def save_refs(path: str, refs_by_tag: dict) -> None:
    np.savez_compressed(path, **refs_by_tag)


def load_refs(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def tag_of(name: str) -> str:
    """'l62_f1_cluster_3_Ant2_p10_site0007.png' -> 'l62_f1_cluster_3_Ant2_p10'."""
    import re
    m = re.match(r"^(.*)_site\d+\.png$", os.path.basename(name))
    return m.group(1) if m else os.path.basename(name)


# --------------------------------------------------------------------------- #
# self-test: the representation's intrinsic cost, before any training
# --------------------------------------------------------------------------- #
def _selftest():
    lut = build_jet_lut(256)
    print(f"jet LUT {lut.shape}  first={lut[0].tolist()} mid={lut[128].tolist()} last={lut[-1].tolist()}")

    try:
        import matplotlib.cm as cm
        mpl = np.round(cm.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3] * 255).astype(int)
        print(f"vs matplotlib jet: max channel diff = {np.abs(mpl - lut.astype(int)).max()}")
    except Exception as e:                                    # noqa: BLE001
        print(f"(matplotlib not comparable here: {e})")

    # 1) injectivity: can a clean encode be decoded exactly?
    v = np.arange(256, dtype=np.float32)
    back = decode_rgb(encode_value(v[None, :], lut), lut)[0]
    dup = len(np.unique(lut.astype(np.int64) @ np.array([65536, 256, 1])))
    print(f"\n[1] round-trip on clean palette colors: max |err| = {np.abs(back - v).max():.3f} "
          f"value-units; distinct LUT colors {dup}/256")

    # 2) the real question: predictions are BLURRY. What does a blurred colormap
    #    decode to, compared with the same blur applied to the scalar field?
    rng = np.random.RandomState(0)
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.zeros((H, W), np.float32)
    for _ in range(6):                       # a few smooth "coverage lobes"
        cy, cx = rng.randint(0, H, 2)
        field += 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.randint(20, 60) ** 2))
    # real coverage maps are NOT smooth: buildings cast hard shadows, and a hard
    # edge is where a colormap hurts most. A purely smooth field flatters RGB.
    for _ in range(12):
        cy, cx = rng.randint(0, H - 40, 2)
        h, w = rng.randint(10, 40, 2)
        field[cy:cy + h, cx:cx + w] *= 0.35
    field = np.clip(field, 0, 255)

    def box_blur(a, k):
        pad = k // 2
        ap = np.pad(a, [(pad, pad)] * 2 + [(0, 0)] * (a.ndim - 2), mode="edge")
        c = np.cumsum(np.cumsum(ap, 0), 1)
        c = np.pad(c, [(1, 0), (1, 0)] + [(0, 0)] * (a.ndim - 2))
        s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        return s / (k * k)

    def nmse(p, g):
        return float(((p - g) ** 2).mean() / (g ** 2).mean())

    print("\n[2] blur robustness -- what a slightly-soft prediction costs:")
    print("    k   NMSE(scalar-blur)   NMSE(decode(rgb-blur))   ratio")
    rgb = encode_value(field, lut).astype(np.float32)
    for k in (3, 5, 9):
        gs = box_blur(field, k)
        rs = decode_rgb(box_blur(rgb, k), lut)
        a, b = nmse(gs, field), nmse(rs, field)
        print(f"    {k:<3} {a:>14.5f}   {b:>20.5f}   {b / max(a, 1e-12):>6.1f}x")

    # 3) additive noise: same perturbation magnitude, the two representations
    print("\n[3] noise robustness (sigma in 0..255 units, applied in each space):")
    print("    sigma  NMSE(scalar)   NMSE(decode(rgb))   ratio")
    for sg in (2, 5, 10):
        gs = field + rng.randn(H, W).astype(np.float32) * sg
        rs = decode_rgb(np.clip(rgb + rng.randn(H, W, 3).astype(np.float32) * sg, 0, 255), lut)
        a, b = nmse(gs, field), nmse(rs, field)
        print(f"    {sg:<6} {a:>11.5f}   {b:>16.5f}   {b / max(a, 1e-12):>6.1f}x")

    print("\nInterpretation: column 'ratio' is the penalty the RGB track pays for the\n"
          "SAME amount of prediction error, purely from the representation. It is the\n"
          "quantitative form of the 2026-07 decision 'RGB is just a colormap'.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.print_help()
