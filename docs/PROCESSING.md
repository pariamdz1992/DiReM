# How the training maps were made from the HTZ exports

Each HTZ export folder holds one *configuration*: one cluster of transmitters, simulated with one
antenna pattern and one power setting (see [GENERATION.md](GENERATION.md)). Processing turns every
transmitter of every configuration into one training sample:

```
export folder (patch / carrier / cluster / antenna / power)
  coverage.BMP + coverage.CSV  ──► colour-to-value table and pixel positions for this folder
  layout.BMP                   ──► input channel 0
  {p}_data.csv (one row per transmitter) ──► input channels 1–2 and metadata
  site####.BMP (one per transmitter)     ──► target (decoded values) and target_rgb (exported colours)
                                   │
                                   ▼
  cond/<name>.png   target/<name>.png   target_rgb/<name>.png   one row of meta.csv
```

`<name>` is `{patch}_{carrier}_{cluster}_{antenna}_{power}_site{NNNN}.png`, for example
`l33_f1_cluster_10_Ant1_p10_site0000.png`. The current release has 122,021 samples from 5,440
configurations.

## 1. Scripts

All scripts are in [`pipeline/`](../pipeline/), as used to build the release. They take the raw
and output folders as arguments.

| script | produces |
|---|---|
| `prepare_dataset.py` | `cond/`, `target/` and the first `meta.csv` |
| `fix_meta_power.py` | the final `meta.csv` (§6) |
| `prepare_rgb_dataset.py`, with `colormap_codec.py` and `regenerate_cond.py` | `target_rgb/` and `colormap_refs.npz` |
| `check_prepared.py`, `verify_rgb_prep.py` | quality-control reports |
| `loaders_prepared.py` | the train / validation / test split (§7) |

## 2. Which folders are used

`prepare_dataset.py` walks the raw store and takes every folder that holds a `coverage.BMP` and
`site*.BMP` files. It reads patch, carrier group, cluster, antenna and power from the folder
path. A folder is used when its antenna folder is `Ant1`–`Ant4` and its number of `site*.BMP`
files equals the number of rows in `{p}_data.csv`.

## 3. Colour-to-value table (per folder)

HTZ colours every export on a scale fitted to that folder's own range, so the same colour means
different values in different folders. Decoding with another folder's table gives a mean error of
48.66 units, against 2.76 with the folder's own table. Each folder is therefore decoded with its
own table:

1. Read `coverage.CSV`: longitude, latitude and field strength (dBµV/m) of every exported pixel.
2. Place each CSV point on `coverage.BMP` by scaling the CSV's longitude and latitude range onto
   the image width and height.
3. For every distinct colour at those pixels, store the mean field strength of the points that
   have it. The tables of all folders are saved in `colormap_refs.npz`.

Check: the pixelwise maximum of a folder's decoded `site*.BMP` files reproduces its decoded
coverage map to within 0.64 units on average. That is the expected result, because the coverage
map is the strongest transmitter at each pixel.

## 4. Target (`target/`)

- Each `site####.BMP` is an 8-bit palette image. Every palette colour is replaced by the value of
  the nearest colour (squared RGB distance) in the folder's table.
- The value map is resized to 256 × 256 with bilinear interpolation, rounded to 8 bits and saved
  as a single-channel PNG.
- **A pixel value is field strength in dBµV/m.**

## 5. Input (`cond/`)

Three channels. Each is built at the export's native resolution (755 pixels wide), clipped to
[0, 1], converted to 8 bits and resized to 256 × 256 with bilinear interpolation. The three are
saved together as one RGB PNG:

| channel | content |
|---|---|
| 0 | `layout.BMP` (the OpenStreetMap basemap) converted to luminance and divided by 255 |
| 1 | a Gaussian at the transmitter's pixel, σ = 6 native pixels (about 2 pixels at 256) |
| 2 | the antenna lobe: clip(P/100, 0, 1) · G(θ) · exp(−d / (0.35 · 755)) |

In channel 2:
- P is `tx_power`;
- G(θ) = 10^(−A(θ)/10), from the horizontal table of the antenna file;
- θ is the bearing from the transmitter minus its azimuth;
- d is the distance from the transmitter in native pixels.

The transmitter's pixel comes from its latitude and longitude, placed with the same scaling as in
§3. The release provides one `cond/` set, used with both `target/` and `target_rgb/`.

## 6. Colour-mapped target (`target_rgb/`) and metadata (`meta.csv`)

**`target_rgb/`**
- Each `site####.BMP` is converted to RGB and resized to 256 × 256 with nearest-neighbour
  resampling. The colours and the basemap stay as HTZ exported them.
- To score an RGB prediction, decode it with its folder's table from `colormap_refs.npz` and
  compare it with `target/`.

**`meta.csv`**, one row per sample:

| column | content |
|---|---|
| `name` | sample file name |
| `patch`, `freq_dir`, `cluster`, `ant`, `power_dir` | the configuration (folder path) |
| `frequency` | carrier (MHz) |
| `power` | `tx_power` as used in the simulation (dBm) |
| `azimuth` | antenna azimuth (degrees from north) |

`fix_meta_power.py` fills `power` from each folder's CSV and adds `power_dir`. It pairs the
`site*.BMP` files, sorted by number, with the CSV rows in order, and checks each pair on azimuth
and frequency. All 5,440 folders pass.

## 7. Split

`loaders_prepared.py` groups the samples by patch and cluster number (160 groups), sorts the
groups, shuffles them with `numpy.random.RandomState(42)` and splits them 70 / 15 / 15:

| split | groups | samples |
|---|---|---|
| train | 112 | 81,297 |
| validation | 24 | 25,613 |
| test | 24 | 15,111 |

The release lists the samples of each split explicitly.

## 8. Quality control

`check_prepared.py` checks that:
- `cond/`, `target/` and `meta.csv` match one to one;
- the metadata values are in range;
- each carrier group's target statistics are plausible.

It also flags near-constant targets and saves image montages for visual inspection.
`verify_rgb_prep.py` checks that the RGB set matches the grayscale set in names, metadata and
inputs.

## 9. EMF maps

EMF maps come from each configuration's `coverage.CSV`:
- The field strength E (dBµV/m) is placed on the export grid.
- E(V/m) = 10^(E/20) × 10⁻⁶.
- The same formula converts a single-transmitter map in `target/` to V/m.
