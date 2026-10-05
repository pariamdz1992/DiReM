# DiReM: Directional Real-deployment Multiband Radio Maps

DiReM is a radio-map dataset built from real Canadian base-station deployments. The transmitter
locations, carriers, powers and sector orientations come from ISED spectrum-licence site data.
The coverage is simulated with ATDI HTZ Communications, a deterministic propagation engine, using
four directional antenna patterns and three transmit-power settings.

DiReM is being built for eight Canadian cities: Toronto, Mississauga, Brampton, Markham, Ottawa,
Aylmer, Peterborough and Woodfibre. Collection is ongoing; the current release contains the
122,021 maps collected so far.

## Data

| record | content | link |
|---|---|---|
| DiReM v1 (frozen) | the 122,021 single-transmitter maps used for training and evaluation in the paper, with inputs, colour-mapped targets, metadata, the paper's split and EMF maps | [huggingface.co/datasets/Pariamdz/DiReM-v1](https://huggingface.co/datasets/Pariamdz/DiReM-v1) |
| DiReM (growing) | all maps collected up to the version date, updated as collection continues | *to come* |

Each sample has:
- an input `cond/<name>.png`, with three channels: basemap, transmitter position and antenna lobe;
- a target `target/<name>.png`, holding field strength in dBµV/m;
- a colour-mapped target `target_rgb/<name>.png`;
- a row in `meta.csv`.

See [docs/PROCESSING.md](docs/PROCESSING.md) for the exact definitions.

## This repository

| path | content |
|---|---|
| [`docs/GENERATION.md`](docs/GENERATION.md) | how the maps were generated: source records, clustering, HTZ settings, exported files |
| [`docs/PROCESSING.md`](docs/PROCESSING.md) | how the HTZ exports became the training maps |
| [`docs/DiReM_Supplementary_Material.pdf`](docs/DiReM_Supplementary_Material.pdf) | supplementary material of the paper: data and supervision, benchmark protocol, metric definitions, output representation and RadioBridge details |
| [`generation/process_and_modify_clusters.py`](generation/process_and_modify_clusters.py) | clusters each carrier CSV into 1 km groups and writes the per-cluster HTZ input files |
| [`generation/htz_automation.py`](generation/htz_automation.py) | drives HTZ to produce the exports |
| [`generation/antennas/`](generation/antennas/) | the four antenna pattern files (`Ant1`–`Ant4`, MSI format) |
| [`pipeline/`](pipeline/) | the processing scripts that built the training maps |
| [`release/`](release/) | the scripts that package and upload the data records |

## Citation

If you use DiReM, please cite:

> P. Mohammadzadeh Hesar, L. Chiaraviglio and H. Tabassum, "Learning-Based Radio Map
> Reconstruction: DiReM Dataset and RadioBridge Framework," under review, 2026.

The paper's supplementary material is in
[docs/DiReM_Supplementary_Material.pdf](docs/DiReM_Supplementary_Material.pdf).

## Sources and attribution

- **Transmitter records:** ISED *Terrestrial Spectrum Licence Site Data Extract* (Innovation,
  Science and Economic Development Canada).
- **Propagation:** HTZ Communications 2026.1 by ATDI. Terrain, clutter and building data come
  from ATDI's map database.
- **Basemap in the exported images:** map data © OpenStreetMap contributors, available under the
  Open Database License (ODbL).

## Licence

- **Data:** CC BY 4.0, with attribution notes in [LICENSE-DATA.md](LICENSE-DATA.md).
- **Code:** MIT ([LICENSE](LICENSE)).
