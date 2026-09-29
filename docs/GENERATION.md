# How DiReM was generated

DiReM's radio maps are simulations of real deployments. The transmitters come from Canada's
spectrum-licence site data, and the propagation is computed with ATDI's HTZ Communications. Two
scripts prepare the inputs and drive HTZ; both are in [`generation/`](../generation/).

```
ISED Terrestrial Spectrum Licence Site Data (all of Canada)
  └── selected areas → patches l{NN} → one CSV per carrier group (f1–f5, fdef)
        └── generation/process_and_modify_clusters.py (one run per carrier CSV)
              l{NN}/f{1..5|def}/cluster_{k}/Ant{1..4}/{p10|p50|pdef}/{p}_data.csv
                └── HTZ Communications, driven by generation/htz_automation.py (one run per CSV)
                      coverage.CSV  coverage.BMP  layout.BMP  object.BMP  site####.BMP
```

## 1. Source records

- **Dataset:** ISED's *Terrestrial Spectrum Licence Site Data Extract* (`Site_Data_Extract_FX.zip`),
  from ISED's
  [Download SMS data](https://ised-isde.canada.ca/site/spectrum-management-system/en/spectrum-management-system-data/download-sms-data)
  page, which is reached from
  [Canada.ca: broadcasting and telecommunications regulation](https://www.canada.ca/en/services/business/permits/federallyregulatedindustrysectors/broadcastingtelecommunicationsregulation.html).
- **Content:** the technical data that spectrum licensees submit for each site. Each row is one
  transmitter. The extract was downloaded in 2025.
- **Columns:** the extract already uses the column names shown in §2.
- **Field definitions**, from ISED's
  [Text Format for Spectrum Licence Site Data Upload](https://ised-isde.canada.ca/site/spectrum-management-system/en/spectrum-management-system-data/text-format-spectrum-licence-site-data-upload):

| column | meaning | unit |
|---|---|---|
| `tx_power` | transmitter TCP-TRP (total conducted or radiated power) | dBm |
| `structure_height` | height of the structure the antennas are mounted on, above ground | m |
| `tx_ant_height` | height of the antenna centre above ground, including the structure or building it is mounted on | m |
| `tx_ant_azimuth` | direction of maximum radiation from true north | ° |
| `tx_ant_gain` | gain relative to isotropic | dBi |
| `bandwidth` | occupied bandwidth | kHz |
| `downlink_allocation` | share of time transmitting | % |
| `site_type` | U underground, O outdoor, I indoor | — |

## 2. From records to folders

1. Selected areas of the national extract were split into regions, and each region was cut into
   patches (`l##`) of about 10 km × 10 km.
2. Within a patch, transmitters are grouped by carrier, one CSV per group:
   - `f1`–`f5`: the five most common carriers, one carrier each;
   - `fdef`: every transmitter on its own carrier.

   These files keep the power from the record.
3. [`process_and_modify_clusters.py`](../generation/process_and_modify_clusters.py) clusters each
   carrier CSV:
   - one point per `location`, at the mean latitude and longitude of its rows;
   - haversine distances between points, then complete-linkage hierarchical clustering cut at
     1.0 km, so any two sites in a cluster are at most 1 km apart;
   - cluster numbers come from `scipy`'s `fcluster`, run on that file.
4. For each cluster, the script writes `cluster_{k}/Ant{1..4}/{p10|p50|pdef}/{p}_data.csv`, plus a
   copy without the header row (`{p}_data_noheader.csv`):
   - `p10` and `p50` set `tx_power` to 10 and 50;
   - `pdef` keeps the recorded value;
   - all four antenna folders get the same CSV, and the antenna pattern is assigned in HTZ (§4).

The same script prepares the columns for HTZ's station import. It adds `height` =
`structure_height` + `tx_ant_height` (the antenna height HTZ uses), writes `bandwidth` twice, and
drops `structure_type`, `tx_ant_manufacturer` and `rx_ant_manufacturer`.

| level | values | meaning |
|---|---|---|
| patch | `l33`, `l61`–`l65`, `l73`–`l78` | geographic area |
| carrier group | `f1`–`f5`, `fdef` | f1 = 1952.5, f2 = 1967.5, f3 = 2115.0, f4 = 2135.0 and f5 = 2142.5 MHz in almost every patch (in `l78`, f5 = 2605.0); `fdef` also contains 2605–2685 MHz |
| cluster | `cluster_{k}` | sites at most 1 km apart, numbered separately in each carrier group; 1 to 266 transmitters |
| antenna | `Ant1`–`Ant4` | one antenna pattern file applied to every transmitter of the cluster (below) |
| power | `p10`, `p50`, `pdef` | 10 dBm, 50 dBm, or the recorded `tx_power` |

**Cities.** DiReM is being built for eight Canadian cities: Toronto, Mississauga, Brampton and
Markham (Ontario); Ottawa (Ontario); Aylmer (Quebec); Peterborough (Ontario); and Woodfibre
(British Columbia). Each city has its own map data (§5). Collection is ongoing. The current release
contains the 122,021 maps collected so far, from 5,440 configurations (patch × carrier group ×
cluster × antenna × power).

**Example carrier CSV** (one patch, 2135.0 MHz, `pdef`):
- 54 transmitters at 16 sites, 3–4 sectors per site, each sector with its own azimuth;
- all HSPA, 5 MHz bandwidth, `tx_power` 40.0;
- the patch spans about 8.0 km × 8.6 km, and its sites are 1.1–4.2 km apart.

**Cluster CSV.** One row per transmitter, in HTZ's station-import format:

```
area_name*, location, station_type, technology, latitude, longitude, site_type, structure_height,
height, tx_frequency, rx_frequency, tx_power, bandwidth, bandwidth, downlink_allocation,
tx_number_antennas, rx_number_antennas, tx_ant_height, rx_ant_height, tx_ant_omni_indicator,
rx_ant_omni_indicator, tx_ant_horiz_beamwidth, rx_ant_horiz_beamwidth, tx_ant_vert_beamwidth,
rx_ant_vert_beamwidth, tx_ant_azimuth, rx_ant_azimuth, tx_ant_elevation_angle,
rx_ant_elevation_angle, tx_ant_gain, rx_ant_gain, tx_line_loss, rx_line_loss
```

**Power.** HTZ reads `tx_power` in dBm: a row with `tx_power` 10 is imported with a nominal power
of 0.01 W.

**Antenna height.** HTZ uses the `height` column as the antenna height above ground. For example,
a row with `structure_height` 2 and `tx_ant_height` 1.0 is imported at 3.00 m.

**Antenna pattern and gain.**
- Each station takes its pattern and gain from the antenna file assigned in §4.
- Azimuth and tilt come from `tx_ant_azimuth` and `tx_ant_elevation_angle`.
- Polarization is vertical, and line loss comes from `tx_line_loss`.
- EIRP = power + antenna gain − line loss. For example, 10 dBm + 25 dBi − 1 dB = 34 dBm (2.51 W).

The four files are in [`generation/antennas/`](../generation/antennas/). Beamwidths and
front-to-back ratios are computed from each file's pattern table.

| folder | antenna (file header) | band in file | gain | horizontal −3 dB | vertical −3 dB | front-to-back |
|---|---|---|---|---|---|---|
| `Ant1` | B5, Airspan | 5.15–5.87 GHz | 25 dBi | ≈12° | ≈12° | 13.6 dB |
| `Ant2` | N5-45x2 sector, Airspan | 4.9–6.4 GHz | 19 dBi | ≈41° | ≈8° | 39.6 dB |
| `Ant3` | R2-5, two-element | 160 MHz | 5 dBi | ≈120° | ≈72° | 16.9 dB |
| `Ant4` | ePMP sector 90° | 5 GHz | 20 dBd | ≈90° | ≈6° | 39.3 dB |

The four files give four distinct pattern shapes and gains.

## 3. Tile (study area)

`calculate_bounding_rectangle()` in
[`htz_automation.py`](../generation/htz_automation.py) sets each cluster's tile:

1. Take the cluster's unique `location`s and their bounding box.
2. Build a square of 1.0 km plus a 50 m margin on each side (1.1 km), centred on that box.
   Degrees are converted with 1° latitude = 111.0 km and 1° longitude = 111.0 · cos(latitude) km.
3. If any two locations are more than 0.935 km apart (0.85 × 1.1 km), the side becomes
   1.2 × their largest separation.
4. Enter the four bounds into HTZ's rectangle tool, to 6 decimals.

The rectangle is entered once per cluster and reused for all of its antenna and power variants.
Tiles measure about 1.1 km on each side; the median extent is 1.15 km × 1.10 km.

## 4. One HTZ run (per folder)

`run_all_sequences()` in `htz_automation.py` drives HTZ's interface. For each folder:

| # | step | result |
|---|---|---|
| 2 | File menu → open the city's template project (for Toronto, `toronto.PRO`) | the map data and calculation settings of §5 |
| 1 | *Layer* toolbar → select the map layer "OSM topo" | the OpenStreetMap basemap |
| 3 | *Rectangle* toolbar → enter the tile bounds (§3) | study area |
| 4 | File menu → import the folder's station CSV | one station per row |
| 5 | *List* toolbar → assign the antenna file (`Ant1`–`Ant4`) to all stations | pattern and gain |
| 6 | *Tools* menu → a fixed sequence of settings | — |
| 7 | *Coverage* menu → coverage parameters (§5) → start the calculation | computed coverage |
| 8 | File menu → export the coverage values as `coverage` | `coverage.CSV` |
| 9 | File menu → export image, basemap only, as `layout` | `layout.BMP` |
| 10 | same dialog, as `coverage` | `coverage.BMP` |
| 11 | same dialog, one image per station, as `site` | `site####.BMP`, one per transmitter |
| 12 | same dialog, base-station markers only, as `object` | `object.BMP` |
| — | File menu → close the project | — |

The script repeats this for every (cluster, antenna, power) folder listed in its `main()`. The
copy in `generation/` holds the settings of one run; the full lists are in its comments.

## 5. HTZ settings

**Coverage parameters**

| setting | value |
|---|---|
| receiver antenna height | 2.00 m |
| distance | 1.000 km |
| minimum stored value | 1 dBµV/m |
| wanted threshold | 35 dBµV/m |

**Propagation model**: HTZ's deterministic model (about 30 MHz to 1 THz).

| part | setting |
|---|---|
| loss | free-space loss + min(diffraction, troposcatter, ducting, reflections, absorption) |
| diffraction | Deygout 94 |
| sub-path attenuation | coarse integration, Fresnel-zone factor 1.00 |
| reflections | off (no 3D multipath, no ground reflections) |
| troposcatter, ducting, gases, fog, rain | off |
| Earth radius | 8500 km (k = 4/3) |
| location / time variability | 50% / 50% |

**Clutter parameters**

| code | class | height (m) | code | class | height (m) |
|---|---|---|---|---|---|
| 0 | open | 0 | 10 | rail | 0 |
| 1 | village | 6 | 11 | road | 0 |
| 2 | suburban | 10 | 12 | airport | 0 |
| 3 | urban | 15 | 13 | tunnel | 0 |
| 4 | dense urban | 20 | 14 | rural | 2 |
| 5 | forest | 12 | 15 | building, plaster | from building data |
| 6 | hydro | 0 | 16 | building, brick | from building data |
| 7 | high urban | 35 | 17 | building, glass | from building data |
| 8 | park/wood | 4 | 18 | building, wood | from building data |
| 9 | building | from building data | 19 | border | 0 |

- **Attenuation:** none (0 dB/km for every class). Clutter acts through its height, as an obstacle
  for diffraction.
- **Per class:** surface and diffraction factors 1.0, standard deviation 5.5 dB, height factor 1.0.
- **Placement:** receivers and transmitters are placed over the ground.
- **Building entry loss:** not applied. Reference frequency 2143 MHz.

**Station defaults:** coverage threshold 35 dBµV/m, receiver threshold 30 dBµV/m, KTBF −97 dBm.
The exports contain the pixels at or above the 35 dBµV/m threshold.

**Software:** HTZ Communications 2026.1 (release 1510, 27 January 2026).

**Map layer:** base map "Map"; map type "OSM topo" (OpenStreetMap); rendering "Map color",
"Maximized map" and "Vectors".

**Map data.**
- Each city's terrain, clutter and building layers come from ATDI's map database, downloaded with
  HTZ's Map Download Manager.
- Each city has one template project and one set of map files. For Toronto these are
  `toronto.PRO`, `toronto.RGE` (terrain), `toronto.RSO` (clutter), `toronto.RBL` (buildings) and
  `toronto.BIM` (image).
- For Toronto, terrain, clutter and buildings are rasters with a 2 m × 2 m step in NAD83 / UTM
  zone 17N. Results are written in decimal degrees (WGS 84), and the basemap image uses Web
  Mercator.

## 6. Output files per folder

| file | content |
|---|---|
| `coverage.CSV` | one line per grid point (about 2 m) at or above 35 dBµV/m: `lon;lat;E_dBuV/m;P_dBm;E_V/m`, e.g. `-79.538575;43.609299;84;-59.8;0.0158`. The V/m column is the EMF layer. |
| `coverage.BMP` | all transmitters combined (the strongest one at each pixel), colour-mapped over the basemap; 8-bit palette, 755 px wide |
| `site####.BMP` | the same map for one transmitter, numbered by HTZ |
| `layout.BMP` | the OpenStreetMap basemap only |
| `object.BMP` | the base-station markers |
| `{p}_data.csv`, `{p}_data_noheader.csv` | the input transmitters (§2) |

**Units.** Field strength (dBµV/m) = received power (dBm) + 77.2 + 20·log₁₀(f / MHz), checked on
72 folders across 12 patches. In `fdef` folders, the conversion uses one carrier for the whole
folder.

**Colour maps.** Each BMP's colour scale is fitted to its folder's own range, so each folder is
decoded separately ([PROCESSING.md](PROCESSING.md)).
