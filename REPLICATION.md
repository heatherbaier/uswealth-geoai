# Replication guide

How to reproduce every finding/figure in this project, file by file. Update
this file whenever a script's inputs/outputs/config change, or a new
analysis is added — it should always describe what's actually in the repo
right now, not what was originally planned.

## The pipeline spans three repos

This repo (`uswealth-geoai`) only holds the *downstream analysis*. Getting
from nothing to a finished analysis output touches three repos in order:

1. **[`geoetl`](https://github.com/heatherbaier/geoetl)** — downloads Sentinel-2 imagery per census tract per
   quarter (via Microsoft Planetary Computer) and writes chips + label/coords
   JSON. Configs live at `configs/tlags/<state>/<state>_q<n>_<year>.yml`.
2. **[`sail`](https://github.com/heatherbaier/sail)** — trains the wealth-regression model (Swin Transformer) on
   those chips, validates it, and runs band-importance attribution. Run via
   `python launch.py --config <path>.yml`, with `task: train` /
   `validate` / `band_importance` in the config.
3. **This repo** — takes `sail`'s output CSVs (predictions, band importance)
   and land-cover data, and runs the actual statistical analysis for the
   paper.

A single config in `geoetl` + a matching pair of configs in `sail` (train +
validate/band_importance) produces the CSVs that everything in
`4_analysis/` consumes. **The `data_root`/`prefix` in a `sail` dataset
config must match the `output.root`/`dataset_name` in the `geoetl` config
that produced that imagery** — see `sail`'s own `configs/tlags/` for
examples and `sail`'s README/PR history for the config schema.

## Folder layout

```
1_data_prep/        tract shapefiles, wealth-index labels, land cover
2_imagery_legacy/    superseded imagery-conversion step (see note below)
3_validation/        per-quarter model accuracy, loaded straight from sail's prediction CSVs
4_analysis/          the actual paper analyses (seasonality, pixel diversity, band importance)
```

Numbering is pipeline order: run `1_data_prep` before you have anything to
validate, `3_validation` before you trust a checkpoint enough to analyze it
further, `4_analysis` last.

---

## 1_data_prep/

Getting tract geometries, wealth labels, and land-cover fractions ready.
None of these scripts talk to `geoetl`/`sail` — they're pure data prep, run
once per state.

### `download_acs_tracts.py`
Downloads Census TIGER/Line tract shapefiles for a given ACS 5-year period
(all 50 states + DC, or a subset). ACS 2015–2019 pairs with TIGER 2019.
```
python download_acs_tracts.py --acs-period 2015-2019 --output-dir ./tracts_2019
```
Output: one shapefile per state under `--output-dir`.

### `clean_wealth_index_{az,ca,ga,pa}.ipynb`
Joins a state's tract shapefile to `final_wealth_indices_2019.csv` (the
ACS-derived wealth index — not produced by anything in this repo; expected
to already exist at the path each notebook's second cell points to), and
writes:
- `<state>_wi2019.csv` — GEOID + `wealth_index_overall_core`
- `<tract_shapefile>_wi.shp` — the tract shapefile with wealth index attached

This is what `geoetl`'s AZ/CA/GA/PA configs point `aoi.path` at
(`tl_2019_<fips>_tract_wi.shp`), and what `params.label_column: wealth_ind`
in those configs reads from. **There is no equivalent notebook for OH in
this repo** — OH's label join was apparently done before this repo's
history starts (OH's `geoetl` configs point at
`tl_2019_39_tract_proj.shp`, a plain-projected shapefile without a `_wi`
suffix, and its `label_column: NAME` is a placeholder, not a real wealth
label — worth double-checking OH's actual label source before writing up
OH results, since it isn't reproducible from anything currently in this
repo).

To do PA/CA/GA end to end, run `download_acs_tracts.py` first, then the
matching notebook here.

### `compute_landcover_oh.py`
**OH-specific and incomplete as a pipeline — read this before using it.**
Downloads ESA WorldCover 10m tiles overlapping Ohio, clips to each tract,
and counts pixels per WorldCover class. This is the "get land cover for
OH" script referenced elsewhere in this doc.

What it actually produces: `./test.json` (yes, that's the literal
filename in the script — rename it if you run this again), mapping
`GEOID -> {ESA class code: pixel count}` using ESA's *numeric* class codes
(10=Tree cover, 30=Grassland, 40=Cropland, 50=Built-up, etc.).

**What's missing:** `4_analysis/analyze_seasonality_lag.py` and
`4_analysis/analyze_band_importance_spatial.py` both expect a `lc.csv`
with *named* fraction columns (`Cropland`, `Built-up`, ...), not this raw
numeric-class-code JSON. There is currently no script in this repo that
does that conversion (remap ESA codes to names, normalize counts to
fractions, write to CSV). If you're re-deriving land cover for a new
state, you'll need to write that conversion step, or ask for it — it's a
short script (rename `test.json`'s class-code keys via the ESA WorldCover
legend, divide by row totals, pivot to one column per class), just not
built yet.

Hardcoded to `./data/tracts2019/OH/tl_2019_39_tract_proj.shp` — copy and
retarget for another state rather than editing in place, or generalize it
to take `--state`/`--shapefile` args.

---

## 2_imagery_legacy/

### `convert_tifs_to_pngs.py`
**Superseded — kept for reference only, not part of the current pipeline.**
Converts 6-band GeoTIFF chips to 3-band PNGs via a *per-image* percentile
contrast stretch. This predates `geoetl`'s current MPC pipeline (which
writes chips directly, with a fixed dataset-wide scale rather than a
per-image stretch — see `geoetl/io/base.py`'s `write_chip()`) and predates
`sail`'s native multiband-TIFF dataloader support (`sail` PR #1, merged),
which reads `geoetl`'s `.tif` chips directly with no PNG step at all. Only
relevant if you're working with imagery chips produced by the *old*
pipeline (e.g. some of the original OH quarters, before the AZ/geoetl/sail
rework this session did). Don't use this for new states.

---

## 3_validation/

Sanity-checking a trained checkpoint before trusting it enough to run the
`4_analysis/` scripts against it. Both notebooks just load `sail`'s
per-quarter prediction CSVs (`epoch<N>_valset_preds.csv` /
`epoch<N>_full_preds.csv`, written by `sail`'s `task: validate` — see
`sail/src/sail/engine.py::run_validation`) and compute R², mean signed
error (bias), and error std per quarter.

### `validation_oh.ipynb`
The original, for OH's (pre-AZ-rework) trained models. Loads 15 quarters
(q2_2016 .. q4_2019) from hardcoded `/data/hbaier/new_data/tlag/imagery/...`
paths.

### `validation_az.ipynb`
**Work in progress — not a finished AZ replication yet.** Only the first
cell was updated to point at the AZ q1_2016 all-bands checkpoint
(`.../az_imagery/q1_2016_s2_allbands/artifacts/az_q1_2016_allbands_v1/epoch174_valset_preds.csv`).
Every cell after that still has the *original OH* paths copy-pasted from
`validation_oh.ipynb`, unedited — running this notebook as-is would mix
one AZ quarter with several OH quarters into the same `stats`/`error`/`std`
lists and plot them together as if they were a single continuous series.
Fix the remaining cells' paths to AZ's other quarters (once trained)
before trusting this notebook's plots.

---

## 4_analysis/

The actual paper analyses. Every script here is standalone (no imports
between them) with a `<<< change me` config block near the top — fill
those in, then `python <script>.py`. Each one writes its outputs to its own
`OUT_DIR`.

### Thread: pixel diversity vs. accuracy
Theory: non-growing-season imagery is spectrally flatter (less pixel-to-
pixel variation), and that flatness costs prediction accuracy —
especially in rural/agricultural tracts.

1. **`compute_pixel_diversity.py`** — run once per quarter, against that
   quarter's *raw* `.tif` chip directory (not PNGs — see the script's
   docstring for why per-image PNG stretching would erase the exact
   signal this measures). Outputs one diversity-metrics CSV per quarter.
   ```
   python compute_pixel_diversity.py --chip-dir <geoetl output.root>/chips \
       --sensor sentinel2 --out ./diversity/q1_2016.csv
   ```
2. **`analyze_pixel_diversity_accuracy.py`** — joins each quarter's
   diversity CSV to that quarter's `sail` prediction CSV (on chip
   filename), then tests: is diversity itself seasonal (H1a); does
   diversity predict per-chip accuracy, both magnitude and signed bias,
   including within a single quarter (H1b); is the effect stronger in
   agricultural tracts (H1c, needs `lc.csv` — see the gap noted under
   `compute_landcover_oh.py` above). Fill in `QUARTERS` or set
   `IMAGERY_ROOT` to auto-discover.

### Thread: seasonal accuracy trends by landscape
### `analyze_seasonality_lag.py`
Takes a wide-format `preds.csv` (`GEOID`, `q<n>_<year>_pred`,
`q<n>_<year>_error` columns) + `lc.csv` (land cover fractions), and tests:
per-quarter R²/RMSE by landscape type; per-quarter *signed bias* by
landscape (does the model systematically over/under-predict wealth in a
given season, and does that differ by urban/rural); seasonal-amplitude and
temporal-slope comparisons; mixed models on both accuracy and bias with
quarter×landscape and time×landscape interactions.

**Known gap:** nothing in this repo currently assembles that wide
`preds.csv` from `sail`'s per-quarter long-format prediction CSVs (the
`epoch<N>_..._preds.csv` files `3_validation/` reads directly) — you'd
need to pivot GEOID×quarter yourself, or ask for that assembly script.
`lc.csv` has the same gap noted above under `compute_landcover_oh.py`.

### Thread: seasonal band contribution
1. In `sail`, run `task: band_importance` per quarter's checkpoint (see
   `sail/src/sail/explain/band_importance.py`) with `band_importance.groups`
   and `band_importance.per_image: true` set, to get all three outputs:
   `band_importance.csv` (per-band, dataset-wide), `band_importance_grouped.csv`
   (per named band-cluster, e.g. visible/vegetation/swir — see that file's
   docstring for why grouped importance can differ a lot from the sum of
   individual bands), and `band_importance_per_image.csv` (per-tract, via
   ablation-to-mean rather than permutation — see the same file's docstring
   for why per-tract needs a different mechanism than the dataset-wide test).
2. **`analyze_band_importance_seasonality.py`** — pools `band_importance.csv`
   across quarters, tests whether each band's importance is itself seasonal
   (growing vs. non-growing) and plots it by quarter. Needs multiple
   quarters trained to say anything (one quarter has no season comparison
   to make).
3. **`analyze_band_importance_spatial.py`** — takes one quarter's
   `band_importance_per_image.csv` + `lc.csv`, tests whether a band's
   per-tract importance differs by landscape (ag vs. urban), and computes
   Moran's I spatial autocorrelation (does a band matter in geographic
   clusters, or is its importance scattered randomly) with a permutation
   significance test.
4. `az_band_importance.ipynb` — scratch notebook, currently just loads and
   sorts `band_importance.csv`. Not a real analysis yet.

**AZ-specific caveat that affects H1a/H2's "growing season" tests:** all
three scripts currently assume `GROWING_QUARTERS = {2, 3}` (Ohio corn-belt
phenology). This is very likely wrong for AZ — irrigated winter cropping
and desert winter-rain green-up don't follow the corn-belt calendar. A
wrong split would hide a real seasonal effect, not fake one. Get real AZ
NDVI/cropping-calendar data before trusting seasonal-split results for AZ.

---

## Open gaps, as of this writing

- No script converts `compute_landcover_oh.py`'s raw ESA class-code JSON
  into the named-fraction `lc.csv` the analysis scripts need.
- No script assembles a wide-format `preds.csv` from `sail`'s per-quarter
  prediction CSVs for `analyze_seasonality_lag.py`.
- `validation_az.ipynb` is a half-finished copy of `validation_oh.ipynb`.
- `GROWING_QUARTERS`/`NON_GROWING_QUARTERS` in the band-importance and
  pixel-diversity scripts need real AZ phenology data, not the Ohio
  assumption they currently default to.
- OH's wealth-index label provenance isn't reproducible from anything in
  this repo (see the note under `clean_wealth_index_*.ipynb`).
