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
once per state. As of this rewrite, every script here (except OH's
already-done label join, which nothing in this repo reproduces) is driven
by `--state` — no more per-state copy-pasted notebooks.

### `download_acs_tracts.py`
Downloads Census TIGER/Line tract shapefiles for a given ACS 5-year period
(all 50 states + DC, or a subset). ACS 2015–2019 pairs with TIGER 2019.
```
python download_acs_tracts.py --acs-period 2015-2019 --output-dir ./data/tracts2019
```
Output: one shapefile per state under `--output-dir` (`<output-dir>/<STATE>/tl_2019_<fips>_tract.shp`).

### `clean_wealth_index.py`
Joins a state's tract shapefile to `final_wealth_indices_2019.csv` (the
ACS-derived wealth index — not produced by anything in this repo; expected
to already exist at the path `--wealth-csv` points to), and writes:
- `<state>_wi2019.csv` — GEOID + `wealth_index_overall_core`
- `tl_2019_<fips>_tract_wi.shp` — the tract shapefile with wealth index attached
```
python clean_wealth_index.py --state az
python clean_wealth_index.py --state ca --tracts-dir ./data/tracts2019 \
    --wealth-csv ./final_wealth_indices_2019.csv
```
This is what `geoetl`'s AZ/CA/GA/PA configs point `aoi.path` at
(`tl_2019_<fips>_tract_wi.shp`), and what `params.label_column: wealth_ind`
in those configs reads from.

Replaces the four near-identical `clean_wealth_index_{az,ca,ga,pa}.ipynb`
notebooks this repo used to have. While merging them into this script, one
of the four (PA) turned out to have a real bug: it wrote its output to
`tl_2019_42_tract.shp` (no `_wi` suffix), silently overwriting the plain
source shapefile instead of writing a separate wealth-index-joined file
like the other three notebooks did. This script always writes the `_wi`
suffix — if PA's original shapefile was already overwritten by that bug,
re-run `download_acs_tracts.py` for PA before trusting its `_wi.shp`.

`--state oh` is rejected on purpose. **There is no equivalent for OH** —
OH's label join was apparently done before this repo's history starts
(OH's `geoetl` configs point at `tl_2019_39_tract_proj.shp`, a
plain-projected shapefile without a `_wi` suffix, and its
`label_column: NAME` is a placeholder, not a real wealth label — worth
double-checking OH's actual label source before writing up OH results,
since it isn't reproducible from anything currently in this repo).

To do a new state end to end: `download_acs_tracts.py` first, then this.

### `compute_landcover.py`
Downloads ESA WorldCover 10m tiles overlapping a state's tracts, clips to
each tract, and writes a ready-to-use `lc.csv` — named fraction columns
(`Cropland`, `Built-up`, `Tree cover`, ...), one row per GEOID.
```
python compute_landcover.py --state oh
python compute_landcover.py --state az --tract-shp ./data/tracts2019/AZ/tl_2019_04_tract_wi.shp
```
Output: `./lc_<state>.csv` (the `lc.csv` that `analyze_seasonality_lag.py`,
`analyze_pixel_diversity_accuracy.py` (H1c), and
`analyze_band_importance_spatial.py` all take as input), plus the raw
per-class pixel counts at `<out-dir>/raw_counts_<state>.json` for
debugging.

Replaces the OH-only `compute_landcover_oh.py`/`lc.py`, which (a) was
hardcoded to OH's shapefile and (b) only ever opened *one* WorldCover
tile after downloading all of them that intersected the AOI — silently
wrong/incomplete for any state whose tracts span more than one 3°x3°
WorldCover tile, which is most states larger than Ohio (AZ/CA/PA
especially). This version clips each tract against every tile it actually
intersects and sums counts across tiles. It also does the ESA
class-code → named-fraction conversion directly, which used to be a
missing manual step (the old script only wrote raw numeric-class-code
counts to a file literally named `./test.json`).

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
`4_analysis/` scripts against it.

### `validate.py`
Auto-discovers every quarter under a state's imagery root that has a
completed `sail` `task: validate` run (`epoch<N>_preds.csv` /
`epoch<N>_valset_preds.csv` / `epoch<N>_full_preds.csv`, written by
`sail/src/sail/engine.py::run_validation`), and computes R², mean signed
error (bias), and error std per quarter.
```
python validate.py --state oh --imagery-root /data/hbaier/new_data/tlag/imagery
python validate.py --state az --imagery-root /data/hbaier/new_data/tlag/az_imagery
```
Output: `validation_summary_<state>.csv` plus `r2_by_quarter.png` /
`bias_by_quarter.png` / `std_by_quarter.png` under `--out-dir` (default
`./out_validation/<state>`).

Replaces `validation_oh.ipynb` and `validation_az.ipynb` — the same
~35-cell notebook copy-pasted per state (and, for AZ, left half-finished
mid-copy: only its first cell had been updated to an AZ path, the rest
still pointed at OH's). Also fixes a bug those notebooks had: their x-axis
tick labels were a hardcoded 15-entry list assuming every state has
exactly q2_2016..q4_2019 trained, which silently produces mislabeled or
truncated plots for any state with a different quarter range (i.e. every
state except OH, today). This script's tick labels are built from
whatever quarters it actually finds.

---

## 4_analysis/

The actual paper analyses. Every script here is standalone (no imports
between them) and driven entirely by CLI arguments (`--state` plus
whatever paths that analysis needs) — `python <script>.py --help` for the
full list. Each one writes its outputs to `--out-dir` (default: a
per-state subfolder under `./out_<analysis_name>/`), so running the same
script for a second state never overwrites the first state's output.

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
2. **`analyze_pixel_diversity_accuracy.py`** — auto-discovers every
   quarter under `--imagery-root`, joins each one's diversity CSV to that
   quarter's `sail` prediction CSV (on chip filename), then tests: is
   diversity itself seasonal (H1a); does diversity predict per-chip
   accuracy, both magnitude and signed bias, including within a single
   quarter (H1b); is the effect stronger in agricultural tracts (H1c,
   needs `--lc-csv` from `compute_landcover.py`).
   ```
   python analyze_pixel_diversity_accuracy.py --state az \
       --imagery-root /data/hbaier/new_data/tlag/az_imagery \
       --diversity-dir ./diversity --lc-csv ./lc_az.csv \
       --growing-quarters 2,3
   ```

### Thread: seasonal accuracy trends by landscape
1. **`assemble_preds_wide.py`** — auto-discovers every quarter under
   `--imagery-root` and pivots `sail`'s per-quarter long-format prediction
   CSVs (GEOID from the chip filename, same convention as everywhere
   else in this doc) into the wide-format `preds.csv`
   (`GEOID`, `q<n>_<year>_pred`, `q<n>_<year>_error`) the next script needs.
   ```
   python assemble_preds_wide.py --state az \
       --imagery-root /data/hbaier/new_data/tlag/az_imagery \
       --out ./preds_az.csv
   ```
2. **`analyze_seasonality_lag.py`** — takes that wide `preds.csv` +
   `lc.csv` (land cover fractions), and tests: per-quarter R²/RMSE by
   landscape type; per-quarter *signed bias* by landscape (does the model
   systematically over/under-predict wealth in a given season, and does
   that differ by urban/rural); seasonal-amplitude and temporal-slope
   comparisons; mixed models on both accuracy and bias with
   quarter×landscape and time×landscape interactions.
   ```
   python analyze_seasonality_lag.py --state az \
       --preds-csv ./preds_az.csv --lc-csv ./lc_az.csv
   ```

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
2. **`analyze_band_importance_seasonality.py`** — auto-discovers
   `band_importance.csv` per quarter under `--artifacts-root`, pools them,
   tests whether each band's importance is itself seasonal (growing vs.
   non-growing) and plots it by quarter. Needs multiple quarters trained
   to say anything (one quarter has no season comparison to make).
   ```
   python analyze_band_importance_seasonality.py --state az \
       --artifacts-root /data/hbaier/new_data/tlag/az_imagery --growing-quarters 2,3
   ```
3. **`analyze_band_importance_spatial.py`** — takes one quarter's
   `band_importance_per_image.csv` + `lc.csv`, tests whether a band's
   per-tract importance differs by landscape (ag vs. urban), and computes
   Moran's I spatial autocorrelation (does a band matter in geographic
   clusters, or is its importance scattered randomly) with a permutation
   significance test.
   ```
   python analyze_band_importance_spatial.py --state az \
       --per-image-csv .../band_importance_per_image.csv --lc-csv ./lc_az.csv
   ```

**AZ-specific caveat that affects H1a/H2's "growing season" tests:** all
three scripts default `--growing-quarters` to `2,3` (Ohio corn-belt
phenology). This is very likely wrong for AZ — irrigated winter cropping
and desert winter-rain green-up don't follow the corn-belt calendar. A
wrong split would hide a real seasonal effect, not fake one. Pass the real
AZ growing quarters via `--growing-quarters` once real
NDVI/cropping-calendar data is available — don't trust the default for AZ.

---

## Open gaps, as of this writing

- `GROWING_QUARTERS`/`--growing-quarters` in the band-importance and
  pixel-diversity scripts still *default* to the Ohio assumption
  (`2,3`) — the flag exists now, but nobody has plugged in real AZ (or
  CA/GA/PA) phenology data yet, so every run for those states needs that
  flag passed explicitly once that data exists.
- OH's wealth-index label provenance isn't reproducible from anything in
  this repo (see the note under `clean_wealth_index.py`).
- CA/GA/PA don't have imagery downloaded/trained yet, so none of
  `3_validation/` or `4_analysis/`'s per-state commands have actually been
  run for them — the scripts should work unchanged (they're state-agnostic
  now), but that's untested until there's real data to point `--imagery-root`
  / `--artifacts-root` at.
