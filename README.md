# uswealth-geoai

Downstream statistical analysis for the wealth-from-satellite-imagery
seasonality project (Sentinel-2 → CV wealth estimates → seasonal/spatial
analysis, across OH/AZ/CA/GA/PA).

See **[REPLICATION.md](./REPLICATION.md)** for how the pipeline fits
together (this repo + `geoetl` + `sail`), what each file in this repo does,
and how to reproduce any finding from scratch. Start there.

## Layout

- `1_data_prep/` — tract shapefiles, wealth-index labels, land cover
- `2_imagery_legacy/` — superseded imagery-conversion step (see REPLICATION.md)
- `3_validation/` — per-quarter model accuracy from `sail`'s prediction CSVs
- `4_analysis/` — the paper's actual analyses (seasonality, pixel diversity, band importance)
