"""
Step 2 of the seasonal-amplitude pipeline: from amplitude_predictions.py's
long (model, eval cell) x GEOID table, compute three per-tract instability
measures, each on raw predictions and on z-scored predictions.

Terminology: "model" = one trained (model_year, model_quarter) checkpoint.
"season"/"imagery quarter" = eval_quarter, the quarter of the IMAGERY fed
into a model (which can differ from the model's own training quarter --
see amplitude_predictions.py's docstring for why the full (model, eval)
grid is collected, not just the diagonal).

  Seasonal amplitude   For a FIXED model and FIXED eval_year, the SD of
                        that model's predictions across its eval_quarter
                        in {1,2,3,4} -- how much does ONE model's output
                        wobble purely from which quarter's imagery it's
                        looking at. Averaged across every (model_year,
                        model_quarter, eval_year) combination available
                        for that tract.

  Noise floor           For a FIXED model and FIXED eval_quarter (season),
                        the SD across eval_year -- how much does ONE
                        model's output wobble across different years of
                        the SAME season's imagery. Season is held fixed
                        here, so this is instability NOT attributable to
                        season at all -- the baseline level seasonal
                        amplitude needs to exceed to mean anything.
                        Averaged across every (model_year, model_quarter,
                        eval_quarter) combination available.

  Model disagreement    Restricted to DIAGONAL cells only (eval_year ==
                        model_year, eval_quarter == model_quarter -- each
                        model evaluated on its OWN season's imagery, its
                        best case). For a fixed model_year, the SD across
                        the (up to) 4 same-year models' diagonal
                        predictions. Averaged across every model_year
                        available. Isolates between-model variance
                        (different training runs/architectures-by-season)
                        from any single model's own seasonal wobble.

ASSUMPTION (z-scoring grouping -- stated explicitly since the original
request's wording is compatible with more than one reading): "z-scoring
predictions within each model and imagery quarter" is implemented as
z-scoring within each (model_year, model_quarter, eval_year, eval_quarter)
CELL -- i.e. each row in amplitude_predictions.py's long table's own
(model, eval) grouping is one z-scoring population, using that cell's
own across-ALL-TEST-TRACTS mean/std. This removes a cell-wide level
shift ("this model read 0.3 high on this exact quarter's imagery for
every tract") before measuring instability, which is the stated intent
("so a global calibration shift is not counted as instability"); a
coarser grouping (e.g. pooling across eval_year too) would still remove
a per-model-per-season shift but NOT a per-cell one specifically.

Missing cells are handled the same way everywhere here: pandas groupby
std()/mean() with the actual count of whatever's present (ddof=1, so a
group with <2 members is NaN, not zero) -- never assumed to be a full
4-quarter or 3-year grid. n_* columns report how many (model, eval)
combinations actually went into each tract's numbers, so a tract's
amplitude computed from only 2 quarters isn't silently treated the same
as one computed from all 16.

Output: one row per GEOID --
    <out-dir>/<state>_<variable>_amplitude.csv (or --out-format parquet)
columns: state, GEOID, seasonal_amplitude_raw, seasonal_amplitude_z,
noise_floor_raw, noise_floor_z, model_disagreement_raw,
model_disagreement_z, n_seasonal_amplitude_groups, n_noise_floor_groups,
n_model_disagreement_groups.

Usage:
    python 3_analysis/compute_amplitude.py \
        --predictions-csv 3_analysis/outputs/amplitude/oh_wealth_index_amplitude_predictions.csv \
        --out-dir 3_analysis/outputs/amplitude
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """z-scores pred within each (model_year, model_quarter, eval_year,
    eval_quarter) cell, across every test tract in that cell (see module
    docstring for why this grouping, not a coarser one)."""
    df = df.copy()
    group_cols = ["model_year", "model_quarter", "eval_year", "eval_quarter"]
    stats = df.groupby(group_cols)["pred"].agg(["mean", "std"]).rename(
        columns={"mean": "_cell_mean", "std": "_cell_std"})
    df = df.merge(stats, on=group_cols, how="left")
    df["pred_z"] = np.where(df["_cell_std"] > 0, (df["pred"] - df["_cell_mean"]) / df["_cell_std"], 0.0)
    return df.drop(columns=["_cell_mean", "_cell_std"])


def seasonal_amplitude(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Per (GEOID, model_year, model_quarter, eval_year): SD of pred_col
    across eval_quarter. Averaged across (model_year, model_quarter,
    eval_year) per GEOID."""
    per_cell = df.groupby(["GEOID", "model_year", "model_quarter", "eval_year"])[pred_col].std(ddof=1)
    return per_cell.groupby("GEOID").agg(["mean", "count"])


def noise_floor(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Per (GEOID, model_year, model_quarter, eval_quarter): SD of
    pred_col across eval_year. Averaged across (model_year, model_quarter,
    eval_quarter) per GEOID."""
    per_cell = df.groupby(["GEOID", "model_year", "model_quarter", "eval_quarter"])[pred_col].std(ddof=1)
    return per_cell.groupby("GEOID").agg(["mean", "count"])


def model_disagreement(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Diagonal cells only. Per (GEOID, model_year): SD of pred_col
    across model_quarter. Averaged across model_year per GEOID."""
    diag = df[(df.model_year == df.eval_year) & (df.model_quarter == df.eval_quarter)]
    per_cell = diag.groupby(["GEOID", "model_year"])[pred_col].std(ddof=1)
    return per_cell.groupby("GEOID").agg(["mean", "count"])


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_zscore(df)

    sa_raw = seasonal_amplitude(df, "pred")
    sa_z = seasonal_amplitude(df, "pred_z")
    nf_raw = noise_floor(df, "pred")
    nf_z = noise_floor(df, "pred_z")
    md_raw = model_disagreement(df, "pred")
    md_z = model_disagreement(df, "pred_z")

    out = pd.DataFrame({
        "seasonal_amplitude_raw": sa_raw["mean"], "seasonal_amplitude_z": sa_z["mean"],
        "noise_floor_raw": nf_raw["mean"], "noise_floor_z": nf_z["mean"],
        "model_disagreement_raw": md_raw["mean"], "model_disagreement_z": md_z["mean"],
        "n_seasonal_amplitude_groups": sa_raw["count"], "n_noise_floor_groups": nf_raw["count"],
        "n_model_disagreement_groups": md_raw["count"],
    })
    return out.reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions-csv", required=True, type=Path,
                    help="amplitude_predictions.py's output for one state")
    p.add_argument("--out-dir", default="3_analysis/outputs/amplitude")
    p.add_argument("--out-format", choices=["csv", "parquet"], default="csv")
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv, dtype={"GEOID": str})
    required = {"state", "GEOID", "model_year", "model_quarter", "eval_year", "eval_quarter", "pred", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{args.predictions_csv} is missing columns: {missing}")

    states = df["state"].unique()
    if len(states) != 1:
        raise SystemExit(f"{args.predictions_csv} has more than one state ({list(states)}) -- "
                          f"run this once per state's own predictions CSV")
    state = states[0]

    amplitude = compute_all(df)
    amplitude.insert(0, "state", state)

    variable_guess = Path(args.predictions_csv).stem.replace(f"{state}_", "").replace("_amplitude_predictions", "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{state}_{variable_guess}_amplitude.{args.out_format}"
    if args.out_format == "csv":
        amplitude.to_csv(out_path, index=False)
    else:
        amplitude.to_parquet(out_path, index=False)

    print(f"Wrote {out_path} ({len(amplitude):,} tracts)")
    print(amplitude[["seasonal_amplitude_raw", "noise_floor_raw", "model_disagreement_raw"]].describe())
    n_low_coverage = int((amplitude["n_seasonal_amplitude_groups"] < 3).sum())
    if n_low_coverage:
        print(f"\nNOTE: {n_low_coverage} tract(s) have seasonal_amplitude averaged over fewer than "
              f"3 (model, eval_year) groups -- check n_seasonal_amplitude_groups before trusting "
              f"those tracts' numbers in step 4.")


if __name__ == "__main__":
    main()
