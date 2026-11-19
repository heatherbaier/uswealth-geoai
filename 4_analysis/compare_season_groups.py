"""
Groups every cell in a state's saved year x quarter R^2/Pearson-r
matrices into four categories and compares them:
  diagonal                  same year, same quarter
  same_season_other_year    same quarter, different year
  other_season_same_year    different quarter, same year
  other_season_other_year   different quarter, different year

Reports mean R^2 and mean Pearson r per group. RQ1 is the
same_season_other_year vs. other_season_same_year comparison, tested
with a permutation test: shuffle which quarter each (year, quarter)
cell counts as (year stays fixed per cell), rebuild the two groups
under that shuffled labeling, and recompute the R^2/Pearson r gap --
repeated --n-perms times to see how often a gap this big happens by
chance.

Reads build_cross_quarter_r2_matrix.py's own output layout directly --
<matrix-root>/<state>_<variable>/r2_matrix.csv and pearson_r_matrix.csv
(its default --out-dir, which is what run_cross_quarter_full.sh already
produces -- no extra step needed beyond that script now also writing
pearson_r_matrix.csv).

Usage:
    python 4_analysis/compare_season_groups.py --states oh pa az ga
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_RE = re.compile(r"^Q(\d)\s+(\d{4})$")


def parse_label(label):
    m = LABEL_RE.match(str(label).strip())
    if not m:
        raise ValueError(f"unrecognized matrix label: {label!r}")
    return int(m.group(1)), int(m.group(2))  # quarter, year


def group_of(model_q, model_y, eval_q, eval_y):
    if model_q == eval_q and model_y == eval_y:
        return "diagonal"
    if model_q == eval_q:
        return "same_season_other_year"
    if model_y == eval_y:
        return "other_season_same_year"
    return "other_season_other_year"


def load_matrix_long(path: Path, value_col: str) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    rows = []
    for row_label in matrix.index:
        model_q, model_y = parse_label(row_label)
        for col_label in matrix.columns:
            eval_q, eval_y = parse_label(col_label)
            val = matrix.loc[row_label, col_label]
            if pd.isna(val):
                continue
            rows.append({
                "model_quarter": model_q, "model_year": model_y,
                "eval_quarter": eval_q, "eval_year": eval_y,
                value_col: float(val),
            })
    return pd.DataFrame(rows)


def build_table(r2_path: Path, pearson_path: Path) -> pd.DataFrame:
    key = ["model_year", "model_quarter", "eval_year", "eval_quarter"]
    r2_df = load_matrix_long(r2_path, "r2")
    pearson_df = load_matrix_long(pearson_path, "pearson_r")
    df = r2_df.merge(pearson_df, on=key, how="outer")
    df["group"] = [group_of(r.model_quarter, r.model_year, r.eval_quarter, r.eval_year)
                    for r in df.itertuples(index=False)]
    return df


def group_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group")[["r2", "pearson_r"]].agg(["mean", "count"])


def permutation_test(df: pd.DataFrame, value_col: str, n_perms: int = 999, seed: int = 0):
    """Shuffles which quarter each (year, quarter) cell counts as (year
    unchanged), rebuilds same_season_other_year vs other_season_same_year
    under that shuffled labeling, and recomputes the mean-value gap."""
    units = sorted(set(zip(df["model_year"], df["model_quarter"])) |
                    set(zip(df["eval_year"], df["eval_quarter"])))
    years = [u[0] for u in units]
    real_quarters = np.array([u[1] for u in units])

    def gap_for(quarters):
        unit_season = {(years[i], units[i][1]): quarters[i] for i in range(len(units))}
        same_season_other_year, other_season_same_year = [], []
        for r in df.itertuples(index=False):
            model_season = unit_season[(r.model_year, r.model_quarter)]
            eval_season = unit_season[(r.eval_year, r.eval_quarter)]
            val = getattr(r, value_col)
            if model_season == eval_season and r.model_year != r.eval_year:
                same_season_other_year.append(val)
            elif model_season != eval_season and r.model_year == r.eval_year:
                other_season_same_year.append(val)
        if not same_season_other_year or not other_season_same_year:
            return np.nan
        return np.mean(same_season_other_year) - np.mean(other_season_same_year)

    observed = gap_for(real_quarters)

    rng = np.random.default_rng(seed)
    null_gaps = np.array([gap_for(rng.permutation(real_quarters)) for _ in range(n_perms)])
    valid_null = null_gaps[~np.isnan(null_gaps)]
    p_value = (1 + int(np.sum(np.abs(valid_null) >= abs(observed)))) / (len(valid_null) + 1)
    return {
        "observed_gap": observed, "null_mean": float(np.nanmean(null_gaps)),
        "null_std": float(np.nanstd(valid_null)), "p_value": p_value, "n_perms": n_perms,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--matrix-dir", default="out_cross_quarter", type=Path,
                    help="Root containing <state>_<variable>/r2_matrix.csv + pearson_r_matrix.csv "
                         "(build_cross_quarter_r2_matrix.py's default --out-dir root)")
    p.add_argument("--out-dir", default="4_analysis/outputs", type=Path)
    p.add_argument("--n-perms", type=int, default=999)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_means, all_rq1 = [], []
    for state in [s.lower() for s in args.states]:
        state_dir = args.matrix_dir / f"{state}_{args.variable}"
        r2_path = state_dir / "r2_matrix.csv"
        pearson_path = state_dir / "pearson_r_matrix.csv"
        if not r2_path.exists() or not pearson_path.exists():
            print(f"[{state}] missing {r2_path} or {pearson_path}, skipping")
            continue

        df = build_table(r2_path, pearson_path)
        means = group_means(df)
        print(f"\n[{state}] group means:")
        print(means)
        means_flat = means.copy()
        means_flat.columns = ["_".join(c) for c in means_flat.columns]
        means_flat = means_flat.reset_index()
        means_flat.insert(0, "state", state)
        all_means.append(means_flat)

        rq1_r2 = permutation_test(df, "r2", n_perms=args.n_perms, seed=args.seed)
        rq1_r = permutation_test(df, "pearson_r", n_perms=args.n_perms, seed=args.seed)
        print(f"[{state}] RQ1 (same_season_other_year - other_season_same_year):")
        print(f"  R^2:       observed={rq1_r2['observed_gap']:+.4f} null_mean={rq1_r2['null_mean']:+.4f} "
              f"p={rq1_r2['p_value']:.4f}")
        print(f"  Pearson r: observed={rq1_r['observed_gap']:+.4f} null_mean={rq1_r['null_mean']:+.4f} "
              f"p={rq1_r['p_value']:.4f}")
        all_rq1.append({"state": state, "metric": "r2", **rq1_r2})
        all_rq1.append({"state": state, "metric": "pearson_r", **rq1_r})

    if not all_means:
        raise SystemExit("No state had both matrix files -- nothing to write.")

    means_path = args.out_dir / "season_group_means.csv"
    rq1_path = args.out_dir / "season_group_rq1_permutation.csv"
    pd.concat(all_means, ignore_index=True).to_csv(means_path, index=False)
    pd.DataFrame(all_rq1).to_csv(rq1_path, index=False)
    print(f"\nWrote {means_path}")
    print(f"Wrote {rq1_path}")


if __name__ == "__main__":
    main()
