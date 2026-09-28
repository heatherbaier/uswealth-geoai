"""
Does a model's cross-quarter transfer failure track land cover -- tree
cover specifically (canopy that fills in over the growing season, so a
model trained on bare-season imagery misses it) versus cropland (also
bare off-season, but agricultural rather than residential) -- for Ohio's
cross-quarter generalization runs (generate_cross_validate_config.py +
sail's task: validate, already run and saved to preds CSVs)?

Steps (see REPLICATION.md's "Thread: cross-quarter generalization" for
where this fits):

  0. Sanity check: merge lc.csv onto the wealth index by GEOID, print
     corr(tree cover, wealth) and corr(built-up, wealth) for all tracts.
  1. Build the per-tract error table from the saved cross-quarter preds
     CSVs, then the transfer penalty: for tract g, year y, training
     quarter t evaluated on quarter e's imagery,
         penalty(g, y, t->e) = sq_err(t model -> e imagery)
                               - sq_err(e model -> e imagery)
     i.e. the transferred model's error, minus what the MATCHED model
     (trained on e itself) gets on that same imagery/tract/year -- this
     differences out how hard a tract is to predict in general, leaving
     just the cost of using the wrong quarter's model.
  2. Main regression: Q1->Q3 penalty ~ tree cover + cropland + built-up +
     C(year), pooled 2017-2019, SEs clustered by GEOID (tracts repeat
     across years). Reports which of tree cover / cropland has the larger
     standardized coefficient.
  3. Dose-response: same regression for Q1->Q2, Q1->Q3, Q1->Q4 (1, 2, and
     3 quarters of seasonal distance from Q1) -- three tree-cover
     coefficients + CIs, plotted in distance order so a gradient (if real)
     is visible. (The task text lists these as "Q1->Q4, Q1->Q2, Q1->Q3" --
     all three are computed and reported regardless, but a dose-response
     plot needs distance on the x-axis to show a gradient at all, so
     that's the plotted order.)
  4. Urban restriction: same regression restricted to built_up > 0.5.
     Cropland is expected to be near-constant there and gets dropped from
     the formula automatically (see run_penalty_regression) -- this is
     the test that separates "residential canopy" from "agricultural
     land" as the mechanism.
  5. Placebo: same regression for the REVERSE direction, Q3->Q1 penalty.
     The hypothesis predicts tree cover should NOT predict failure here
     (a growing-season model is expected to handle bare/dormant
     landscapes fine) -- a null result here is what makes step 2-4's
     result directional rather than "tree cover just correlates with
     transfer penalty somehow."
  6. NDVI alternative (only if --ndvi-csv given): seasonal amplitude =
     mean NDVI(Q3) - mean NDVI(Q1) per tract, substituted for tree cover
     in the step-2 regression. Reports both versions side by side.

Ends with one summary CSV of every regression's coefficients and a short
printed interpretation.

Two methodology points worth being explicit about, both fixed after an
earlier version of this script's own printed interpretation got them
wrong on a real Ohio run:
  - Land-cover fractions are compositional (sum to ~1). Every regression
    here includes EVERY class present in --lc-csv except one, explicitly
    chosen as the omitted reference category (resolve_reference_class) --
    not just tree_cover/cropland/built_up with everything else silently
    absorbed into the intercept. Three negative coefficients don't mean
    "less of everything is better"; they mean "more of the OMITTED class
    is worse", and the reference class is always printed so that's never
    ambiguous.
  - The hypothesis is directional: more tree cover should mean a BIGGER
    transfer penalty (a positive coefficient). A significant coefficient
    with the WRONG sign is evidence AGAINST the hypothesis, not for it --
    every printed claim here is gated on sign as well as significance
    (see _direction_note), and a subsample regression fitting several
    parameters against too few clusters (e.g. step 4 with under
    --min-urban-tracts, default 30) is skipped rather than reported as if
    it were a finding.

Inputs this needs, none of which this script produces:
  --wealth-csv   GEOID + a wealth index column. For OH specifically there
                 is no clean_wealth_index.py output (see REPLICATION.md --
                 OH's label provenance predates this repo and isn't
                 reproducible from anything in it) -- point this at
                 whatever CSV OH's actual wealth labels come from.
  --lc-csv       1_data_prep/compute_landcover.py's output
                 (`python compute_landcover.py --state oh` -> ./lc_oh.csv).
  --ndvi-csv     optional, only for step 6 -- fine to just leave out
                 entirely, step 6 then prints one line saying it was
                 skipped and every other step runs as normal. Long format:
                 GEOID, year, quarter, ndvi_mean, one row per tract per
                 quarter per year. Nothing in this repo computes this today.
  cross-quarter preds CSVs -- three ways to point at these, in priority
  order:
    (a) --preds-manifest: an explicit CSV with columns year,
        train_quarter, eval_quarter, preds_csv, one row per already-
        validated cell.
    (b) (default) auto-discovered from --data-root-template/
        --base-prefix-template, which already default to the convention
        every state's checkpoints actually use on disk (confirmed against
        OH's real q2_2017 checkpoint: .../oh_imagery/q2_2017_s2_allbands/
        artifacts/oh_q2_2017_wealth_index_v1/cross_quarter/epoch167_
        valset_oh_2017_q1_s2_allbands_wealth_index_preds.csv) -- this
        works for OH today without needing a state_registry.yml entry at
        all, and for any other state following the same layout.
    (c) --use-state-registry: same discovery, but pointed at
        pipeline_configs/state_registry.yml instead (needs a full entry
        there -- spatial_block_deg/band stats too, not just the path
        templates -- so this only applies to states already migrated,
        e.g. AZ/GA/PA).
  Reuses model_ckpt_dir/imagery_prefix/find_preds_csv/latest_existing_version
  from build_cross_quarter_r2_matrix.py / generate_validate_config.py
  rather than reimplementing that path logic a third time.

Required (train_quarter, eval_quarter) cells, per year -- Q1's full row
(needed for steps 2-4, all of which transfer FROM Q1), every quarter's own
diagonal (needed as the "matched model" baseline for every penalty), and
Q3->Q1 for the step-5 placebo:
    (1,1) (1,2) (1,3) (1,4)   (2,2)   (3,3) (3,1)   (4,4)

Usage:
    # Sanity-check which cells are actually found first (no --wealth-csv/
    # --lc-csv needed yet):
    python 4_analysis/analyze_transfer_penalty_landcover.py \
        --state oh --years 2017 2018 2019 --build-manifest-only \
        --out-dir ./out_transfer_penalty/oh

    # Full run, --ndvi-csv omitted (step 6 just gets skipped):
    python 4_analysis/analyze_transfer_penalty_landcover.py \
        --state oh --years 2017 2018 2019 \
        --wealth-csv ./oh_wi2019.csv --wealth-col wealth_index_overall_core \
        --lc-csv ./lc_oh.csv --out-dir ./out_transfer_penalty/oh

    # Or point directly at an explicit manifest instead of auto-discovery:
    python 4_analysis/analyze_transfer_penalty_landcover.py \
        --state oh --years 2017 2018 2019 \
        --wealth-csv ./oh_wi2019.csv --wealth-col wealth_index_overall_core \
        --lc-csv ./lc_oh.csv --preds-manifest ./oh_preds_manifest.csv \
        --out-dir ./out_transfer_penalty/oh
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
from build_cross_quarter_r2_matrix import (  # noqa: E402
    model_ckpt_dir, imagery_prefix, find_preds_csv, load_registry,
)
from generate_validate_config import latest_existing_version  # noqa: E402
from update_ys_labels import normalize_geoid  # noqa: E402

NAMED_LC_CLASSES = {"tree_cover": "Tree cover", "cropland": "Cropland", "built_up": "Built-up"}
REQUIRED_CELLS = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (3, 3), (4, 4), (3, 1)]
MIN_CLUSTERS_WARNING = 30
ALPHA = 0.05


def _direction_note(coef: float, p: float) -> str:
    """The hypothesis is directional (more tree cover -> BIGGER transfer
    penalty, i.e. a POSITIVE coefficient), not just "tree cover matters
    somehow" -- a significant result with the wrong sign is evidence
    AGAINST the hypothesis, not for it. Never collapse this to "p<0.05 so
    it holds"."""
    if p is None:
        return "not estimable"
    if coef > 0 and p < ALPHA:
        return f"significant and POSITIVE (p={p:.4g}) -- consistent with the hypothesis"
    if coef < 0 and p < ALPHA:
        return (f"significant but NEGATIVE (p={p:.4g}) -- OPPOSITE the hypothesis "
                f"(more of this predictor associated with a SMALLER transfer penalty here)")
    return f"not significant (p={p:.4g}, coef={coef:+.5f}) -- no support either way"

# Same directory/naming convention every state (AZ/GA/PA, and OH's actual
# on-disk layout too, confirmed against a real `ls` of OH's q2_2017
# checkpoint) already uses -- see generate_download_config.py/
# generate_train_config.py's identical templates. Parameterized here
# directly (not via state_registry.yml's resolve_state_settings) because
# that function also demands spatial_block_deg/band_mean/band_std/
# in_channels be filled in, which a pre-registry state like OH doesn't
# have and doesn't need just to locate already-written preds CSVs.
DEFAULT_DATA_ROOT_TEMPLATE = "/data/hbaier/new_data/tlag/{state}_imagery/q{quarter}_{year}_s2_allbands/"
DEFAULT_BASE_PREFIX_TEMPLATE = "{state}_{year}_q{quarter}_s2_allbands"


# ---------------------------------------------------------------------
# Step 0 -- sanity check
# ---------------------------------------------------------------------
def load_wealth(wealth_csv, wealth_col) -> pd.DataFrame:
    df = pd.read_csv(wealth_csv, dtype=str)
    if wealth_col not in df.columns:
        raise SystemExit(f"--wealth-col {wealth_col!r} not in {wealth_csv} (columns: {list(df.columns)})")
    geoid_col = "GEOID" if "GEOID" in df.columns else df.columns[0]
    out = pd.DataFrame({
        "GEOID": df[geoid_col].apply(normalize_geoid),
        "wealth_index": pd.to_numeric(df[wealth_col], errors="coerce"),
    })
    return out.dropna(subset=["GEOID", "wealth_index"]).drop_duplicates(subset="GEOID")


def snake_case(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()


def load_landcover(lc_csv) -> pd.DataFrame:
    """Keeps EVERY class column present, not just tree_cover/cropland/
    built_up -- land cover fractions are compositional (sum to ~1), so a
    regression that only includes 3 of N classes leaves the other N-3
    implicitly folded into the intercept as an unstated reference
    category. resolve_reference_class() below picks one of the extras
    (e.g. Grassland) to serve as that reference explicitly, so every
    printed coefficient can be stated as "relative to <X>" instead of
    "relative to whatever's left over"."""
    lc = pd.read_csv(lc_csv, dtype={0: str}, index_col=0)
    lc.index.name = "GEOID"
    lc = lc.reset_index()
    lc["GEOID"] = lc["GEOID"].apply(normalize_geoid)

    class_cols = [c for c in lc.columns if c != "GEOID"]
    row_sums = lc[class_cols].sum(axis=1)
    if (row_sums > 2).any():
        lc[class_cols] = lc[class_cols].div(row_sums, axis=0)

    lc = lc.rename(columns={c: snake_case(c) for c in class_cols})
    snake_cols = [snake_case(c) for c in class_cols]

    missing = [snake for snake in NAMED_LC_CLASSES if snake not in snake_cols]
    if missing:
        have = [NAMED_LC_CLASSES.get(c, c) for c in snake_cols]
        raise SystemExit(f"--lc-csv is missing expected class column(s) for {missing} (has: {have})")

    return lc[["GEOID"] + snake_cols].drop_duplicates(subset="GEOID")


def resolve_reference_class(features_df: pd.DataFrame, reference_class=None) -> str:
    """Picks (or validates) the omitted reference land-cover category.
    Every OTHER class present gets included as an explicit regressor --
    this is the fix for a real bug: with only tree_cover/cropland/built_up
    in the model, three negative coefficients don't mean "less of
    everything is better", they mean "more of the OMITTED class (whatever
    that silently ends up being) is worse". Making it explicit removes
    that ambiguity."""
    lc_classes = [c for c in features_df.columns if c != "GEOID"]
    if reference_class is not None:
        if reference_class not in lc_classes:
            raise SystemExit(f"--lc-reference-class {reference_class!r} not among land-cover "
                              f"columns: {lc_classes}")
        return reference_class
    non_named = [c for c in lc_classes if c not in NAMED_LC_CLASSES]
    if not non_named:
        raise SystemExit(
            "No land-cover class left over to serve as an omitted reference category -- --lc-csv "
            "only has tree_cover/cropland/built_up. Add more ESA WorldCover classes (e.g. "
            "Grassland, Shrubland) to lc.csv, or pass --lc-reference-class explicitly (it will "
            "then be dropped as a regressor, same as any reference category)."
        )
    # Default: the largest-mean leftover class -- the most natural "everything else" baseline.
    return features_df[non_named].mean().idxmax()


def step0_sanity_check(wealth_df: pd.DataFrame, lc_df: pd.DataFrame):
    merged = wealth_df.merge(lc_df, on="GEOID", how="inner")
    print(f"[Step 0] Merged wealth x land cover: {len(merged)} tracts "
          f"({len(wealth_df)} wealth rows, {len(lc_df)} land cover rows)")
    r_tree, p_tree = scipy_stats.pearsonr(merged["tree_cover"], merged["wealth_index"])
    r_built, p_built = scipy_stats.pearsonr(merged["built_up"], merged["wealth_index"])
    print(f"[Step 0] corr(tree cover, wealth index) = {r_tree:+.4f}  (p={p_tree:.4g}, n={len(merged)})")
    print(f"[Step 0] corr(built-up,   wealth index) = {r_built:+.4f}  (p={p_built:.4g}, n={len(merged)})")
    return merged


# ---------------------------------------------------------------------
# Step 1 -- error table + transfer penalty
# ---------------------------------------------------------------------
def build_manifest_from_registry(state, variable, years, version, registry) -> pd.DataFrame:
    rows, missing = [], []
    for year in years:
        for train_q, eval_q in REQUIRED_CELLS:
            try:
                ckpt_dir = model_ckpt_dir(state, variable, year, train_q, version, registry)
            except SystemExit as e:
                missing.append((year, train_q, eval_q, str(e)))
                continue
            prefix = imagery_prefix(state, variable, year, eval_q, registry)
            csv_path = find_preds_csv(ckpt_dir, prefix)
            if csv_path is None:
                missing.append((year, train_q, eval_q, f"not validated under {ckpt_dir}"))
                continue
            rows.append({"year": year, "train_quarter": train_q, "eval_quarter": eval_q, "preds_csv": csv_path})
    if missing:
        print(f"[Step 1] WARNING: {len(missing)}/{len(years) * len(REQUIRED_CELLS)} required "
              f"(year, train_q, eval_q) cell(s) not found -- dropped from the analysis:")
        for year, tq, eq, reason in missing:
            print(f"    {year} Q{tq}->Q{eq}: {reason}")
    return pd.DataFrame(rows, columns=["year", "train_quarter", "eval_quarter", "preds_csv"])


def _template_ckpt_dir(state, variable, year, quarter, version, data_root_template):
    data_root = data_root_template.format(state=state, year=year, quarter=quarter)
    data_root = data_root if data_root.endswith("/") else data_root + "/"
    output_dir = data_root + "artifacts/"
    exp_base = f"{state}_q{quarter}_{year}_{variable}"
    v = version or latest_existing_version(output_dir, exp_base)
    return os.path.join(output_dir, f"{exp_base}_{v}")


def _template_imagery_prefix(state, variable, year, quarter, base_prefix_template):
    base_prefix = base_prefix_template.format(state=state, year=year, quarter=quarter)
    return f"{base_prefix}_{variable}"


def build_manifest_from_templates(state, variable, years, version,
                                   data_root_template, base_prefix_template) -> pd.DataFrame:
    """Same discovery as build_manifest_from_registry, but pointed at
    --data-root-template/--base-prefix-template directly instead of a
    state_registry.yml entry -- the path for a state (OH) that isn't in
    the registry but already writes checkpoints/preds CSVs in the exact
    same layout as the states that are."""
    rows, missing = [], []
    for year in years:
        for train_q, eval_q in REQUIRED_CELLS:
            ckpt_dir = _template_ckpt_dir(state, variable, year, train_q, version, data_root_template)
            prefix = _template_imagery_prefix(state, variable, year, eval_q, base_prefix_template)
            csv_path = find_preds_csv(ckpt_dir, prefix)
            if csv_path is None:
                missing.append((year, train_q, eval_q, f"not found under {ckpt_dir}"))
                continue
            rows.append({"year": year, "train_quarter": train_q, "eval_quarter": eval_q, "preds_csv": csv_path})
    if missing:
        print(f"[Step 1] WARNING: {len(missing)}/{len(years) * len(REQUIRED_CELLS)} required "
              f"(year, train_q, eval_q) cell(s) not found -- dropped from the analysis:")
        for year, tq, eq, reason in missing:
            print(f"    {year} Q{tq}->Q{eq}: {reason}")
    return pd.DataFrame(rows, columns=["year", "train_quarter", "eval_quarter", "preds_csv"])


def load_manifest_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"year", "train_quarter", "eval_quarter", "preds_csv"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"--preds-manifest is missing column(s): {sorted(missing)}")
    return df


def build_error_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (GEOID, year, train_quarter, eval_quarter)
    with that pair's squared error. GEOID comes from the chip filename
    stem in the preds CSV's 'name' column (full chip path) -- same
    convention as assemble_preds_wide.py / compute_pixel_diversity.py,
    since geoetl writes chips to <data_root>/chips/{GEOID}.tif and sail's
    run_validation writes that exact path into every preds row."""
    if manifest.empty:
        raise SystemExit("Empty preds manifest -- nothing to build an error table from.")
    parts = []
    for row in manifest.itertuples(index=False):
        df = pd.read_csv(row.preds_csv)
        part = pd.DataFrame({
            "GEOID": df["name"].apply(lambda p: normalize_geoid(Path(p).stem)),
            "year": row.year,
            "train_quarter": row.train_quarter,
            "eval_quarter": row.eval_quarter,
            "sq_err": (df["label"] - df["pred"]) ** 2,
        })
        n_before = len(part)
        part = part.drop_duplicates(subset="GEOID")
        if len(part) != n_before:
            print(f"  {row.year} Q{row.train_quarter}->Q{row.eval_quarter}: "
                  f"dropped {n_before - len(part)} duplicate GEOID row(s)")
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    print(f"[Step 1] Built error table: {len(long)} (tract, year, train_q, eval_q) rows "
          f"across {manifest['year'].nunique()} year(s), {len(manifest)} cell(s)")
    return long


def transfer_penalty(error_long: pd.DataFrame, train_q: int, eval_q: int) -> pd.DataFrame:
    """penalty(GEOID, year) = sq_err(train_q model -> eval_q imagery)
                              - sq_err(eval_q model -> eval_q imagery)
    Differences out per-tract baseline difficulty: a tract that's
    genuinely noisy/hard to predict inflates both terms equally and
    cancels out, leaving just the extra cost of using the wrong quarter's
    model."""
    transferred = error_long[(error_long["train_quarter"] == train_q) & (error_long["eval_quarter"] == eval_q)]
    matched = error_long[(error_long["train_quarter"] == eval_q) & (error_long["eval_quarter"] == eval_q)]
    if transferred.empty:
        raise SystemExit(f"No rows for Q{train_q}->Q{eval_q} in the error table -- "
                          f"that cell wasn't in the preds manifest.")
    if matched.empty:
        raise SystemExit(f"No rows for the matched Q{eval_q}->Q{eval_q} model -- "
                          f"needed as the baseline for the Q{train_q}->Q{eval_q} transfer penalty.")
    merged = transferred.merge(
        matched[["GEOID", "year", "sq_err"]], on=["GEOID", "year"], how="inner",
        suffixes=("_transfer", "_matched"),
    )
    merged["penalty"] = merged["sq_err_transfer"] - merged["sq_err_matched"]
    merged["train_quarter"] = train_q
    merged["eval_quarter"] = eval_q
    return merged[["GEOID", "year", "train_quarter", "eval_quarter", "penalty"]]


# ---------------------------------------------------------------------
# Regression core, reused by steps 2-6
# ---------------------------------------------------------------------
def run_penalty_regression(penalty_df: pd.DataFrame, features_df: pd.DataFrame,
                            primary_col: str, label: str, reference_class: str,
                            min_std=1e-8, min_clusters=MIN_CLUSTERS_WARNING,
                            print_summary=True, full_summary=False):
    """penalty ~ primary_col + (every land-cover class in features_df
    except primary_col and reference_class) + C(year), SEs clustered by
    GEOID (tracts repeat across years, so residuals within a tract across
    years aren't independent -- plain OLS SEs would be too small).

    Controls are EVERY other land-cover class present, not a hardcoded
    pair -- reference_class is the one deliberately left out (see
    resolve_reference_class), so every coefficient here is interpretable
    as "relative to a tract that's entirely reference_class", stated
    explicitly rather than left as an implicit intercept effect.
    """
    lc_classes = [c for c in features_df.columns if c != "GEOID"]
    controls = [c for c in lc_classes if c not in (primary_col, reference_class)]

    d = penalty_df.merge(features_df, on="GEOID", how="inner")
    needed = [primary_col] + controls
    d = d.dropna(subset=["penalty", "year", *needed])

    if d[primary_col].std() < min_std:
        print(f"  WARNING: {primary_col} is ~constant in this sample (std={d[primary_col].std():.2g}) "
              f"-- its coefficient will be poorly identified.")

    terms = [primary_col]
    dropped = []
    for c in controls:
        if d[c].std() < min_std:
            dropped.append(c)
            continue
        terms.append(c)

    n_clusters = d["GEOID"].nunique()
    formula = f"penalty ~ {' + '.join(terms)} + C(year)"
    model = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["GEOID"]})
    n_params = len(model.params)

    # Standardized coefficient = coef * std(x) / std(y): puts predictors
    # on different natural scales (a land-cover fraction in [0,1] vs. an
    # NDVI amplitude) on the same footing -- "SDs of penalty per SD of x".
    y_std = d["penalty"].std()
    standardized = {}
    for c in terms:
        raw = model.params.get(c)
        if raw is not None and y_std > 0:
            standardized[c] = raw * d[c].std() / y_std

    result = {
        "label": label, "n": len(d), "n_tracts": d["GEOID"].nunique(), "n_clusters": n_clusters,
        "n_params": n_params, "reference_class": reference_class,
        "model": model, "formula": formula, "terms": terms,
        "standardized": standardized, "dropped_controls": dropped,
    }

    if print_summary:
        print(f"\n[{label}] {formula}  (n={result['n']}, n_tracts={result['n_tracts']}, "
              f"reference={reference_class!r})")
        if dropped:
            print(f"  (dropped near-constant control(s) in this subsample: {dropped})")
        if n_clusters < min_clusters:
            print(f"  WARNING: only {n_clusters} tract(s)/cluster(s) fitting {n_params} parameters "
                  f"-- cluster-robust SEs need substantially more clusters than that to be "
                  f"trustworthy (rule of thumb: >= {min_clusters}). Treat this result as "
                  f"illustrative, not a finding.")
        for c in terms:
            coef, se, pval = model.params[c], model.bse[c], model.pvalues[c]
            std_b = standardized.get(c)
            std_str = f", standardized={std_b:+.4f}" if std_b is not None else ""
            print(f"    {c:<12} coef={coef:+.5f}  se={se:.5f}  p={pval:.4g}{std_str}")
        print(f"    R^2 = {model.rsquared:.4f}")
        if full_summary:
            print(model.summary())

    return result


# ---------------------------------------------------------------------
# Step 2 -- main regression
# ---------------------------------------------------------------------
def step2_main_regression(pen_13: pd.DataFrame, features_df: pd.DataFrame, reference_class: str):
    res = run_penalty_regression(pen_13, features_df, "tree_cover", "Q1->Q3 (main)",
                                  reference_class, full_summary=True)
    tree_coef = res["model"].params.get("tree_cover")
    tree_p = res["model"].pvalues.get("tree_cover")
    tree_std = res["standardized"].get("tree_cover")
    crop_std = res["standardized"].get("cropland")
    if tree_coef is not None and tree_p is not None:
        print(f"\n[Step 2] Tree cover: {_direction_note(tree_coef, tree_p)}")
    if tree_std is not None and crop_std is not None:
        bigger = "tree cover" if abs(tree_std) > abs(crop_std) else "cropland"
        print(f"[Step 2] Larger standardized MAGNITUDE (not necessarily more supportive of the "
              f"hypothesis -- check the direction note above): {bigger} "
              f"(tree cover={tree_std:+.4f} SD, cropland={crop_std:+.4f} SD)")
    return res


# ---------------------------------------------------------------------
# Step 3 -- dose-response
# ---------------------------------------------------------------------
def step3_dose_response(error_long: pd.DataFrame, features_df: pd.DataFrame, out_dir: Path,
                         reference_class: str, precomputed=None):
    precomputed = precomputed or {}
    pairs = [(1, 2), (1, 3), (1, 4)]
    results, rows = {}, []
    for train_q, eval_q in pairs:
        res = precomputed.get((train_q, eval_q))
        if res is None:
            pen = transfer_penalty(error_long, train_q, eval_q)
            res = run_penalty_regression(pen, features_df, "tree_cover", f"Q{train_q}->Q{eval_q}",
                                          reference_class)
        results[(train_q, eval_q)] = res
        ci = res["model"].conf_int().loc["tree_cover"]
        rows.append({
            "transfer": f"Q{train_q}->Q{eval_q}", "distance_quarters": eval_q - train_q,
            "coef": res["model"].params["tree_cover"], "se": res["model"].bse["tree_cover"],
            "ci_lower": ci[0], "ci_upper": ci[1], "n": res["n"],
            "significant": bool(ci[0] > 0 or ci[1] < 0),
        })
    dose_df = pd.DataFrame(rows).sort_values("distance_quarters").reset_index(drop=True)

    print("\n[Step 3] Tree cover coefficient by transfer distance (distance order):")
    print(dose_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = list(range(len(dose_df)))
    yerr = [dose_df["coef"] - dose_df["ci_lower"], dose_df["ci_upper"] - dose_df["coef"]]
    ax.errorbar(x, dose_df["coef"], yerr=yerr, fmt="o-", capsize=4, color="tab:green")
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(dose_df["transfer"])
    ax.set_xlabel("Transfer (training quarter -> evaluation quarter)")
    ax.set_ylabel("Tree cover coefficient on transfer penalty")
    ax.set_title("Dose-response: tree cover effect on transfer penalty\nby seasonal distance from Q1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "dose_response_tree_cover.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Step 3] Wrote {out_path}")
    return dose_df, results


# ---------------------------------------------------------------------
# Step 4 -- urban restriction
# ---------------------------------------------------------------------
def step4_urban_restriction(error_long: pd.DataFrame, features_df: pd.DataFrame,
                             built_up_threshold: float, reference_class: str,
                             min_urban_tracts=MIN_CLUSTERS_WARNING):
    urban_geoids = features_df.loc[features_df["built_up"] > built_up_threshold, "GEOID"]
    pen = transfer_penalty(error_long, 1, 3)
    pen_urban = pen[pen["GEOID"].isin(urban_geoids)]
    n_urban_tracts = pen_urban["GEOID"].nunique()
    print(f"\n[Step 4] Urban restriction (built_up > {built_up_threshold}): "
          f"{n_urban_tracts} / {pen['GEOID'].nunique()} tracts")
    if n_urban_tracts < min_urban_tracts:
        print(f"  Only {n_urban_tracts} urban tract(s) -- below --min-urban-tracts="
              f"{min_urban_tracts}. A clustered regression here would fit several parameters "
              f"against that many clusters (or fewer, after tracts repeat across years get "
              f"counted once); the result would be overfitting noise dressed up as a coefficient, "
              f"not a finding. Skipping. Pass --min-urban-tracts to override if you want to see it "
              f"anyway (with that caveat).")
        return None
    res = run_penalty_regression(pen_urban, features_df, "tree_cover", "Q1->Q3 (urban-restricted)",
                                  reference_class)
    coef = res["model"].params.get("tree_cover")
    p = res["model"].pvalues.get("tree_cover")
    print(f"[Step 4] Tree cover: {_direction_note(coef, p)}")
    return res


# ---------------------------------------------------------------------
# Step 5 -- reverse-direction placebo
# ---------------------------------------------------------------------
def step5_placebo(error_long: pd.DataFrame, features_df: pd.DataFrame, reference_class: str):
    pen = transfer_penalty(error_long, 3, 1)
    res = run_penalty_regression(pen, features_df, "tree_cover", "Q3->Q1 (placebo)", reference_class)
    coef = res["model"].params.get("tree_cover")
    p = res["model"].pvalues.get("tree_cover")
    if p is None:
        print("[Step 5] tree cover not estimable")
    else:
        # Unlike steps 2/4, a placebo pass is sign-agnostic: significance
        # in EITHER direction here is bad news for the hypothesis (it
        # would mean tree cover predicts transfer failure even where the
        # hypothesis says it shouldn't), so this checks p alone on purpose.
        is_null = p >= ALPHA
        print(f"[Step 5] Placebo {'PASSES' if is_null else 'FAILS'} (tree cover coef={coef:+.5f}, "
              f"{'not' if is_null else 'IS'} significant in the reverse direction, p={p:.4g})")
    return res


# ---------------------------------------------------------------------
# Step 6 -- NDVI alternative
# ---------------------------------------------------------------------
def load_ndvi_amplitude(ndvi_csv) -> pd.DataFrame:
    """--ndvi-csv: long format GEOID, year, quarter, ndvi_mean. Amplitude
    = mean NDVI(Q3) - mean NDVI(Q1), averaged across whatever years are
    present for that tract -- a single tract-level trait, matching how
    tree_cover/cropland/built_up are one static value per tract rather
    than one per tract-year."""
    ndvi = pd.read_csv(ndvi_csv)
    required = {"GEOID", "year", "quarter", "ndvi_mean"}
    missing = required - set(ndvi.columns)
    if missing:
        raise SystemExit(f"--ndvi-csv is missing column(s): {sorted(missing)}")
    ndvi = ndvi.copy()
    ndvi["GEOID"] = ndvi["GEOID"].apply(normalize_geoid)
    q1 = ndvi[ndvi["quarter"] == 1].groupby("GEOID")["ndvi_mean"].mean()
    q3 = ndvi[ndvi["quarter"] == 3].groupby("GEOID")["ndvi_mean"].mean()
    return (q3 - q1).rename("amplitude").reset_index()


def step6_ndvi_alternative(pen_13: pd.DataFrame, features_df: pd.DataFrame, ndvi_csv, reference_class: str):
    if ndvi_csv is None:
        print("\n[Step 6] --ndvi-csv not provided -- skipping NDVI-amplitude alternative.")
        return None
    amp = load_ndvi_amplitude(ndvi_csv)
    features_with_amp = features_df.merge(amp, on="GEOID", how="left")
    n_missing = features_with_amp["amplitude"].isna().sum()
    if n_missing:
        print(f"[Step 6] {n_missing} tract(s) missing NDVI amplitude, dropped from this regression")
    res = run_penalty_regression(pen_13, features_with_amp, "amplitude", "Q1->Q3 (NDVI amplitude)",
                                  reference_class)
    coef = res["model"].params.get("amplitude")
    p = res["model"].pvalues.get("amplitude")
    print(f"[Step 6] Amplitude: {_direction_note(coef, p)}")
    return res


# ---------------------------------------------------------------------
# Summary + interpretation
# ---------------------------------------------------------------------
def summarize(results, out_dir: Path) -> pd.DataFrame:
    rows = []
    for res in results:
        if res is None:
            continue
        m = res["model"]
        for term in res["terms"]:
            rows.append({
                "regression": res["label"], "term": term,
                "coef": m.params[term], "se": m.bse[term], "p": m.pvalues[term],
                "standardized": res["standardized"].get(term),
                "n": res["n"], "n_tracts": res["n_tracts"], "n_clusters": res["n_clusters"],
                "n_params": res["n_params"], "reference_class": res["reference_class"],
                "low_cluster_count": res["n_clusters"] < MIN_CLUSTERS_WARNING,
                "r2": m.rsquared,
            })
    summary = pd.DataFrame(rows)
    out_path = out_dir / "summary_table.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[Summary] Wrote {out_path}")
    print(summary.to_string(index=False))
    return summary


def print_interpretation(main_res, dose_df, urban_res, placebo_res, ndvi_res):
    """Every claim here is gated on BOTH significance AND sign matching
    the hypothesis (more of the predictor -> bigger transfer penalty) --
    "significant" alone is not "supports the hypothesis": a significant
    coefficient with the wrong sign is evidence AGAINST it. See
    _direction_note(). Also never calls a pattern a "gradient" without
    checking the CIs actually exclude zero and the pattern holds in
    |magnitude|, not just in signed value (a sequence that crosses zero
    can be "monotonic" in raw value while its true effect size is
    shrinking -- see step 3's own construction)."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    lines = []

    tree_coef = main_res["model"].params.get("tree_cover")
    tree_p = main_res["model"].pvalues.get("tree_cover")
    tree_std = main_res["standardized"].get("tree_cover")
    crop_std = main_res["standardized"].get("cropland")
    if tree_coef is not None and tree_p is not None:
        lines.append(f"Main regression (Q1->Q3): tree cover is {_direction_note(tree_coef, tree_p)} "
                     f"(reference category: {main_res['reference_class']}).")
    if tree_std is not None and crop_std is not None:
        bigger = "tree cover" if abs(tree_std) > abs(crop_std) else "cropland"
        lines.append(f"By raw standardized MAGNITUDE (not the same as support for the hypothesis -- "
                     f"see the line above), {bigger} is larger (tree cover={tree_std:+.3f} SD, "
                     f"cropland={crop_std:+.3f} SD).")

    if dose_df is not None and len(dose_df) == 3:
        rows = list(dose_df.sort_values("distance_quarters").itertuples())
        coefs = [r.coef for r in rows]
        sig = [bool(r.significant) for r in rows]
        desc = ", ".join(f"{r.transfer}={r.coef:+.4f}{'*' if s else ''}"
                          for r, s in zip(rows, sig))
        if not any(sig):
            lines.append(f"Dose-response ({desc}, * = CI excludes 0): NONE of the three distances "
                         f"have a CI excluding zero -- there is no reliable dose-response signal "
                         f"here, regardless of how the raw point estimates are ordered.")
        else:
            mags = [abs(c) for c in coefs]
            mag_increasing = all(a <= b for a, b in zip(mags, mags[1:]))
            same_sign = len({c > 0 for c in coefs}) == 1
            if mag_increasing and same_sign and coefs[0] > 0:
                lines.append(f"Dose-response ({desc}): |effect| grows with distance and stays "
                             f"positive throughout -- consistent with a real, hypothesis-direction "
                             f"gradient.")
            else:
                lines.append(f"Dose-response ({desc}): raw values are NOT a clean magnitude "
                             f"gradient in the hypothesized direction (|coef| by distance: "
                             f"{', '.join(f'{m:.4f}' for m in mags)}"
                             f"{'; crosses sign' if not same_sign else ''}) -- do not read this as "
                             f"support for the canopy hypothesis just because the raw values happen "
                             f"to be ordered.")

    if urban_res is not None:
        coef = urban_res["model"].params.get("tree_cover")
        p = urban_res["model"].pvalues.get("tree_cover")
        if coef is not None and p is not None:
            caveat = (f" CAVEAT: only {urban_res['n_clusters']} tract(s)/cluster(s) fitting "
                      f"{urban_res['n_params']} parameters -- treat as illustrative, not a finding."
                      if urban_res["n_clusters"] < MIN_CLUSTERS_WARNING else "")
            lines.append(f"Urban restriction: tree cover is {_direction_note(coef, p)}.{caveat}")
    else:
        lines.append("Urban restriction: skipped (too few urban tracts for a trustworthy "
                     "clustered regression -- see Step 4's own message for the count).")

    if placebo_res is not None:
        coef = placebo_res["model"].params.get("tree_cover")
        p = placebo_res["model"].pvalues.get("tree_cover")
        if p is not None:
            is_null = p >= ALPHA
            lines.append(f"The Q3->Q1 placebo (coef={coef:+.5f}, p={p:.4g}) "
                         f"{'is null as predicted' if is_null else 'is NOT null'}, "
                         f"{'strengthening' if is_null else 'weakening'} the directional "
                         f"(growing-season-model-handles-bare-landscapes) story.")

    if ndvi_res is not None:
        amp_coef = ndvi_res["model"].params.get("amplitude")
        amp_p = ndvi_res["model"].pvalues.get("amplitude")
        if amp_coef is not None and amp_p is not None:
            lines.append(f"NDVI-amplitude alternative: {_direction_note(amp_coef, amp_p)}.")

    for line in lines:
        print("- " + line)


# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--years", required=True, nargs="+", type=int)
    p.add_argument("--version", default=None,
                    help="Trained version per row, e.g. 'v1' (default: auto-detect the latest "
                         "existing version under each quarter's artifacts/ dir)")
    p.add_argument("--wealth-csv", default=None, help="Required unless --build-manifest-only")
    p.add_argument("--wealth-col", default="wealth_index_overall_core")
    p.add_argument("--lc-csv", default=None, help="Required unless --build-manifest-only")
    p.add_argument("--ndvi-csv", default=None, help="Optional, enables step 6 -- fine to omit entirely")
    p.add_argument("--preds-manifest", default=None,
                    help="CSV with columns year,train_quarter,eval_quarter,preds_csv. If given, "
                         "used as-is and nothing below in this paragraph applies. Otherwise the "
                         "manifest is built automatically from --data-root-template/"
                         "--base-prefix-template (default: every state's existing on-disk "
                         "convention, OH included -- state_registry.yml not required), or, with "
                         "--use-state-registry, from pipeline_configs/state_registry.yml instead "
                         "(needs a full registry entry: spatial_block_deg/band stats too, not just "
                         "the path templates).")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Build the manifest via pipeline_configs/state_registry.yml instead of "
                         "--data-root-template/--base-prefix-template")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Python .format() string with {{state}}/{{year}}/{{quarter}} "
                         f"(default: {DEFAULT_DATA_ROOT_TEMPLATE!r})")
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE,
                    help=f"Same idea for dataset.prefix (default: {DEFAULT_BASE_PREFIX_TEMPLATE!r})")
    p.add_argument("--build-manifest-only", action="store_true",
                    help="Discover and write preds_manifest.csv, then stop -- skips the regression "
                         "steps, so --wealth-csv/--lc-csv aren't needed. Useful to sanity-check "
                         "which cells were found before those are ready.")
    p.add_argument("--urban-built-up-threshold", type=float, default=0.5)
    p.add_argument("--min-urban-tracts", type=int, default=MIN_CLUSTERS_WARNING,
                    help=f"Skip step 4 (urban restriction) if fewer than this many urban tracts "
                         f"are present -- a clustered regression needs substantially more clusters "
                         f"than parameters to be trustworthy (default: {MIN_CLUSTERS_WARNING}).")
    p.add_argument("--lc-reference-class", default=None,
                    help="Land-cover fractions are compositional (sum to ~1), so every regression "
                         "needs one class explicitly omitted as the reference category -- every "
                         "other class present in --lc-csv is included as a control. Default: "
                         "auto-picks the largest-mean class among whatever's left over after "
                         "tree_cover/cropland/built_up (e.g. Grassland, if present).")
    p.add_argument("--out-dir", default=None, help="Default: ./out_transfer_penalty/<state>")
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"./out_transfer_penalty/{args.state.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preds_manifest:
        manifest = load_manifest_csv(args.preds_manifest)
    elif args.use_state_registry:
        registry = load_registry()
        manifest = build_manifest_from_registry(
            args.state.lower(), args.variable, args.years, args.version, registry
        )
    else:
        manifest = build_manifest_from_templates(
            args.state.lower(), args.variable, args.years, args.version,
            args.data_root_template, args.base_prefix_template,
        )
    if manifest.empty:
        raise SystemExit("No preds CSVs found/loaded -- nothing to analyze. Run the cross-quarter "
                          "validate configs first (generate_cross_validate_config.py), or pass "
                          "--preds-manifest.")
    manifest_path = out_dir / "preds_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path} ({len(manifest)} cell(s))")

    if args.build_manifest_only:
        print("--build-manifest-only set -- stopping here.")
        return

    if not args.wealth_csv or not args.lc_csv:
        raise SystemExit("--wealth-csv and --lc-csv are required unless --build-manifest-only is set.")

    wealth_df = load_wealth(args.wealth_csv, args.wealth_col)
    lc_df = load_landcover(args.lc_csv)
    step0_sanity_check(wealth_df, lc_df)
    features_df = lc_df  # GEOID + every land-cover class present

    reference_class = resolve_reference_class(features_df, args.lc_reference_class)
    all_classes = [c for c in features_df.columns if c != "GEOID"]
    print(f"\n[Land cover] {len(all_classes)} class(es) present: {all_classes}")
    print(f"[Land cover] Reference category (omitted from every regression's right-hand side): "
          f"{reference_class!r} -- every land-cover coefficient below is relative to a tract "
          f"that's entirely {reference_class}.")

    error_long = build_error_table(manifest)

    pen_13 = transfer_penalty(error_long, 1, 3)
    main_res = step2_main_regression(pen_13, features_df, reference_class)
    dose_df, dose_results = step3_dose_response(
        error_long, features_df, out_dir, reference_class, precomputed={(1, 3): main_res}
    )
    urban_res = step4_urban_restriction(error_long, features_df, args.urban_built_up_threshold,
                                         reference_class, args.min_urban_tracts)
    placebo_res = step5_placebo(error_long, features_df, reference_class)
    ndvi_res = step6_ndvi_alternative(pen_13, features_df, args.ndvi_csv, reference_class)

    all_results = [dose_results[(1, 2)], dose_results[(1, 3)], dose_results[(1, 4)],
                   urban_res, placebo_res, ndvi_res]
    summarize(all_results, out_dir)
    print_interpretation(main_res, dose_df, urban_res, placebo_res, ndvi_res)


if __name__ == "__main__":
    main()
