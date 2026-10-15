"""
run_wealth_index.py — Individual State Wealth Index
=====================================================
Computes six income-weighted composite wealth indices for a specified state:

  wealth_index              — 8 variables (Property: Core + Satellite union)  [PRIMARY]
  wealth_index_sat          — 6 variables (Satellite-visible only; includes mobile_home_pct)
  wealth_index_housing_core — 4 variables (ownership + crowding)
  wealth_index_transport    — 2 variables (vehicle access)
  wealth_index_financial    — 4 variables (poverty, unemployment, housing cost burden)
  wealth_index_utilities    — 3 variables (broadband, heating; broadband 2019 only)

WEIGHTING STRATEGY — NATIONAL WEIGHTS:
  Weights and z-score parameters are always computed from the full national
  dataset (73,056 US tracts) and then applied to the target state. 

  Scores are in national units: 0 = US average, +1 = 1 std dev above US average.

Validation thresholds: income r > +0.70  AND  poverty r < -0.70
Independent validation: per_capita_income (never used in weight computation)

Run from the Individual_States/ directory:
  python run_wealth_index.py

To switch state, edit STATE_FIPS / STATE_NAME / STATE_ABBR below.
For the full USA, set STATE_FIPS = None.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Configuration — edit these to change state ────────────────────────────────

STATE_FIPS = '39'       # Ohio=39, Pennsylvania=42, Michigan=26, California=06, Arizona=04
                        # Set to None to run for the full United States
STATE_NAME = 'Ohio'
STATE_ABBR = 'OH'

DATA_DIR   = 'data'
OUTPUT_DIR = os.path.join('output', STATE_ABBR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Index variable sets ───────────────────────────────────────────────────────

# 8-variable Wealth Index 
INDEX_WEALTH = [
    'log_home_value',        # Core + Satellite (shared)
    'owner_pct',             # Core: homeownership rate
    'crowded_pct',           # Core: mild crowding (1.01-1.50 persons/room)
    'severely_crowded_pct',  # Core: severe crowding (1.51+ persons/room)
    'log_gross_rent',        # Satellite: rent level
    'median_rooms',          # Satellite: unit size
    'single_family_pct',     # Satellite: building type
    'median_year_built',     # Satellite: housing age
]

# 6-variable Satellite Index 
INDEX_SAT = [
    'log_home_value',        # visible: lot/building size, condition
    'log_gross_rent',        # visible: neighborhood desirability signals
    'median_rooms',          # visible: building footprint/size
    'single_family_pct',     # visible: building type geometry
    'median_year_built',     # visible: roof condition / architectural era
    'mobile_home_pct',       # visible: mobile home parks visually distinct from all other housing types
]

# 4-variable Housing Core
INDEX_HOUSING_CORE = [
    'log_home_value', 'owner_pct', 'crowded_pct', 'severely_crowded_pct',
]

# 2-variable Transport 
INDEX_TRANSPORT = [
    'pct_no_vehicle', 'pct_2plus_vehicles',
]

# 4-variable Financial
INDEX_FINANCIAL = [
    'poverty_rate', 'unemployment_rate',
    'rent_burdened_30plus_pct', 'owner_burdened_30plus_pct',
]

# 3-variable Utilities
INDEX_UTILITIES = [
    'broadband_pct', 'gas_heat_pct', 'poor_heat_pct',
]


# ── Feature derivation ────────────────────────────────────────────────────────

def calculate_features(df):
    df = df.copy()
    df['log_home_value']      = np.log1p(df['median_home_value'].clip(lower=0))
    df['owner_pct']           = df['owner_occupied_units'] / df['total_housing_units'] * 100
    mild   = (df['owner_crowded_1_01_to_1_50_occupants'] +
              df['renter_crowded_1_01_to_1_50_occupants'])
    severe = (df['owner_severely_crowded_1_51_plus_occupants'] +
              df['renter_severely_crowded_1_51_plus_occupants'])
    df['crowded_pct']          = mild   / df['crowding_total_households'] * 100
    df['severely_crowded_pct'] = severe / df['crowding_total_households'] * 100
    df['log_gross_rent']       = np.log1p(df['median_gross_rent'].clip(lower=0))
    df['single_family_pct']    = (df['units_1_detached'] + df['units_1_attached']) / df['total_units_in_structure'] * 100
    df['poverty_rate']         = df['population_below_poverty_line'] / df['poverty_total_population'] * 100
    with_veh = (df['owner_1_vehicle'] + df['owner_2_vehicles'] + df['owner_3_plus_vehicles'] +
                df['renter_1_vehicle'] + df['renter_2_vehicles'] + df['renter_3_plus_vehicles'])
    df['pct_no_vehicle']     = (df['vehicles_total_households'] - with_veh) / df['vehicles_total_households'] * 100
    df['pct_2plus_vehicles'] = (df['owner_2_vehicles'] + df['owner_3_plus_vehicles'] +
                                df['renter_2_vehicles'] + df['renter_3_plus_vehicles']) / df['vehicles_total_households'] * 100
    df['poor_heat_pct']      = (df['fuel_oil_kerosene_heat'] + df['wood_heat']) / df['heating_total_households'] * 100
    df['gas_heat_pct']       = df['utility_gas_heat'] / df['heating_total_households'] * 100
    df['unemployment_rate']  = df['unemployed'] / df['labor_force_total'] * 100
    rent_b  = (df['rent_burdened_30_34'] + df['rent_burdened_35_39'] +
               df['rent_burdened_40_49'] + df['rent_burdened_50_plus'])
    df['rent_burdened_30plus_pct']  = rent_b / df['rent_burden_universe'] * 100
    owner_b = (df['owner_mortgaged_30_34'] + df['owner_mortgaged_35_39'] +
               df['owner_mortgaged_40_49'] + df['owner_mortgaged_50_plus'])
    df['owner_burdened_30plus_pct'] = owner_b / df['owner_mortgaged_units'] * 100
    if 'broadband_internet' in df.columns and 'internet_total_households' in df.columns:
        df['broadband_pct'] = df['broadband_internet'] / df['internet_total_households'] * 100
    else:
        df['broadband_pct'] = np.nan
    # Units-in-structure breakdown (B25024) — all satellite-visible
    if 'units_2' in df.columns:
        tot = df['total_units_in_structure']
        df['units_2_pct']      = df['units_2']      / tot * 100
        df['units_3_4_pct']    = df['units_3_4']    / tot * 100
        df['units_5_9_pct']    = df['units_5_9']    / tot * 100
        df['units_10_19_pct']  = df['units_10_19']  / tot * 100
        df['units_20_49_pct']  = df['units_20_49']  / tot * 100
        df['units_50plus_pct'] = df['units_50plus'] / tot * 100
        df['mobile_home_pct']  = df['mobile_home']  / tot * 100
    # Additional heating fuels (B25040)
    if 'propane_heat' in df.columns:
        ht = df['heating_total_households']
        df['propane_pct']  = df['propane_heat']  / ht * 100
        df['no_fuel_pct']  = df['no_fuel_heat']  / ht * 100
        df['solar_heat_pct'] = df['solar_heat']  / ht * 100  # in dataset only; excluded from all feature sets
    # Commute mode (B08301)
    if 'commute_universe' in df.columns:
        cu = df['commute_universe']
        df['car_alone_pct']       = df['commute_car_alone']  / cu * 100
        df['transit_pct']         = df['commute_transit']    / cu * 100
        df['walk_pct']            = df['commute_walk']       / cu * 100
        df['work_from_home_pct']  = df['commute_work_home']  / cu * 100
    return df


# ── National weight + scaler computation ─────────────────────────────────────

def compute_national_weights(nat_df, features):
    """Compute income-correlation weights from the national dataset."""
    avail = [f for f in features if f in nat_df.columns and nat_df[f].notna().sum() > 1000]
    v = nat_df[avail + ['median_household_income']].dropna()
    return {f: v[f].corr(v['median_household_income']) for f in avail}


def fit_national_scaler(nat_df, features):
    """Fit StandardScaler on the national dataset."""
    avail = [f for f in features if f in nat_df.columns and nat_df[f].notna().sum() > 1000]
    clean = nat_df[avail].dropna()
    scaler = StandardScaler().fit(clean)
    return scaler, avail


def compute_national_score_params(nat_df, weights, scaler, avail_features):
    """
    Compute the national mean and std of the weighted composite score.
    Used to re-standardize any dataset so that 0 = US average and 1 = 1 national std dev.
    This ensures OH file scores == USA file Ohio-row scores exactly.
    """
    clean = nat_df[avail_features].dropna()
    z = pd.DataFrame(scaler.transform(clean), index=clean.index, columns=avail_features)
    raw_score = sum(weights[f] * z[f] for f in avail_features)
    return raw_score.mean(), raw_score.std()


# ── Apply national weights to target data ────────────────────────────────────

def apply_index(df, weights, scaler, avail_features, nat_mean, nat_std, index_col):
    """
    Apply pre-fitted national scaler and pre-computed national weights to df,
    then re-standardize using the NATIONAL distribution of scores (not state-level).
    Result: 0 = US average, 1 = 1 national std dev above average.
    Ohio scores in _OH files will exactly match Ohio rows in _USA files.
    """
    clean = df[avail_features].dropna()
    z = pd.DataFrame(
        scaler.transform(clean),
        index=clean.index, columns=avail_features
    )
    raw_score = sum(weights[f] * z[f] for f in avail_features)
    score = (raw_score - nat_mean) / nat_std   # re-standardize with national params

    df_out = df.copy()
    df_out[index_col] = np.nan
    df_out.loc[clean.index, index_col] = score.values

    quintile_col = index_col.replace('wealth_index', 'wealth_quintile')
    mask = df_out[index_col].notna()
    df_out[quintile_col] = np.nan
    df_out.loc[mask, quintile_col] = pd.qcut(
        df_out.loc[mask, index_col], q=5,
        labels=['Poorest', 'Poor', 'Middle', 'Rich', 'Richest'],
        duplicates='drop'
    ).astype(str)

    return df_out


# ── Validation ────────────────────────────────────────────────────────────────

def validate(df, index_col, label):
    v  = df[[index_col, 'median_household_income', 'poverty_rate']].dropna()
    ir = v[index_col].corr(v['median_household_income'])
    pr = v[index_col].corr(v['poverty_rate'])
    si = 'PASS' if ir >  0.70 else 'FAIL'
    sp = 'PASS' if pr < -0.70 else 'FAIL'
    pci_r = np.nan
    if 'per_capita_income' in df.columns:
        v2 = df[[index_col, 'per_capita_income']].dropna()
        if len(v2) > 10:
            pci_r = v2[index_col].corr(v2['per_capita_income'])
    passed = '  PASS' if (ir > 0.70 and pr < -0.70) else '  FAIL'
    pci_s  = f'  pci r={pci_r:+.3f}' if not np.isnan(pci_r) else ''
    print(f'    {label:<40} income r={ir:+.3f} {si}   poverty r={pr:+.3f} {sp}{pci_s}{passed}')
    return ir, pr, pci_r


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_diagnostic_plot(df19, df14, index_col, title, output_dir, state_abbr):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title}\n({state_abbr} — national weights)', fontsize=12, fontweight='bold')

    for row, (df, year) in enumerate([(df19, '2019'), (df14, '2014')]):
        ax = axes[row, 0]
        df[index_col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(0, color='red', lw=1, ls='--', label='US avg = 0')
        ax.set_title(f'Score Distribution ({year})')
        ax.set_xlabel('Wealth Score (national units)'); ax.set_ylabel('Tracts'); ax.legend()

        ax = axes[row, 1]
        v = df[[index_col, 'median_household_income']].dropna()
        ax.scatter(v['median_household_income'], v[index_col], alpha=0.3, s=8, color='steelblue')
        ir = v[index_col].corr(v['median_household_income'])
        ax.set_title(f'vs Median HH Income ({year})  r={ir:+.3f}')
        ax.set_xlabel('Median HH Income ($)'); ax.set_ylabel('Wealth Score')

        ax = axes[row, 2]
        v2 = df[[index_col, 'poverty_rate']].dropna()
        ax.scatter(v2['poverty_rate'], v2[index_col], alpha=0.3, s=8, color='tomato')
        pr = v2[index_col].corr(v2['poverty_rate'])
        ax.set_title(f'vs Poverty Rate ({year})  r={pr:+.3f}')
        ax.set_xlabel('Poverty Rate (%)'); ax.set_ylabel('Wealth Score')

    plt.tight_layout()
    tag   = index_col.replace('wealth_index_', 'sat_').replace('wealth_index', 'wealth')
    fname = os.path.join(output_dir, f'{tag}_{state_abbr}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved plot: {fname}')


def save_weights_plot(weights19, weights14, index_col, title, output_dir, state_abbr):
    features = list(weights19.keys())
    w19 = [weights19[f] for f in features]
    w14 = [weights14.get(f, 0) for f in features]
    x   = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x + 0.2, w19, 0.4, label='2019 national weights', color='steelblue', alpha=0.85)
    ax.barh(x - 0.2, w14, 0.4, label='2014 national weights', color='tomato',    alpha=0.85)
    ax.set_yticks(x); ax.set_yticklabels(features)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Pearson r with median_household_income (national)')
    ax.set_title(f'{title} — National Feature Weights')
    ax.legend()
    plt.tight_layout()
    tag   = index_col.replace('wealth_index_', 'sat_weights_').replace('wealth_index', 'weights')
    fname = os.path.join(output_dir, f'{tag}_{state_abbr}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved plot: {fname}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{"="*65}')
    if STATE_FIPS:
        print(f'  Individual State Wealth Index — {STATE_NAME} ({STATE_ABBR})')
        print(f'  Weights: national (73,056 US tracts)')
    else:
        print(f'  Wealth Index — United States (all tracts)')
        print(f'  Weights: national (same dataset)')
    print(f'{"="*65}')

    # ── Step 1: Load full national data and compute national weights/scalers ──
    print(f'\nStep 1: Computing national weights from full US dataset...')
    raw19_nat = pd.read_csv(f'{DATA_DIR}/acs_wealth_variables_2015_2019_US.csv')
    raw14_nat = pd.read_csv(f'{DATA_DIR}/acs_wealth_variables_2010_2014_US.csv')
    nat19 = calculate_features(raw19_nat)
    nat14 = calculate_features(raw14_nat)
    print(f'  National 2019: {len(nat19):,} tracts')
    print(f'  National 2014: {len(nat14):,} tracts')

    # Wealth Index — national weights, scalers, and score normalization params
    w_wealth19 = compute_national_weights(nat19, INDEX_WEALTH)
    w_wealth14 = compute_national_weights(nat14, INDEX_WEALTH)
    scaler_wealth19, avail_wealth19 = fit_national_scaler(nat19, INDEX_WEALTH)
    scaler_wealth14, avail_wealth14 = fit_national_scaler(nat14, INDEX_WEALTH)
    nat_mean_w19, nat_std_w19 = compute_national_score_params(nat19, w_wealth19, scaler_wealth19, avail_wealth19)
    nat_mean_w14, nat_std_w14 = compute_national_score_params(nat14, w_wealth14, scaler_wealth14, avail_wealth14)

    # Satellite Index — national weights, scalers, and score normalization params
    w_sat19 = compute_national_weights(nat19, INDEX_SAT)
    w_sat14 = compute_national_weights(nat14, INDEX_SAT)
    scaler_sat19, avail_sat19 = fit_national_scaler(nat19, INDEX_SAT)
    scaler_sat14, avail_sat14 = fit_national_scaler(nat14, INDEX_SAT)
    nat_mean_s19, nat_std_s19 = compute_national_score_params(nat19, w_sat19, scaler_sat19, avail_sat19)
    nat_mean_s14, nat_std_s14 = compute_national_score_params(nat14, w_sat14, scaler_sat14, avail_sat14)

    # Housing Core Index
    w_hc19 = compute_national_weights(nat19, INDEX_HOUSING_CORE)
    w_hc14 = compute_national_weights(nat14, INDEX_HOUSING_CORE)
    scaler_hc19, avail_hc19 = fit_national_scaler(nat19, INDEX_HOUSING_CORE)
    scaler_hc14, avail_hc14 = fit_national_scaler(nat14, INDEX_HOUSING_CORE)
    nat_mean_hc19, nat_std_hc19 = compute_national_score_params(nat19, w_hc19, scaler_hc19, avail_hc19)
    nat_mean_hc14, nat_std_hc14 = compute_national_score_params(nat14, w_hc14, scaler_hc14, avail_hc14)

    # Transport Index
    w_tr19 = compute_national_weights(nat19, INDEX_TRANSPORT)
    w_tr14 = compute_national_weights(nat14, INDEX_TRANSPORT)
    scaler_tr19, avail_tr19 = fit_national_scaler(nat19, INDEX_TRANSPORT)
    scaler_tr14, avail_tr14 = fit_national_scaler(nat14, INDEX_TRANSPORT)
    nat_mean_tr19, nat_std_tr19 = compute_national_score_params(nat19, w_tr19, scaler_tr19, avail_tr19)
    nat_mean_tr14, nat_std_tr14 = compute_national_score_params(nat14, w_tr14, scaler_tr14, avail_tr14)

    # Financial Index
    w_fin19 = compute_national_weights(nat19, INDEX_FINANCIAL)
    w_fin14 = compute_national_weights(nat14, INDEX_FINANCIAL)
    scaler_fin19, avail_fin19 = fit_national_scaler(nat19, INDEX_FINANCIAL)
    scaler_fin14, avail_fin14 = fit_national_scaler(nat14, INDEX_FINANCIAL)
    nat_mean_fin19, nat_std_fin19 = compute_national_score_params(nat19, w_fin19, scaler_fin19, avail_fin19)
    nat_mean_fin14, nat_std_fin14 = compute_national_score_params(nat14, w_fin14, scaler_fin14, avail_fin14)

    # Utilities Index
    w_util19 = compute_national_weights(nat19, INDEX_UTILITIES)
    w_util14 = compute_national_weights(nat14, INDEX_UTILITIES)
    scaler_util19, avail_util19 = fit_national_scaler(nat19, INDEX_UTILITIES)
    scaler_util14, avail_util14 = fit_national_scaler(nat14, INDEX_UTILITIES)
    nat_mean_util19, nat_std_util19 = compute_national_score_params(nat19, w_util19, scaler_util19, avail_util19)
    nat_mean_util14, nat_std_util14 = compute_national_score_params(nat14, w_util14, scaler_util14, avail_util14)

    print(f'\n  National weights — Wealth Index (8 var), 2019:')
    for f, w in sorted(w_wealth19.items(), key=lambda x: -abs(x[1])):
        print(f'    {f:<35} {w:+.3f}')

    print(f'\n  National weights — Wealth Index Sat (6 var), 2019:')
    for f, w in sorted(w_sat19.items(), key=lambda x: -abs(x[1])):
        print(f'    {f:<35} {w:+.3f}')

    # ── Step 2: Filter to target state (or keep all if STATE_FIPS is None) ────
    print(f'\nStep 2: Filtering to {STATE_NAME}...')
    if STATE_FIPS:
        acs19 = nat19[nat19['state'].astype(str).str.zfill(2) == STATE_FIPS.zfill(2)].copy()
        acs14 = nat14[nat14['state'].astype(str).str.zfill(2) == STATE_FIPS.zfill(2)].copy()
    else:
        acs19 = nat19.copy()
        acs14 = nat14.copy()
    print(f'  2015–2019: {len(acs19):,} tracts')
    print(f'  2010–2014: {len(acs14):,} tracts')

    # ── Step 3: Apply national weights to state data ──────────────────────────
    print(f'\nStep 3: Applying national weights to {STATE_NAME} tracts...')

    # Wealth Index
    acs19 = apply_index(acs19, w_wealth19, scaler_wealth19, avail_wealth19, nat_mean_w19, nat_std_w19, 'wealth_index')
    acs14 = apply_index(acs14, w_wealth14, scaler_wealth14, avail_wealth14, nat_mean_w14, nat_std_w14, 'wealth_index')

    # Satellite Index
    acs19 = apply_index(acs19, w_sat19, scaler_sat19, avail_sat19, nat_mean_s19, nat_std_s19, 'wealth_index_sat')
    acs14 = apply_index(acs14, w_sat14, scaler_sat14, avail_sat14, nat_mean_s14, nat_std_s14, 'wealth_index_sat')

    # Housing Core Index
    acs19 = apply_index(acs19, w_hc19, scaler_hc19, avail_hc19, nat_mean_hc19, nat_std_hc19, 'wealth_index_housing_core')
    acs14 = apply_index(acs14, w_hc14, scaler_hc14, avail_hc14, nat_mean_hc14, nat_std_hc14, 'wealth_index_housing_core')

    # Transport Index
    acs19 = apply_index(acs19, w_tr19, scaler_tr19, avail_tr19, nat_mean_tr19, nat_std_tr19, 'wealth_index_transport')
    acs14 = apply_index(acs14, w_tr14, scaler_tr14, avail_tr14, nat_mean_tr14, nat_std_tr14, 'wealth_index_transport')

    # Financial Index
    acs19 = apply_index(acs19, w_fin19, scaler_fin19, avail_fin19, nat_mean_fin19, nat_std_fin19, 'wealth_index_financial')
    acs14 = apply_index(acs14, w_fin14, scaler_fin14, avail_fin14, nat_mean_fin14, nat_std_fin14, 'wealth_index_financial')

    # Utilities Index
    acs19 = apply_index(acs19, w_util19, scaler_util19, avail_util19, nat_mean_util19, nat_std_util19, 'wealth_index_utilities')
    acs14 = apply_index(acs14, w_util14, scaler_util14, avail_util14, nat_mean_util14, nat_std_util14, 'wealth_index_utilities')

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    print(f'\n{"─"*65}')
    print(f'  Validation — {STATE_NAME} (national weights)')
    print(f'{"─"*65}')
    print('\n  Wealth Index (8 var):')
    r_wi19 = validate(acs19, 'wealth_index',     f'{STATE_ABBR} 2019')
    r_wi14 = validate(acs14, 'wealth_index',     f'{STATE_ABBR} 2014')
    print('\n  Wealth Index Satellite (6 var):')
    r_sat19 = validate(acs19, 'wealth_index_sat', f'{STATE_ABBR} 2019')
    r_sat14 = validate(acs14, 'wealth_index_sat', f'{STATE_ABBR} 2014')
    print('\n  Wealth Index Housing Core (4 var):')
    r_hc19 = validate(acs19, 'wealth_index_housing_core', f'{STATE_ABBR} 2019')
    r_hc14 = validate(acs14, 'wealth_index_housing_core', f'{STATE_ABBR} 2014')
    print('\n  Wealth Index Transport (2 var):')
    r_tr19 = validate(acs19, 'wealth_index_transport', f'{STATE_ABBR} 2019')
    r_tr14 = validate(acs14, 'wealth_index_transport', f'{STATE_ABBR} 2014')
    print('\n  Wealth Index Financial (4 var):')
    r_fin19 = validate(acs19, 'wealth_index_financial', f'{STATE_ABBR} 2019')
    r_fin14 = validate(acs14, 'wealth_index_financial', f'{STATE_ABBR} 2014')
    print('\n  Wealth Index Utilities (3 var; broadband 2019 only):')
    r_util19 = validate(acs19, 'wealth_index_utilities', f'{STATE_ABBR} 2019')
    r_util14 = validate(acs14, 'wealth_index_utilities', f'{STATE_ABBR} 2014')

    # ── Step 5: Cross-period weight stability ─────────────────────────────────
    # Tests whether the national feature WEIGHTS are stable across periods.
    # Each period is z-scored with its OWN national scaler (period-appropriate params),
    # then the OTHER period's weights are applied. High income r = weights are stable.
    # (Applying the 2014 scaler to 2019 raw data fails because home prices shifted
    # substantially 2014→2019; that tests parameter transfer, not weight stability.)
    print(f'\n{"─"*65}')
    print('  Cross-period weight stability (own-period z-scores, cross-period weights)')
    print(f'{"─"*65}')
    for label, apply_df, own_scaler, own_avail, cross_weights in [
        ('2019 z-scores × 2014 nat weights', acs19, scaler_wealth19, avail_wealth19, w_wealth14),
        ('2014 z-scores × 2019 nat weights', acs14, scaler_wealth14, avail_wealth14, w_wealth19),
    ]:
        common = [f for f in own_avail if f in cross_weights]
        valid_rows = apply_df[common].dropna().index
        z_all = pd.DataFrame(
            own_scaler.transform(apply_df.loc[valid_rows, own_avail]),
            index=valid_rows, columns=own_avail
        )
        z = z_all[common]
        s = sum(cross_weights[f] * z[f] for f in common)
        tmp = apply_df.loc[valid_rows, ['median_household_income', 'poverty_rate']].copy()
        tmp['cross'] = s.values
        v = tmp.dropna()
        ir = v['cross'].corr(v['median_household_income'])
        pr = v['cross'].corr(v['poverty_rate'])
        print(f'    {label}:  income r={ir:+.3f}   poverty r={pr:+.3f}')

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print(f'\n{"─"*65}')
    print(f'  Summary — {STATE_NAME} with national weights')
    print(f'{"─"*65}')
    print(f'  {"Index":<30} {"Year":<6} {"Income r":>10} {"Poverty r":>10} {"PCI r":>8} {"Pass?":>6}')
    print(f'  {"─"*30} {"─"*6} {"─"*10} {"─"*10} {"─"*8} {"─"*6}')
    for label, r, year in [
        ('Wealth Index (8 var)',          r_wi19,  '2019'),
        ('Wealth Index (8 var)',          r_wi14,  '2014'),
        ('Wealth Index Sat (6 var)',      r_sat19, '2019'),
        ('Wealth Index Sat (6 var)',      r_sat14, '2014'),
        ('Housing Core (4 var)',          r_hc19,  '2019'),
        ('Housing Core (4 var)',          r_hc14,  '2014'),
        ('Transport (2 var)',             r_tr19,  '2019'),
        ('Transport (2 var)',             r_tr14,  '2014'),
        ('Financial (4 var)',             r_fin19, '2019'),
        ('Financial (4 var)',             r_fin14, '2014'),
        ('Utilities (3 var)',             r_util19,'2019'),
        ('Utilities (3 var; heat only)',  r_util14,'2014'),
    ]:
        ir, pr, pci = r
        passed = 'PASS' if (ir > 0.70 and pr < -0.70) else 'FAIL'
        pci_s  = f'{pci:+.3f}' if not np.isnan(pci) else '  n/a'
        print(f'  {label:<30} {year:<6} {ir:>+10.3f} {pr:>+10.3f} {pci_s:>8} {passed:>6}')

    print(f'\n  Mean wealth score ({STATE_NAME} relative to US average):')
    for df, year in [(acs19, '2019'), (acs14, '2014')]:
        wi  = df['wealth_index'].mean()
        sat = df['wealth_index_sat'].mean()
        print(f'    {year}  Wealth Index: {wi:+.3f}  |  Wealth Index Sat: {sat:+.3f}  (0 = US avg)')

    print(f'\n  Correlation between Wealth Index and Wealth Index Sat:')
    for df, year in [(acs19, '2019'), (acs14, '2014')]:
        v = df[['wealth_index', 'wealth_index_sat']].dropna()
        r = v['wealth_index'].corr(v['wealth_index_sat'])
        print(f'    {year}: r={r:+.3f}  ({len(v):,} tracts)')

    # ── Step 7: Save output CSVs ──────────────────────────────────────────────
    print(f'\n{"─"*65}')
    print('  Saving output CSVs...')
    for df, year in [(acs19, '2019'), (acs14, '2014')]:
        out_path = os.path.join(OUTPUT_DIR, f'wealth_indices_{year}_{STATE_ABBR}.csv')
        df.to_csv(out_path, index=False)
        print(f'  Saved: {out_path}  ({len(df):,} tracts × {len(df.columns)} cols)')

    # ── Step 8: Plots ─────────────────────────────────────────────────────────
    save_diagnostic_plot(acs19, acs14, 'wealth_index',
                         'Wealth Index (8 variables)', OUTPUT_DIR, STATE_ABBR)
    save_diagnostic_plot(acs19, acs14, 'wealth_index_sat',
                         'Wealth Index Satellite (6 variables)', OUTPUT_DIR, STATE_ABBR)
    save_weights_plot(w_wealth19, w_wealth14, 'wealth_index',
                      'Wealth Index (8 variables)', OUTPUT_DIR, STATE_ABBR)
    save_weights_plot(w_sat19, w_sat14, 'wealth_index_sat',
                      'Wealth Index Satellite (6 variables)', OUTPUT_DIR, STATE_ABBR)
    save_diagnostic_plot(acs19, acs14, 'wealth_index_housing_core',
                         'Wealth Index Housing Core (4 variables)', OUTPUT_DIR, STATE_ABBR)
    save_weights_plot(w_hc19, w_hc14, 'wealth_index_housing_core',
                      'Wealth Index Housing Core (4 variables)', OUTPUT_DIR, STATE_ABBR)
    save_diagnostic_plot(acs19, acs14, 'wealth_index_transport',
                         'Wealth Index Transport (2 variables)', OUTPUT_DIR, STATE_ABBR)
    save_weights_plot(w_tr19, w_tr14, 'wealth_index_transport',
                      'Wealth Index Transport (2 variables)', OUTPUT_DIR, STATE_ABBR)
    save_diagnostic_plot(acs19, acs14, 'wealth_index_financial',
                         'Wealth Index Financial (4 variables)', OUTPUT_DIR, STATE_ABBR)
    save_weights_plot(w_fin19, w_fin14, 'wealth_index_financial',
                      'Wealth Index Financial (4 variables)', OUTPUT_DIR, STATE_ABBR)
    save_diagnostic_plot(acs19, acs14, 'wealth_index_utilities',
                         'Wealth Index Utilities (3 variables; broadband 2019 only)', OUTPUT_DIR, STATE_ABBR)
    save_weights_plot(w_util19, w_util14, 'wealth_index_utilities',
                      'Wealth Index Utilities (3 variables)', OUTPUT_DIR, STATE_ABBR)

    print(f'\n{"="*65}')
    print(f'  Done. Output in: {OUTPUT_DIR}/')
    print(f'{"="*65}\n')


if __name__ == '__main__':
    main()
